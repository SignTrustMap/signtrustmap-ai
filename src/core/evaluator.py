from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
import torch
from src.infrastructure.db import sync_engine
from src.infrastructure.models import EvaluationEmbedding, ModelClass, ModelEvaluationMetric


class Evaluator:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    def load_evaluation_dataset(self, split: str = "TEST") -> Tuple[torch.Tensor, torch.Tensor]:
        split_upper = split.upper()
        with Session(sync_engine) as session:
            stmt = (
                select(ModelClass.class_index, EvaluationEmbedding.embedding)
                .join(ModelClass, EvaluationEmbedding.class_id == ModelClass.id)
                .where(EvaluationEmbedding.split == split_upper)
                .order_by(EvaluationEmbedding.id)
            )
            rows = session.execute(stmt).fetchall()

        if not rows:
            logger.warning(f"No evaluation samples found for split: {split_upper}")
            return torch.empty((0, 512), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

        targets = torch.tensor([r[0] for r in rows], dtype=torch.long)
        embs_np = np.array([r[1] for r in rows], dtype=np.float32)
        features = torch.tensor(embs_np, dtype=torch.float32)
        features = features / torch.clamp(features.norm(dim=-1, keepdim=True), min=1e-8)
        logger.debug(f"Loaded {len(targets)} samples for split {split_upper} from PostgreSQL")
        return features, targets

    def compute_metrics(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        latency_ms: float = 0.0,
    ) -> Dict[str, float]:
        if logits.numel() == 0 or targets.numel() == 0:
            return {
                "top1_accuracy": 0.0,
                "top5_accuracy": 0.0,
                "macro_f1": 0.0,
                "latency_ms": latency_ms,
            }

        preds = logits.argmax(dim=-1)
        top1_acc = float((preds == targets).float().mean().item() * 100.0)

        k = min(5, logits.shape[-1])
        _, top5_idx = logits.topk(k, dim=-1)
        top5_acc = float((top5_idx == targets.unsqueeze(-1)).any(dim=-1).float().mean().item() * 100.0)

        num_classes = logits.shape[-1]
        f1_list = []
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().float()
            fp = ((preds == c) & (targets != c)).sum().float()
            fn = ((preds != c) & (targets == c)).sum().float()
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2.0 * prec * rec / (prec + rec + 1e-8)
            f1_list.append(f1.item())
        macro_f1 = float(sum(f1_list) / max(1, len(f1_list)))

        return {
            "top1_accuracy": round(top1_acc, 2),
            "top5_accuracy": round(top5_acc, 2),
            "macro_f1": round(macro_f1, 4),
            "latency_ms": round(latency_ms, 4),
        }

    def evaluate_classifier_weights(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor,
        split: str = "TEST",
        batch_size: int = 1024,
    ) -> Dict[str, float]:
        features, targets = self.load_evaluation_dataset(split=split)
        n_samples = len(targets)
        if n_samples == 0:
            return {"top1_accuracy": 0.0, "top5_accuracy": 0.0, "macro_f1": 0.0, "latency_ms": 0.0}

        dev = self.device
        w = weights.to(dev).float()
        b = bias.to(dev).float()

        logits_list = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for i in range(0, n_samples, batch_size):
                batch_feat = features[i : i + batch_size].to(dev)
                batch_logits = batch_feat @ w + b
                logits_list.append(batch_logits.cpu())

        total_time_ms = (time.perf_counter() - t0) * 1000.0
        latency_ms = total_time_ms / max(1, n_samples)

        all_logits = torch.cat(logits_list, dim=0)
        metrics = self.compute_metrics(all_logits, targets, latency_ms=latency_ms)
        logger.info(
            f"Evaluation on {split} ({n_samples} samples): "
            f"Top1={metrics['top1_accuracy']}% Top5={metrics['top5_accuracy']}% F1={metrics['macro_f1']} Latency={metrics['latency_ms']}ms"
        )
        return metrics

    def persist_metrics(
        self,
        model_id: int,
        metrics: Dict[str, float],
        split: str = "test",
        note: Optional[str] = None,
    ) -> None:
        split_lower = split.lower()
        now = datetime.now(timezone.utc)
        records = [
            {
                "model_id": model_id,
                "metric_name": k,
                "metric_value": v,
                "split": split_lower,
                "note": note,
                "created_at": now,
            }
            for k, v in metrics.items()
        ]

        with Session(sync_engine) as session:
            for rec in records:
                stmt = insert(ModelEvaluationMetric).values(**rec)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["model_id", "metric_name", "split"],
                    set_={
                        "metric_value": stmt.excluded.metric_value,
                        "note": stmt.excluded.note,
                        "created_at": stmt.excluded.created_at,
                    },
                )
                session.execute(stmt)
            session.commit()
        logger.debug(f"Persisted {len(records)} evaluation metrics for model_id={model_id} split={split_lower}")

    def evaluate_and_persist(
        self,
        model_id: int,
        weights: torch.Tensor,
        bias: torch.Tensor,
        split: str = "TEST",
        note: Optional[str] = None,
    ) -> Dict[str, float]:
        metrics = self.evaluate_classifier_weights(weights=weights, bias=bias, split=split)
        self.persist_metrics(model_id=model_id, metrics=metrics, split=split, note=note)
        return metrics

    def get_metrics_for_model(self, model_id: int, split: Optional[str] = None) -> List[Dict[str, Any]]:
        with Session(sync_engine) as session:
            stmt = select(ModelEvaluationMetric).where(ModelEvaluationMetric.model_id == model_id)
            if split:
                stmt = stmt.where(ModelEvaluationMetric.split == split.lower())
            stmt = stmt.order_by(ModelEvaluationMetric.id.asc())
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "metric_name": r.metric_name,
                    "metric_value": r.metric_value,
                    "split": r.split,
                    "note": r.note,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]


evaluator = Evaluator()
