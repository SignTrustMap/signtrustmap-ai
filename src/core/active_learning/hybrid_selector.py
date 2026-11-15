from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.config_manager import config_manager
from src.core.few_shot_manager import few_shot_manager
from src.core.model_manager import model_manager
from src.core.active_learning.strategies.fdal import FDALSelector
from src.core.active_learning.strategies.r3_cec import R3CECSelector
from src.infrastructure.db import get_sync_db
from src.infrastructure.models import (
    ActiveLearningContextualPrior,
    ActiveLearningRecord,
    CandidateEmbedding,
    ModelVersion,
)


class HybridALSelector:
    def __init__(
        self,
        fdal_selector: Optional[FDALSelector] = None,
        r3_cec_selector: Optional[R3CECSelector] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.device = torch.device(device if device is not None else "cpu")
        self.fdal_selector = fdal_selector or FDALSelector(device=self.device)
        self.r3_cec_selector = r3_cec_selector or R3CECSelector()
        self._active_classifier_model_id: Optional[int] = None

    def get_active_classifier_model_id(self) -> int:
        if self._active_classifier_model_id is not None:
            return self._active_classifier_model_id
        try:
            with get_sync_db() as session:
                stmt = select(ModelVersion.id).where(
                    ModelVersion.model_type == "CLASSIFIER",
                    ModelVersion.is_active == True,
                ).limit(1)
                row = session.execute(stmt).scalar_one_or_none()
                if row is not None:
                    self._active_classifier_model_id = int(row)
                    return self._active_classifier_model_id
        except Exception as e:
            logger.warning(f"Could not fetch active classifier model_id: {e}")
        return 2

    def get_timo_parameters(self, model_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        classifier = model_manager.get_classifier()
        if classifier is not None:
            try:
                w, b = classifier.get_timo_parameters()
                if w is not None and b is not None:
                    return w.to(self.device), b.to(self.device)
            except Exception as e:
                logger.warning(f"Error fetching TIMO parameters from loaded classifier: {e}")

        w_cache, b_cache = few_shot_manager.get_timo_classifier(model_id=model_id or self.get_active_classifier_model_id())
        if w_cache is not None and b_cache is not None:
            return w_cache.to(self.device), b_cache.to(self.device)

        w_built, b_built = few_shot_manager.build_timo_classifier(
            model_id=model_id or self.get_active_classifier_model_id(),
            save_cache=True,
        )
        return w_built.to(self.device), b_built.to(self.device)

    def get_or_create_prior_snapshot(
        self,
        run_id: str,
        model_id: int,
        probabilities: np.ndarray,
        top_n: int = 5,
    ) -> np.ndarray:
        with get_sync_db() as session:
            stmt = select(ActiveLearningContextualPrior).where(
                ActiveLearningContextualPrior.run_id == run_id,
                ActiveLearningContextualPrior.model_id == model_id,
            ).limit(1)
            existing = session.execute(stmt).scalar_one_or_none()
            if existing is not None:
                logger.debug(f"Loaded frozen contextual prior snapshot for run_id='{run_id}', model_id={model_id}")
                return np.asarray(existing.p0_prior, dtype=np.float64)

            _, p0 = self.r3_cec_selector.calibrate_probabilities(probabilities, top_n=top_n)
            pool_size = int(probabilities.shape[0])

            insert_stmt = (
                pg_insert(ActiveLearningContextualPrior)
                .values(
                    run_id=run_id,
                    model_id=model_id,
                    pool_size=pool_size,
                    top_n_per_class=int(top_n),
                    p0_prior=p0.tolist(),
                )
                .on_conflict_do_nothing(index_elements=["run_id", "model_id"])
            )
            session.execute(insert_stmt)
            session.commit()
            logger.info(
                f"Created contextual prior snapshot for run_id='{run_id}', "
                f"model_id={model_id}, pool_size={pool_size}, top_n={top_n}"
            )
            return p0

    def score_candidates_by_ids(
        self,
        candidate_ids: List[Union[str, uuid.UUID]],
        strategy: str = "fdal",
        save_db: bool = True,
    ) -> List[Dict[str, Any]]:
        if not candidate_ids:
            return []

        clean_ids = [uuid.UUID(str(c)) for c in candidate_ids]
        model_id = self.get_active_classifier_model_id()
        timo_weights, timo_bias = self.get_timo_parameters(model_id=model_id)
        class_anchors = self.fdal_selector.load_class_anchors(model_id=model_id)

        candidates_data = []
        with get_sync_db() as session:
            stmt = (
                select(
                    CandidateEmbedding.candidate_id,
                    CandidateEmbedding.embedding,
                    CandidateEmbedding.s3_key,
                    ActiveLearningRecord.box_xyxy,
                    ActiveLearningRecord.detect_confidence,
                    ActiveLearningRecord.classify_confidence,
                )
                .outerjoin(ActiveLearningRecord, CandidateEmbedding.candidate_id == ActiveLearningRecord.candidate_id)
                .where(CandidateEmbedding.candidate_id.in_(clean_ids))
            )
            rows = session.execute(stmt).fetchall()
            seen_cids = set()
            for r in rows:
                cid = r[0]
                if cid in seen_cids or r[1] is None:
                    continue
                seen_cids.add(cid)
                candidates_data.append({
                    "candidate_id": cid,
                    "embedding": np.asarray(r[1], dtype=np.float32),
                    "s3_key": r[2],
                    "box_xyxy": r[3] or [0.0, 0.0, 0.0, 0.0],
                    "detect_confidence": float(r[4] or 0.0),
                    "classify_confidence": float(r[5] or 0.0) if r[5] is not None else None,
                })

        if not candidates_data:
            logger.warning("No visual embeddings found for requested candidate_ids.")
            return []

        embs = np.stack([c["embedding"] for c in candidates_data], axis=0)
        embs_tensor = torch.from_numpy(embs).to(self.device)
        logits = torch.matmul(embs_tensor, timo_weights) + timo_bias
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        pred_classes = torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()

        scored_records = []
        if strategy == "fdal":
            fdal_res = self.fdal_selector.score_candidates_batch(
                visual_embeddings=embs,
                predicted_classes=pred_classes,
                class_probabilities=probs,
                timo_weights=timo_weights,
                timo_bias=timo_bias,
                anchors=class_anchors,
            )
            for idx, c in enumerate(candidates_data):
                f_item = fdal_res[idx]
                acq = float(f_item["fdal_priority"])
                scored_records.append({
                    "candidate_id": str(c["candidate_id"]),
                    "class_id": int(pred_classes[idx]),
                    "acquisition_score": acq,
                    "flip_count": int(f_item["flip_count"]),
                    "flip_classes": list(f_item["flipped_anchor_classes"]),
                    "calibrated_entropy": float(f_item["fallback_entropy"]),
                    "detect_confidence": c["detect_confidence"],
                    "classify_confidence": float(probs[idx, pred_classes[idx]]),
                    "box_xyxy": c["box_xyxy"],
                })
        else:
            _, p0 = self.r3_cec_selector.calibrate_probabilities(probs)
            h_hat, _, _ = self.r3_cec_selector.compute_calibrated_uncertainty(
                features=embs,
                probabilities=probs,
                reference_prior=p0,
            )
            for idx, c in enumerate(candidates_data):
                acq = float(h_hat[idx])
                scored_records.append({
                    "candidate_id": str(c["candidate_id"]),
                    "class_id": int(pred_classes[idx]),
                    "acquisition_score": acq,
                    "flip_count": 0,
                    "flip_classes": [],
                    "calibrated_entropy": acq,
                    "detect_confidence": c["detect_confidence"],
                    "classify_confidence": float(probs[idx, pred_classes[idx]]),
                    "box_xyxy": c["box_xyxy"],
                })

        if save_db:
            with get_sync_db() as session:
                for s in scored_records:
                    cid = uuid.UUID(s["candidate_id"])
                    stmt = select(ActiveLearningRecord).where(
                        ActiveLearningRecord.candidate_id == cid,
                        ActiveLearningRecord.model_id == model_id,
                    ).limit(1)
                    existing = session.execute(stmt).scalar_one_or_none()
                    if existing:
                        existing.strategy = strategy
                        existing.acquisition_score = s["acquisition_score"]
                        existing.flip_count = s["flip_count"]
                        existing.flip_classes = s["flip_classes"]
                        existing.calibrated_entropy = s["calibrated_entropy"]
                        existing.class_id = s["class_id"]
                        existing.classify_confidence = s["classify_confidence"]
                    else:
                        rec = ActiveLearningRecord(
                            candidate_id=cid,
                            model_id=model_id,
                            class_id=s["class_id"],
                            box_xyxy=s["box_xyxy"],
                            detect_confidence=s["detect_confidence"],
                            classify_confidence=s["classify_confidence"],
                            strategy=strategy,
                            acquisition_score=s["acquisition_score"],
                            flip_count=s["flip_count"],
                            flip_classes=s["flip_classes"],
                            calibrated_entropy=s["calibrated_entropy"],
                        )
                        session.add(rec)
                session.commit()
                logger.debug(f"Persisted {len(scored_records)} scored candidate records to database.")

        return scored_records

    def select_candidates(
        self,
        run_id: str,
        candidate_ids: List[Union[str, uuid.UUID]],
        query_budget: int,
        labeled_counts: Optional[Dict[Union[str, int], int]] = None,
        shortlist_factor: int = 3,
        top_n_prior: int = 5,
        save_db: bool = True,
    ) -> Dict[str, Any]:
        n_input = len(candidate_ids)
        budget = int(query_budget)

        if n_input == 0 or budget <= 0:
            logger.info(f"Empty candidate pool (N={n_input}) or zero budget (K={budget}). Returning empty selection.")
            return {
                "run_id": run_id,
                "total_pool_size": n_input,
                "shortlist_size": 0,
                "query_budget": budget,
                "selected_count": 0,
                "selected_candidates": [],
            }

        clean_ids = [uuid.UUID(str(c)) for c in candidate_ids]
        model_id = self.get_active_classifier_model_id()
        timo_weights, timo_bias = self.get_timo_parameters(model_id=model_id)
        class_anchors = self.fdal_selector.load_class_anchors(model_id=model_id)

        candidates_data = []
        with get_sync_db() as session:
            stmt = (
                select(
                    CandidateEmbedding.candidate_id,
                    CandidateEmbedding.embedding,
                    CandidateEmbedding.s3_key,
                    ActiveLearningRecord.box_xyxy,
                    ActiveLearningRecord.detect_confidence,
                    ActiveLearningRecord.classify_confidence,
                    ActiveLearningRecord.acquisition_score,
                    ActiveLearningRecord.strategy,
                )
                .outerjoin(ActiveLearningRecord, CandidateEmbedding.candidate_id == ActiveLearningRecord.candidate_id)
                .where(CandidateEmbedding.candidate_id.in_(clean_ids))
            )
            rows = session.execute(stmt).fetchall()
            seen_cids = set()
            for r in rows:
                cid = r[0]
                if cid in seen_cids or r[1] is None:
                    continue
                seen_cids.add(cid)
                candidates_data.append({
                    "candidate_id": cid,
                    "embedding": np.asarray(r[1], dtype=np.float32),
                    "s3_key": r[2],
                    "box_xyxy": r[3] or [0.0, 0.0, 0.0, 0.0],
                    "detect_confidence": float(r[4] or 0.0),
                    "classify_confidence": float(r[5] or 0.0) if r[5] is not None else None,
                    "existing_score": float(r[6]) if r[6] is not None else None,
                    "strategy": r[7],
                })

        n_samples = len(candidates_data)
        if n_samples == 0:
            logger.warning("No candidate records with valid embeddings found in PostgreSQL AI.")
            return {
                "run_id": run_id,
                "total_pool_size": 0,
                "shortlist_size": 0,
                "query_budget": budget,
                "selected_count": 0,
                "selected_candidates": [],
            }

        embs = np.stack([c["embedding"] for c in candidates_data], axis=0)
        embs_tensor = torch.from_numpy(embs).to(self.device)
        logits = torch.matmul(embs_tensor, timo_weights) + timo_bias
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        pred_classes = torch.argmax(logits, dim=-1).detach().cpu().numpy()

        p0 = self.get_or_create_prior_snapshot(
            run_id=run_id,
            model_id=model_id,
            probabilities=probs,
            top_n=top_n_prior,
        )

        fdal_res = self.fdal_selector.score_candidates_batch(
            visual_embeddings=embs,
            predicted_classes=pred_classes.tolist(),
            class_probabilities=probs,
            timo_weights=timo_weights,
            timo_bias=timo_bias,
            anchors=class_anchors,
            reference_prior=p0,
        )

        for idx, c in enumerate(candidates_data):
            f_item = fdal_res[idx]
            c["acquisition_score"] = float(f_item["fdal_priority"])
            c["flip_count"] = int(f_item["flip_count"])
            c["flip_classes"] = list(f_item["flipped_anchor_classes"])
            c["calibrated_entropy"] = float(f_item["fallback_entropy"])
            c["predicted_class_id"] = int(pred_classes[idx])
            c["classify_confidence"] = float(probs[idx, pred_classes[idx]])

        k_eff = min(budget, n_samples)
        m_shortlist = min(int(shortlist_factor) * budget, n_samples)

        if n_samples <= budget:
            logger.info(f"Corner Case (0 < N <= K): Selecting all {n_samples} candidates without clustering.")
            selected_items = []
            for idx, c in enumerate(candidates_data):
                item = {
                    "candidate_id": str(c["candidate_id"]),
                    "predicted_class_id": c["predicted_class_id"],
                    "acquisition_score": c["acquisition_score"],
                    "uncertainty_utility": c["calibrated_entropy"],
                    "residual_priority": 1.0,
                    "cluster_id": 0,
                    "tri_rank_res": 1,
                    "tri_rank_info": 1,
                    "tri_rank_rep": 1,
                    "aggregate_rank": 3,
                    "selection_status": "ALL_SELECTED",
                }
                selected_items.append(item)

            if save_db:
                self._persist_selection_results(
                    run_id=run_id,
                    model_id=model_id,
                    selected_items=selected_items,
                    candidates_data=candidates_data,
                    all_selected=True,
                )

            return {
                "run_id": run_id,
                "total_pool_size": n_samples,
                "shortlist_size": n_samples,
                "query_budget": budget,
                "selected_count": len(selected_items),
                "selected_candidates": selected_items,
            }

        scores_arr = np.array([c["acquisition_score"] for c in candidates_data], dtype=np.float32)
        if n_samples < int(shortlist_factor) * budget:
            logger.info(f"Corner Case (K < N < 3K): Taking all {n_samples} candidates into K-Means.")
            shortlist_indices = np.arange(n_samples)
        else:
            logger.info(f"Corner Case (N >= 3K): Taking Top-{m_shortlist} FDAL candidates into K-Means.")
            shortlist_indices = np.argsort(-scores_arr)[:m_shortlist]

        shortlist_features = embs[shortlist_indices]
        shortlist_probs = probs[shortlist_indices]
        shortlist_preds = pred_classes[shortlist_indices]

        sel_indices_local, detail_records = self.r3_cec_selector.select_candidates_with_details(
            visual_features=shortlist_features,
            probabilities=shortlist_probs,
            labeled_counts=labeled_counts,
            query_budget=k_eff,
            reference_prior=p0,
        )

        selected_items = []
        detail_map = {d["sample_index"]: d for d in detail_records}

        for local_idx in sel_indices_local:
            global_c = candidates_data[shortlist_indices[local_idx]]
            d = detail_map.get(local_idx, {})
            item = {
                "candidate_id": str(global_c["candidate_id"]),
                "predicted_class_id": global_c["predicted_class_id"],
                "acquisition_score": global_c["acquisition_score"],
                "uncertainty_utility": float(d.get("uncertainty_utility", 0.0)),
                "residual_priority": float(d.get("residual_priority", 1.0)),
                "cluster_id": int(d.get("cluster_id", 0)),
                "tri_rank_res": d.get("tri_rank_res"),
                "tri_rank_info": d.get("tri_rank_info"),
                "tri_rank_rep": d.get("tri_rank_rep"),
                "aggregate_rank": d.get("aggregate_rank"),
                "selection_status": d.get("selection_status", "CLUSTER_WINNER"),
            }
            selected_items.append(item)

        if save_db:
            shortlist_details = []
            for local_idx, d in detail_map.items():
                g_c = candidates_data[shortlist_indices[local_idx]]
                shortlist_details.append((g_c, d))
            self._persist_selection_results(
                run_id=run_id,
                model_id=model_id,
                selected_items=selected_items,
                candidates_data=candidates_data,
                shortlist_details=shortlist_details,
                all_selected=False,
            )

        return {
            "run_id": run_id,
            "total_pool_size": n_samples,
            "shortlist_size": len(shortlist_indices),
            "query_budget": budget,
            "selected_count": len(selected_items),
            "selected_candidates": selected_items,
        }

    def _persist_selection_results(
        self,
        run_id: str,
        model_id: int,
        selected_items: List[Dict[str, Any]],
        candidates_data: List[Dict[str, Any]],
        shortlist_details: Optional[List[Tuple[Dict[str, Any], Dict[str, Any]]]] = None,
        all_selected: bool = False,
    ) -> None:
        try:
            with get_sync_db() as session:
                selected_cid_set = {uuid.UUID(s["candidate_id"]) for s in selected_items}
                item_map = {uuid.UUID(s["candidate_id"]): s for s in selected_items}

                for c in candidates_data:
                    cid = c["candidate_id"]
                    stmt = select(ActiveLearningRecord).where(
                        ActiveLearningRecord.candidate_id == cid,
                        ActiveLearningRecord.model_id == model_id,
                    ).limit(1)
                    rec = session.execute(stmt).scalar_one_or_none()
                    is_sel = cid in selected_cid_set
                    item_meta = item_map.get(cid)

                    if rec:
                        rec.batch_run_id = run_id
                        rec.is_selected = is_sel
                        rec.strategy = "hybrid_fdal_r3_cec"
                        rec.acquisition_score = c.get("acquisition_score", rec.acquisition_score)
                        rec.flip_count = c.get("flip_count", rec.flip_count)
                        rec.flip_classes = c.get("flip_classes", rec.flip_classes)
                        rec.calibrated_entropy = c.get("calibrated_entropy", rec.calibrated_entropy)
                        rec.class_id = c.get("predicted_class_id", rec.class_id)
                        rec.classify_confidence = c.get("classify_confidence", rec.classify_confidence)
                        if item_meta:
                            rec.cluster_id = item_meta.get("cluster_id")
                            rec.tri_rank_res = item_meta.get("tri_rank_res")
                            rec.tri_rank_info = item_meta.get("tri_rank_info")
                            rec.tri_rank_rep = item_meta.get("tri_rank_rep")
                            rec.aggregate_rank = item_meta.get("aggregate_rank")
                    else:
                        new_rec = ActiveLearningRecord(
                            candidate_id=cid,
                            model_id=model_id,
                            class_id=c.get("predicted_class_id", 0),
                            box_xyxy=c.get("box_xyxy", [0.0, 0.0, 0.0, 0.0]),
                            detect_confidence=c.get("detect_confidence", 0.0),
                            classify_confidence=c.get("classify_confidence", 0.0),
                            strategy="hybrid_fdal_r3_cec",
                            acquisition_score=c.get("acquisition_score", 0.0),
                            flip_count=c.get("flip_count", 0),
                            flip_classes=c.get("flip_classes", []),
                            calibrated_entropy=c.get("calibrated_entropy", 0.0),
                            batch_run_id=run_id,
                            cluster_id=item_meta.get("cluster_id") if item_meta else None,
                            tri_rank_res=item_meta.get("tri_rank_res") if item_meta else None,
                            tri_rank_info=item_meta.get("tri_rank_info") if item_meta else None,
                            tri_rank_rep=item_meta.get("tri_rank_rep") if item_meta else None,
                            aggregate_rank=item_meta.get("aggregate_rank") if item_meta else None,
                            is_selected=is_sel,
                        )
                        session.add(new_rec)

                if shortlist_details and not all_selected:
                    for g_c, d in shortlist_details:
                        s_cid = g_c["candidate_id"]
                        if s_cid in selected_cid_set:
                            continue
                        stmt = select(ActiveLearningRecord).where(
                            ActiveLearningRecord.candidate_id == s_cid,
                            ActiveLearningRecord.model_id == model_id,
                        ).limit(1)
                        s_rec = session.execute(stmt).scalar_one_or_none()
                        if s_rec:
                            s_rec.batch_run_id = run_id
                            s_rec.cluster_id = d.get("cluster_id")
                            s_rec.tri_rank_res = d.get("tri_rank_res")
                            s_rec.tri_rank_info = d.get("tri_rank_info")
                            s_rec.tri_rank_rep = d.get("tri_rank_rep")
                            s_rec.aggregate_rank = d.get("aggregate_rank")
                            s_rec.is_selected = False

                session.commit()
                logger.info(f"Persisted AL selection results for run_id='{run_id}', selected={len(selected_items)}.")
        except Exception as e:
            logger.error(f"Failed to persist AL selection results for run_id='{run_id}': {e}")


hybrid_al_selector = HybridALSelector()
