import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger
from src.core.config_manager import config_manager


class ActiveLearningStore:
    def __init__(self, custom_path: Optional[Union[str, Path]] = None):
        self._lock = threading.Lock()
        self._custom_path: Optional[Path] = Path(custom_path) if custom_path else None

    def set_store_path(self, path: Optional[Union[str, Path]]) -> None:
        self._custom_path = Path(path) if path else None

    def get_file_path(self) -> Path:
        if self._custom_path is not None:
            return self._custom_path
        raw_path = config_manager.get("active_learning.dataset_store_path", "runs/active_learning/dataset.json")
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = base_dir / resolved
        return resolved

    def _get_empty_schema(self) -> Dict[str, Any]:
        return {
            "detector": {"pending": [], "trained": []},
            "classifier": {"pending": [], "trained": []},
            "clip": {"pending": [], "trained": []},
            "train_history": [],
        }

    def load_data(self) -> Dict[str, Any]:
        path = self.get_file_path()
        if not path.exists():
            return self._get_empty_schema()

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            schema = self._get_empty_schema()
            for key in ("detector", "classifier", "clip"):
                if key in data and isinstance(data[key], dict):
                    schema[key]["pending"] = list(data[key].get("pending", []))
                    schema[key]["trained"] = list(data[key].get("trained", []))
            if "train_history" in data and isinstance(data["train_history"], list):
                schema["train_history"] = list(data["train_history"])
            return schema
        except Exception as e:
            logger.error(f"Failed to read Active Learning store at '{path}': {e}")
            return self._get_empty_schema()

    def save_data(self, data: Dict[str, Any]) -> None:
        path = self.get_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = path.parent / f".tmp_{uuid.uuid4().hex}.json"

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, path)
            logger.debug(f"Saved Active Learning dataset store to '{path}'")
        except Exception as e:
            if temp_file.exists():
                try:
                    os.remove(temp_file)
                except Exception:
                    pass
            logger.error(f"Failed to save Active Learning store to '{path}': {e}")
            raise e

    def add_detector_sample(
        self,
        object_name: str,
        box_xyxy: List[float],
        pred_confidence: float,
        oracle_label: str = "NOT_A_SIGN",
    ) -> Dict[str, Any]:
        with self._lock:
            data = self.load_data()
            sample = {
                "id": f"det_{uuid.uuid4().hex[:8]}",
                "object_name": object_name,
                "box_xyxy": box_xyxy,
                "pred_confidence": float(pred_confidence),
                "oracle_label": oracle_label,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            data["detector"]["pending"].append(sample)
            self.save_data(data)
            logger.info(f"Added detector hard-negative sample '{sample['id']}' for '{object_name}'")
            return sample

    def add_classifier_sample(
        self,
        object_name: str,
        box_xyxy: List[float],
        pred_class_id: int,
        pred_class_name: str,
        pred_confidence: float,
        oracle_class_id: int,
        oracle_class_name: str,
    ) -> Dict[str, Any]:
        with self._lock:
            data = self.load_data()
            sample = {
                "id": f"cls_{uuid.uuid4().hex[:8]}",
                "object_name": object_name,
                "box_xyxy": box_xyxy,
                "pred_class_id": int(pred_class_id),
                "pred_class_name": pred_class_name,
                "pred_confidence": float(pred_confidence),
                "oracle_class_id": int(oracle_class_id),
                "oracle_class_name": oracle_class_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            data["classifier"]["pending"].append(sample)
            self.save_data(data)
            logger.info(f"Added classifier corrected sample '{sample['id']}' for '{object_name}' (Pred={pred_class_id} -> Oracle={oracle_class_id})")
            return sample

    def add_clip_sample(
        self,
        object_name: str,
        oracle_class_name: str,
        box_xyxy: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            data = self.load_data()
            sample = {
                "id": f"clip_{uuid.uuid4().hex[:8]}",
                "object_name": object_name,
                "box_xyxy": box_xyxy,
                "oracle_class_name": oracle_class_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            data["clip"]["pending"].append(sample)
            self.save_data(data)
            logger.info(f"Added CLIP staging sample '{sample['id']}' for '{object_name}'")
            return sample

    def get_counts(self) -> Dict[str, Any]:
        with self._lock:
            data = self.load_data()
            return {
                "detector": {
                    "pending": len(data["detector"]["pending"]),
                    "trained": len(data["detector"]["trained"]),
                },
                "classifier": {
                    "pending": len(data["classifier"]["pending"]),
                    "trained": len(data["classifier"]["trained"]),
                },
                "clip": {
                    "pending": len(data["clip"]["pending"]),
                    "trained": len(data["clip"]["trained"]),
                },
                "total_pending": (
                    len(data["detector"]["pending"])
                    + len(data["classifier"]["pending"])
                    + len(data["clip"]["pending"])
                ),
                "total_trained": (
                    len(data["detector"]["trained"])
                    + len(data["classifier"]["trained"])
                    + len(data["clip"]["trained"])
                ),
                "train_history_count": len(data.get("train_history", [])),
                "store_path": str(self.get_file_path()),
            }

    def get_pending_samples(self, target: str) -> List[Dict[str, Any]]:
        target_norm = target.lower()
        with self._lock:
            data = self.load_data()
            if target_norm not in data:
                logger.warning(f"Unknown target '{target}'. Returning empty list.")
                return []
            return list(data[target_norm].get("pending", []))

    def mark_as_trained(self, target: str, sample_ids: List[str]) -> int:
        target_norm = target.lower()
        with self._lock:
            data = self.load_data()
            if target_norm not in data:
                logger.warning(f"Unknown target '{target}' to mark as trained.")
                return 0

            pending = data[target_norm].get("pending", [])
            trained = data[target_norm].get("trained", [])
            ids_set = set(sample_ids)

            remaining_pending = []
            moved_count = 0

            for item in pending:
                if item.get("id") in ids_set:
                    item_copy = dict(item)
                    item_copy["trained_at"] = datetime.now(timezone.utc).isoformat()
                    trained.append(item_copy)
                    moved_count += 1
                else:
                    remaining_pending.append(item)

            data[target_norm]["pending"] = remaining_pending
            data[target_norm]["trained"] = trained
            self.save_data(data)
            logger.info(f"Marked {moved_count} sample(s) as trained in '{target_norm}'.")
            return moved_count

    def record_train_history(
        self,
        target: str,
        sample_count: int,
        status: str,
        weights_name: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            data = self.load_data()
            record = {
                "id": f"tr_{uuid.uuid4().hex[:8]}",
                "target_model": target.upper(),
                "sample_count": int(sample_count),
                "status": status,
                "weights_name": weights_name,
                "error": error,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
            data.setdefault("train_history", []).append(record)
            self.save_data(data)
            logger.info(f"Recorded training history record '{record['id']}' for '{target.upper()}' (status={status})")
            return record

    def clear_store(self, target: Optional[str] = None) -> None:
        with self._lock:
            if target is None:
                self.save_data(self._get_empty_schema())
                logger.info("Cleared entire Active Learning dataset store.")
            else:
                target_norm = target.lower()
                data = self.load_data()
                if target_norm in data:
                    data[target_norm] = {"pending": [], "trained": []}
                    self.save_data(data)
                    logger.info(f"Cleared Active Learning store for target '{target_norm}'.")


al_store = ActiveLearningStore()
