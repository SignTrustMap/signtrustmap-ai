import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger
from src.core.config_manager import config_manager
from src.core.active_learning.store import al_store
from src.core.active_learning.retrain import trigger_retrain


class FeedbackProcessor:
    def process_feedback_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        object_name = str(item.get("object_name", ""))
        box_xyxy = list(item.get("box_xyxy", [0.0, 0.0, 0.0, 0.0]))
        evaluation = str(item.get("evaluation", "")).strip().upper()
        pred_class_id = item.get("pred_class_id")
        pred_class_name = str(item.get("pred_class_name", ""))
        pred_confidence = float(item.get("pred_confidence", 0.0))
        oracle_class_id = item.get("oracle_class_id")
        oracle_class_name = str(item.get("oracle_class_name", "")).strip()
        is_clip = bool(item.get("is_clip", False))

        if evaluation == "NOT_GIVEN":
            sample = al_store.add_detector_sample(
                object_name=object_name,
                box_xyxy=box_xyxy,
                pred_confidence=pred_confidence,
                oracle_label="NOT_GIVEN",
            )
            return {"category": "detector", "sample_id": sample["id"], "action": "added_hard_negative"}

        if evaluation == "FALSE":
            if is_clip:
                sample = al_store.add_clip_sample(
                    object_name=object_name,
                    oracle_class_name=oracle_class_name,
                    box_xyxy=box_xyxy,
                )
                return {"category": "clip", "sample_id": sample["id"], "action": "added_clip_staging"}

            sample = al_store.add_classifier_sample(
                object_name=object_name,
                box_xyxy=box_xyxy,
                pred_class_id=int(pred_class_id) if pred_class_id is not None else -1,
                pred_class_name=pred_class_name,
                pred_confidence=pred_confidence,
                oracle_class_id=int(oracle_class_id) if oracle_class_id is not None else -1,
                oracle_class_name=oracle_class_name,
            )
            return {"category": "classifier", "sample_id": sample["id"], "action": "added_classifier_correction"}

        if evaluation == "TRUE":
            return {"category": "correct", "sample_id": None, "action": "verified_correct"}

        logger.warning(f"Unknown evaluation status '{evaluation}' for object '{object_name}'. Skipping.")
        return {"category": "unknown", "sample_id": None, "action": "skipped_unknown_evaluation"}

    def process_feedback_batch(
        self,
        items: List[Dict[str, Any]],
        auto_trigger: Optional[bool] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Processing Active Learning feedback batch of {len(items)} item(s)...")
        results = []
        detector_count = 0
        classifier_count = 0
        clip_count = 0
        correct_count = 0

        for item in items:
            res = self.process_feedback_item(item)
            results.append(res)
            cat = res["category"]
            if cat == "detector":
                detector_count += 1
            elif cat == "classifier":
                classifier_count += 1
            elif cat == "clip":
                clip_count += 1
            elif cat == "correct":
                correct_count += 1

        counts = al_store.get_counts()
        should_auto_trigger = (
            auto_trigger
            if auto_trigger is not None
            else bool(config_manager.get("active_learning.auto_trigger_training", True))
        )

        triggered_tasks = []
        min_batch_detector = int(config_manager.get("active_learning.min_batch_size_detector", 50))
        min_batch_classifier = int(config_manager.get("active_learning.min_batch_size_classifier", 100))

        if should_auto_trigger:
            if counts["detector"]["pending"] >= min_batch_detector:
                logger.info(f"Detector pending samples ({counts['detector']['pending']}) reached threshold ({min_batch_detector}). Triggering retrain.")
                asyncio.create_task(trigger_retrain("detector"))
                triggered_tasks.append("detector")

            if counts["classifier"]["pending"] >= min_batch_classifier:
                logger.info(f"Classifier pending samples ({counts['classifier']['pending']}) reached threshold ({min_batch_classifier}). Triggering retrain.")
                asyncio.create_task(trigger_retrain("classifier"))
                triggered_tasks.append("classifier")

        logger.info(
            f"Active Learning feedback processed: detector={detector_count}, "
            f"classifier={classifier_count}, clip={clip_count}, correct={correct_count}, triggered={triggered_tasks}"
        )

        return {
            "total_processed": len(items),
            "detector_added": detector_count,
            "classifier_added": classifier_count,
            "clip_added": clip_count,
            "correct_count": correct_count,
            "dataset_counts": counts,
            "triggered_retrain": triggered_tasks,
        }


feedback_processor = FeedbackProcessor()
