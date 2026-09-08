import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional
from loguru import logger
from src.core.config import settings
from src.core.config_manager import config_manager
from src.core.model_manager import model_manager
from src.infrastructure.minio_client import minio_client
from src.core.active_learning.base import BaseActiveLearningScorer
import src.core.active_learning.strategies


class DetectionActiveLearningEngine:
    def __init__(self):
        self._strategies: Dict[str, BaseActiveLearningScorer] = {}
        self._discover_strategies()

    def _discover_strategies(self) -> None:
        for cls in BaseActiveLearningScorer.__subclasses__():
            instance = cls()
            self._strategies[instance.name] = instance
            logger.debug(f"Auto-registered Active Learning strategy: '{instance.name}'")

    def get_scorer(self, strategy_name: Optional[str] = None):
        default_strat = config_manager.get("active_learning.default_strategy", "least_confidence")
        target = (strategy_name or default_strat).lower()
        scorer = self._strategies.get(target)
        if scorer is None:
            fallback = list(self._strategies.keys())[0] if self._strategies else "least_confidence"
            logger.warning(f"Strategy '{target}' not found. Falling back to '{fallback}'.")
            target = fallback
            scorer = self._strategies.get(target)
        return target, scorer

    def get_supported_strategies(self) -> Dict[str, Any]:
        return {
            "strategies": list(self._strategies.keys()),
            "aggregation_methods": BaseActiveLearningScorer.get_supported_aggregations(),
            "default_strategy": config_manager.get("active_learning.default_strategy", "least_confidence"),
            "default_aggregation": config_manager.get("active_learning.default_aggregation", "avg"),
            "default_top_k": config_manager.get("active_learning.default_top_k", 50),
        }

    def score_single(
        self,
        object_name: str,
        strategy: Optional[str] = None,
        aggregation_method: Optional[str] = None,
        conf_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Scoring single image from MinIO: '{object_name}'")
        temp_dir = tempfile.gettempdir()
        file_ext = os.path.splitext(object_name)[1] or ".jpg"
        temp_image_path = os.path.join(temp_dir, f"al_score_{uuid.uuid4().hex}{file_ext}")

        success = minio_client.download_file(settings.MINIO_BUCKET, object_name, temp_image_path)
        if not success or not os.path.exists(temp_image_path):
            logger.error(f"Failed to fetch image '{object_name}' from MinIO storage.")
            raise FileNotFoundError(f"Image '{object_name}' could not be downloaded from MinIO storage.")

        effective_strategy, scorer = self.get_scorer(strategy)
        effective_agg = str(aggregation_method or config_manager.get("active_learning.default_aggregation", "avg")).lower()
        effective_conf = conf_threshold if conf_threshold is not None else config_manager.default_detector_conf
        effective_device = config_manager.default_device
        no_detection_score = float(config_manager.get("active_learning.no_detection_score", 0.0))

        try:
            detector = model_manager.get_detector()
            detections = detector.detect(temp_image_path, conf=effective_conf, device=effective_device)
            image_score, scored_detections = scorer.score_detections(
                detections=detections,
                aggregation_method=effective_agg,
                no_detection_score=no_detection_score,
            )

            return {
                "object_name": object_name,
                "score": image_score,
                "total_detections": len(scored_detections),
                "detections": scored_detections,
                "strategy": effective_strategy,
                "aggregation_method": effective_agg,
            }
        finally:
            if os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                    logger.debug(f"Removed temporary AL image: {temp_image_path}")
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_image_path}: {e}")

    def score_batch(
        self,
        object_names: List[str],
        top_k: Optional[int] = None,
        strategy: Optional[str] = None,
        aggregation_method: Optional[str] = None,
        conf_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Starting batch Active Learning scoring for {len(object_names)} object(s).")
        effective_strategy, scorer = self.get_scorer(strategy)
        effective_agg = str(aggregation_method or config_manager.get("active_learning.default_aggregation", "avg")).lower()
        effective_conf = conf_threshold if conf_threshold is not None else config_manager.default_detector_conf
        effective_device = config_manager.default_device
        no_detection_score = float(config_manager.get("active_learning.no_detection_score", 0.0))

        results: List[Dict[str, Any]] = []
        detector = model_manager.get_detector()
        temp_dir = tempfile.gettempdir()

        for idx, object_name in enumerate(object_names):
            file_ext = os.path.splitext(object_name)[1] or ".jpg"
            temp_image_path = os.path.join(temp_dir, f"al_batch_{uuid.uuid4().hex}{file_ext}")

            try:
                success = minio_client.download_file(settings.MINIO_BUCKET, object_name, temp_image_path)
                if not success or not os.path.exists(temp_image_path):
                    logger.warning(f"Batch item [{idx + 1}/{len(object_names)}]: '{object_name}' download failed. Skipping.")
                    continue

                detections = detector.detect(temp_image_path, conf=effective_conf, device=effective_device)
                image_score, scored_detections = scorer.score_detections(
                    detections=detections,
                    aggregation_method=effective_agg,
                    no_detection_score=no_detection_score,
                )

                results.append({
                    "object_name": object_name,
                    "score": image_score,
                    "total_detections": len(scored_detections),
                    "detections": scored_detections,
                })
                logger.trace(f"Batch [{idx + 1}/{len(object_names)}]: '{object_name}' -> score={image_score}")

            except Exception as e:
                logger.error(f"Error scoring batch item '{object_name}': {e}")
            finally:
                if os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {temp_image_path}: {e}")

        results.sort(key=lambda x: x["score"], reverse=True)

        configured_top_k = int(config_manager.get("active_learning.default_top_k", 50))
        target_top_k = top_k if top_k is not None else configured_top_k

        if target_top_k == -1:
            ranked_samples = results
        else:
            ranked_samples = results[:max(0, target_top_k)]

        logger.info(
            f"Completed AL batch scoring: evaluated={len(results)}/{len(object_names)}, "
            f"top_k_requested={target_top_k}, returned={len(ranked_samples)}"
        )

        return {
            "total_evaluated": len(results),
            "top_k_returned": len(ranked_samples),
            "strategy": effective_strategy,
            "aggregation_method": effective_agg,
            "ranked_samples": ranked_samples,
        }


al_engine = DetectionActiveLearningEngine()
