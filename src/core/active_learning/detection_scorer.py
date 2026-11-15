import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import torch
from loguru import logger
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.core.config import settings
from src.core.config_manager import config_manager
from src.core.model_manager import model_manager
from src.infrastructure.minio_client import minio_client
from src.infrastructure.db import get_sync_db
from src.infrastructure.models import CandidateEmbedding, ActiveLearningRecord, ModelVersion
from src.core.active_learning.base import BaseActiveLearningScorer
import src.core.active_learning.strategies


class DetectionActiveLearningEngine:
    def __init__(self):
        self._strategies: Dict[str, BaseActiveLearningScorer] = {}
        self._discover_strategies()
        self._active_classifier_model_id: Optional[int] = None

    def _discover_strategies(self) -> None:
        for cls in BaseActiveLearningScorer.__subclasses__():
            instance = cls()
            self._strategies[instance.name] = instance
            logger.debug(f"Auto-registered Active Learning strategy: '{instance.name}'")

    def _get_active_classifier_model_id(self) -> Optional[int]:
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

    def get_scorer(self, strategy_name: Optional[str] = None):
        default_strat = config_manager.get("active_learning.strategy", "r3_cec")
        target = (strategy_name or default_strat).lower()
        scorer = self._strategies.get(target)
        if scorer is None:
            fallback = list(self._strategies.keys())[0] if self._strategies else "least_confidence"
            logger.warning(f"Strategy '{target}' not found. Falling back to '{fallback}'.")
            target = fallback
            scorer = self._strategies.get(target)
        return target, scorer

    def get_supported_strategies(self) -> Dict[str, Any]:
        default_strat = config_manager.get("active_learning.strategy", "r3_cec")
        default_agg = config_manager.get("active_learning.aggregation", "avg")
        default_budget = int(config_manager.get("active_learning.query_budget", 50))
        return {
            "strategies": list(self._strategies.keys()),
            "aggregation_methods": BaseActiveLearningScorer.get_supported_aggregations(),
            "strategy": default_strat,
            "aggregation": default_agg,
            "query_budget": default_budget,
            "default_strategy": default_strat,
            "default_aggregation": default_agg,
            "default_top_k": default_budget,
        }

    def _persist_evaluation_records(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        try:
            with get_sync_db() as session:
                for item in records:
                    if item.get("embedding") is not None:
                        emb = item["embedding"]
                        emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                        stmt = (
                            pg_insert(CandidateEmbedding)
                            .values(
                                candidate_id=item["candidate_id"],
                                s3_key=item["s3_key"],
                                model_id=item.get("model_id"),
                                embedding=emb_list,
                            )
                            .on_conflict_do_nothing(index_elements=["candidate_id"])
                        )
                        session.execute(stmt)

                    rec = ActiveLearningRecord(
                        candidate_id=item["candidate_id"],
                        model_id=item.get("model_id"),
                        class_id=item.get("class_id"),
                        box_xyxy=item.get("box_xyxy"),
                        detect_confidence=item.get("detect_confidence"),
                        classify_confidence=item.get("classify_confidence"),
                        strategy=item["strategy"],
                        acquisition_score=item["acquisition_score"],
                        flip_count=item.get("flip_count", 0),
                        flip_classes=item.get("flip_classes", []),
                        calibrated_entropy=item.get("calibrated_entropy", 0.0),
                    )
                    session.add(rec)
                session.commit()
                logger.debug(f"Persisted {len(records)} Active Learning record(s) to PostgreSQL AI.")
        except Exception as e:
            logger.error(f"Failed to persist Active Learning records to database: {e}")

    def _evaluate_and_persist_detections(
        self,
        image_path: str,
        s3_key: str,
        detections: List[Dict[str, Any]],
        strategy_name: str,
        scorer: BaseActiveLearningScorer,
    ) -> List[Dict[str, Any]]:
        if not detections:
            return []

        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not read image for crop extraction: {image_path}")
            _, scored_dets = scorer.score_detections(detections=detections)
            return scored_dets

        h, w = image.shape[:2]
        crops = []
        for det in detections:
            box = det.get("box_xyxy", [0, 0, 0, 0])
            bx1 = max(0, min(w - 1, int(round(box[0]))))
            by1 = max(0, min(h - 1, int(round(box[1]))))
            bx2 = max(bx1 + 1, min(w, int(round(box[2]))))
            by2 = max(by1 + 1, min(h, int(round(box[3]))))
            crops.append(image[by1:by2, bx1:bx2])

        classifier = model_manager.get_classifier()
        classified_results = None
        if classifier is not None:
            try:
                classified_results = classifier.classify_crops_batch(crops, return_features=True)
            except Exception as e:
                logger.warning(f"Batch crop classification failed: {e}")

        model_id = self._get_active_classifier_model_id()
        records_to_persist = []
        scored_detections = []

        visual_embeddings = []
        predicted_classes = []
        class_probabilities = []

        for idx, det in enumerate(detections):
            cls_res = classified_results[idx] if classified_results and idx < len(classified_results) else {}
            v_emb = cls_res.get("visual_embedding")
            pred_cls = cls_res.get("class_id", det.get("class_id", 0))
            probs = cls_res.get("probabilities")
            classify_conf = cls_res.get("classify_confidence", det.get("classify_confidence"))

            visual_embeddings.append(v_emb)
            predicted_classes.append(int(pred_cls) if pred_cls is not None else 0)
            class_probabilities.append(probs)

        fdal_scored_map = {}
        if strategy_name == "fdal" and classifier is not None:
            try:
                from src.core.active_learning.strategies.fdal import FDALSelector
                selector = FDALSelector(device="cpu")
                w_timo, b_timo = classifier.get_timo_parameters()
                fdal_batch_res = selector.score_candidates_batch(
                    visual_embeddings=visual_embeddings,
                    predicted_classes=predicted_classes,
                    class_probabilities=class_probabilities,
                    timo_weights=w_timo,
                    timo_bias=b_timo,
                )
                for res_item in fdal_batch_res:
                    fdal_scored_map[res_item["candidate_index"]] = res_item
            except Exception as e:
                logger.warning(f"FDAL batch scoring failed, falling back to base scorer: {e}")

        for idx, det in enumerate(detections):
            v_emb = visual_embeddings[idx]
            cls_res = classified_results[idx] if classified_results and idx < len(classified_results) else {}
            classify_conf = cls_res.get("classify_confidence", det.get("classify_confidence"))
            box_coords = det.get("box_xyxy", [0.0, 0.0, 0.0, 0.0])
            candidate_seed = f"{s3_key}:{idx}:{box_coords[0]}_{box_coords[1]}_{box_coords[2]}_{box_coords[3]}"
            candidate_id = uuid.uuid5(uuid.NAMESPACE_DNS, candidate_seed)

            if idx in fdal_scored_map:
                fd_item = fdal_scored_map[idx]
                acq_score = float(fd_item["fdal_priority"])
                flip_cnt = int(fd_item["flip_count"])
                flip_clss = list(fd_item["flipped_anchor_classes"])
                calib_ent = float(fd_item["fallback_entropy"])
            else:
                acq_score = round(scorer.compute_box_score(det), 4)
                flip_cnt = 0
                flip_clss = []
                calib_ent = 0.0

            det_copy = dict(det)
            det_copy["candidate_id"] = str(candidate_id)
            det_copy["score"] = acq_score
            det_copy["acquisition_score"] = acq_score
            det_copy["flip_count"] = flip_cnt
            det_copy["flip_classes"] = flip_clss
            det_copy["calibrated_entropy"] = calib_ent
            det_copy["detect_confidence"] = det.get("detect_confidence", det.get("confidence", 0.0))
            det_copy["classify_confidence"] = classify_conf
            scored_detections.append(det_copy)

            records_to_persist.append({
                "candidate_id": candidate_id,
                "s3_key": s3_key,
                "model_id": model_id,
                "class_id": det_copy.get("class_id"),
                "box_xyxy": box_coords,
                "detect_confidence": det_copy["detect_confidence"],
                "classify_confidence": classify_conf,
                "strategy": strategy_name,
                "acquisition_score": acq_score,
                "flip_count": flip_cnt,
                "flip_classes": flip_clss,
                "calibrated_entropy": calib_ent,
                "embedding": v_emb,
            })

        self._persist_evaluation_records(records_to_persist)
        del image, crops
        return scored_detections

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
        effective_agg = str(aggregation_method or config_manager.get("active_learning.aggregation", "avg")).lower()
        effective_conf = conf_threshold if conf_threshold is not None else config_manager.detector_conf_thresh
        effective_device = config_manager.cuda_device
        no_detection_score = float(config_manager.get("active_learning.no_detection_score", 0.0))

        try:
            detector = model_manager.get_detector()
            detections = detector.detect(temp_image_path, conf=effective_conf, device=effective_device)
            scored_detections = self._evaluate_and_persist_detections(
                image_path=temp_image_path,
                s3_key=object_name,
                detections=detections,
                strategy_name=effective_strategy,
                scorer=scorer,
            )

            if scored_detections:
                box_scores = [d["score"] for d in scored_detections]
                image_score = scorer.aggregate(box_scores, effective_agg)
            else:
                image_score = round(float(no_detection_score), 4)

            return {
                "object_name": object_name,
                "s3_key": object_name,
                "score": image_score,
                "acquisition_score": image_score,
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
        effective_agg = str(aggregation_method or config_manager.get("active_learning.aggregation", "avg")).lower()
        effective_conf = conf_threshold if conf_threshold is not None else config_manager.detector_conf_thresh
        effective_device = config_manager.cuda_device
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
                scored_detections = self._evaluate_and_persist_detections(
                    image_path=temp_image_path,
                    s3_key=object_name,
                    detections=detections,
                    strategy_name=effective_strategy,
                    scorer=scorer,
                )

                if scored_detections:
                    box_scores = [d["score"] for d in scored_detections]
                    image_score = scorer.aggregate(box_scores, effective_agg)
                else:
                    image_score = round(float(no_detection_score), 4)

                results.append({
                    "object_name": object_name,
                    "s3_key": object_name,
                    "score": image_score,
                    "acquisition_score": image_score,
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

        configured_budget = int(config_manager.get("active_learning.query_budget", 50))
        target_top_k = top_k if top_k is not None else configured_budget

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
