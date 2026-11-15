import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.models import (
    AddedClassDiffItem,
    ChangedClassDiffItem,
    ClassDistributionDiffResponse,
    ClassDistributionItem,
    ClassDistributionResponse,
    ClassifierPoolStatus,
    DetectorPoolStatus,
    RemovedClassDiffItem,
    TrainingDataStatusResponse,
)
from src.core.config_manager import config_manager
from src.infrastructure.db import get_async_db
from src.infrastructure.models import (
    CandidateEmbedding,
    ModelClass,
    ModelVersion,
    SupportShot,
)

router = APIRouter(tags=["Training Data"])


def _load_catalog() -> List[Dict[str, Any]]:
    base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    cat_file = base_dir / "configs" / "vietnam_traffic_sign_catalog.json"
    if cat_file.exists():
        try:
            with open(cat_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read catalog file: {e}")
    return []


def _load_verified_cache() -> Dict[str, Any]:
    base_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
    cache_file = base_dir / "working" / "data" / "verified_signs_cache.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read verified cache: {e}")
    return []


@router.get("/status", response_model=TrainingDataStatusResponse)
async def get_training_data_status(db: AsyncSession = Depends(get_async_db)):
    logger.debug("Retrieving unified training data status")
    v_stmt = select(ModelVersion).where(ModelVersion.is_active == True)
    v_res = await db.execute(v_stmt)
    active_versions = v_res.scalars().all()

    active_cls = next((v for v in active_versions if v.model_type == "CLASSIFIER"), None)
    active_det = next((v for v in active_versions if v.model_type == "DETECTOR"), None)

    verified_map = _load_verified_cache()
    total_verified = len(verified_map)

    cand_stmt = select(CandidateEmbedding.candidate_id).where(CandidateEmbedding.model_id == (active_cls.id if active_cls else 2))
    cand_res = await db.execute(cand_stmt)
    ai_cand_ids = [str(r[0]) for r in cand_res.fetchall()]

    matched_count = sum(1 for cid in ai_cand_ids if cid in verified_map)
    affected_codes = set()
    for cid in ai_cand_ids:
        if cid in verified_map:
            sc = verified_map[cid].get("sign_code")
            if sc:
                affected_codes.add(sc)

    cfg_cls_thresh = int(config_manager.get("active_learning.min_samples_classifier", 100))
    cfg_det_thresh = int(config_manager.get("active_learning.min_samples_detector", 50))

    timo_cfg = config_manager.get("timo", {})
    supp_cfg = config_manager.get("few_shot_support", {})
    current_cls_hyperparams = {
        "alpha": float(timo_cfg.get("alpha", 250.0)),
        "beta": float(timo_cfg.get("beta", 0.8)),
        "shots_per_class": int(supp_cfg.get("shots_per_class", 16)),
        "num_anchors": int(supp_cfg.get("num_anchors", 4)),
        "num_medoids": int(supp_cfg.get("num_medoids", 12)),
        "min_sharpness": float(supp_cfg.get("min_sharpness", 15.0)),
    }

    current_det_hyperparams = {
        "epochs": 100,
        "batch_size": 16,
        "imgsz": 640,
    }

    cls_status = ClassifierPoolStatus(
        active_version_id=active_cls.id if active_cls else 2,
        active_version_tag=active_cls.version_tag if active_cls else "mobileclip_s2",
        total_verified_crops_in_ai_db=matched_count,
        new_verified_crops_since_active_model=matched_count,
        affected_classes_count=len(affected_codes),
        min_samples_threshold=cfg_cls_thresh,
        is_eligible_for_rebuild=matched_count >= cfg_cls_thresh,
        current_hyperparams=current_cls_hyperparams,
    )

    det_status = DetectorPoolStatus(
        active_version_id=active_det.id if active_det else 1,
        active_version_tag=active_det.version_tag if active_det else "yolo26n_vtsdb_mtsd_dfg",
        data_source="backend_survey_submissions",
        distinct_full_frames_count=len(ai_cand_ids),
        total_bounding_box_annotations=total_verified,
        min_samples_threshold=cfg_det_thresh,
        is_eligible_for_retrain=total_verified >= cfg_det_thresh,
        current_hyperparams=current_det_hyperparams,
    )

    return TrainingDataStatusResponse(
        status="SUCCESS",
        total_verified_signs_platform=total_verified,
        classifier_pool=cls_status,
        detector_pool=det_status,
    )


@router.get("/class-distribution", response_model=ClassDistributionResponse)
async def get_class_distribution():
    logger.debug("Retrieving full catalog class distribution")
    catalog = _load_catalog()
    verified_map = _load_verified_cache()

    code_counts: Dict[str, int] = {}
    for item in verified_map.values():
        sc = item.get("sign_code")
        if sc:
            code_counts[sc] = code_counts.get(sc, 0) + 1

    distribution = []
    classes_with_signs = 0
    for sign in catalog:
        sc = sign.get("signCode", "")
        cnt = code_counts.get(sc, 0)
        if cnt > 0:
            classes_with_signs += 1
        cat_info = sign.get("category", {})
        cat_code = cat_info.get("code") if isinstance(cat_info, dict) else "UNKNOWN"
        distribution.append(
            ClassDistributionItem(
                class_id=sign.get("id", 0),
                sign_code=sc,
                name_vi=sign.get("nameVi") or "",
                name_en=sign.get("nameEn") or "",
                category=cat_code or "UNKNOWN",
                verified_count=cnt,
            )
        )

    distribution.sort(key=lambda x: (x.verified_count == 0, -x.verified_count, x.sign_code))

    return ClassDistributionResponse(
        status="SUCCESS",
        total_catalog_classes=len(catalog),
        classes_with_verified_signs=classes_with_signs,
        distribution=distribution,
    )


@router.get("/class-distribution/diff", response_model=ClassDistributionDiffResponse)
async def get_class_distribution_diff(
    compare_with_version_id: Optional[int] = Query(None, description="Model version ID to compare against"),
    db: AsyncSession = Depends(get_async_db),
):
    target_id = compare_with_version_id
    if target_id is None:
        stmt = select(ModelVersion.id).where(ModelVersion.model_type == "CLASSIFIER", ModelVersion.is_active == True)
        res = await db.execute(stmt)
        target_id = res.scalar_one_or_none() or 2

    v_stmt = select(ModelVersion).where(ModelVersion.id == target_id)
    v_res = await db.execute(v_stmt)
    target_version = v_res.scalar_one_or_none()
    if not target_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version #{target_id} not found",
        )

    class_stmt = select(ModelClass.sign_code, ModelClass.class_name).where(ModelClass.model_id == target_id)
    class_res = await db.execute(class_stmt)
    model_classes = {r[0]: r[1] for r in class_res.fetchall()}

    supp_stmt = (
        select(ModelClass.sign_code, func.count(SupportShot.id))
        .join(ModelClass, SupportShot.class_id == ModelClass.id)
        .where(SupportShot.model_id == target_id)
        .group_by(ModelClass.sign_code)
    )
    supp_res = await db.execute(supp_stmt)
    previous_counts = {r[0]: int(r[1]) for r in supp_res.fetchall()}

    verified_map = _load_verified_cache()
    current_counts: Dict[str, int] = {}
    sign_meta: Dict[str, Dict[str, str]] = {}
    for item in verified_map.values():
        sc = item.get("sign_code")
        if sc:
            current_counts[sc] = current_counts.get(sc, 0) + 1
            if sc not in sign_meta:
                sign_meta[sc] = {
                    "name_vi": item.get("name_vi", ""),
                    "name_en": item.get("name_en", ""),
                }

    catalog = _load_catalog()
    for item in catalog:
        sc = item.get("signCode")
        if sc and sc not in sign_meta:
            sign_meta[sc] = {
                "name_vi": item.get("nameVi", ""),
                "name_en": item.get("nameEn", ""),
            }

    added_classes = []
    removed_classes = []
    changed_classes = []
    unchanged_count = 0

    for sc, curr_cnt in current_counts.items():
        if sc not in model_classes:
            added_classes.append(
                AddedClassDiffItem(
                    sign_code=sc,
                    name_vi=sign_meta.get(sc, {}).get("name_vi", ""),
                    name_en=sign_meta.get(sc, {}).get("name_en", ""),
                    current_verified_count=curr_cnt,
                )
            )

    for sc in model_classes:
        prev_cnt = previous_counts.get(sc, 16)
        curr_cnt = current_counts.get(sc, 0)
        meta = sign_meta.get(sc, {})

        if curr_cnt == 0 and prev_cnt > 0:
            removed_classes.append(
                RemovedClassDiffItem(
                    sign_code=sc,
                    name_vi=meta.get("name_vi", model_classes[sc]),
                    name_en=meta.get("name_en", model_classes[sc]),
                    previous_samples_count=prev_cnt,
                )
            )
        elif curr_cnt != prev_cnt:
            changed_classes.append(
                ChangedClassDiffItem(
                    sign_code=sc,
                    name_vi=meta.get("name_vi", model_classes[sc]),
                    name_en=meta.get("name_en", model_classes[sc]),
                    previous_samples_count=prev_cnt,
                    current_samples_count=curr_cnt,
                    delta_samples=curr_cnt - prev_cnt,
                )
            )
        else:
            unchanged_count += 1

    added_classes.sort(key=lambda x: -x.current_verified_count)
    changed_classes.sort(key=lambda x: -abs(x.delta_samples))

    return ClassDistributionDiffResponse(
        status="SUCCESS",
        compared_with_version_id=target_id,
        compared_version_tag=target_version.version_tag,
        total_classes_in_compared_model=len(model_classes),
        added_classes=added_classes,
        removed_classes=removed_classes,
        changed_classes=changed_classes,
        unchanged_classes_count=unchanged_count,
    )
