from datetime import datetime, timezone
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
    return {}


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

    active_cls_cutoff = active_cls.created_at if active_cls and active_cls.created_at else None

    new_verified_crops = 0
    affected_codes = set()

    for item in verified_map.values():
        v_at_str = item.get("verified_at") or item.get("created_at")
        if not v_at_str:
            if active_cls_cutoff is None:
                new_verified_crops += 1
                sc = item.get("sign_code")
                if sc:
                    affected_codes.add(sc)
            continue
        try:
            v_at = datetime.fromisoformat(v_at_str.replace("Z", "+00:00"))
            if active_cls_cutoff is None or v_at > active_cls_cutoff:
                new_verified_crops += 1
                sc = item.get("sign_code")
                if sc:
                    affected_codes.add(sc)
        except Exception:
            continue

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
        total_verified_crops_in_ai_db=total_verified,
        new_verified_crops_since_active_model=new_verified_crops,
        affected_classes_count=len(affected_codes),
        min_samples_threshold=cfg_cls_thresh,
        is_eligible_for_rebuild=new_verified_crops >= cfg_cls_thresh,
        current_hyperparams=current_cls_hyperparams,
    )

    det_status = DetectorPoolStatus(
        active_version_id=active_det.id if active_det else 1,
        active_version_tag=active_det.version_tag if active_det else "yolo26n_vtsdb_mtsd_dfg",
        data_source="backend_survey_submissions",
        distinct_full_frames_count=total_verified,
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
    compare_with_version_id: Optional[int] = Query(None, description="Model version ID to compare against (base model, defaults to active)"),
    target_version_id: Optional[int] = Query(None, description="Target model version ID to compare against (defaults to current pool)"),
    db: AsyncSession = Depends(get_async_db),
):
    base_id = compare_with_version_id
    if base_id is None:
        stmt = select(ModelVersion.id).where(ModelVersion.model_type == "CLASSIFIER", ModelVersion.is_active == True)
        res = await db.execute(stmt)
        base_id = res.scalar_one_or_none() or 2

    base_stmt = select(ModelVersion).where(ModelVersion.id == base_id)
    base_res = await db.execute(base_stmt)
    base_version = base_res.scalar_one_or_none()
    if not base_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version #{base_id} not found",
        )

    target_version = None
    if target_version_id is not None:
        t_stmt = select(ModelVersion).where(ModelVersion.id == target_version_id)
        t_res = await db.execute(t_stmt)
        target_version = t_res.scalar_one_or_none()
        if not target_version:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Target model version #{target_version_id} not found",
            )

    base_cutoff = base_version.created_at
    target_cutoff = target_version.created_at if target_version else None

    class_stmt = select(ModelClass.sign_code, ModelClass.class_name).where(ModelClass.model_id == base_id)
    class_res = await db.execute(class_stmt)
    model_classes = {r[0]: r[1] for r in class_res.fetchall()}

    verified_map = _load_verified_cache()
    sign_meta: Dict[str, Dict[str, str]] = {}

    catalog = _load_catalog()
    for item in catalog:
        sc = item.get("signCode")
        if sc and sc not in sign_meta:
            sign_meta[sc] = {
                "name_vi": item.get("nameVi", ""),
                "name_en": item.get("nameEn", ""),
            }

    base_counts: Dict[str, int] = {}
    target_counts: Dict[str, int] = {}

    for item in verified_map.values():
        sc = item.get("sign_code")
        if not sc:
            continue
        if sc not in sign_meta:
            sign_meta[sc] = {
                "name_vi": item.get("name_vi", ""),
                "name_en": item.get("name_en", ""),
            }

        v_at_str = item.get("verified_at") or item.get("created_at")
        v_at = None
        if v_at_str:
            try:
                v_at = datetime.fromisoformat(v_at_str.replace("Z", "+00:00"))
            except Exception:
                pass

        if v_at is not None and base_cutoff is not None:
            if v_at <= base_cutoff:
                base_counts[sc] = base_counts.get(sc, 0) + 1
        elif base_cutoff is None:
            base_counts[sc] = base_counts.get(sc, 0) + 1

        if target_cutoff is not None:
            if v_at is not None and v_at <= target_cutoff:
                target_counts[sc] = target_counts.get(sc, 0) + 1
        else:
            target_counts[sc] = target_counts.get(sc, 0) + 1

    added_classes = []
    removed_classes = []
    changed_classes = []
    unchanged_count = 0

    for sc, curr_cnt in target_counts.items():
        if sc not in model_classes:
            prev_cnt = base_counts.get(sc, 0)
            if curr_cnt > prev_cnt:
                added_classes.append(
                    AddedClassDiffItem(
                        sign_code=sc,
                        name_vi=sign_meta.get(sc, {}).get("name_vi", ""),
                        name_en=sign_meta.get(sc, {}).get("name_en", ""),
                        current_verified_count=curr_cnt - prev_cnt,
                    )
                )

    for sc in model_classes:
        prev_cnt = base_counts.get(sc, 0)
        curr_cnt = target_counts.get(sc, 0)
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
        compared_with_version_id=base_id,
        compared_version_tag=base_version.version_tag,
        target_version_id=target_version.id if target_version else None,
        target_version_tag=target_version.version_tag if target_version else "current_pool",
        total_classes_in_compared_model=len(model_classes),
        added_classes=added_classes,
        removed_classes=removed_classes,
        changed_classes=changed_classes,
        unchanged_classes_count=unchanged_count,
    )
