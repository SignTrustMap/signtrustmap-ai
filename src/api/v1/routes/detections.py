from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from src.infrastructure.db import get_async_db
from src.infrastructure.models import CandidateDetectionMetadata
from src.api.schemas.detections import (
    BatchDetectionMetadataRequest,
    DetectionMetadataItem,
    DetectionMetadataResponse,
)

router = APIRouter(tags=["Detections"])


@router.post("/metadata", response_model=DetectionMetadataResponse)
async def upsert_detection_metadata(
    request: BatchDetectionMetadataRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """Upsert original candidate detection metadata into AI Database."""
    try:
        count = 0
        for item in request.items:
            existing = await db.get(CandidateDetectionMetadata, item.candidate_id)
            if existing:
                existing.frame_number = item.frame_number
                existing.bbox_x = item.bbox_x
                existing.bbox_y = item.bbox_y
                existing.bbox_width = item.bbox_width
                existing.bbox_height = item.bbox_height
                existing.tracking_id = item.tracking_id
                if item.detector_model_version_id is not None:
                    existing.detector_model_version_id = item.detector_model_version_id
            else:
                db_item = CandidateDetectionMetadata(
                    candidate_id=item.candidate_id,
                    frame_number=item.frame_number,
                    bbox_x=item.bbox_x,
                    bbox_y=item.bbox_y,
                    bbox_width=item.bbox_width,
                    bbox_height=item.bbox_height,
                    tracking_id=item.tracking_id,
                    detector_model_version_id=item.detector_model_version_id,
                )
                db.add(db_item)
            count += 1
        await db.commit()
        logger.info(f"Successfully upserted {count} detection metadata record(s) into AI DB.")
        return DetectionMetadataResponse(status="SUCCESS", count=count)
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to upsert detection metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upsert detection metadata: {str(e)}",
        )


@router.get("/{candidate_id}", response_model=DetectionMetadataItem)
async def get_candidate_detection_metadata(
    candidate_id: UUID,
    db: AsyncSession = Depends(get_async_db),
):
    """Retrieve original detection metadata for a candidate from AI Database."""
    record = await db.get(CandidateDetectionMetadata, candidate_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Detection metadata for candidate '{candidate_id}' not found in AI DB.",
        )
    return DetectionMetadataItem(
        candidate_id=record.candidate_id,
        frame_number=record.frame_number,
        bbox_x=record.bbox_x,
        bbox_y=record.bbox_y,
        bbox_width=record.bbox_width,
        bbox_height=record.bbox_height,
        tracking_id=record.tracking_id,
        detector_model_version_id=record.detector_model_version_id,
    )


@router.get("", response_model=List[DetectionMetadataItem])
async def list_detection_metadata(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_async_db),
):
    """List original detection metadata records with pagination."""
    stmt = select(CandidateDetectionMetadata).limit(min(limit, 500)).offset(offset)
    result = await db.execute(stmt)
    records = result.scalars().all()
    return [
        DetectionMetadataItem(
            candidate_id=r.candidate_id,
            frame_number=r.frame_number,
            bbox_x=r.bbox_x,
            bbox_y=r.bbox_y,
            bbox_width=r.bbox_width,
            bbox_height=r.bbox_height,
            tracking_id=r.tracking_id,
            detector_model_version_id=r.detector_model_version_id,
        )
        for r in records
    ]