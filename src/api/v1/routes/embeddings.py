from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from loguru import logger

from src.infrastructure.db import get_async_db
from src.infrastructure.models import CandidateEmbedding
from src.api.schemas.embeddings import (
    EmbeddingItem,
    EmbeddingBatchRequest,
    EmbeddingSimilarityQuery,
    EmbeddingSimilarityResult,
)

router = APIRouter(tags=["Embeddings"])


@router.post("", status_code=status.HTTP_201_CREATED)
async def store_candidate_embedding(
    request: EmbeddingItem,
    db: AsyncSession = Depends(get_async_db),
):
    """Store or update a visual embedding for a candidate in AI Database."""
    try:
        existing = await db.get(CandidateEmbedding, request.candidate_id)
        if existing:
            existing.clip_embedding = request.embedding
            if request.embedding_model_version_id is not None:
                existing.embedding_model_version_id = request.embedding_model_version_id
        else:
            db_item = CandidateEmbedding(
                candidate_id=request.candidate_id,
                clip_embedding=request.embedding,
                embedding_model_version_id=request.embedding_model_version_id,
            )
            db.add(db_item)
        await db.commit()
        return {"status": "SUCCESS", "candidate_id": str(request.candidate_id)}
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to store embedding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store embedding: {str(e)}",
        )


@router.post("/batch", status_code=status.HTTP_201_CREATED)
async def store_embeddings_batch(
    request: EmbeddingBatchRequest,
    db: AsyncSession = Depends(get_async_db),
):
    """Store a batch of candidate visual embeddings into AI Database."""
    try:
        count = 0
        for item in request.items:
            existing = await db.get(CandidateEmbedding, item.candidate_id)
            if existing:
                existing.clip_embedding = item.embedding
                if item.embedding_model_version_id is not None:
                    existing.embedding_model_version_id = item.embedding_model_version_id
            else:
                db_item = CandidateEmbedding(
                    candidate_id=item.candidate_id,
                    clip_embedding=item.embedding,
                    embedding_model_version_id=item.embedding_model_version_id,
                )
                db.add(db_item)
            count += 1
        await db.commit()
        return {"status": "SUCCESS", "count": count}
    except Exception as e:
        await db.rollback()
        logger.error(f"Failed to store embeddings batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store embeddings batch: {str(e)}",
        )


@router.get("/{candidate_id}")
async def get_candidate_embedding(
    candidate_id: UUID,
    db: AsyncSession = Depends(get_async_db),
):
    """Retrieve visual embedding for a candidate from AI Database."""
    record = await db.get(CandidateEmbedding, candidate_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Embedding for candidate '{candidate_id}' not found in AI DB.",
        )
    return {
        "candidate_id": str(record.candidate_id),
        "embedding": list(record.clip_embedding) if hasattr(record.clip_embedding, "__iter__") else record.clip_embedding,
        "embedding_model_version_id": record.embedding_model_version_id,
        "generated_at": record.generated_at.isoformat() if record.generated_at else None,
    }


@router.post("/similarity", response_model=List[EmbeddingSimilarityResult])
async def search_similar_candidates(
    query: EmbeddingSimilarityQuery,
    db: AsyncSession = Depends(get_async_db),
):
    """Search nearest neighbor candidates using pgvector cosine/L2 distance."""
    try:
        vec_str = "[" + ",".join(str(x) for x in query.embedding) + "]"
        raw_sql = text("""
            SELECT candidate_id, embedding_model_version_id,
                   (clip_embedding <=> :vec) AS distance
            FROM candidate_embeddings
            ORDER BY clip_embedding <=> :vec
            LIMIT :k
        """)
        result = await db.execute(raw_sql, {"vec": vec_str, "k": query.top_k})
        rows = result.fetchall()
        return [
            EmbeddingSimilarityResult(
                candidate_id=r[0],
                embedding_model_version_id=r[1],
                similarity_score=float(1.0 - r[2]),
            )
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity search failed: {str(e)}",
        )