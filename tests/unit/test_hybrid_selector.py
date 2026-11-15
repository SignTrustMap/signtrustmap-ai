from typing import Any, Dict, List
import uuid
import numpy as np
import pytest
import torch
from loguru import logger
from sqlalchemy import select

from src.core.active_learning.hybrid_selector import HybridALSelector
from src.infrastructure.db import get_sync_db
from src.infrastructure.models import (
    ActiveLearningContextualPrior,
    ActiveLearningRecord,
    CandidateEmbedding,
)


@pytest.fixture
def hybrid_selector():
    return HybridALSelector(device="cpu")


def _insert_test_candidates(cids: List[uuid.UUID], feature_dim: int = 512) -> None:
    rng = np.random.RandomState(42)
    with get_sync_db() as session:
        for idx, cid in enumerate(cids):
            vec = rng.randn(feature_dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            ce = CandidateEmbedding(
                candidate_id=cid,
                s3_key=f"test_run/cand_{idx}.jpg",
                model_id=2,
                embedding=vec.tolist(),
            )
            session.merge(ce)
        session.commit()


def _cleanup_test_data(cids: List[uuid.UUID], run_ids: List[str]) -> None:
    with get_sync_db() as session:
        for cid in cids:
            session.query(ActiveLearningRecord).filter(ActiveLearningRecord.candidate_id == cid).delete()
            session.query(CandidateEmbedding).filter(CandidateEmbedding.candidate_id == cid).delete()
        for rid in run_ids:
            session.query(ActiveLearningContextualPrior).filter(ActiveLearningContextualPrior.run_id == rid).delete()
        session.commit()


def test_ai_hybrid_01_corner_case_n0_or_k0(hybrid_selector: HybridALSelector):
    res_n0 = hybrid_selector.select_candidates(
        run_id="test_run_n0",
        candidate_ids=[],
        query_budget=10,
        save_db=False,
    )
    assert res_n0["total_pool_size"] == 0
    assert res_n0["selected_count"] == 0
    assert res_n0["selected_candidates"] == []

    res_k0 = hybrid_selector.select_candidates(
        run_id="test_run_k0",
        candidate_ids=[str(uuid.uuid4())],
        query_budget=0,
        save_db=False,
    )
    assert res_k0["selected_count"] == 0
    assert res_k0["selected_candidates"] == []


def test_ai_hybrid_02_corner_case_n_le_k(hybrid_selector: HybridALSelector):
    run_id = f"test_run_n_le_k_{uuid.uuid4().hex[:8]}"
    test_cids = [uuid.uuid4() for _ in range(4)]
    _insert_test_candidates(test_cids)

    try:
        res = hybrid_selector.select_candidates(
            run_id=run_id,
            candidate_ids=[str(c) for c in test_cids],
            query_budget=10,
            save_db=True,
        )
        assert res["total_pool_size"] == 4
        assert res["selected_count"] == 4
        assert len(res["selected_candidates"]) == 4

        for item in res["selected_candidates"]:
            assert item["selection_status"] == "ALL_SELECTED"
            assert item["cluster_id"] == 0
            assert item["aggregate_rank"] == 3

        with get_sync_db() as session:
            stmt = select(ActiveLearningRecord).where(ActiveLearningRecord.batch_run_id == run_id)
            db_records = session.execute(stmt).scalars().all()
            assert len(db_records) == 4
            for r in db_records:
                assert r.is_selected is True
    finally:
        _cleanup_test_data(test_cids, [run_id])


def test_ai_hybrid_03_corner_case_k_lt_n_lt_3k(hybrid_selector: HybridALSelector):
    run_id = f"test_run_between_{uuid.uuid4().hex[:8]}"
    test_cids = [uuid.uuid4() for _ in range(8)]
    _insert_test_candidates(test_cids)

    try:
        res = hybrid_selector.select_candidates(
            run_id=run_id,
            candidate_ids=[str(c) for c in test_cids],
            query_budget=4,
            shortlist_factor=3,
            save_db=True,
        )
        assert res["total_pool_size"] == 8
        assert res["shortlist_size"] == 8
        assert res["selected_count"] == 4
        assert len(res["selected_candidates"]) == 4

        selected_ids = {s["candidate_id"] for s in res["selected_candidates"]}
        assert len(selected_ids) == 4

        with get_sync_db() as session:
            stmt = select(ActiveLearningRecord).where(ActiveLearningRecord.batch_run_id == run_id)
            db_records = session.execute(stmt).scalars().all()
            assert len(db_records) == 8
            selected_count = sum(1 for r in db_records if r.is_selected)
            assert selected_count == 4
    finally:
        _cleanup_test_data(test_cids, [run_id])


def test_ai_hybrid_04_corner_case_n_ge_3k(hybrid_selector: HybridALSelector):
    run_id = f"test_run_ge_3k_{uuid.uuid4().hex[:8]}"
    test_cids = [uuid.uuid4() for _ in range(12)]
    _insert_test_candidates(test_cids)

    try:
        res = hybrid_selector.select_candidates(
            run_id=run_id,
            candidate_ids=[str(c) for c in test_cids],
            query_budget=3,
            shortlist_factor=3,
            save_db=True,
        )
        assert res["total_pool_size"] == 12
        assert res["shortlist_size"] == 9
        assert res["selected_count"] == 3
        assert len(res["selected_candidates"]) == 3

        selected_ids = {s["candidate_id"] for s in res["selected_candidates"]}
        assert len(selected_ids) == 3

        with get_sync_db() as session:
            stmt = select(ActiveLearningRecord).where(ActiveLearningRecord.batch_run_id == run_id)
            db_records = session.execute(stmt).scalars().all()
            assert len(db_records) == 12
            selected_count = sum(1 for r in db_records if r.is_selected)
            assert selected_count == 3
    finally:
        _cleanup_test_data(test_cids, [run_id])


def test_ai_hybrid_05_prior_snapshot_and_freeze(hybrid_selector: HybridALSelector):
    run_id = f"test_prior_snap_{uuid.uuid4().hex[:8]}"
    model_id = 2
    probs = np.random.RandomState(42).dirichlet(np.ones(10), size=15)

    try:
        p0_first = hybrid_selector.get_or_create_prior_snapshot(
            run_id=run_id,
            model_id=model_id,
            probabilities=probs,
            top_n=3,
        )
        assert len(p0_first) == 10

        with get_sync_db() as session:
            stmt = select(ActiveLearningContextualPrior).where(
                ActiveLearningContextualPrior.run_id == run_id,
                ActiveLearningContextualPrior.model_id == model_id,
            )
            rows = session.execute(stmt).scalars().all()
            assert len(rows) == 1
            assert rows[0].pool_size == 15

        dummy_probs = np.ones((5, 10)) / 10.0
        p0_second = hybrid_selector.get_or_create_prior_snapshot(
            run_id=run_id,
            model_id=model_id,
            probabilities=dummy_probs,
            top_n=3,
        )
        np.testing.assert_allclose(p0_first, p0_second, atol=1e-7)

        with get_sync_db() as session:
            stmt = select(ActiveLearningContextualPrior).where(
                ActiveLearningContextualPrior.run_id == run_id,
                ActiveLearningContextualPrior.model_id == model_id,
            )
            rows = session.execute(stmt).scalars().all()
            assert len(rows) == 1
    finally:
        _cleanup_test_data([], [run_id])


def test_ai_hybrid_06_score_candidates_by_ids(hybrid_selector: HybridALSelector):
    test_cids = [uuid.uuid4() for _ in range(3)]
    _insert_test_candidates(test_cids)

    try:
        scored = hybrid_selector.score_candidates_by_ids(
            candidate_ids=[str(c) for c in test_cids],
            strategy="fdal",
            save_db=True,
        )
        assert len(scored) == 3
        for item in scored:
            assert "candidate_id" in item
            assert "acquisition_score" in item
            assert "flip_count" in item
            assert "flip_classes" in item
            assert "calibrated_entropy" in item

        with get_sync_db() as session:
            for cid in test_cids:
                stmt = select(ActiveLearningRecord).where(ActiveLearningRecord.candidate_id == cid)
                rec = session.execute(stmt).scalar_one_or_none()
                assert rec is not None
                assert rec.strategy == "fdal"
                assert rec.acquisition_score >= 0.0
    finally:
        _cleanup_test_data(test_cids, [])
