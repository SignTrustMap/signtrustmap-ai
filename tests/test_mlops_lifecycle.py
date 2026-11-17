import os
from typing import Any, Dict
import pytest
import requests

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")


def test_training_data_status():
    resp = requests.get(f"{BASE_URL}/training-data/status", timeout=10)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["total_verified_signs_platform"] >= 400
    assert "classifier_pool" in data
    assert "detector_pool" in data
    cls_pool = data["classifier_pool"]
    assert cls_pool["total_verified_crops_in_ai_db"] >= 400
    assert cls_pool["min_samples_threshold"] == 100
    assert cls_pool["is_eligible_for_rebuild"] is True


def test_training_data_class_distribution():
    resp = requests.get(f"{BASE_URL}/training-data/class-distribution", timeout=10)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["total_catalog_classes"] == 429
    assert data["classes_with_verified_signs"] >= 40
    dist = data["distribution"]
    assert len(dist) == 429
    assert dist[0]["verified_count"] > 0


def test_training_data_class_distribution_diff():
    resp = requests.get(f"{BASE_URL}/training-data/class-distribution/diff?compare_with_version_id=2", timeout=10)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert data["compared_with_version_id"] == 2
    assert len(data["added_classes"]) >= 1
    added_codes = [c["sign_code"] for c in data["added_classes"]]
    assert "I.407b" in added_codes
    assert len(data["changed_classes"]) >= 10


def test_classifier_multi_version_mlops_lifecycle():
    rebuild_payload_v3 = {
        "version_tag": "classifier-v2.1",
        "note": "Multi-version simulation batch 1 (240 samples)",
        "sample_limit": 240,
        "alpha": 250.0,
        "beta": 0.8,
        "shots_per_class": 16,
    }
    resp_v3 = requests.post(f"{BASE_URL}/classifier/rebuild", json=rebuild_payload_v3, timeout=60)
    assert resp_v3.status_code == 201, resp_v3.text
    v3_data = resp_v3.json()
    assert v3_data["status"] == "SUCCESS"
    v3_id = v3_data["new_model"]["version_id"]
    assert v3_data["new_model"]["version_tag"] == "classifier-v2.1"
    assert v3_data["new_model"]["is_active"] is False
    assert v3_data["evaluation_metrics"]["sample_count"] == 5911
    assert v3_data["evaluation_metrics"]["top1_accuracy"] > 65.0

    eval_v3_resp = requests.get(f"{BASE_URL}/classifier/versions/{v3_id}/evaluation", timeout=10)
    assert eval_v3_resp.status_code == 200, eval_v3_resp.text
    eval_v3_data = eval_v3_resp.json()
    assert eval_v3_data["version_id"] == v3_id
    assert len(eval_v3_data["metrics"]) >= 4

    deploy_v3_resp = requests.post(f"{BASE_URL}/classifier/versions/{v3_id}/deploy", timeout=15)
    assert deploy_v3_resp.status_code == 200, deploy_v3_resp.text
    deploy_v3_data = deploy_v3_resp.json()
    assert deploy_v3_data["deployed_version_id"] == v3_id
    assert deploy_v3_data["is_active"] is True

    versions_resp = requests.get(f"{BASE_URL}/classifier/versions", timeout=10)
    assert versions_resp.status_code == 200, versions_resp.text
    versions_list = versions_resp.json()
    v3_rec = next((v for v in versions_list if v["id"] == v3_id), None)
    assert v3_rec is not None
    assert v3_rec["is_active"] is True
    v2_rec = next((v for v in versions_list if v["id"] == 2), None)
    if v2_rec:
        assert v2_rec["is_active"] is False

    rebuild_payload_v4 = {
        "version_tag": "classifier-v2.2",
        "note": "Multi-version simulation batch 2 (489 samples)",
        "sample_limit": 489,
        "alpha": 250.0,
        "beta": 0.8,
        "shots_per_class": 16,
    }
    resp_v4 = requests.post(f"{BASE_URL}/classifier/rebuild", json=rebuild_payload_v4, timeout=60)
    assert resp_v4.status_code == 201, resp_v4.text
    v4_data = resp_v4.json()
    assert v4_data["status"] == "SUCCESS"
    v4_id = v4_data["new_model"]["version_id"]
    assert v4_data["new_model"]["version_tag"] == "classifier-v2.2"
    assert v4_data["new_model"]["is_active"] is False

    deploy_v4_resp = requests.post(f"{BASE_URL}/classifier/versions/{v4_id}/deploy", timeout=15)
    assert deploy_v4_resp.status_code == 200, deploy_v4_resp.text
    deploy_v4_data = deploy_v4_resp.json()
    assert deploy_v4_data["deployed_version_id"] == v4_id
    assert deploy_v4_data["is_active"] is True

    diff_v4_resp = requests.get(f"{BASE_URL}/training-data/class-distribution/diff", timeout=10)
    assert diff_v4_resp.status_code == 200, diff_v4_resp.text
    assert len(diff_v4_resp.json()["changed_classes"]) == 0

    rollback_resp = requests.post(f"{BASE_URL}/classifier/rollback", timeout=15)
    assert rollback_resp.status_code == 200, rollback_resp.text
    rollback_data = rollback_resp.json()
    assert rollback_data["status"] == "SUCCESS"
    assert rollback_data["rolled_back_from"] == v4_id
    assert rollback_data["active_version_id"] == v3_id

    diff_v3_resp = requests.get(f"{BASE_URL}/training-data/class-distribution/diff", timeout=10)
    assert diff_v3_resp.status_code == 200, diff_v3_resp.text
    assert len(diff_v3_resp.json()["changed_classes"]) > 0

    diff_v3_v4_resp = requests.get(f"{BASE_URL}/training-data/class-distribution/diff?compare_with_version_id={v3_id}&target_version_id={v4_id}", timeout=10)
    assert diff_v3_v4_resp.status_code == 200, diff_v3_v4_resp.text
    v3_v4_data = diff_v3_v4_resp.json()
    assert v3_v4_data["target_version_tag"] == "classifier-v2.2"
    assert len(v3_v4_data["changed_classes"]) > 0

    switch_v2_resp = requests.post(f"{BASE_URL}/classifier/switch/2", timeout=15)
    assert switch_v2_resp.status_code == 200, switch_v2_resp.text
    assert switch_v2_resp.json()["switched_to_version_id"] == 2

    diff_v2_resp = requests.get(f"{BASE_URL}/training-data/class-distribution/diff", timeout=10)
    assert diff_v2_resp.status_code == 200, diff_v2_resp.text
    assert len(diff_v2_resp.json()["changed_classes"]) >= 10

    switch_v4_resp = requests.post(f"{BASE_URL}/classifier/switch/{v4_id}", timeout=15)
    assert switch_v4_resp.status_code == 200, switch_v4_resp.text
    assert switch_v4_resp.json()["switched_to_version_id"] == v4_id


def test_detector_mlops_endpoints():
    versions_resp = requests.get(f"{BASE_URL}/detector/versions", timeout=10)
    assert versions_resp.status_code == 200, versions_resp.text
    data = versions_resp.json()
    assert len(data) >= 1
    det_id = data[0]["id"]

    eval_resp = requests.get(f"{BASE_URL}/detector/versions/{det_id}/evaluation", timeout=10)
    assert eval_resp.status_code == 200, eval_resp.text
    assert eval_resp.json()["version_id"] == det_id

    deploy_resp = requests.post(f"{BASE_URL}/detector/versions/{det_id}/deploy", timeout=15)
    assert deploy_resp.status_code == 200, deploy_resp.text
    assert deploy_resp.json()["deployed_version_id"] == det_id

    switch_resp = requests.post(f"{BASE_URL}/detector/switch/{det_id}", timeout=15)
    assert switch_resp.status_code == 200, switch_resp.text
    assert switch_resp.json()["switched_to_version_id"] == det_id


def test_feedback_consensus_handling():
    from src.core.active_learning.feedback_processor import feedback_processor

    res_unchanged = feedback_processor.process_consensus_feedback(
        candidate_id="538643d8-4304-5f33-b358-578b0153d24b",
        outcome="TP_BBOX_UNCHANGED",
        final_sign_code="P.102",
    )
    assert res_unchanged["action"] == "candidate_retained_in_pool"

    res_edited = feedback_processor.process_consensus_feedback(
        candidate_id="538643d8-4304-5f33-b358-578b0153d24b",
        outcome="TP_BBOX_EDITED",
        final_sign_code="P.102",
        final_box_xyxy=[15.0, 25.0, 110.0, 160.0],
    )
    assert res_edited["action"] == "bbox_updated"

    res_fp = feedback_processor.process_consensus_feedback(
        candidate_id="538643d8-4304-5f33-b358-578b0153d24b",
        outcome="FALSE_POSITIVE",
    )
    assert res_fp["action"] == "candidate_discarded"
