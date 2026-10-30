import shutil
from pathlib import Path
from typing import Dict, Any, List
import pytest
from fastapi.testclient import TestClient
from loguru import logger
from main import app
from src.core.config import settings


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_ai_it_01_real_image_inference(api_client: TestClient, verified_images_dir: Path, mock_minio: Path):
    src_img = verified_images_dir / "11288.jpg"
    dest_img = mock_minio / "signtrustmap" / "raw" / "11288.jpg"
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_img, dest_img)

    payload = {
        "media_objects": [
            {"media_type": "IMAGE", "bucket": "signtrustmap", "object_key": "raw/11288.jpg"}
        ]
    }
    resp = api_client.post(
        "/api/v1/inference/submissions/sub_it_01",
        json=payload,
        headers={"X-Internal-Token": settings.INTERNAL_API_TOKEN},
    )

    assert resp.status_code == 200, f"Error: {resp.status_code} - {resp.text}"
    data = resp.json()
    assert data["submission_id"] == "sub_it_01"
    assert data["media_type"] == "IMAGE"
    assert len(data["signs"]) >= 1

    found_sign = any(s.get("name_vi") is not None for s in data["signs"])
    assert found_sign

    crop_path = mock_minio / data["signs"][0]["sign_crop_key"]
    assert crop_path.exists()
    logger.info(f"AI-IT-01 verified: End-to-end image inference detected sign and stored crop.")


def test_ai_it_02_real_video_inference(api_client: TestClient, real_videos_dir: Path, mock_minio: Path):
    src_vid = real_videos_dir / "clip_000505.mp4"
    dest_vid = mock_minio / "signtrustmap" / "raw" / "clip_000505.mp4"
    dest_vid.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_vid, dest_vid)

    payload = {
        "media_objects": [
            {"media_type": "VIDEO", "bucket": "signtrustmap", "object_key": "raw/clip_000505.mp4"}
        ]
    }
    resp = api_client.post(
        "/api/v1/inference/submissions/sub_it_02",
        json=payload,
        headers={"X-Internal-Token": settings.INTERNAL_API_TOKEN},
    )

    assert resp.status_code == 200, f"Error: {resp.status_code} - {resp.text}"
    data = resp.json()
    assert data["submission_id"] == "sub_it_02"
    assert data["media_type"] == "VIDEO"
    assert len(data["signs"]) == 3

    track_ids = {s["track_id"] for s in data["signs"]}
    assert len(track_ids) == 3

    for s in data["signs"]:
        frame_file = mock_minio / s["best_frame_key"]
        crop_file = mock_minio / s["sign_crop_key"]
        assert frame_file.exists()
        assert crop_file.exists()
    logger.info("AI-IT-02 verified: End-to-end video inference processed 3 tracks with storage artifacts.")


def test_ai_it_03_blank_canvas_zero_artifacts(api_client: TestClient, verified_images_dir: Path, mock_minio: Path):
    src_blank = verified_images_dir / "blank_canvas.jpg"
    dest_blank = mock_minio / "signtrustmap" / "raw" / "blank.jpg"
    dest_blank.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_blank, dest_blank)

    payload = {
        "media_objects": [
            {"media_type": "IMAGE", "bucket": "signtrustmap", "object_key": "raw/blank.jpg"}
        ]
    }
    resp = api_client.post(
        "/api/v1/inference/submissions/sub_it_03",
        json=payload,
        headers={"X-Internal-Token": settings.INTERNAL_API_TOKEN},
    )

    assert resp.status_code == 200, f"Error: {resp.status_code} - {resp.text}"
    data = resp.json()
    assert data["signs"] == []

    candidates_dir = mock_minio / "signtrustmap" / "candidates" / "sub_it_03"
    assert not candidates_dir.exists()
    logger.info("AI-IT-03 verified: Blank canvas produces zero artifacts on MinIO.")


def test_ai_it_04_missing_media_download_error(api_client: TestClient):
    payload = {
        "media_objects": [
            {"media_type": "IMAGE", "bucket": "signtrustmap", "object_key": "raw/missing_file.jpg"}
        ]
    }
    resp = api_client.post(
        "/api/v1/inference/submissions/sub_it_04",
        json=payload,
        headers={"X-Internal-Token": settings.INTERNAL_API_TOKEN},
    )
    assert resp.status_code == 500
    logger.info("AI-IT-04 verified: Missing object download returns 500 error.")

