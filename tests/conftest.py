import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Generator
import pytest
import torch
from loguru import logger
from src.core.active_learning.store import al_store
from src.core.classifier import SignClassifier
from src.core.config import settings
from src.core.detector import SignDetector
from src.infrastructure.minio_client import minio_client


@pytest.fixture(autouse=True)
def isolate_al_store(tmp_path: Path) -> Generator[Path, None, None]:
    temp_store = tmp_path / "test_active_learning_dataset.json"
    al_store.set_store_path(temp_store)
    yield temp_store
    al_store.set_store_path(None)


@pytest.fixture(autouse=True)
def cleanup_resources() -> Generator[None, None, None]:
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def detector() -> SignDetector:
    logger.info("Initializing session SignDetector")
    return SignDetector()


@pytest.fixture(scope="session")
def classifier() -> SignClassifier:
    logger.info("Initializing session SignClassifier")
    return SignClassifier()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def verified_images_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "verified_images"


@pytest.fixture(scope="session")
def verified_labels_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "verified_labels"


@pytest.fixture(scope="session")
def real_videos_dir(fixtures_dir: Path) -> Path:
    return fixtures_dir / "real_videos"


@pytest.fixture(scope="session")
def real_features_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "real_features" / "features_test_slice.pt"


@pytest.fixture(scope="session")
def verified_tracks_metadata(real_videos_dir: Path) -> Dict[str, Any]:
    meta_path = real_videos_dir / "verified_tracks_clip_000505.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def output_dirs() -> Dict[str, Path]:
    base = Path(__file__).resolve().parent / "outputs" / "latest"
    dirs = {
        "predictions": base / "predictions",
        "crops": base / "crops",
        "annotated_frames": base / "annotated_frames",
        "reports": base / "reports",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture(autouse=True)
def mock_minio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    storage_root = tmp_path / "minio_mock_storage"
    storage_root.mkdir(parents=True, exist_ok=True)

    def mock_upload(local_file_path: str, object_name: str, bucket_name: str = None, content_type: str = None):
        bucket = bucket_name or settings.MINIO_OUTPUT_BUCKET or "stm-sign-crops"
        dest = storage_root / bucket / object_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_file_path, dest)
        return True, f"{bucket}/{object_name}"

    def mock_download(bucket_name: str, object_name: str, local_file_path: str):
        src = storage_root / bucket_name / object_name
        if not src.exists():
            matches = list(storage_root.glob(f"*/{object_name}"))
            if matches:
                src = matches[0]
            else:
                return False
        Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, local_file_path)
        return True

    def mock_is_configured():
        return True

    monkeypatch.setattr(minio_client, "upload_file", mock_upload)
    monkeypatch.setattr(minio_client, "download_file", mock_download)
    monkeypatch.setattr(minio_client, "is_configured", mock_is_configured)
    return storage_root

