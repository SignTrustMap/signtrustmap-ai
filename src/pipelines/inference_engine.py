import os
import tempfile
import threading
from pathlib import Path
from typing import Iterable

from src.core.config_manager import config_manager
from src.core.model_manager import model_manager
from src.infrastructure.minio_client import minio_client
from src.pipelines.image_pipeline import infer_image
from src.pipelines.video_pipeline import infer_video


class SubmissionInferenceEngine:
    """Stateless adapter from MinIO media references to model inference."""

    def __init__(self) -> None:
        self._inference_lock = threading.Lock()

    def process(self, submission_id: str, media_objects: Iterable[dict]) -> dict:
        with self._inference_lock:
            return self._process(submission_id, media_objects)

    def _process(self, submission_id: str, media_objects: Iterable[dict]) -> dict:
        objects = list(media_objects)
        primary = self._primary_media(objects)
        suffix = Path(primary["object_key"]).suffix

        with tempfile.TemporaryDirectory(prefix=f"stm-input-{submission_id}-") as temp_dir:
            local_path = os.path.join(temp_dir, f"input{suffix}")
            downloaded = minio_client.download_file(
                primary["bucket"],
                primary["object_key"],
                local_path,
            )
            if not downloaded:
                raise RuntimeError(
                    f"Unable to download {primary['bucket']}/{primary['object_key']}"
                )

            detector = model_manager.get_detector()
            classifier = model_manager.get_classifier()
            common = {
                "submission_id": submission_id,
                "detector": detector,
                "classifier": classifier,
                "conf": config_manager.default_detector_conf,
                "device": config_manager.default_device,
            }
            if primary["media_type"] == "IMAGE":
                result = infer_image(
                    **common,
                    image_path=local_path,
                    source_object_key=f"{primary['bucket']}/{primary['object_key']}",
                )
            else:
                result = infer_video(**common, video_path=local_path)

        return {
            "submission_id": submission_id,
            "media_type": primary["media_type"],
            **result,
        }

    @staticmethod
    def _primary_media(media_objects: list[dict]) -> dict:
        for media_type in ("VIDEO", "IMAGE"):
            for media_object in media_objects:
                if media_object["media_type"] == media_type:
                    return media_object
        raise ValueError("A VIDEO or IMAGE MinIO object is required for inference")


inference_engine = SubmissionInferenceEngine()
