import uuid
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from src.api.schemas.inference import SubmissionSignResult
from src.core.config import STM_CANDIDATE_NAMESPACE
from src.pipelines.image_pipeline import infer_image
from src.pipelines.video_pipeline import infer_video


def test_candidate_id_deterministic_image_seed():
    sub_id = "test-submission-001"
    box = [100.23, 150.56, 300.78, 450.91]
    box_values = [round(float(v), 1) for v in box[:4]]
    seed1 = f"{sub_id}:img:0:{box_values[0]}_{box_values[1]}_{box_values[2]}_{box_values[3]}"
    id1 = uuid.uuid5(STM_CANDIDATE_NAMESPACE, seed1)

    seed2 = f"{sub_id}:img:0:{box_values[0]}_{box_values[1]}_{box_values[2]}_{box_values[3]}"
    id2 = uuid.uuid5(STM_CANDIDATE_NAMESPACE, seed2)
    assert id1 == id2

    diff_box = [100.23, 150.56, 301.0, 450.91]
    diff_values = [round(float(v), 1) for v in diff_box[:4]]
    diff_seed = f"{sub_id}:img:0:{diff_values[0]}_{diff_values[1]}_{diff_values[2]}_{diff_values[3]}"
    id_diff = uuid.uuid5(STM_CANDIDATE_NAMESPACE, diff_seed)
    assert id1 != id_diff


def test_candidate_id_deterministic_video_seed():
    sub_id = "test-submission-001"
    seed1 = f"{sub_id}:track:42"
    id1 = uuid.uuid5(STM_CANDIDATE_NAMESPACE, seed1)

    seed2 = f"{sub_id}:track:42"
    id2 = uuid.uuid5(STM_CANDIDATE_NAMESPACE, seed2)
    assert id1 == id2

    seed_diff = f"{sub_id}:track:43"
    id_diff = uuid.uuid5(STM_CANDIDATE_NAMESPACE, seed_diff)
    assert id1 != id_diff


def test_image_pipeline_generates_candidate_id(verified_images_dir: Path, mock_minio):
    fake_detector = MagicMock()
    fake_detector.detect.return_value = [
        {
            "class_id": 1,
            "class_name": "P.102",
            "confidence": 0.9,
            "detect_confidence": 0.9,
            "box_xyxy": [10.0, 10.0, 50.0, 50.0],
        }
    ]
    fake_classifier = MagicMock()
    fake_classifier.crop_box.return_value = MagicMock()
    fake_classifier.classify_crops_batch.return_value = [
        {
            "top1": {"class_id": 1, "class_name": "P.102", "confidence": 0.95, "classify_confidence": 0.95},
            "class_id": 1,
            "class_name": "P.102",
            "classify_confidence": 0.95,
        }
    ]

    img_path = str(verified_images_dir / "1007.jpg")
    sub_id = "sub_candidate_id_img_test"
    res1 = infer_image(
        submission_id=sub_id,
        detector=fake_detector,
        image_path=img_path,
        source_object_key="raw/1007.jpg",
        classifier=fake_classifier,
    )

    res2 = infer_image(
        submission_id=sub_id,
        detector=fake_detector,
        image_path=img_path,
        source_object_key="raw/1007.jpg",
        classifier=fake_classifier,
    )

    assert len(res1["signs"]) == 1
    assert "candidate_id" in res1["signs"][0]
    cand_id_str1 = res1["signs"][0]["candidate_id"]
    cand_id_str2 = res2["signs"][0]["candidate_id"]

    assert cand_id_str1 == cand_id_str2
    parsed = uuid.UUID(cand_id_str1)
    assert parsed.version == 5


def test_submission_sign_result_schema_validates_candidate_id():
    cand_id = uuid.uuid4()
    item = SubmissionSignResult(
        candidate_id=cand_id,
        track_id=1,
        class_id=1,
        class_name="P.102",
        confidence=0.9,
        detect_confidence=0.9,
        box_xyxy=(10.0, 10.0, 50.0, 50.0),
        best_frame_key="raw/frame.jpg",
        sign_crop_key="crops/crop.jpg",
        frame_index=0,
        timestamp_seconds=1.0,
    )
    assert item.candidate_id == cand_id
    assert item.model_dump()["candidate_id"] == cand_id


def test_video_pipeline_generates_candidate_id(monkeypatch: pytest.MonkeyPatch, mock_minio):
    import numpy as np

    fake_detector = MagicMock()
    fake_detector.track.return_value = [
        (
            0,
            [
                {
                    "id": 1,
                    "class_id": 1,
                    "class_name": "P.102",
                    "confidence": 0.9,
                    "detect_confidence": 0.9,
                    "box_xyxy": [10.0, 10.0, 50.0, 50.0],
                }
            ],
        )
    ]
    fake_classifier = MagicMock()
    fake_classifier.crop_box.return_value = MagicMock()
    fake_classifier.classify_crop.return_value = {
        "top1": {"class_id": 1, "class_name": "P.102", "confidence": 0.95, "classify_confidence": 0.95},
        "class_id": 1,
        "class_name": "P.102",
        "classify_confidence": 0.95,
    }

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    fake_cap = MagicMock()
    fake_cap.isOpened.return_value = True
    fake_cap.get.side_effect = lambda prop: 30.0 if prop == 5 else 100.0
    def mock_extract(vpath, fidx, outpath):
        Path(outpath).touch()
        return True

    def mock_imwrite(outpath, img):
        Path(outpath).touch()
        return True

    monkeypatch.setattr("cv2.VideoCapture", lambda *args, **kwargs: fake_cap)
    monkeypatch.setattr("src.pipelines.video_pipeline.extract_video_frame", mock_extract)
    monkeypatch.setattr("cv2.imread", lambda *args, **kwargs: dummy_frame)
    monkeypatch.setattr("cv2.imwrite", mock_imwrite)

    sub_id = "sub_candidate_id_vid_test"
    res = infer_video(
        submission_id=sub_id,
        detector=fake_detector,
        video_path="dummy.mp4",
        classifier=fake_classifier,
    )

    assert len(res["signs"]) == 1
    assert "candidate_id" in res["signs"][0]
    expected_seed = f"{sub_id}:track:1"
    expected_id = uuid.uuid5(STM_CANDIDATE_NAMESPACE, expected_seed)
    assert res["signs"][0]["candidate_id"] == str(expected_id)
