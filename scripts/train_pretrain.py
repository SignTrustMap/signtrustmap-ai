import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import ultralytics.data.utils as data_utils
import ultralytics.engine.exporter as exporter
import ultralytics.engine.trainer as trainer
import ultralytics.engine.validator as validator
from loguru import logger
from ultralytics import YOLO
from ultralytics.data.base import BaseDataset
from src.training import monkeypatch

dataset_yaml_path = "configs/training/all.yaml"
active_classes = []

with open(dataset_yaml_path, "r") as f:
    for line in f:
        if ":" in line and "# blacklist" not in line and "names:" not in line and "path:" not in line:
            parts = line.strip().split(":")
            if len(parts) >= 2:
                try:
                    active_classes.append(int(parts[0].strip()))
                except ValueError:
                    pass

_orig_check = data_utils.check_det_dataset


def _patched_check(dataset, autodownload=True):
    data = _orig_check(dataset, autodownload=autodownload)
    if isinstance(data, dict):
        data["names"] = {0: "main_sign", 1: "supplementary_sign", 2: "text_sign"}
        data["nc"] = 3
    return data


data_utils.check_det_dataset = _patched_check
trainer.check_det_dataset = _patched_check
validator.check_det_dataset = _patched_check
exporter.check_det_dataset = _patched_check

_orig_patched_update = BaseDataset.update_labels


def _patched_3cls_labels(self, include_class=None):
    _orig_patched_update(self, active_classes)
    for label in self.labels:
        cls_arr = label["cls"]
        if cls_arr.size == 0:
            continue
        new_cls = []
        for c in cls_arr.flatten():
            c = int(c)
            if c == 300 or (302 <= c <= 346):
                new_cls.append(1)
            elif c == 301 or (673 <= c <= 675):
                new_cls.append(2)
            else:
                new_cls.append(0)
        label["cls"] = np.array(new_cls, dtype=cls_arr.dtype).reshape(-1, 1)


BaseDataset.update_labels = _patched_3cls_labels

if __name__ == "__main__":
    model = YOLO("weights/yolo26n.pt")

    results = model.train(
        data=dataset_yaml_path,
        epochs=100,
        batch=128,
        imgsz=640,
        device=[0],
        workers=8,
        cache="disk",
        project="runs/pretrain",
        name="yolo26n_3cls",
        optimizer="auto",
        save=True,
        pretrained=True,
        plots=True,
        channels_last=True,
        compile=True,
        fliplr=0.0,
        flipud=0.0,
        degrees=10.0,
    )

    test_results = model.val(
        data=dataset_yaml_path,
        split="test",
        batch=512,
        imgsz=640,
        device=[0],
        workers=8,
        plots=True,
    )

    logger.info(f"Test Results: {test_results.results_dict}")