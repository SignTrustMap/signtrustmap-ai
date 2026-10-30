import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import torch
import ultralytics.data.dataset as dataset_mod
import ultralytics.data.utils as data_utils
import ultralytics.engine.exporter as exporter
import ultralytics.engine.trainer as trainer
import ultralytics.engine.validator as validator
from loguru import logger
from ultralytics import YOLO
from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import DetectionModel
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


def _patched_check(*args, **kwargs):
    data = _orig_check(*args, **kwargs)
    if isinstance(data, dict):
        data["names"] = {0: "main_sign", 1: "supplementary_sign", 2: "text_sign"}
        data["nc"] = 3
    return data


data_utils.check_det_dataset = _patched_check
trainer.check_det_dataset = _patched_check
validator.check_det_dataset = _patched_check
exporter.check_det_dataset = _patched_check

_orig_verify = data_utils.verify_image_label


def _patched_verify_image_label(args):
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim, single_cls = args
    return _orig_verify((im_file, lb_file, prefix, keypoint, float("inf"), nkpt, ndim, single_cls))


data_utils.verify_image_label = _patched_verify_image_label
dataset_mod.verify_image_label = _patched_verify_image_label

_orig_verify_args = YOLODataset.verify_args


def _patched_verify_args(self):
    func, args_iter = _orig_verify_args(self)
    return _patched_verify_image_label, args_iter


YOLODataset.verify_args = _patched_verify_args

_orig_load_cache = data_utils.load_dataset_cache_file


def _patched_load_dataset_cache_file(path):
    try:
        return _orig_load_cache(path)
    except Exception as e:
        raise FileNotFoundError(f"Corrupted cache file {path}: {e}")


data_utils.load_dataset_cache_file = _patched_load_dataset_cache_file
dataset_mod.load_dataset_cache_file = _patched_load_dataset_cache_file

_orig_load_or_scan = YOLODataset._load_or_scan_cache


def _patched_load_or_scan_cache(self, cache_path, cache_hash):
    try:
        return _orig_load_or_scan(self, cache_path, cache_hash)
    except Exception as e:
        logger.warning(f"Cache file {cache_path} corrupted ({e}), rescanning in memory...")
        return self.cache_labels(cache_path), False


YOLODataset._load_or_scan_cache = _patched_load_or_scan_cache

_orig_save_cache = data_utils.save_dataset_cache_file


def _patched_save_dataset_cache_file(prefix: str, path: Path, x: dict, version: str):
    try:
        return _orig_save_cache(prefix, path, x, version)
    except Exception as e:
        logger.warning(f"Could not save cache to {path}: {e}")


data_utils.save_dataset_cache_file = _patched_save_dataset_cache_file
dataset_mod.save_dataset_cache_file = _patched_save_dataset_cache_file

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

_orig_load_image = BaseDataset.load_image


def _patched_load_image(self, i, *args, **kwargs):
    self.npy_files[i] = Path("/dev/null/dummy.npy")
    return _orig_load_image(self, i, *args, **kwargs)


BaseDataset.load_image = _patched_load_image

_orig_init_criterion = DetectionModel.init_criterion


def _patched_init_criterion(self):
    if not hasattr(self, "class_weights") or self.class_weights is None:
        self.class_weights = torch.tensor([1.0, 2.5, 4.0])
    return _orig_init_criterion(self)


DetectionModel.init_criterion = _patched_init_criterion

if __name__ == "__main__":
    model = YOLO("weights/yolo26n.pt")

    results = model.train(
        data=dataset_yaml_path,
        epochs=100,
        batch=128,
        imgsz=640,
        device=[0],
        workers=8,
        cache=False,
        project="runs/pretrain",
        name="yolo26n_3cls",
        optimizer="auto",
        save=True,
        pretrained=True,
        plots=True,
        channels_last=True,
        compile=False,
        fliplr=0.0,
        flipud=0.0,
        degrees=10.0,
        scale=0.8,
        hsv_s=0.85,
        hsv_v=0.6,
        cls=1.2,
        mixup=0.15,
        copy_paste=0.1,
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