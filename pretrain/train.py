import pretrain.monkeypatch
import os

os.environ["MPLBACKEND"] = "Agg"

import yaml
from ultralytics import YOLO

dataset_yaml_path = "pretrain/all.yaml"
active_classes = []

with open(dataset_yaml_path, "r") as f:
    for line in f:
        if ":" in line and "# blacklist" not in line and "names:" not in line and "path:" not in line:
            parts = line.strip().split(":")
            if len(parts) >= 2:
                try:
                    class_id = int(parts[0].strip())
                    active_classes.append(class_id)
                except ValueError:
                    pass

model = YOLO("weights/yolo26n.pt")

results = model.train(
    data=dataset_yaml_path,
    classes=active_classes,
    single_cls=True,
    epochs=100,
    batch=128,
    imgsz=640,
    device=[0],
    workers=8,
    cache=True,
    project="runs/pretrain",
    name="yolo26n",
    optimizer="auto",
    save=True,
    pretrained=True,
    plots=True,
    channels_last=True,
    compile=True,
    fliplr=0.0,
    flipud=0.0,
    degrees=10.0
)