import argparse
import os
from pathlib import Path
from loguru import logger
from ultralytics import YOLO


def export_model(
    weights_path: str,
    export_format: str,
    half: bool = True,
    device: int = 0,
    workspace: int = 2,
    dynamic: bool = False,
    opset: int = 12,
) -> None:
    path = Path(weights_path)
    if not path.is_file():
        logger.error(f"Weights file not found: {weights_path}")
        raise FileNotFoundError(f"Weights file not found at {weights_path}")

    logger.info(f"Loading YOLO model from {weights_path}...")
    model = YOLO(str(path))

    formats_to_export = ["onnx", "engine"] if export_format == "all" else [export_format]

    for fmt in formats_to_export:
        logger.info(f"Starting export for format: {fmt}")
        try:
            if fmt == "onnx":
                exported_path = model.export(
                    format="onnx",
                    dynamic=dynamic,
                    opset=opset,
                    half=half,
                    device=device,
                )
            elif fmt == "engine":
                exported_path = model.export(
                    format="engine",
                    half=half,
                    device=device,
                    workspace=workspace,
                    dynamic=dynamic,
                )
            else:
                exported_path = model.export(
                    format=fmt,
                    half=half,
                    device=device,
                )
            logger.info(f"Successfully exported to {exported_path}")
        except Exception as e:
            logger.error(f"Export failed for format '{fmt}': {e}")
            raise e


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch (.pt) models to ONNX / TensorRT Engine format.")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolo12n_VTSDB100_v1.0.pt",
        help="Path to .pt weights file",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["onnx", "engine", "all"],
        default="all",
        help="Export format (onnx, engine, or all)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Use FP16 half precision",
    )
    parser.add_argument(
        "--no-half",
        dest="half",
        action="store_false",
        help="Disable FP16 half precision",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device index",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=2,
        help="TensorRT workspace size in GB",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="Enable dynamic axes for ONNX / Engine",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version",
    )

    args = parser.parse_args()
    export_model(
        weights_path=args.weights,
        export_format=args.format,
        half=args.half,
        device=args.device,
        workspace=args.workspace,
        dynamic=args.dynamic,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
