import argparse
import os
from pathlib import Path
import subprocess
import torch
from loguru import logger
from ultralytics import YOLO
import open_clip
import clip


class CLIPVisualWrapper(torch.nn.Module):
    def __init__(self, visual_model):
        super().__init__()
        self.visual = visual_model

    def forward(self, image):
        feats = self.visual(image)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats


def export_detector(
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

    logger.info(f"Loading detector model from {weights_path}...")
    try:
        model = YOLO(str(path))
        is_ultralytics = True
    except Exception as e:
        logger.debug(f"Not an Ultralytics model: {e}. Attempting generic PyTorch export...")
        is_ultralytics = False

    formats_to_export = ["onnx", "engine"] if export_format == "all" else [export_format]

    if is_ultralytics:
        for fmt in formats_to_export:
            logger.info(f"Starting detector export for format: {fmt}")
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
                logger.error(f"Detector export failed for format '{fmt}': {e}")
                raise e
    else:
        device_str = f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu"
        pt_model = torch.load(str(path), map_location=device_str)
        if hasattr(pt_model, "eval"):
            pt_model.eval()
        dummy_input = torch.randn(1, 3, 640, 640, device=device_str)
        stem = path.stem
        output_dir = path.parent
        onnx_path = output_dir / f"{stem}.onnx"
        engine_path = output_dir / f"{stem}.engine"

        if "onnx" in formats_to_export or "engine" in formats_to_export:
            logger.info(f"Exporting PyTorch detector to ONNX: {onnx_path}...")
            torch.onnx.export(
                pt_model,
                dummy_input,
                str(onnx_path),
                input_names=["images"],
                output_names=["output"],
                dynamic_axes={"images": {0: "batch_size"}} if dynamic else None,
                opset_version=opset,
                dynamo=False,
            )

        if "engine" in formats_to_export:
            logger.info(f"Building TensorRT Engine from {onnx_path} -> {engine_path}...")
            trtexec_bin = "/usr/src/tensorrt/bin/trtexec"
            if not os.path.exists(trtexec_bin):
                trtexec_bin = "trtexec"
            cmd = [
                trtexec_bin,
                f"--onnx={onnx_path}",
                f"--saveEngine={engine_path}",
                f"--memPoolSize=workspace:{workspace * 1024}",
            ]
            if half:
                cmd.append("--fp16")
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"TensorRT build failed: {res.stderr}")
            logger.info(f"Successfully exported TensorRT detector engine to {engine_path}")


def export_classifier(
    weights_path: str,
    export_format: str,
    half: bool = True,
    device: int = 0,
    workspace: int = 2,
    dynamic: bool = False,
    opset: int = 17,
) -> None:
    path = Path(weights_path)
    if not path.is_file():
        logger.error(f"Classifier weights file not found: {weights_path}")
        raise FileNotFoundError(f"Classifier weights file not found at {weights_path}")

    stem = path.stem
    output_dir = path.parent

    logger.info(f"Loading classifier visual model from {weights_path}...")
    if "mobileclip" in weights_path.lower():
        arch = "MobileCLIP-S2"
        if "s0" in weights_path.lower():
            arch = "MobileCLIP-S0"
        elif "s1" in weights_path.lower():
            arch = "MobileCLIP-S1"
        elif "s2" in weights_path.lower():
            arch = "MobileCLIP-S2"
        elif "b" in weights_path.lower():
            arch = "MobileCLIP-B"
        model, _, _ = open_clip.create_model_and_transforms(
            arch,
            pretrained=str(path),
            device="cpu",
        )
        img_size = 256
    else:
        model, _ = clip.load(str(path), device="cpu")
        img_size = 224

    model.eval()
    visual_wrapper = CLIPVisualWrapper(model.visual).to("cpu")
    visual_wrapper.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device="cpu")
    onnx_path = output_dir / f"{stem}.onnx"
    engine_path = output_dir / f"{stem}.engine"

    formats_to_export = ["onnx", "engine"] if export_format == "all" else [export_format]

    if "onnx" in formats_to_export or "engine" in formats_to_export:
        logger.info(f"Exporting classifier to ONNX: {onnx_path}...")
        dynamic_axes = {"images": {0: "batch_size"}, "features": {0: "batch_size"}}
        torch.onnx.export(
            visual_wrapper,
            dummy_input,
            str(onnx_path),
            input_names=["images"],
            output_names=["features"],
            dynamic_axes=dynamic_axes if dynamic else None,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
        logger.info(f"Successfully exported ONNX classifier to {onnx_path}")

    if "engine" in formats_to_export:
        logger.info(f"Building TensorRT Engine from {onnx_path} -> {engine_path}...")
        trtexec_bin = "/usr/src/tensorrt/bin/trtexec"
        if not os.path.exists(trtexec_bin):
            trtexec_bin = "trtexec"

        cmd = [
            trtexec_bin,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--device={device}",
            "--memPoolSize=workspace:256",
            "--builderOptimizationLevel=2",
        ]
        if half:
            cmd.append("--fp16")
        if dynamic:
            cmd.extend([
                f"--minShapes=images:1x3x{img_size}x{img_size}",
                f"--optShapes=images:1x3x{img_size}x{img_size}",
                f"--maxShapes=images:1x3x{img_size}x{img_size}",
            ])
        else:
            cmd.append(f"--shapes=images:1x3x{img_size}x{img_size}")

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.error(f"trtexec build failed: {res.stderr or res.stdout}")
            raise RuntimeError(f"TensorRT build failed with code {res.returncode}: {res.stderr}")
        logger.info(f"Successfully exported TensorRT classifier engine to {engine_path}")


def export_model(
    weights_path: str,
    export_format: str,
    half: bool = True,
    device: int = 0,
    workspace: int = 2,
    dynamic: bool = False,
    opset: int = 17,
) -> None:
    path = Path(weights_path)
    if "classifiers" in str(path) or "clip" in path.name.lower():
        export_classifier(
            weights_path=weights_path,
            export_format=export_format,
            half=half,
            device=device,
            workspace=workspace,
            dynamic=dynamic,
            opset=opset,
        )
    else:
        export_detector(
            weights_path=weights_path,
            export_format=export_format,
            half=half,
            device=device,
            workspace=workspace,
            dynamic=dynamic,
            opset=opset,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch (.pt) models to ONNX / TensorRT Engine format.")
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/detectors/yolo26n_vtsdb_mtsd_dfg.pt",
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
