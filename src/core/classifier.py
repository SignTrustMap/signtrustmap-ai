import gc
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
try:
    import tensorrt as trt
except ImportError:
    trt = None
from loguru import logger
import clip
import open_clip
from src.core.config import settings
from src.core.class_manager import class_manager
from src.infrastructure.minio_client import minio_client


class TensorRTVisualRunner:
    def __init__(self, engine_path: str, device: str = "cuda:0"):
        if trt is None:
            raise RuntimeError(
                "TensorRT is not installed in this environment. "
                "Cannot load TensorRT engine (.engine). Use a .pt or ONNX model, or install tensorrt."
            )
        self.device = torch.device(device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def encode_image(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        if not batch_tensor.is_cuda:
            batch_tensor = batch_tensor.to(self.device)

        batch_size = batch_tensor.shape[0]
        self.context.set_input_shape("images", tuple(batch_tensor.shape))
        output_tensor = torch.empty((batch_size, 512), dtype=torch.float32, device=self.device)

        bindings = [batch_tensor.data_ptr(), output_tensor.data_ptr()]
        self.context.execute_v2(bindings)
        output_tensor = output_tensor / output_tensor.norm(dim=-1, keepdim=True)
        return output_tensor


class SignClassifier:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        prompt_template: Optional[str] = None,
        device: Optional[Union[int, str]] = None,
    ):
        self.device = self._resolve_device(device)
        self.prompt_template = prompt_template or settings.CLASSIFIER_PROMPT_TEMPLATE
        self.weights_path = self._resolve_weights_path(weights_path)
        self._is_mobileclip = False
        self._tokenizer = None
        self._class_map = class_manager.get_classes_map()
        self._class_ids = sorted(self._class_map.keys())

        logger.info(f"Loading SignClassifier on {self.device} ({self.weights_path})")
        self.model, self.preprocess = self._load_model(self.weights_path)
        self.text_features = self._load_or_build_text_features(self.prompt_template)
        logger.info(f"SignClassifier initialized ({len(self._class_ids)} classes).")

    def _resolve_device(self, device: Optional[Union[int, str]]) -> str:
        if device is not None:
            if isinstance(device, int):
                return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            return str(device)
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _normalize_name(name: str) -> Tuple[str, str]:
        clean = os.path.basename(name).strip()
        clip_name = clean.removesuffix(".pt").removesuffix(".engine").removesuffix(".onnx").replace("-", "/")
        file_name = f"{clip_name.replace('/', '-')}.pt"
        return clip_name, file_name

    def _resolve_weights_path(self, weights_path: Optional[str]) -> str:
        base_dir = Path(__file__).resolve().parent.parent.parent
        classifiers_dir = base_dir / "weights" / "classifiers"
        classifiers_dir.mkdir(parents=True, exist_ok=True)

        if weights_path is None:
            if settings.CLASSIFIER_S3_KEY:
                model_filename = os.path.basename(settings.CLASSIFIER_S3_KEY)
                local_path = classifiers_dir / model_filename
                if not local_path.exists():
                    logger.info(f"Downloading classifier '{model_filename}' from MinIO...")
                    success = minio_client.download_file(
                        settings.MINIO_BUCKET,
                        settings.CLASSIFIER_S3_KEY,
                        str(local_path),
                    )
                    if not success:
                        logger.warning(f"Failed MinIO download for '{settings.CLASSIFIER_S3_KEY}'. Falling back to local default.")
                        fallback_path = base_dir / (settings.FALLBACK_CLASSIFIER_PATH or "weights/classifiers/mobileclip_s2.engine")
                        weights_path = str(fallback_path)
                    else:
                        weights_path = str(local_path)
                else:
                    weights_path = str(local_path)
            else:
                fallback_path = base_dir / (settings.FALLBACK_CLASSIFIER_PATH or "weights/classifiers/mobileclip_s2.engine")
                weights_path = str(fallback_path)

        target_path = Path(weights_path)
        if target_path.is_file():
            return str(target_path.resolve())

        clean_clip_name, clean_file_name = self._normalize_name(weights_path)
        local_file = classifiers_dir / clean_file_name
        if local_file.is_file():
            return str(local_file.resolve())

        if "mobileclip" in weights_path.lower():
            return weights_path

        if clean_clip_name in clip.available_models():
            logger.info(f"Downloading CLIP model '{clean_clip_name}'...")
            clip.load(clean_clip_name, device="cpu", download_root=str(classifiers_dir))
            if local_file.is_file():
                return str(local_file.resolve())
            return clean_clip_name

        raise FileNotFoundError(f"Classifier weights '{weights_path}' not found in {classifiers_dir}")

    def _load_model(self, weights_path: str):
        base_dir = Path(__file__).resolve().parent.parent.parent
        download_root = str(base_dir / "weights" / "classifiers")

        if weights_path.endswith(".engine"):
            self._is_mobileclip = "mobileclip" in weights_path.lower()
            img_size = 256 if self._is_mobileclip else 224
            preprocess = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)) if self._is_mobileclip else transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ])
            runner = TensorRTVisualRunner(weights_path, device=self.device)
            return runner, preprocess

        if "mobileclip" in weights_path.lower():
            self._is_mobileclip = True
            arch = "MobileCLIP-S2"
            if "s0" in weights_path.lower():
                arch = "MobileCLIP-S0"
            elif "s1" in weights_path.lower():
                arch = "MobileCLIP-S1"
            elif "s2" in weights_path.lower():
                arch = "MobileCLIP-S2"
            elif "b" in weights_path.lower():
                arch = "MobileCLIP-B"

            model, _, preprocess = open_clip.create_model_and_transforms(
                arch,
                pretrained=weights_path if os.path.isfile(weights_path) else "datacompdr",
                device=self.device,
            )
            model.eval()
            self._tokenizer = open_clip.get_tokenizer(arch)
            return model, preprocess

        self._is_mobileclip = False
        self._tokenizer = None

        if os.path.isfile(weights_path):
            model, preprocess = clip.load(weights_path, device=self.device)
            model.eval()
            return model, preprocess

        clip_name, _ = self._normalize_name(weights_path)
        model, preprocess = clip.load(clip_name, device=self.device, download_root=download_root)
        model.eval()
        return model, preprocess

    def _get_embedding_cache_path(self) -> Path:
        base_dir = Path(__file__).resolve().parent.parent.parent
        cache_dir = base_dir / "weights" / "embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.weights_path).stem.replace("/", "-")
        return cache_dir / f"text_embeddings_{stem}.pt"

    def _encode_prompts(self, prompts: List[str]) -> torch.Tensor:
        features_list = []
        batch_size = 32

        has_text_encoder = hasattr(self.model, "encode_text") and (
            (self._is_mobileclip and hasattr(self.model, "text"))
            or (not self._is_mobileclip and hasattr(self.model, "token_embedding"))
        )

        if has_text_encoder:

            if self._is_mobileclip and self._tokenizer is not None:
                text_tokens = self._tokenizer(prompts)
            else:
                text_tokens = clip.tokenize(prompts)

            with torch.inference_mode():
                for i in range(0, len(text_tokens), batch_size):
                    chunk_tokens = text_tokens[i : i + batch_size].to(self.device)
                    chunk_feats = self.model.encode_text(chunk_tokens)
                    chunk_feats = chunk_feats / chunk_feats.norm(dim=-1, keepdim=True)
                    features_list.append(chunk_feats)
                return torch.cat(features_list, dim=0)

        logger.info("Instantiating temporary text encoder to compute prompt embeddings...")
        base_dir = Path(__file__).resolve().parent.parent.parent
        classifiers_dir = base_dir / "weights" / "classifiers"

        if self._is_mobileclip:
            arch = "MobileCLIP-S2"
            if "s0" in self.weights_path.lower():
                arch = "MobileCLIP-S0"
            elif "s1" in self.weights_path.lower():
                arch = "MobileCLIP-S1"
            elif "s2" in self.weights_path.lower():
                arch = "MobileCLIP-S2"
            elif "b" in self.weights_path.lower():
                arch = "MobileCLIP-B"

            stem = Path(self.weights_path).stem
            local_pt = classifiers_dir / f"{stem}.pt"
            fallback_pt = classifiers_dir / "mobileclip_s2.pt"
            pretrained_target = (
                str(local_pt)
                if local_pt.is_file()
                else (str(fallback_pt) if fallback_pt.is_file() else "datacompdr")
            )

            temp_model, _, _ = open_clip.create_model_and_transforms(
                arch,
                pretrained=pretrained_target,
                device=self.device,
            )
            temp_model.eval()
            tokenizer = open_clip.get_tokenizer(arch)
            text_tokens = tokenizer(prompts)

            with torch.inference_mode():
                for i in range(0, len(text_tokens), batch_size):
                    chunk_tokens = text_tokens[i : i + batch_size].to(self.device)
                    chunk_feats = temp_model.encode_text(chunk_tokens)
                    chunk_feats = chunk_feats / chunk_feats.norm(dim=-1, keepdim=True)
                    features_list.append(chunk_feats)

            del temp_model
        else:
            clean_clip_name, clean_file_name = self._normalize_name(self.weights_path)
            local_clip_pt = classifiers_dir / clean_file_name
            target_clip = str(local_clip_pt) if local_clip_pt.is_file() else clean_clip_name
            temp_model, _ = clip.load(target_clip, device=self.device)
            temp_model.eval()
            text_tokens = clip.tokenize(prompts)

            with torch.inference_mode():
                for i in range(0, len(text_tokens), batch_size):
                    chunk_tokens = text_tokens[i : i + batch_size].to(self.device)
                    chunk_feats = temp_model.encode_text(chunk_tokens)
                    chunk_feats = chunk_feats / chunk_feats.norm(dim=-1, keepdim=True)
                    features_list.append(chunk_feats)

            del temp_model


        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        return torch.cat(features_list, dim=0)

    def _load_or_build_text_features(self, prompt_template: str) -> torch.Tensor:
        self._class_map = class_manager.get_classes_map()
        self._class_ids = sorted(self._class_map.keys())
        expected_count = len(self._class_ids)

        cache_path = self._get_embedding_cache_path()

        if cache_path.exists():
            try:
                logger.debug(f"Loading text embeddings: {cache_path}")
                text_features = torch.load(cache_path, map_location=self.device)
                if isinstance(text_features, torch.Tensor) and text_features.shape[0] == expected_count:
                    logger.debug(f"Loaded text embeddings shape: {text_features.shape}")
                    self._prune_text_encoder()
                    return text_features
            except Exception as e:
                logger.warning(f"Could not load cached text embeddings from {cache_path}: {e}")

        logger.info(f"Computing text embeddings for {expected_count} classes...")
        prompts = class_manager.get_prompts(template=prompt_template)
        text_features = self._encode_prompts(prompts)

        try:
            torch.save(text_features.cpu(), cache_path)
            logger.info(f"Saved text embeddings to {cache_path}")
        except Exception as e:
            logger.error(f"Failed to persist text embeddings to {cache_path}: {e}")

        self._prune_text_encoder()
        return text_features.to(self.device)

    def rebuild_text_features(self, template: Optional[str] = None) -> torch.Tensor:
        active_template = template or self.prompt_template
        self.prompt_template = active_template
        self._class_map = class_manager.get_classes_map()
        self._class_ids = sorted(self._class_map.keys())
        expected_count = len(self._class_ids)

        logger.info(f"Rebuilding text embeddings for {expected_count} classes (template: '{active_template}')...")
        prompts = class_manager.get_prompts(template=active_template)
        text_features = self._encode_prompts(prompts)
        self.text_features = text_features.to(self.device)

        cache_path = self._get_embedding_cache_path()
        try:
            torch.save(self.text_features.cpu(), cache_path)
            logger.info(f"Updated cached text embeddings at {cache_path}")
        except Exception as e:
            logger.error(f"Failed to save text embeddings to {cache_path}: {e}")

        return self.text_features

    def _prune_text_encoder(self) -> None:
        try:
            if hasattr(self.model, "text"):
                del self.model.text
            if hasattr(self.model, "transformer"):
                del self.model.transformer
            if hasattr(self.model, "token_embedding"):
                del self.model.token_embedding
            if hasattr(self.model, "positional_embedding"):
                del self.model.positional_embedding
            if hasattr(self.model, "ln_final"):
                del self.model.ln_final
            if hasattr(self.model, "text_projection"):
                del self.model.text_projection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.debug("Pruned CLIP text transformer.")
        except Exception as e:
            logger.trace(f"Text encoder attributes not present: {e}")

    @staticmethod
    def crop_box(
        image: np.ndarray,
        box_xyxy: List[float],
        padding_ratio: float = 0.1,
    ) -> Optional[np.ndarray]:
        if image is None or len(box_xyxy) < 4:
            return None

        h, w = image.shape[:2]
        xmin, ymin, xmax, ymax = box_xyxy[:4]
        box_w = max(1.0, xmax - xmin)
        box_h = max(1.0, ymax - ymin)

        pad_w = box_w * padding_ratio
        pad_h = box_h * padding_ratio

        x1 = max(0, int(round(xmin - pad_w)))
        y1 = max(0, int(round(ymin - pad_h)))
        x2 = min(w, int(round(xmax + pad_w)))
        y2 = min(h, int(round(ymax + pad_h)))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def classify_crop(self, crop: np.ndarray) -> Dict[str, Any]:
        if crop is None or crop.size == 0:
            return {
                "class_id": None,
                "class_name": "unknown",
                "classify_confidence": 0.0,
            }

        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            image_features = self.model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_feats = self.text_features.to(device=image_features.device, dtype=image_features.dtype)
            similarity = (100.0 * image_features @ text_feats.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)

        idx = int(indices[0].item())
        confidence = float(values[0].item())
        class_id = self._class_ids[idx] if idx < len(self._class_ids) else idx
        class_name = self._class_map.get(class_id, f"class_{class_id}")

        logger.trace(f"Crop classified: {class_name} ({confidence:.4f})")
        return {
            "class_id": class_id,
            "class_name": class_name,
            "classify_confidence": round(confidence, 4),
        }

    def classify_crops_batch(
        self,
        crops: List[Optional[np.ndarray]],
        batch_size: int = 4,
    ) -> List[Dict[str, Any]]:
        if not crops:
            return []

        results: List[Dict[str, Any]] = [
            {"class_id": None, "class_name": "unknown", "classify_confidence": 0.0}
            for _ in range(len(crops))
        ]

        valid_indices = [i for i, c in enumerate(crops) if c is not None and c.size > 0]
        if not valid_indices:
            return results

        logger.debug(f"Classifying {len(valid_indices)} crop(s) (batch_size={batch_size})...")

        for start_idx in range(0, len(valid_indices), batch_size):
            chunk_indices = valid_indices[start_idx : start_idx + batch_size]
            tensors = []

            for idx in chunk_indices:
                crop = crops[idx]
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensors.append(self.preprocess(pil_img))

            batch_tensor = torch.stack(tensors, dim=0).to(self.device)

            with torch.inference_mode():
                image_features = self.model.encode_image(batch_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_feats = self.text_features.to(device=image_features.device, dtype=image_features.dtype)
                similarity = (100.0 * image_features @ text_feats.T).softmax(dim=-1)
                values, indices = similarity.topk(1, dim=-1)

            for i, global_idx in enumerate(chunk_indices):
                idx = int(indices[i][0].item())
                conf = float(values[i][0].item())
                cls_id = self._class_ids[idx] if idx < len(self._class_ids) else idx
                name = self._class_map.get(cls_id, f"class_{cls_id}")
                results[global_idx] = {
                    "class_id": cls_id,
                    "class_name": name,
                    "classify_confidence": round(conf, 4),
                }

        logger.debug(f"Batch classification complete ({len(valid_indices)} crops).")
        return results

    def unload(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "text_features"):
            del self.text_features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SignClassifier unloaded.")
