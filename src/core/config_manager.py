import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
from src.core.config import settings


class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                self[key] = ConfigDict(value)

    def __getattr__(self, name: str) -> Any:
        if name in self:
            val = self[name]
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                val = ConfigDict(val)
                self[name] = val
            return val
        raise AttributeError(f"'ConfigDict' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
        self[name] = value

    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for key, val in self.items():
            if isinstance(val, ConfigDict):
                result[key] = val.to_dict()
            elif isinstance(val, dict):
                result[key] = ConfigDict(val).to_dict()
            else:
                result[key] = val
        return result


class ConfigManager:
    def __init__(self):
        self._base_dir = Path(__file__).resolve().parent.parent.parent
        self._config_file = self._base_dir / "configs" / "app_config.json"
        self._config: ConfigDict = ConfigDict(self._get_default_config())

    def _get_default_config(self) -> Dict[str, Dict[str, Any]]:
        detector_fallback = os.path.basename(settings.FALLBACK_DETECTOR_PATH)
        classifier_fallback = (
            "mobileclip_s2.engine"
            if (self._base_dir / "weights" / "classifiers" / "mobileclip_s2.engine").exists()
            else ("mobileclip_s2.pt" if (self._base_dir / "weights" / "classifiers" / "mobileclip_s2.pt").exists() else "ViT-B-16.pt")
        )
        return {
            "active_learning": {
                "default_top_k": 50,
                "default_strategy": "r3_cec",
                "default_aggregation": "avg",
                "no_detection_score": 0.0,
                "min_batch_size_detector": 50,
                "min_batch_size_classifier": 100,
                "discrepancy_threshold": 0.5,
                "auto_trigger_training": True,
                "dataset_store_path": "runs/active_learning/dataset.json",
                "r3_cec": {
                    "k_neighbors": 5,
                    "temperature": 0.5,
                    "query_budget": 16,
                    "apply_calibration": True,
                    "random_state": 42,
                },
                "least_confidence": {
                    "aggregation_method": "avg",
                    "no_detection_score": 0.0,
                },
            },
            "protofuse": {
                "fusion_weight_alpha": 0.5,
                "prompt_template": settings.CLASSIFIER_PROMPT_TEMPLATE or "a photo of a {} traffic sign.",
                "normalize_embeddings": True,
                "prototypes_cache_file": "weights/classifiers/protofuse_prototypes_100.pt",
            },
            "few_shot_support": {
                "total_shots_per_class": 16,
                "num_anchors": 4,
                "num_medoids": 12,
                "min_crop_resolution": 20,
                "min_sharpness_laplacian": 15.0,
                "replacement_policy": "diversity_kmeans",
                "preserve_pinned_shots": True,
            },
            "detector": {
                "active_model": detector_fallback,
                "default_conf": 0.6,
            },
            "classifier": {
                "active_model": classifier_fallback,
                "default_conf": 0.6,
                "prompt_template": settings.CLASSIFIER_PROMPT_TEMPLATE,
            },
            "system": {
                "default_device": 0,
                "model_idle_timeout_seconds": settings.MODEL_IDLE_TIMEOUT_SECONDS,
                "log_level": settings.LOG_LEVEL,
            },
            "storage": {
                "minio_endpoint": settings.MINIO_ENDPOINT,
                "minio_bucket": settings.MINIO_BUCKET,
            },
        }

    def _normalize_key(self, key: str) -> str:
        return key.strip().lower().replace("-", "_")

    def __getattr__(self, name: str) -> Any:
        norm = self._normalize_key(name)
        if norm in self._config:
            return self._config[norm]
        raise AttributeError(f"'ConfigManager' object has no attribute '{name}'")

    def load_config(self) -> None:
        if self._config_file.exists():
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    loaded_data: Dict[str, Any] = json.load(f)

                for group_key, group_val in loaded_data.items():
                    norm_group = self._normalize_key(group_key)
                    if isinstance(group_val, dict):
                        if norm_group not in self._config:
                            self._config[norm_group] = ConfigDict()
                        self._config[norm_group].update(group_val)

                logger.info(f"Loaded persistent runtime config groups: {list(self._config.keys())}")
            except Exception as e:
                logger.warning(f"Failed to load persistent config from {self._config_file}: {e}")

    def save_config(self) -> None:
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(self._config.to_dict(), f, indent=2)
            logger.debug(f"Saved runtime configuration to {self._config_file}")
        except Exception as e:
            logger.error(f"Failed to persist runtime config to {self._config_file}: {e}")

    def get_all(self) -> Dict[str, Any]:
        return self._config.to_dict()

    def get_group(self, group_name: str) -> Optional[Dict[str, Any]]:
        norm = self._normalize_key(group_name)
        group = self._config.get(norm)
        if group is not None:
            return group.to_dict() if isinstance(group, ConfigDict) else copy.deepcopy(group)
        return None

    def get(self, path: str, default: Any = None) -> Any:
        parts = path.split(".")
        curr: Any = self._config
        for part in parts:
            if isinstance(curr, dict):
                norm = self._normalize_key(part)
                curr = curr.get(norm, default)
            else:
                return default
        return curr

    def update_group(self, group_name: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        norm = self._normalize_key(group_name)
        if norm not in self._config:
            self._config[norm] = ConfigDict()

        for key, val in updates.items():
            if val is not None:
                self._config[norm][key] = val

        self.save_config()
        logger.info(f"Updated group '{norm}' configuration: {self._config[norm]}")
        return self.get_group(norm)

    def update_bulk(self, bulk_updates: Dict[str, Any]) -> Dict[str, Any]:
        for group_name, group_updates in bulk_updates.items():
            if isinstance(group_updates, dict):
                norm = self._normalize_key(group_name)
                if norm not in self._config:
                    self._config[norm] = ConfigDict()
                for k, v in group_updates.items():
                    if v is not None:
                        self._config[norm][k] = v

        self.save_config()
        logger.info(f"Updated bulk configurations: {list(bulk_updates.keys())}")
        return self.get_all()

    def reset_config(self) -> Dict[str, Any]:
        logger.info("Resetting runtime configuration back to factory default settings.")
        self._config = ConfigDict(self._get_default_config())

        if self._config_file.exists():
            try:
                os.remove(self._config_file)
                logger.debug(f"Removed persistent config file: {self._config_file}")
            except Exception as e:
                logger.warning(f"Could not remove {self._config_file}: {e}")

        return self.get_all()

    @property
    def active_detector_model(self) -> str:
        return self.get("detector.active_model")

    @property
    def active_classifier_model(self) -> Optional[str]:
        return self.get("classifier.active_model")

    @property
    def default_detector_conf(self) -> float:
        return float(self.get("detector.default_conf", 0.6))

    @property
    def default_classifier_conf(self) -> float:
        return float(self.get("classifier.default_conf", 0.6))

    @property
    def classifier_prompt_template(self) -> str:
        return str(self.get("classifier.prompt_template", settings.CLASSIFIER_PROMPT_TEMPLATE))

    @property
    def default_device(self) -> int:
        return int(self.get("system.default_device", 0))

    @property
    def model_idle_timeout_seconds(self) -> int:
        return int(self.get("system.model_idle_timeout_seconds", 1800))

    @property
    def log_level(self) -> str:
        return str(self.get("system.log_level", "INFO"))

    def set_active_detector(self, weights_name: str) -> None:
        self.update_group("detector", {"active_model": os.path.basename(weights_name)})

    def set_active_classifier(self, weights_name: Optional[str]) -> None:
        val = os.path.basename(weights_name) if weights_name else None
        self.update_group("classifier", {"active_model": val})

    def get_active_detector_path(self) -> str:
        model_name = self.active_detector_model
        if model_name:
            weights_path = self._base_dir / "weights" / "detectors" / model_name
            if weights_path.exists():
                return str(weights_path)
        return str(self._base_dir / settings.FALLBACK_DETECTOR_PATH)

    def get_active_classifier_path(self) -> Optional[str]:
        model_name = self.active_classifier_model
        if not model_name:
            fallback = self._base_dir / "weights" / "classifiers" / "mobileclip_s2.pt"
            if fallback.exists():
                return str(fallback)
            fallback_vit = self._base_dir / "weights" / "classifiers" / "ViT-B-16.pt"
            if fallback_vit.exists():
                return str(fallback_vit)
            return None
        weights_path = self._base_dir / "weights" / "classifiers" / model_name
        if weights_path.exists():
            return str(weights_path)
        fallback = self._base_dir / "weights" / "classifiers" / "mobileclip_s2.pt"
        if fallback.exists():
            return str(fallback)
        fallback_vit = self._base_dir / "weights" / "classifiers" / "ViT-B-16.pt"
        if fallback_vit.exists():
            return str(fallback_vit)
        return None


config_manager = ConfigManager()
