import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from src.core.config import settings
from src.infrastructure.db import sync_engine
from src.infrastructure.models import AiSystemConfig


def normalize_config_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


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
        self.load_config()

    def _get_default_config(self) -> Dict[str, Dict[str, Any]]:
        detector_fallback = os.path.basename(settings.FALLBACK_DETECTOR_PATH)
        classifier_fallback = (
            "mobileclip_s2.engine"
            if (self._base_dir / "weights" / "classifiers" / "mobileclip_s2.engine").exists()
            else ("mobileclip_s2.pt" if (self._base_dir / "weights" / "classifiers" / "mobileclip_s2.pt").exists() else "ViT-B-16.pt")
        )
        return {
            "detector": {
                "active_model": detector_fallback,
                "conf_thresh": 0.6,
            },
            "classifier": {
                "active_model": classifier_fallback,
                "conf_thresh": 0.6,
                "prompt_template": settings.CLASSIFIER_PROMPT_TEMPLATE or "a photo of a {} traffic sign.",
                "temperature": 7.0,
            },
            "timo": {
                "alpha": 250.0,
                "beta": 0.8,
            },
            "few_shot_support": {
                "shots_per_class": 16,
                "num_anchors": 4,
                "num_medoids": 12,
                "min_crop_size": 20,
                "min_sharpness": 15.0,
            },
            "active_learning": {
                "strategy": "r3_cec",
                "query_budget": 50,
                "aggregation": "avg",
                "no_detection_score": 0.0,
                "min_samples_detector": 50,
                "min_samples_classifier": 100,
                "discrepancy_thresh": 0.5,
                "auto_trigger_retrain": True,
                "r3_cec_k_neighbors": 5,
                "r3_cec_temperature": 0.5,
            },
            "system": {
                "model_idle_timeout": settings.MODEL_IDLE_TIMEOUT_SECONDS,
                "cuda_device": 0,
            },
        }

    def _normalize_key(self, key: str) -> str:
        return normalize_config_key(key)

    def __getattr__(self, name: str) -> Any:
        norm = self._normalize_key(name)
        if norm in self._config:
            return self._config[norm]
        raise AttributeError(f"'ConfigManager' object has no attribute '{name}'")

    def load_config(self) -> None:
        loaded_from_db = False
        try:
            with Session(sync_engine) as session:
                db_configs = session.query(AiSystemConfig).all()
                for cfg in db_configs:
                    if "." in cfg.config_key:
                        group, key = cfg.config_key.split(".", 1)
                        norm_group = self._normalize_key(group)
                        if norm_group not in self._config:
                            self._config[norm_group] = ConfigDict()
                        self._config[norm_group][key] = cfg.config_value
                if db_configs:
                    loaded_from_db = True
                    logger.info(f"Loaded {len(db_configs)} system configs from AI Database")
        except Exception as e:
            logger.warning("================================================================================")
            logger.warning(f"FALLBACK WARNING: Database query for system configs failed ({e})! Falling back to local configuration.")
            logger.warning("================================================================================")

        if not loaded_from_db and self._config_file.exists():
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    loaded_data: Dict[str, Any] = json.load(f)
                for group_key, group_val in loaded_data.items():
                    norm_group = self._normalize_key(group_key)
                    if isinstance(group_val, dict):
                        if norm_group not in self._config:
                            self._config[norm_group] = ConfigDict()
                        self._config[norm_group].update(group_val)
                logger.info(f"Loaded fallback persistent runtime config from {self._config_file}")
            except Exception as e:
                logger.warning("================================================================================")
                logger.warning(f"FALLBACK WARNING: Failed to load persistent config from {self._config_file}: {e}")
                logger.warning("================================================================================")

        self.save_config()

    def _persist_to_db(self, updates_by_group: Dict[str, Dict[str, Any]]) -> None:
        try:
            with Session(sync_engine) as session:
                for group_name, group_updates in updates_by_group.items():
                    norm_group = self._normalize_key(group_name)
                    for key, val in group_updates.items():
                        norm_key = self._normalize_key(key)
                        full_key = f"{norm_group}.{norm_key}"
                        db_cfg = session.query(AiSystemConfig).filter_by(config_key=full_key).first()
                        if db_cfg:
                            db_cfg.config_value = val
                            db_cfg.updated_at = func.now()
                session.commit()
                logger.debug("Successfully persisted configuration updates to AI Database.")
        except Exception as e:
            logger.warning("================================================================================")
            logger.warning(f"FALLBACK WARNING: Failed to persist config updates to Database ({e}). Storing in local cache.")
            logger.warning("================================================================================")

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

        normalized_updates = {}
        for key, val in updates.items():
            if val is not None:
                norm_key = self._normalize_key(key)
                normalized_updates[norm_key] = val
                self._config[norm][norm_key] = val

        self._persist_to_db({norm: normalized_updates})
        self.save_config()
        logger.info(f"Updated group '{norm}' configuration: {self._config[norm]}")
        return self.get_group(norm)

    def update_bulk(self, bulk_updates: Dict[str, Any]) -> Dict[str, Any]:
        normalized_bulk = {}
        for group_name, group_updates in bulk_updates.items():
            if isinstance(group_updates, dict):
                norm = self._normalize_key(group_name)
                if norm not in self._config:
                    self._config[norm] = ConfigDict()
                norm_updates = {}
                for k, v in group_updates.items():
                    if v is not None:
                        norm_k = self._normalize_key(k)
                        norm_updates[norm_k] = v
                        self._config[norm][norm_k] = v
                normalized_bulk[norm] = norm_updates

        self._persist_to_db(normalized_bulk)
        self.save_config()
        logger.info(f"Updated bulk configurations: {list(bulk_updates.keys())}")
        return self.get_all()

    def reset_config(self) -> Dict[str, Any]:
        logger.info("Resetting runtime configuration back to factory default settings.")
        defaults = self._get_default_config()
        self._config = ConfigDict(defaults)

        try:
            with Session(sync_engine) as session:
                for group, group_dict in defaults.items():
                    if isinstance(group_dict, dict):
                        for k, v in group_dict.items():
                            full_key = f"{group}.{k}"
                            db_cfg = session.query(AiSystemConfig).filter_by(config_key=full_key).first()
                            if db_cfg:
                                db_cfg.config_value = v
                                db_cfg.updated_at = func.now()
                session.commit()
                logger.info("Restored default configs in AI Database.")
        except Exception as e:
            logger.warning(f"Failed to reset configs in Database ({e}). Resetting local state.")

        self.save_config()
        return self.get_all()

    @property
    def active_detector_model(self) -> str:
        return self.get("detector.active_model")

    @property
    def active_classifier_model(self) -> Optional[str]:
        return self.get("classifier.active_model")

    @property
    def detector_conf_thresh(self) -> float:
        return float(self.get("detector.conf_thresh", 0.6))

    @property
    def classifier_conf_thresh(self) -> float:
        return float(self.get("classifier.conf_thresh", 0.6))

    @property
    def classifier_temperature(self) -> float:
        return float(self.get("classifier.temperature", 7.0))

    @property
    def classifier_prompt_template(self) -> str:
        return str(self.get("classifier.prompt_template", settings.CLASSIFIER_PROMPT_TEMPLATE or "a photo of a {} traffic sign."))

    @property
    def cuda_device(self) -> int:
        return int(self.get("system.cuda_device", 0))

    @property
    def model_idle_timeout(self) -> int:
        return int(self.get("system.model_idle_timeout", settings.MODEL_IDLE_TIMEOUT_SECONDS))

    @property
    def log_level(self) -> str:
        return str(settings.LOG_LEVEL)

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
