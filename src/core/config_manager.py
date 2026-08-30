import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from loguru import logger
from src.core.config import settings
from src.api.schemas.config import AppConfigResponse, AppConfigUpdateRequest


class ConfigManager:
    def __init__(self):
        self._base_dir = Path(__file__).resolve().parent.parent.parent
        self._config_file = self._base_dir / "configs" / "app_config.json"
        
        self.active_detector_model: str = os.path.basename(settings.FALLBACK_DETECTOR_PATH)
        self.active_classifier_model: Optional[str] = None
        self.default_detector_conf: float = 0.6
        self.default_classifier_conf: float = 0.6
        self.model_idle_timeout_seconds: int = settings.MODEL_IDLE_TIMEOUT_SECONDS
        self.default_device: int = 0
        self.log_level: str = settings.LOG_LEVEL

    def load_config(self) -> None:
        if self._config_file.exists():
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    data: Dict[str, Any] = json.load(f)
                    self.active_detector_model = data.get("active_detector_model", data.get("active_model", self.active_detector_model))
                    self.active_classifier_model = data.get("active_classifier_model", self.active_classifier_model)
                    self.default_detector_conf = float(data.get("default_detector_conf", data.get("default_conf", self.default_detector_conf)))
                    self.default_classifier_conf = float(data.get("default_classifier_conf", self.default_classifier_conf))
                    self.model_idle_timeout_seconds = int(data.get("model_idle_timeout_seconds", self.model_idle_timeout_seconds))
                    self.default_device = int(data.get("default_device", self.default_device))
                    self.log_level = data.get("log_level", self.log_level)
                logger.info(
                    f"Loaded persistent runtime config: active_detector='{self.active_detector_model}', "
                    f"active_classifier='{self.active_classifier_model}', timeout={self.model_idle_timeout_seconds}s"
                )
            except Exception as e:
                logger.warning(f"Failed to load persistent config from {self._config_file}: {e}")

    def save_config(self) -> None:
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "active_detector_model": self.active_detector_model,
                "active_classifier_model": self.active_classifier_model,
                "default_detector_conf": self.default_detector_conf,
                "default_classifier_conf": self.default_classifier_conf,
                "model_idle_timeout_seconds": self.model_idle_timeout_seconds,
                "default_device": self.default_device,
                "log_level": self.log_level,
            }
            with open(self._config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved runtime configuration to {self._config_file}")
        except Exception as e:
            logger.error(f"Failed to persist runtime config to {self._config_file}: {e}")

    def get_config_response(self) -> AppConfigResponse:
        return AppConfigResponse(
            active_detector_model=self.active_detector_model,
            active_classifier_model=self.active_classifier_model,
            default_detector_conf=self.default_detector_conf,
            default_classifier_conf=self.default_classifier_conf,
            model_idle_timeout_seconds=self.model_idle_timeout_seconds,
            default_device=self.default_device,
            log_level=self.log_level,
            minio_endpoint=settings.MINIO_ENDPOINT,
            minio_bucket=settings.MINIO_BUCKET,
        )

    def update_config(self, req: AppConfigUpdateRequest) -> AppConfigResponse:
        if req.active_detector_model is not None:
            self.set_active_detector(req.active_detector_model)

        if req.active_classifier_model is not None:
            self.set_active_classifier(req.active_classifier_model)

        if req.default_detector_conf is not None:
            self.default_detector_conf = req.default_detector_conf

        if req.default_classifier_conf is not None:
            self.default_classifier_conf = req.default_classifier_conf

        if req.model_idle_timeout_seconds is not None:
            self.model_idle_timeout_seconds = req.model_idle_timeout_seconds
            settings.MODEL_IDLE_TIMEOUT_SECONDS = req.model_idle_timeout_seconds

        if req.default_device is not None:
            self.default_device = req.default_device

        if req.log_level is not None:
            self.log_level = req.log_level.upper()

        self.save_config()
        logger.info(f"Updated dynamic runtime configuration: {self.get_config_response().model_dump_json()}")
        return self.get_config_response()

    def set_active_detector(self, weights_name: str) -> None:
        clean_name = os.path.basename(weights_name)
        weights_path = self._base_dir / "weights" / "detectors" / clean_name
        if not weights_path.exists():
            raise FileNotFoundError(f"Detector weights file '{clean_name}' not found in weights/detectors/ directory.")
        self.active_detector_model = clean_name
        self.save_config()

    def set_active_classifier(self, weights_name: Optional[str]) -> None:
        if weights_name is None or weights_name == "":
            self.active_classifier_model = None
            self.save_config()
            return

        clean_name = os.path.basename(weights_name)
        weights_path = self._base_dir / "weights" / "classifiers" / clean_name
        if not weights_path.exists():
            raise FileNotFoundError(f"Classifier weights file '{clean_name}' not found in weights/classifiers/ directory.")
        self.active_classifier_model = clean_name
        self.save_config()

    def get_active_detector_path(self) -> str:
        weights_path = self._base_dir / "weights" / "detectors" / self.active_detector_model
        if weights_path.exists():
            return str(weights_path)
        fallback = self._base_dir / settings.FALLBACK_DETECTOR_PATH
        return str(fallback)

    def get_active_classifier_path(self) -> Optional[str]:
        if not self.active_classifier_model:
            return None
        weights_path = self._base_dir / "weights" / "classifiers" / self.active_classifier_model
        if weights_path.exists():
            return str(weights_path)
        return None

    def reset_config(self) -> AppConfigResponse:
        logger.info("Resetting runtime configuration back to factory default settings.")
        self.active_detector_model = os.path.basename(settings.FALLBACK_DETECTOR_PATH)
        self.active_classifier_model = None
        self.default_detector_conf = 0.6
        self.default_classifier_conf = 0.6
        self.model_idle_timeout_seconds = 1800
        settings.MODEL_IDLE_TIMEOUT_SECONDS = 1800
        self.default_device = 0
        self.log_level = "INFO"

        if self._config_file.exists():
            try:
                os.remove(self._config_file)
                logger.debug(f"Removed persistent config file: {self._config_file}")
            except Exception as e:
                logger.warning(f"Could not remove {self._config_file}: {e}")

        return self.get_config_response()


config_manager = ConfigManager()
