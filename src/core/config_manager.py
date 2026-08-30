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
        
        self.active_model: str = os.path.basename(settings.FALLBACK_MODEL_PATH)
        self.default_conf: float = 0.6
        self.model_idle_timeout_seconds: int = settings.MODEL_IDLE_TIMEOUT_SECONDS
        self.default_device: int = 0
        self.log_level: str = settings.LOG_LEVEL

    def load_config(self) -> None:
        if self._config_file.exists():
            try:
                with open(self._config_file, "r", encoding="utf-8") as f:
                    data: Dict[str, Any] = json.load(f)
                    self.active_model = data.get("active_model", self.active_model)
                    self.default_conf = float(data.get("default_conf", self.default_conf))
                    self.model_idle_timeout_seconds = int(data.get("model_idle_timeout_seconds", self.model_idle_timeout_seconds))
                    self.default_device = int(data.get("default_device", self.default_device))
                    self.log_level = data.get("log_level", self.log_level)
                logger.info(f"Loaded persistent runtime config: active_model='{self.active_model}', conf={self.default_conf}, timeout={self.model_idle_timeout_seconds}s")
            except Exception as e:
                logger.warning(f"Failed to load persistent config from {self._config_file}: {e}")

    def save_config(self) -> None:
        try:
            self._config_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "active_model": self.active_model,
                "default_conf": self.default_conf,
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
            active_model=self.active_model,
            default_conf=self.default_conf,
            model_idle_timeout_seconds=self.model_idle_timeout_seconds,
            default_device=self.default_device,
            log_level=self.log_level,
            minio_endpoint=settings.MINIO_ENDPOINT,
            minio_bucket=settings.MINIO_BUCKET,
        )

    def update_config(self, req: AppConfigUpdateRequest) -> AppConfigResponse:
        if req.active_model is not None:
            self.set_active_model(req.active_model)

        if req.default_conf is not None:
            self.default_conf = req.default_conf

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

    def set_active_model(self, weights_name: str) -> None:
        clean_name = os.path.basename(weights_name)
        weights_path = self._base_dir / "weights" / clean_name
        if not weights_path.exists():
            raise FileNotFoundError(f"Model weights file '{clean_name}' not found in weights directory.")
        self.active_model = clean_name
        self.save_config()

    def get_active_weights_path(self) -> str:
        weights_path = self._base_dir / "weights" / self.active_model
        if weights_path.exists():
            return str(weights_path)
        fallback = self._base_dir / settings.FALLBACK_MODEL_PATH
        return str(fallback)

    def reset_config(self) -> AppConfigResponse:
        logger.info("Resetting runtime configuration back to factory default settings.")
        self.active_model = os.path.basename(settings.FALLBACK_MODEL_PATH)
        self.default_conf = 0.6
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
