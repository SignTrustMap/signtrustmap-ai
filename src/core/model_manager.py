import asyncio
import gc
import os
import time
from typing import Optional
import torch
from loguru import logger
from src.core.config import settings
from src.core.detector import SignDetector


class ModelManager:
    def __init__(self):
        self._detector: Optional[SignDetector] = None
        self._loaded_weights_path: Optional[str] = None
        self._last_accessed: float = 0.0
        self._lock = asyncio.Lock()
        self._watchdog_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

    def get_detector(self, weights_path: Optional[str] = None) -> SignDetector:
        if self._detector is None or (weights_path and weights_path != self._loaded_weights_path):
            logger.info(f"Initializing SignDetector with weights: {weights_path or 'default'}...")
            self._detector = SignDetector(weights_path=weights_path)
            self._loaded_weights_path = weights_path or settings.FALLBACK_MODEL_PATH

        self._last_accessed = time.time()
        return self._detector

    def touch(self) -> None:
        self._last_accessed = time.time()

    def unload_model(self) -> bool:
        if self._detector is not None:
            weights_name = os.path.basename(self._loaded_weights_path) if self._loaded_weights_path else "model"
            logger.info(f"Unloading model '{weights_name}' from RAM/VRAM to reclaim memory.")

            try:
                if hasattr(self._detector, "model"):
                    del self._detector.model
            except Exception as e:
                logger.warning(f"Error deleting detector model attribute: {e}")

            self._detector = None
            self._loaded_weights_path = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            logger.info("Model successfully unloaded. Memory reclaimed.")
            return True
        return False

    def is_loaded(self) -> bool:
        return self._detector is not None

    def get_loaded_model_filename(self) -> Optional[str]:
        if self._detector is not None and self._loaded_weights_path:
            return os.path.basename(self._loaded_weights_path)
        return None

    def get_idle_seconds(self) -> float:
        if self._detector is not None and self._last_accessed > 0:
            return round(time.time() - self._last_accessed, 1)
        return 0.0

    async def _watchdog_loop(self) -> None:
        logger.debug(f"Model idle watchdog started (timeout={settings.MODEL_IDLE_TIMEOUT_SECONDS}s).")
        while self._is_running:
            try:
                await asyncio.sleep(15)
                if not self._is_running:
                    break

                if self._detector is not None and settings.MODEL_IDLE_TIMEOUT_SECONDS > 0:
                    idle_duration = time.time() - self._last_accessed
                    if idle_duration >= settings.MODEL_IDLE_TIMEOUT_SECONDS:
                        logger.info(
                            f"Model has been idle for {idle_duration:.1f}s (threshold: {settings.MODEL_IDLE_TIMEOUT_SECONDS}s). "
                            f"Triggering automatic model unload."
                        )
                        self.unload_model()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in model watchdog loop: {e}")

    def start_watchdog(self) -> None:
        if not self._is_running:
            self._is_running = True
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    def stop_watchdog(self) -> None:
        self._is_running = False
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()


model_manager = ModelManager()
