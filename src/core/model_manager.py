import asyncio
import gc
import os
import time
from pathlib import Path
from typing import Optional
import torch
from loguru import logger
from src.core.config import settings
from src.core.config_manager import config_manager
from src.core.detector import SignDetector


class ModelManager:
    def __init__(self):
        self._detector: Optional[SignDetector] = None
        self._loaded_detector_path: Optional[str] = None
        
        self._classifier = None
        self._loaded_classifier_path: Optional[str] = None

        self._last_accessed: float = 0.0
        self._lock = asyncio.Lock()
        self._watchdog_task: Optional[asyncio.Task] = None
        self._is_running: bool = False

    def get_detector(self, weights_path: Optional[str] = None) -> SignDetector:
        target_weights = weights_path or config_manager.get_active_detector_path()

        if self._detector is None or target_weights != self._loaded_detector_path:
            logger.info(f"Initializing SignDetector with weights: {target_weights}...")
            self._detector = SignDetector(weights_path=target_weights)
            self._loaded_detector_path = target_weights

        self._last_accessed = time.time()
        return self._detector

    def get_classifier(self, weights_path: Optional[str] = None):
        target_weights = weights_path or config_manager.get_active_classifier_path()
        if not target_weights:
            return None

        if self._classifier is None or target_weights != self._loaded_classifier_path:
            logger.info(f"Initializing Classifier with weights: {target_weights}...")
            from ultralytics import YOLO
            self._classifier = YOLO(target_weights)
            self._loaded_classifier_path = target_weights

        self._last_accessed = time.time()
        return self._classifier

    def load_detector(self, weights_name: str) -> str:
        base_dir = Path(__file__).resolve().parent.parent.parent
        clean_name = os.path.basename(weights_name)
        weights_path = base_dir / "weights" / "detectors" / clean_name

        if not weights_path.exists():
            raise FileNotFoundError(f"Detector weights '{clean_name}' not found in weights/detectors/ directory.")

        if self._detector is not None:
            self.unload_detector()

        logger.info(f"Explicitly loading detector '{clean_name}' into memory...")
        self._detector = SignDetector(weights_path=str(weights_path))
        self._loaded_detector_path = str(weights_path)
        self._last_accessed = time.time()

        config_manager.set_active_detector(clean_name)
        return clean_name

    def load_classifier(self, weights_name: str) -> str:
        base_dir = Path(__file__).resolve().parent.parent.parent
        clean_name = os.path.basename(weights_name)
        weights_path = base_dir / "weights" / "classifiers" / clean_name

        if not weights_path.exists():
            raise FileNotFoundError(f"Classifier weights '{clean_name}' not found in weights/classifiers/ directory.")

        if self._classifier is not None:
            self.unload_classifier()

        logger.info(f"Explicitly loading classifier '{clean_name}' into memory...")
        from ultralytics import YOLO
        self._classifier = YOLO(str(weights_path))
        self._loaded_classifier_path = str(weights_path)
        self._last_accessed = time.time()

        config_manager.set_active_classifier(clean_name)
        return clean_name

    def touch(self) -> None:
        self._last_accessed = time.time()

    def unload_detector(self) -> bool:
        if self._detector is not None:
            name = os.path.basename(self._loaded_detector_path) if self._loaded_detector_path else "detector"
            logger.info(f"Unloading detector '{name}' from memory.")
            try:
                if hasattr(self._detector, "model"):
                    del self._detector.model
            except Exception as e:
                logger.warning(f"Error deleting detector model attribute: {e}")

            self._detector = None
            self._loaded_detector_path = None
            self._cleanup_memory()
            return True
        return False

    def unload_classifier(self) -> bool:
        if self._classifier is not None:
            name = os.path.basename(self._loaded_classifier_path) if self._loaded_classifier_path else "classifier"
            logger.info(f"Unloading classifier '{name}' from memory.")
            self._classifier = None
            self._loaded_classifier_path = None
            self._cleanup_memory()
            return True
        return False

    def unload_all(self) -> bool:
        unloaded_det = self.unload_detector()
        unloaded_cls = self.unload_classifier()
        return unloaded_det or unloaded_cls

    def _cleanup_memory(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.debug("Memory garbage collection and CUDA cache empty completed.")

    def is_detector_loaded(self) -> bool:
        return self._detector is not None

    def is_classifier_loaded(self) -> bool:
        return self._classifier is not None

    def get_loaded_detector_filename(self) -> Optional[str]:
        if self._detector is not None and self._loaded_detector_path:
            return os.path.basename(self._loaded_detector_path)
        return None

    def get_loaded_classifier_filename(self) -> Optional[str]:
        if self._classifier is not None and self._loaded_classifier_path:
            return os.path.basename(self._loaded_classifier_path)
        return None

    def get_idle_seconds(self) -> float:
        if (self._detector is not None or self._classifier is not None) and self._last_accessed > 0:
            return round(time.time() - self._last_accessed, 1)
        return 0.0

    async def _watchdog_loop(self) -> None:
        logger.debug(f"Multi-model idle watchdog started (timeout={config_manager.model_idle_timeout_seconds}s).")
        while self._is_running:
            try:
                await asyncio.sleep(15)
                if not self._is_running:
                    break

                timeout = config_manager.model_idle_timeout_seconds
                if (self._detector is not None or self._classifier is not None) and timeout > 0:
                    idle_duration = time.time() - self._last_accessed
                    if idle_duration >= timeout:
                        logger.info(
                            f"Models have been idle for {idle_duration:.1f}s (threshold: {timeout}s). "
                            f"Triggering automatic unload of all models."
                        )
                        self.unload_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in multi-model watchdog loop: {e}")

    def start_watchdog(self) -> None:
        if not self._is_running:
            self._is_running = True
            self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    def stop_watchdog(self) -> None:
        self._is_running = False
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()


model_manager = ModelManager()
