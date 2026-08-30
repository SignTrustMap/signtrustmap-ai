import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger
from src.core.config import settings
from src.core.config_manager import config_manager
from src.core.classes import DEFAULT_CLASSES
from src.api.schemas.classes import SignClassItem, ClassifierClassesResponse


class ClassManager:
    def __init__(self):
        self._base_dir = Path(__file__).resolve().parent.parent.parent
        self._classes_file = self._base_dir / "configs" / "classes.json"
        self._classes: Dict[int, str] = {}
        self.load_classes()

    def load_classes(self) -> None:
        if self._classes_file.exists():
            try:
                with open(self._classes_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    loaded_classes = {int(k): str(v) for k, v in raw_data.items()}
                    self._classes = loaded_classes
                logger.info(f"Loaded {len(self._classes)} classes from {self._classes_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to read classes from {self._classes_file}: {e}. Falling back to defaults.")

        self._classes = dict(DEFAULT_CLASSES)
        self.save_classes()
        logger.info(f"Initialized {len(self._classes)} default classes at {self._classes_file}")

    def save_classes(self) -> None:
        try:
            self._classes_file.parent.mkdir(parents=True, exist_ok=True)
            serializable = {str(k): v for k, v in sorted(self._classes.items())}
            with open(self._classes_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2, ensure_ascii=False)
            logger.debug(f"Persisted {len(self._classes)} classes to {self._classes_file}")
        except Exception as e:
            logger.error(f"Failed to persist classes to {self._classes_file}: {e}")

    def get_classes_map(self) -> Dict[int, str]:
        return dict(self._classes)

    def format_item(self, class_id: int, class_name: str, template: Optional[str] = None) -> SignClassItem:
        active_template = template or config_manager.classifier_prompt_template or settings.CLASSIFIER_PROMPT_TEMPLATE
        human_readable = class_name.replace("_", " ")
        prompt = active_template.format(human_readable)
        return SignClassItem(
            class_id=class_id,
            class_name=class_name,
            human_readable_name=human_readable,
            prompt=prompt,
        )

    def get_all_classes_response(self, template: Optional[str] = None) -> ClassifierClassesResponse:
        active_template = template or config_manager.classifier_prompt_template or settings.CLASSIFIER_PROMPT_TEMPLATE
        items = [
            self.format_item(cid, self._classes[cid], active_template)
            for cid in sorted(self._classes.keys())
        ]
        return ClassifierClassesResponse(
            total_classes=len(items),
            prompt_template=active_template,
            classes=items,
        )

    def get_class_by_id(self, class_id: int, template: Optional[str] = None) -> Optional[SignClassItem]:
        if class_id not in self._classes:
            return None
        return self.format_item(class_id, self._classes[class_id], template)

    def add_class(self, class_name: str, custom_id: Optional[int] = None, template: Optional[str] = None) -> SignClassItem:
        clean_name = class_name.strip().replace(" ", "_")
        if custom_id is not None:
            if custom_id in self._classes:
                raise ValueError(f"Class ID {custom_id} already exists with name '{self._classes[custom_id]}'.")
            target_id = custom_id
        else:
            target_id = max(self._classes.keys()) + 1 if self._classes else 0

        self._classes[target_id] = clean_name
        self.save_classes()
        logger.info(f"Added class ID {target_id}: '{clean_name}'")
        return self.format_item(target_id, clean_name, template)

    def update_class(self, class_id: int, class_name: str, template: Optional[str] = None) -> SignClassItem:
        if class_id not in self._classes:
            raise KeyError(f"Class ID {class_id} not found.")
        clean_name = class_name.strip().replace(" ", "_")
        self._classes[class_id] = clean_name
        self.save_classes()
        logger.info(f"Updated class ID {class_id} to '{clean_name}'")
        return self.format_item(class_id, clean_name, template)

    def delete_class(self, class_id: int) -> bool:
        if class_id not in self._classes:
            raise KeyError(f"Class ID {class_id} not found.")
        removed_name = self._classes.pop(class_id)
        self.save_classes()
        logger.info(f"Deleted class ID {class_id}: '{removed_name}'")
        return True

    def reset_to_defaults(self, template: Optional[str] = None) -> ClassifierClassesResponse:
        self._classes = dict(DEFAULT_CLASSES)
        self.save_classes()
        logger.info("Reset classes configuration to default 100 classes.")
        return self.get_all_classes_response(template)

    def get_prompts(self, template: Optional[str] = None) -> List[str]:
        active_template = template or config_manager.classifier_prompt_template or settings.CLASSIFIER_PROMPT_TEMPLATE
        prompts = []
        for cid in sorted(self._classes.keys()):
            raw_name = self._classes[cid]
            human_readable = raw_name.replace("_", " ")
            prompts.append(active_template.format(human_readable))
        return prompts


class_manager = ClassManager()
