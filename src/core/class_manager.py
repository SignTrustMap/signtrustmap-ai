import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger
from sqlalchemy.orm import Session
from src.core.config import settings
from src.core.config_manager import config_manager
from src.infrastructure.db import sync_engine
from src.infrastructure.models import ModelVersion, ModelClass
from src.api.schemas.classes import (
    CategoryItem,
    CategoryListResponse,
    SignClassItem,
    ClassifierClassesResponse,
)

class ClassManager:
    def __init__(self):
        self._base_dir = Path(__file__).resolve().parent.parent.parent
        self._classes_file = self._base_dir / "configs" / "classes.json"
        self._metadata_file = self._base_dir / "configs" / "sign_metadata.json"
        self._categories_file = self._base_dir / "configs" / "categories.json"
        self._classes: Dict[int, str] = {}
        self._metadata: Dict[int, Dict[str, Any]] = {}
        self._categories: List[Dict[str, Any]] = []
        self.load_classes()

    def load_classes(self) -> None:
        db_loaded = False
        try:
            with Session(sync_engine) as session:
                active_classifier = (
                    session.query(ModelVersion)
                    .filter_by(model_type="CLASSIFIER", is_active=True)
                    .first()
                )
                if active_classifier:
                    db_classes = (
                        session.query(ModelClass)
                        .filter_by(model_id=active_classifier.id)
                        .order_by(ModelClass.class_index.asc())
                        .all()
                    )
                    if db_classes:
                        self._classes = {r.class_index: r.class_name for r in db_classes}
                        db_loaded = True
                        logger.info(
                            f"Loaded {len(self._classes)} classes from AI Database for active classifier '{active_classifier.version_tag}'"
                        )
        except Exception as e:
            logger.warning("================================================================================")
            logger.warning(f"FALLBACK WARNING: Database query for model classes failed ({e})! Falling back to local JSON files.")
            logger.warning("================================================================================")

        if not db_loaded:
            logger.warning("================================================================================")
            logger.warning("FALLBACK WARNING: Active classifier classes not loaded from database! Falling back to configs/classes.json.")
            logger.warning("================================================================================")
            if self._classes_file.exists():
                try:
                    with open(self._classes_file, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
                        self._classes = {int(k): str(v) for k, v in raw_data.items()}
                    logger.info(f"Loaded {len(self._classes)} classes from {self._classes_file}")
                except Exception as e:
                    logger.warning("================================================================================")
                    logger.warning(f"FALLBACK WARNING: Failed to read classes from {self._classes_file}: {e}.")
                    logger.warning("================================================================================")
                    self._classes = {}
            else:
                self._classes = {}

        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, "r", encoding="utf-8") as f:
                    raw_meta = json.load(f)
                    self._metadata = {
                        int(k): {
                            "category": v.get("category"),
                            "name_vi": v.get("name_vi"),
                            "name_en": v.get("name_en"),
                            "standard_code": v.get("standard_code"),
                        }
                        for k, v in raw_meta.items()
                    }
                logger.info(f"Loaded {len(self._metadata)} sign metadata entries from {self._metadata_file}")
            except Exception as e:
                logger.warning("================================================================================")
                logger.warning(f"FALLBACK WARNING: Failed to read metadata from {self._metadata_file}: {e}.")
                logger.warning("================================================================================")
                self._metadata = {}
        else:
            self._metadata = {}

        if self._categories_file.exists():
            try:
                with open(self._categories_file, "r", encoding="utf-8") as f:
                    self._categories = json.load(f)
                logger.info(f"Loaded {len(self._categories)} categories from {self._categories_file}")
            except Exception as e:
                logger.warning("================================================================================")
                logger.warning(f"FALLBACK WARNING: Failed to read categories from {self._categories_file}: {e}.")
                logger.warning("================================================================================")
                self._categories = []
        else:
            self._categories = []

        self.save_classes()

    def save_classes(self) -> None:
        try:
            self._classes_file.parent.mkdir(parents=True, exist_ok=True)
            serializable_classes = {str(k): v for k, v in sorted(self._classes.items())}
            with open(self._classes_file, "w", encoding="utf-8") as f:
                json.dump(serializable_classes, f, indent=2, ensure_ascii=False)

            serializable_meta = {
                str(k): {
                    "class_id": k,
                    "class_name": self._classes.get(k, ""),
                    "category": v.get("category"),
                    "name_vi": v.get("name_vi"),
                    "name_en": v.get("name_en"),
                    "standard_code": v.get("standard_code"),
                }
                for k, v in sorted(self._metadata.items())
            }
            with open(self._metadata_file, "w", encoding="utf-8") as f:
                json.dump(serializable_meta, f, indent=2, ensure_ascii=False)

            with open(self._categories_file, "w", encoding="utf-8") as f:
                json.dump(self._categories, f, indent=2, ensure_ascii=False)

            logger.debug(f"Persisted classes, metadata, and categories configs")
        except Exception as e:
            logger.error(f"Failed to persist classes configurations: {e}")

    def get_classes_map(self) -> Dict[int, str]:
        return dict(self._classes)

    def get_sign_metadata(self, class_identifier: Union[int, str]) -> Dict[str, Any]:
        if isinstance(class_identifier, int):
            return dict(self._metadata.get(class_identifier, {}))
        target_name = str(class_identifier).strip()
        for cid, name in self._classes.items():
            if name == target_name:
                return dict(self._metadata.get(cid, {}))
        return {}

    def get_all_categories_response(self) -> CategoryListResponse:
        items = [CategoryItem(**item) for item in self._categories]
        return CategoryListResponse(
            total_categories=len(items),
            categories=items,
        )

    def format_item(self, class_id: int, class_name: str, template: Optional[str] = None) -> SignClassItem:
        active_template = template or config_manager.classifier_prompt_template or settings.CLASSIFIER_PROMPT_TEMPLATE
        human_readable = class_name.replace("_", " ")
        prompt = active_template.format(human_readable)
        meta = self._metadata.get(class_id, {})
        return SignClassItem(
            class_id=class_id,
            class_name=class_name,
            human_readable_name=human_readable,
            prompt=prompt,
            category=meta.get("category"),
            name_vi=meta.get("name_vi"),
            name_en=meta.get("name_en"),
            standard_code=meta.get("standard_code"),
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

    def add_class(
        self,
        class_name: str,
        custom_id: Optional[int] = None,
        category: Optional[str] = None,
        name_vi: Optional[str] = None,
        name_en: Optional[str] = None,
        standard_code: Optional[str] = None,
        template: Optional[str] = None,
    ) -> SignClassItem:
        clean_name = class_name.strip().replace(" ", "_")
        if custom_id is not None:
            if custom_id in self._classes:
                raise ValueError(f"Class ID {custom_id} already exists with name '{self._classes[custom_id]}'.")
            target_id = custom_id
        else:
            target_id = max(self._classes.keys()) + 1 if self._classes else 0

        self._classes[target_id] = clean_name
        self._metadata[target_id] = {
            "category": category,
            "name_vi": name_vi,
            "name_en": name_en,
            "standard_code": standard_code,
        }
        self.save_classes()
        logger.info(f"Added class ID {target_id}: '{clean_name}' (category={category}, standard={standard_code})")
        return self.format_item(target_id, clean_name, template)

    def update_class(
        self,
        class_id: int,
        class_name: Optional[str] = None,
        category: Optional[str] = None,
        name_vi: Optional[str] = None,
        name_en: Optional[str] = None,
        standard_code: Optional[str] = None,
        template: Optional[str] = None,
    ) -> SignClassItem:
        if class_id not in self._classes:
            raise KeyError(f"Class ID {class_id} not found.")
        if class_name is not None:
            clean_name = class_name.strip().replace(" ", "_")
            self._classes[class_id] = clean_name
        else:
            clean_name = self._classes[class_id]

        if class_id not in self._metadata:
            self._metadata[class_id] = {}

        if category is not None:
            self._metadata[class_id]["category"] = category
        if name_vi is not None:
            self._metadata[class_id]["name_vi"] = name_vi
        if name_en is not None:
            self._metadata[class_id]["name_en"] = name_en
        if standard_code is not None:
            self._metadata[class_id]["standard_code"] = standard_code

        self.save_classes()
        logger.info(f"Updated class ID {class_id} to '{clean_name}'")
        return self.format_item(class_id, clean_name, template)

    def delete_class(self, class_id: int) -> bool:
        if class_id not in self._classes:
            raise KeyError(f"Class ID {class_id} not found.")
        removed_name = self._classes.pop(class_id)
        self._metadata.pop(class_id, None)
        self.save_classes()
        logger.info(f"Deleted class ID {class_id}: '{removed_name}'")
        return True

    def reset_to_defaults(self, template: Optional[str] = None) -> ClassifierClassesResponse:
        self.load_classes()
        logger.info("Reloaded classes from active classifier in database.")
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
