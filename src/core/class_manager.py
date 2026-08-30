import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger
from src.core.config import settings
from src.core.config_manager import config_manager
from src.api.schemas.classes import SignClassItem, ClassifierClassesResponse

DEFAULT_CLASSES: Dict[int, str] = {
    0: "No_entry",
    1: "No_cars",
    2: "No_cars_turn_right",
    3: "No_cars_turn_left",
    4: "No_motorcycles",
    5: "No_trucks",
    6: "No_trucks_over_2.5T",
    7: "No_pedestrians",
    8: "Height_limit",
    9: "No_turn_left",
    10: "No_turn_right",
    11: "No_U_turn",
    12: "No_turn_left_and_U_turn",
    13: "Speed_limit_40",
    14: "Speed_limit_50",
    15: "Speed_limit_60",
    16: "Speed_limit_70",
    17: "Speed_limit_80",
    18: "No_honking",
    19: "No_stopping_and_parking",
    20: "No_parking",
    21: "No_parking_odd_days",
    22: "No_parking_even_days",
    23: "No_straight_ahead",
    24: "No_turn_left_and_right",
    25: "Dangerous_curve_left",
    26: "Road_narrows",
    27: "Crossroad",
    28: "Intersection_with_non_priority_road",
    29: "Give_way",
    30: "Road_bump",
    31: "Pedestrian_crossing",
    32: "Children_crossing",
    33: "Road_works",
    34: "Overhead_power_lines",
    35: "Clearance_height",
    36: "Slow_down",
    37: "Go_straight",
    38: "Keep_right",
    39: "Roundabout",
    40: "Main_road",
    41: "End_main_road",
    42: "Pedestrian_crossing_zone",
    43: "Hospital",
    44: "Bus_stop",
    45: "Market",
    46: "Other_prohibition_signs",
    47: "No_slow_moving_vehicles",
    48: "U_turn_area",
    49: "Pedestrian_overpass",
    50: "Width_limit",
    51: "Weight_limit",
    52: "No_trailers",
    53: "No_vehicles_carrying_dangerous_goods",
    54: "Tunnel",
    55: "No_bicycles",
    56: "Cliff_ahead",
    57: "Min_speed_limit_30",
    58: "Min_distance_between_vehicles",
    59: "No_overtaking",
    60: "End_min_speed_limit_30",
    61: "No_motorcycles_and_slow_vehicles",
    62: "End_dual_carriageway",
    63: "No_buses_and_trucks",
    64: "Two_way_traffic",
    65: "Accident_area",
    66: "Turn_left_only",
    67: "Turn_right_only",
    68: "One_way_road",
    69: "Railway_crossing_with_gates",
    70: "Underpass",
    71: "Residential_area",
    72: "End_residential_area",
    73: "No_cars_U_turn",
    74: "Dangerous_curve_right",
    75: "Multiple_curves",
    76: "Road_narrows_left",
    77: "Road_narrows_right",
    78: "Intersection_with_non_priority_road_right",
    79: "T_junction",
    80: "T_junction_right",
    81: "T_junction_left",
    82: "Intersection_with_non_priority_road_left",
    83: "End_no_overtaking",
    84: "End_all_prohibitions",
    85: "End_tunnel",
    86: "Speed_limit_90",
    87: "Speed_limit_100",
    88: "Road_reserved_for_cars",
    89: "Expressway",
    90: "Axle_weight_limit",
    91: "Road_closed",
    92: "Dead_end",
    93: "Reserved_for_disabled",
    94: "Checkpoint",
    95: "Speed_limit_120",
    96: "Parking_area",
    97: "Min_speed_limit_60",
    98: "Steep_descent_5_percent",
    99: "Steep_ascent_5_percent",
}


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
