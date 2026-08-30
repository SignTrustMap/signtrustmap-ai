from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from loguru import logger

AGGREGATION_FUNCS = {
    "avg": lambda vals: sum(vals) / len(vals) if vals else 0.0,
    "max": max,
    "min": min,
    "sum": sum,
}


class BaseActiveLearningScorer(ABC):
    name: str = "base"

    @classmethod
    def get_supported_aggregations(cls) -> List[str]:
        return list(AGGREGATION_FUNCS.keys())

    def aggregate(self, values: List[float], method: str = "avg") -> float:
        func = AGGREGATION_FUNCS.get(str(method).lower(), AGGREGATION_FUNCS["avg"])
        return round(float(func(values)), 4)

    @abstractmethod
    def compute_box_score(self, detection: Dict[str, Any]) -> float:
        pass

    def score_detections(
        self,
        detections: List[Dict[str, Any]],
        aggregation_method: str = "avg",
        no_detection_score: float = 0.0,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        if not detections:
            logger.debug("Zero detections found. Returning default no_detection_score.")
            return round(float(no_detection_score), 4), []

        scored_detections = []
        box_scores = []

        for det in detections:
            box_score = round(self.compute_box_score(det), 4)
            box_scores.append(box_score)

            det_copy = dict(det)
            det_copy["score"] = box_score
            scored_detections.append(det_copy)

        image_score = self.aggregate(box_scores, aggregation_method)
        logger.debug(f"Computed {self.name} score: {image_score} (agg={aggregation_method}, count={len(detections)})")
        return image_score, scored_detections
