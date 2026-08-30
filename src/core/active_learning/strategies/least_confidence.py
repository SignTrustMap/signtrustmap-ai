from typing import Any, Dict
from src.core.active_learning.base import BaseActiveLearningScorer


class LeastConfidenceScorer(BaseActiveLearningScorer):
    name: str = "least_confidence"

    def compute_box_score(self, detection: Dict[str, Any]) -> float:
        conf = float(detection.get("confidence", 0.0))
        return max(0.0, min(1.0, 1.0 - conf))
