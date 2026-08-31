import asyncio
from typing import Any, Dict, Optional
from loguru import logger
from src.core.active_learning.store import al_store


async def trigger_retrain(
    target: str = "classifier",
    dry_run: bool = False,
    mock_duration_seconds: float = 1.0,
) -> Dict[str, Any]:
    target_norm = target.lower().strip()
    if target_norm not in ("detector", "classifier", "clip"):
        msg = f"Invalid retraining target '{target}'. Must be one of: detector, classifier, clip."
        logger.error(msg)
        return {"status": "FAILED", "target": target_norm, "error": msg}

    pending_samples = al_store.get_pending_samples(target_norm)
    sample_count = len(pending_samples)

    if sample_count == 0:
        logger.warning(f"Retraining triggered for '{target_norm}' but zero pending samples found.")
        return {
            "status": "SKIPPED",
            "target": target_norm,
            "sample_count": 0,
            "message": "No pending samples available for retraining.",
        }

    logger.info(f"Initiating mock retraining for '{target_norm}' with {sample_count} sample(s) (dry_run={dry_run})...")

    if not dry_run:
        await asyncio.sleep(mock_duration_seconds)

    new_weights_name = f"{target_norm}_al_mock_latest.pt"
    sample_ids = [s["id"] for s in pending_samples if "id" in s]

    if not dry_run:
        al_store.mark_as_trained(target_norm, sample_ids)
        al_store.record_train_history(
            target=target_norm,
            sample_count=sample_count,
            status="SUCCESS",
            weights_name=new_weights_name,
        )

    logger.info(f"Mock retraining complete for '{target_norm}'. {sample_count} sample(s) processed.")
    return {
        "status": "SUCCESS",
        "target": target_norm,
        "sample_count": sample_count,
        "new_weights": new_weights_name,
        "dry_run": dry_run,
    }
