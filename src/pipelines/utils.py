import httpx
from loguru import logger

def send_callback(callback_url: str, payload: dict):
    logger.info(f"Sending webhook callback to: {callback_url}")
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(callback_url, json=payload)
            logger.debug(f"Webhook callback response status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to send webhook callback to {callback_url}: {e}")
