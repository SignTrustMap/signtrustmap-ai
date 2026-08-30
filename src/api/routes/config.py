from fastapi import APIRouter, HTTPException, status
from loguru import logger
from src.core.config_manager import config_manager
from src.api.schemas.config import (
    AppConfigResponse,
    AppConfigUpdateRequest,
    ConfigResetResponse,
)

router = APIRouter(tags=["Config"])


@router.get("", response_model=AppConfigResponse)
async def get_system_configuration():
    logger.debug("Retrieving current system configuration.")
    return config_manager.get_config_response()


@router.patch("", response_model=AppConfigResponse)
async def update_system_configuration(request: AppConfigUpdateRequest):
    logger.info(f"Received configuration update request: {request.model_dump(exclude_unset=True)}")
    try:
        updated_config = config_manager.update_config(request)
        return updated_config
    except FileNotFoundError as e:
        logger.warning(f"Failed to update configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Configuration update failed: {str(e)}",
        )


@router.post("/reset", response_model=ConfigResetResponse)
async def reset_system_configuration():
    logger.info("Received request to reset system configuration to defaults.")
    reset_config = config_manager.reset_config()
    return {
        "status": "RESET_SUCCESSFUL",
        "message": "Configuration restored to factory default settings.",
        "config": reset_config,
    }
