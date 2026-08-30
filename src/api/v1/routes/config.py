from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status, Body
from loguru import logger
from src.core.config_manager import config_manager
from src.api.schemas.config import ConfigResetResponse

router = APIRouter(tags=["Config"])


@router.get("", response_model=Dict[str, Dict[str, Any]])
async def get_all_configuration():
    logger.debug("Retrieving all system configuration groups.")
    return config_manager.get_all()


@router.patch("", response_model=Dict[str, Dict[str, Any]])
async def update_all_configuration(updates: Dict[str, Any] = Body(...)):
    logger.info(f"Received bulk configuration update request: {updates}")
    try:
        return config_manager.update_bulk(updates)
    except (KeyError, ValueError, FileNotFoundError) as e:
        logger.warning(f"Failed to update configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration update failed: {str(e)}",
        )


@router.get("/{group_name}", response_model=Dict[str, Any])
async def get_group_configuration(group_name: str):
    logger.debug(f"Retrieving configuration for group '{group_name}'.")
    group_config = config_manager.get_group(group_name)
    if group_config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration group '{group_name}' not found.",
        )
    return group_config


@router.patch("/{group_name}", response_model=Dict[str, Any])
async def update_group_configuration(group_name: str, updates: Dict[str, Any] = Body(...)):
    logger.info(f"Updating configuration for group '{group_name}': {updates}")
    try:
        return config_manager.update_group(group_name, updates)
    except KeyError as e:
        logger.warning(f"Group not found: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except FileNotFoundError as e:
        logger.warning(f"File not found during update: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except ValueError as e:
        logger.warning(f"Invalid field or value: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error updating group '{group_name}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
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
