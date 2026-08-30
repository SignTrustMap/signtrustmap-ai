from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status
from loguru import logger
from src.core.class_manager import class_manager
from src.core.model_manager import model_manager
from src.api.schemas.classes import (
    ClassifierClassesResponse,
    SignClassDetailResponse,
    CreateClassRequest,
    UpdateClassRequest,
    ClassActionResponse,
    ClassSyncRequest,
    ClassSyncResponse,
)

router = APIRouter(tags=["Classes"])


@router.get("", response_model=ClassifierClassesResponse)
async def list_classes(
    template: Optional[str] = Query(None, description="Optional prompt template override"),
):
    logger.debug(f"Retrieving all classifier classes (template override: {template})")
    return class_manager.get_all_classes_response(template=template)


@router.get("/{class_id}", response_model=SignClassDetailResponse)
async def get_class(
    class_id: int,
    template: Optional[str] = Query(None, description="Optional prompt template override"),
):
    logger.debug(f"Retrieving class details for class ID: {class_id}")
    item = class_manager.get_class_by_id(class_id, template=template)
    if item is None:
        logger.warning(f"Class ID {class_id} not found.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Class with ID {class_id} not found.",
        )
    return SignClassDetailResponse(class_item=item)


@router.post("", response_model=ClassActionResponse, status_code=status.HTTP_201_CREATED)
async def create_class(request: CreateClassRequest):
    logger.info(f"Received request to create class: name='{request.class_name}', custom_id={request.custom_id}")
    try:
        created_item = class_manager.add_class(
            class_name=request.class_name,
            custom_id=request.custom_id,
        )
        if model_manager.is_classifier_loaded():
            model_manager.sync_classifier_classes()
        return ClassActionResponse(
            status="CREATED",
            message=f"Class '{created_item.class_name}' with ID {created_item.class_id} successfully created.",
            class_item=created_item,
        )
    except ValueError as e:
        logger.warning(f"Failed to create class: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error creating class: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Class creation failed: {str(e)}",
        )


@router.put("/{class_id}", response_model=ClassActionResponse)
async def update_class(class_id: int, request: UpdateClassRequest):
    logger.info(f"Received request to update class ID {class_id} to name='{request.class_name}'")
    try:
        updated_item = class_manager.update_class(
            class_id=class_id,
            class_name=request.class_name,
        )
        if model_manager.is_classifier_loaded():
            model_manager.sync_classifier_classes()
        return ClassActionResponse(
            status="UPDATED",
            message=f"Class ID {class_id} successfully updated to '{updated_item.class_name}'.",
            class_item=updated_item,
        )
    except KeyError as e:
        logger.warning(f"Class ID {class_id} not found for update: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Class with ID {class_id} not found.",
        )
    except Exception as e:
        logger.error(f"Error updating class ID {class_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Class update failed: {str(e)}",
        )


@router.delete("/{class_id}", response_model=ClassActionResponse)
async def delete_class(class_id: int):
    logger.info(f"Received request to delete class ID {class_id}")
    try:
        target_item = class_manager.get_class_by_id(class_id)
        if target_item is None:
            raise KeyError(f"Class ID {class_id} not found.")

        class_manager.delete_class(class_id)
        if model_manager.is_classifier_loaded():
            model_manager.sync_classifier_classes()
        return ClassActionResponse(
            status="DELETED",
            message=f"Class ID {class_id} ('{target_item.class_name}') successfully deleted.",
            class_item=target_item,
        )
    except KeyError as e:
        logger.warning(f"Class ID {class_id} not found for deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Class with ID {class_id} not found.",
        )
    except Exception as e:
        logger.error(f"Error deleting class ID {class_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Class deletion failed: {str(e)}",
        )


@router.post("/reset", response_model=ClassActionResponse)
async def reset_classes():
    logger.info("Received request to reset classes configuration to default 100 classes.")
    try:
        class_manager.reset_to_defaults()
        if model_manager.is_classifier_loaded():
            model_manager.sync_classifier_classes()
        return ClassActionResponse(
            status="RESET_SUCCESSFUL",
            message="Classes configuration successfully restored to default 100 classes.",
            class_item=None,
        )
    except Exception as e:
        logger.error(f"Error resetting classes: {e}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Class reset failed: {str(e)}",
        )


@router.post("/sync", response_model=ClassSyncResponse)
async def sync_classifier_embeddings(
    request: Optional[ClassSyncRequest] = None,
):
    req = request or ClassSyncRequest()
    logger.info(f"Received request to sync classifier text embeddings (model='{req.model_name}', template='{req.prompt_template}')")
    try:
        is_loaded, target_model = model_manager.sync_classifier_classes(
            model_name=req.model_name,
            template=req.prompt_template,
        )
        total_classes = len(class_manager.get_classes_map())

        msg = (
            f"Classifier '{target_model}' text embeddings successfully rebuilt in RAM/GPU and cached."
            if is_loaded
            else f"Classifier '{target_model}' text embeddings successfully pre-computed and cached to disk."
        )

        return ClassSyncResponse(
            status="SYNC_SUCCESSFUL",
            target_model=target_model,
            total_classes_synced=total_classes,
            is_classifier_loaded=is_loaded,
            message=msg,
        )
    except Exception as e:
        logger.error(f"Error syncing classifier embeddings: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to sync text embeddings: {str(e)}",
        )

