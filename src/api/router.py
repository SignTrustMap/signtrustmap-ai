from fastapi import APIRouter
from src.api.routes import inference, models, health

api_router = APIRouter()

api_router.include_router(inference.router, prefix="/api/inference")
api_router.include_router(models.router, prefix="/api/models")
api_router.include_router(health.router)
