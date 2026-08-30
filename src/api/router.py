from fastapi import APIRouter
from src.api.routes import process, models, health

api_router = APIRouter()

api_router.include_router(process.router, prefix="/api/process", tags=["Processing"])
api_router.include_router(models.router, prefix="/api/models", tags=["Models"])
api_router.include_router(health.router, tags=["Health"])
