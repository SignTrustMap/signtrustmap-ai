from fastapi import APIRouter
from src.api.v1.routes import inference, models, system, config

v1_router = APIRouter()

v1_router.include_router(inference.router, prefix="/inference")
v1_router.include_router(models.router, prefix="/models")
v1_router.include_router(system.router, prefix="/system")
v1_router.include_router(config.router, prefix="/config")
