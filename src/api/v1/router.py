from fastapi import APIRouter
from src.api.v1.routes import inference, models, system, config, classes, active_learning

v1_router = APIRouter()

v1_router.include_router(inference.router, prefix="/inference")
v1_router.include_router(models.router, prefix="/models")
v1_router.include_router(active_learning.router, prefix="/active-learning")
v1_router.include_router(classes.router, prefix="/classes")
v1_router.include_router(config.router, prefix="/config")
v1_router.include_router(system.router, prefix="/system")
