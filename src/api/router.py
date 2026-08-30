from fastapi import APIRouter
from src.api.routes import inference, models, system, config
from src.api.schemas.system import HealthResponse

api_router = APIRouter()

api_router.include_router(inference.router, prefix="/api/inference")
api_router.include_router(models.router, prefix="/api/models")
api_router.include_router(system.router, prefix="/api/system")
api_router.include_router(config.router, prefix="/api/config")
api_router.add_api_route("/health", system.health_check, methods=["GET"], tags=["System"], response_model=HealthResponse)
