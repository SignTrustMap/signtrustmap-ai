from fastapi import APIRouter
from src.api.v1.router import v1_router
from src.api.v1.routes.system import health_check
from src.api.schemas.system import HealthResponse

api_router = APIRouter()

api_router.include_router(v1_router, prefix="/api/v1")
api_router.add_api_route("/health", health_check, methods=["GET"], tags=["System"], response_model=HealthResponse)
