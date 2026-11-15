from fastapi import APIRouter
from src.api.v1.routes import (
    active_learning,
    classifier,
    classes,
    config,
    detector,
    inference,
    models,
    system,
    training_data,
)

v1_router = APIRouter()

v1_router.include_router(training_data.router, prefix="/training-data")
v1_router.include_router(classifier.router, prefix="/classifier")
v1_router.include_router(detector.router, prefix="/detector")
v1_router.include_router(inference.router, prefix="/inference")
v1_router.include_router(models.router, prefix="/models")
v1_router.include_router(active_learning.router, prefix="/active-learning")
v1_router.include_router(classes.router, prefix="/classes")
v1_router.include_router(config.router, prefix="/config")
v1_router.include_router(system.router, prefix="/system")
