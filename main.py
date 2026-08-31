import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from src.core.logger import setup_logger
setup_logger()

from src.core.utils import get_reload_excludes
from src.core.config_manager import config_manager
from src.core.model_manager import model_manager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from src.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_manager.load_config()
    model_manager.start_watchdog()
    yield
    model_manager.stop_watchdog()
    model_manager.unload_all()


app = FastAPI(title="SignTrustMap AI Processing Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    excludes = get_reload_excludes()
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_config=None,
        reload_excludes=excludes,
        timeout_graceful_shutdown=3,
    )
