import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from src.core.logger import setup_logger
setup_logger()

from src.core.utils import get_reload_excludes
from src.core.model_manager import model_manager
from fastapi import FastAPI
import uvicorn
from src.api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_manager.start_watchdog()
    yield
    model_manager.stop_watchdog()
    model_manager.unload_model()


app = FastAPI(title="SignTrustMap AI Processing Service", lifespan=lifespan)

app.include_router(api_router)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    excludes = get_reload_excludes()
    uvicorn.run("main:app", host=host, port=port, reload=reload, log_config=None, reload_excludes=excludes)
