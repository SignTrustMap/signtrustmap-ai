import os
from dotenv import load_dotenv
load_dotenv()

from src.core.logger import setup_logger
setup_logger()

from src.core.utils import get_reload_excludes
from fastapi import FastAPI
import uvicorn
from src.api.endpoints import router

app = FastAPI(title="SignTrustMap AI Processing Service")

app.include_router(router)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    excludes = get_reload_excludes()
    uvicorn.run("main:app", host=host, port=port, reload=reload, log_config=None, reload_excludes=excludes)

