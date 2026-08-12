import os
from dotenv import load_dotenv
load_dotenv()

from src.core.logger import setup_logger
setup_logger()

from fastapi import FastAPI
from src.api.endpoints import router

app = FastAPI(title="SignTrustMap AI Processing Service")

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "SignTrustMap AI Processing Service is online"}
