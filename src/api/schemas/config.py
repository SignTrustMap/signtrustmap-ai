from typing import Any, Dict
from pydantic import BaseModel, Field


class ConfigResetResponse(BaseModel):
    status: str = Field(..., description="Reset status", examples=["RESET_SUCCESSFUL"])
    message: str = Field(..., description="Confirmation detail", examples=["Configuration restored to factory default settings."])
    config: Dict[str, Dict[str, Any]] = Field(..., description="Active configuration state after reset")
