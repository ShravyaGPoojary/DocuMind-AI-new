from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok", description="Current health status of the backend")
    app: str = Field(default="DocuMind AI", description="Application name")
    version: str = Field(default="1.0.0", description="Application release version")
    stage: str = Field(default="Stage 1 - Complete UI + Backend Foundation", description="Project development stage")
    environment: str = Field(default="development", description="Execution environment")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of the health check")
    uptime_message: str = Field(default="FastAPI backend operational and ready for Stage 2 RAG integration")
