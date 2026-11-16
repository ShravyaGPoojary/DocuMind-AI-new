from datetime import datetime, timezone
from fastapi import APIRouter
from app.core.config import settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="System Health Check")
async def get_health():
    """
    Returns the system health, version, stage, and operational status.
    Used by frontend health monitoring and external uptime checks.
    """
    return HealthResponse(
        status="ok",
        app=settings.PROJECT_NAME,
        version=settings.VERSION,
        stage=settings.STAGE,
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(timezone.utc),
        uptime_message="FastAPI backend operational with Stage 5 Intelligent Document Summarization",
    )
