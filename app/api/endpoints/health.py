"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import logging
from datetime import datetime

from app.api.dependencies import get_database_session
from app.models.pydantic_models import HealthResponse

router = APIRouter(tags=["health"])

logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health_check(
    db: AsyncSession = Depends(get_database_session)
) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns service status including database connectivity.
    """
    try:
        # Simple database check
        await db.execute(text("SELECT 1"))
        db_status = "connected"
        details = {"database": "healthy"}
        
        logger.info("Health check passed")
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        db_status = "disconnected"
        details = {"database": "unhealthy", "error": str(e)}
    
    return HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        service="QuantumTron API",
        version="0.1.0",
        timestamp=datetime.now(),
        details=details
    )


@router.get("/health/live")
async def liveness_probe() -> dict:
    """
    Liveness probe for Kubernetes/container orchestration.
    
    Returns a simple response indicating the service is alive.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health/ready")
async def readiness_probe(
    db: AsyncSession = Depends(get_database_session)
) -> dict:
    """
    Readiness probe for Kubernetes/container orchestration.
    
    Returns readiness status including database connectivity.
    """
    try:
        await db.execute(text("SELECT 1"))
        return {
            "status": "ready",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "not_ready",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }