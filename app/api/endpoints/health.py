"""
Health check endpoints
"""

from fastapi import APIRouter
from app.schemas.florence import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from app.services.florence import (
        deepseek_client,
        knowledge_base,
        statistical_engine,
        response_formatter
    )
    
    components = {
        "deepseek": bool(deepseek_client),
        "knowledge_base": bool(knowledge_base),
        "statistical_engine": bool(statistical_engine),
        "response_formatter": bool(response_formatter)
    }
    
    all_healthy = all(components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="1.0.0",
        components=components
    )


@router.get("/ready")
async def readiness_check():
    """Readiness check for load balancers"""
    from app.services.florence import deepseek_client
    
    if deepseek_client and deepseek_client.api_key:
        return {"status": "ready"}
    else:
        return {"status": "not_ready", "reason": "DeepSeek not configured"}
