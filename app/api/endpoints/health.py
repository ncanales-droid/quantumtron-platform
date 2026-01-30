from fastapi import APIRouter

router = APIRouter()

@router.get("/health", tags=["health"])
async def health():
    return {
        "status": "healthy",
        "service": "quantumtron-api",
        "version": "1.0.0"
    }

@router.get("/ready", tags=["health"])
async def ready():
    return {"status": "ready"}
