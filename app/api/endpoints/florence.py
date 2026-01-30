from fastapi import APIRouter

router = APIRouter()

@router.get("/florence/health", tags=["florence"])
async def florence_health():
    return {"status": "healthy", "component": "florence"}

@router.post("/florence/chat", tags=["florence"])
async def chat():
    return {
        "message": "Chat endpoint",
        "response": "Florence AI is ready to assist",
        "status": "operational"
    }
