"""
Florence API Endpoints - VERSIÓN CORREGIDA
"""

import time
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse

from app.schemas.florence import (
    ResearchQuestionRequest,
    StatisticalAnalysisRequest,
    KnowledgeBaseRequest,
    ChatRequest,
    ResearchAnalysisResponse,
    StatisticalResultResponse,
    KnowledgeBaseResponse,
    ChatResponse,
    ErrorResponse
)

from app.api.deps import (
    get_deepseek_client,
    get_knowledge_base,
    get_statistical_engine,
    get_response_formatter,
    get_florence_orchestrator
)

# Importar base de datos
from app.db_simple import florence_db

router = APIRouter(prefix="/florence", tags=["florence"])

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint for Florence"""
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
        "response_formatter": bool(response_formatter),
        "database": bool(florence_db),
        "api_version": "1.0.0"
    }

    all_healthy = all([components["deepseek"], components["database"]])

    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": components,
        "message": "Florence PhD Research Agent"
    }

@router.post("/chat", response_model=ChatResponse)
async def chat_with_florence(
    request: ChatRequest,
    deepseek_client = Depends(get_deepseek_client)
):
    """
    Chat with Florence PhD Agent
    """
    try:
        start_time = time.time()

        response = deepseek_client.research_assistance(
            prompt=request.message,
            context=request.context or "",
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        processing_time_ms = (time.time() - start_time) * 1000
        
        # Save query to SQLite database (SOLO UNA VEZ)
        query_id = "not_saved"
        if florence_db:
            try:
                query_id = florence_db.save_query(
                    question=request.message,
                    response=response
                )
                print(f"💾 Query saved to SQLite: {query_id}")
            except Exception as db_error:
                print(f"⚠️  Could not save to database: {db_error}")
        else:
            print("⚠️  Database not available")

        return ChatResponse(
            message=request.message,
            response=response,
            model=deepseek_client.model,
            query_id=query_id,
            processing_time_ms=round(processing_time_ms, 2)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )

# ... (el resto del código permanece igual, pero con datetime importado)
