"""
Dependencies for Florence API
"""

from typing import Generator
from fastapi import Depends, HTTPException, status
from app.services.florence import (
    deepseek_client,
    knowledge_base,
    statistical_engine,
    response_formatter,
    FlorenceOrchestrator
)


def get_deepseek_client():
    """Get DeepSeek client dependency"""
    if not deepseek_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DeepSeek service not available"
        )
    return deepseek_client


def get_knowledge_base():
    """Get knowledge base dependency"""
    if not knowledge_base:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge Base not available"
        )
    return knowledge_base


def get_statistical_engine():
    """Get statistical engine dependency"""
    if not statistical_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Statistical Engine not available"
        )
    return statistical_engine


def get_response_formatter():
    """Get response formatter dependency"""
    if not response_formatter:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Response Formatter not available"
        )
    return response_formatter


def get_florence_orchestrator() -> FlorenceOrchestrator:
    """Get Florence orchestrator dependency"""
    try:
        return FlorenceOrchestrator()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize Florence: {str(e)}"
        )
