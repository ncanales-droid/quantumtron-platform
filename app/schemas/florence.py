"""
Pydantic schemas for Florence API
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime


# ===== REQUEST SCHEMAS =====

class ResearchQuestionRequest(BaseModel):
    """Request for research question analysis"""
    question: str = Field(..., description="Research question to analyze")
    context: Optional[str] = Field(None, description="Additional context")
    data: Optional[Dict[str, Any]] = Field(None, description="Dataset for analysis")
    analysis_type: Optional[str] = Field("auto", description="Type of analysis to perform")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "Is there a significant difference between control and treatment groups?",
                "context": "Clinical trial for new drug",
                "data": {
                    "control": [65, 70, 68, 72, 75],
                    "treatment": [80, 85, 82, 78, 88]
                },
                "analysis_type": "t_test"
            }
        }


class StatisticalAnalysisRequest(BaseModel):
    """Request for statistical analysis"""
    data: Dict[str, List[float]] = Field(..., description="Data for analysis")
    analysis_type: str = Field(..., description="Type of statistical test")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis options")
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "group_a": [1.2, 2.3, 3.4, 4.5, 5.6],
                    "group_b": [2.1, 3.2, 4.3, 5.4, 6.5]
                },
                "analysis_type": "t_test",
                "options": {"test_type": "independent", "alternative": "two-sided"}
            }
        }


class KnowledgeBaseRequest(BaseModel):
    """Request for knowledge base operations"""
    operation: str = Field(..., description="Operation: add, search, query, clear")
    content: Optional[str] = Field(None, description="Text content for add operation")
    query: Optional[str] = Field(None, description="Query for search operation")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "operation": "search",
                "query": "statistical significance",
                "metadata": {"source": "user_query"}
            }
        }


class ChatRequest(BaseModel):
    """Request for chat with Florence"""
    message: str = Field(..., description="Message to Florence")
    context: Optional[str] = Field(None, description="Context for the conversation")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for response")
    max_tokens: Optional[int] = Field(1000, ge=1, le=4000, description="Maximum tokens in response")
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Explain p-values in simple terms",
                "context": "For a graduate student new to statistics",
                "temperature": 0.7,
                "max_tokens": 500
            }
        }


# ===== RESPONSE SCHEMAS =====

class StatisticalResultResponse(BaseModel):
    """Response for statistical analysis"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    interpretation: str
    assumptions: Dict[str, bool]
    recommendations: List[str]
    raw_result: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "test_name": "Student's t-test (independent)",
                "statistic": 2.345,
                "p_value": 0.021,
                "effect_size": 0.45,
                "confidence_interval": [0.12, 0.78],
                "interpretation": "The difference between means is significant...",
                "assumptions": {"normality": True, "equal_variance": True},
                "recommendations": ["Consider sample size", "Check normality assumption"],
                "raw_result": {"df": 48, "alternative": "two-sided"}
            }
        }


class KnowledgeBaseResponse(BaseModel):
    """Response for knowledge base operations"""
    operation: str
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    documents: Optional[List[Dict[str, Any]]] = None
    statistics: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "operation": "search",
                "success": True,
                "message": "Found 3 relevant documents",
                "documents": [
                    {"id": "doc1", "text": "Statistical significance...", "score": 0.85}
                ],
                "statistics": {"total_documents": 5, "search_time_ms": 45}
            }
        }


class ChatResponse(BaseModel):
    """Response for chat with Florence"""
    message: str
    response: str
    model: str
    query_id: Optional[str] = None
    usage: Optional[Dict[str, int]] = None
    processing_time_ms: float
    
    class Config:
        schema_extra = {
            "example": {
                "message": "Explain p-values",
                "response": "A p-value is the probability...",
                "model": "deepseek-chat",
                "usage": {"prompt_tokens": 25, "completion_tokens": 150},
                "processing_time_ms": 1250.5
            }
        }


class ResearchAnalysisResponse(BaseModel):
    """Complete research analysis response"""
    question: str
    statistical_analysis: Optional[StatisticalResultResponse] = None
    ai_interpretation: str
    knowledge_context: Optional[List[Dict[str, Any]]] = None
    formatted_report: Optional[str] = None
    recommendations: List[str]
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "question": "Is treatment effective?",
                "ai_interpretation": "Based on the data analysis...",
                "recommendations": ["Increase sample size", "Consider longitudinal study"],
                "processing_time_ms": 3450.2,
                "timestamp": "2024-01-27T15:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    components: Dict[str, bool]
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "deepseek": True,
                    "statistical_engine": True,
                    "knowledge_base": True,
                    "response_formatter": True
                },
                "timestamp": "2024-01-27T15:30:00"
            }
        }


# ===== ERROR SCHEMAS =====

class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Validation Error",
                "detail": "Invalid data format provided",
                "timestamp": "2024-01-27T15:30:00"
            }
        }
