"""
Pydantic models for data validation in QuantumTron API.

These models define the structure of request/response data
and provide automatic validation, serialization, and documentation.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List, Union  # <-- AÑADE 'Union' aquí
from pydantic import BaseModel, Field, validator


# ============ REQUEST MODELS ============

class DatasetCreate(BaseModel):
    """
    Request model for creating a new dataset.
    
    Used by: POST /api/v1/datasets
    """
    name: str = Field(
        ...,
        description="Name of the dataset",
        min_length=1,
        max_length=255
    )
    description: Optional[str] = Field(
        None,
        description="Optional description of the dataset"
    )
    file_path: str = Field(
        ...,
        description="Path to the dataset file",
        max_length=500
    )
    file_type: str = Field(
        ...,
        description="Type of the file",
        examples=["csv", "json", "parquet", "excel"]
    )
    row_count: Optional[int] = Field(
        None,
        description="Number of rows in the dataset",
        ge=0
    )
    column_count: Optional[int] = Field(
        None,
        description="Number of columns in the dataset",
        ge=0
    )
    dataset_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for the dataset"
    )
    created_by: Optional[str] = Field(
        None,
        description="User who created the dataset",
        max_length=255
    )
    
    @validator('file_type')
    def validate_file_type(cls, v):
        valid_types = ["csv", "json", "parquet", "excel"]
        if v.lower() not in valid_types:
            raise ValueError(f"file_type must be one of: {valid_types}")
        return v.lower()


class DatasetUpdate(BaseModel):
    """
    Request model for updating an existing dataset.
    """
    name: Optional[str] = Field(
        None,
        description="Name of the dataset",
        min_length=1,
        max_length=255
    )
    description: Optional[str] = Field(
        None,
        description="Optional description of the dataset"
    )
    file_path: Optional[str] = Field(
        None,
        description="Path to the dataset file",
        max_length=500
    )
    file_type: Optional[str] = Field(
        None,
        description="Type of the file",
        examples=["csv", "json", "parquet", "excel"]
    )
    row_count: Optional[int] = Field(
        None,
        description="Number of rows in the dataset",
        ge=0
    )
    column_count: Optional[int] = Field(
        None,
        description="Number of columns in the dataset",
        ge=0
    )
    dataset_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for the dataset"
    )
    is_active: Optional[bool] = Field(
        None,
        description="Whether the dataset is active"
    )
    
    @validator('file_type')
    def validate_file_type(cls, v):
        if v is None:
            return v
        valid_types = ["csv", "json", "parquet", "excel"]
        if v.lower() not in valid_types:
            raise ValueError(f"file_type must be one of: {valid_types}")
        return v.lower()


class DiagnosticRequest(BaseModel):
    """
    Request model for diagnostic analysis on existing datasets.
    
    Used by: POST /api/v1/diagnostics
    """
    dataset_id: int = Field(
        ...,
        description="ID of the dataset to analyze",
        gt=0
    )
    analysis_type: str = Field(
        ...,
        description="Type of analysis to perform",
        examples=["descriptive", "correlation", "outlier_detection", "normality"]
    )
    parameters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional parameters for the analysis"
    )
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        valid_types = ["descriptive", "correlation", "outlier_detection", "normality"]
        if v not in valid_types:
            raise ValueError(f"analysis_type must be one of: {valid_types}")
        return v


class UploadDiagnosticRequest(BaseModel):
    """
    Request model for direct file upload diagnostics.
    
    Used by: POST /api/v1/diagnostics/upload
    """
    target_column: Optional[str] = Field(
        None,
        description="Optional target column for regression analysis"
    )
    analysis_name: Optional[str] = Field(
        None,
        description="Optional custom name for this analysis"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "target_column": "sales",
                "analysis_name": "Q1 Sales Analysis"
            }
        }


# ============ RESPONSE MODELS ============

class DatasetResponse(BaseModel):
    """
    Response model for dataset information.
    """
    id: int = Field(..., description="Dataset ID")
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    file_path: str = Field(..., description="File path")
    file_type: str = Field(..., description="File type", examples=["csv", "parquet", "json", "excel"])
    row_count: Optional[int] = Field(None, description="Number of rows")
    column_count: Optional[int] = Field(None, description="Number of columns")
    dataset_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the dataset")
    is_active: bool = Field(..., description="Whether dataset is active")
    created_by: Optional[str] = Field(None, description="User who created the dataset")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        from_attributes = True


class DiagnosticResultResponse(BaseModel):
    """
    Response model for diagnostic analysis results.
    
    Used by: All diagnostic endpoints
    """
    id: int = Field(..., description="Diagnostic result ID")
    dataset_id: int = Field(..., description="Dataset ID")
    analysis_type: str = Field(..., description="Type of analysis performed")
    result_data: Dict[str, Any] = Field(..., description="Analysis results")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Analysis parameters")
    status: str = Field(..., description="Analysis status", examples=["completed", "failed", "pending"])
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_by: Optional[str] = Field(None, description="User who created the analysis")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    
    class Config:
        from_attributes = True  # Allows conversion from SQLAlchemy models


class HealthResponse(BaseModel):
    """
    Response model for health check endpoints.
    """
    status: str = Field(..., description="Service status", examples=["healthy", "unhealthy"])
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")


class StatisticalSummary(BaseModel):
    """
    Model for statistical summary in responses.
    """
    mean: Optional[float] = Field(None, description="Mean value")
    std: Optional[float] = Field(None, description="Standard deviation")
    min: Optional[float] = Field(None, description="Minimum value")
    max: Optional[float] = Field(None, description="Maximum value")
    count: int = Field(..., description="Number of observations")


# ============ ERROR MODELS ============

class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    detail: str = Field(..., description="Error detail message")
    error_type: Optional[str] = Field(None, description="Type of error")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Dataset not found",
                "error_type": "NotFoundError",
                "timestamp": "2024-01-26T10:30:00Z"
            }
        }


# ============ VALIDATION MODELS ============

class ColumnValidation(BaseModel):
    """
    Model for column validation rules.
    """
    column_name: str = Field(..., description="Column name")
    expected_type: str = Field(..., description="Expected data type")
    required: bool = Field(True, description="Whether column is required")
    valid_values: Optional[List[Any]] = Field(None, description="List of valid values")


class AnalysisParameters(BaseModel):
    """
    Model for analysis parameter validation.
    """
    confidence_level: float = Field(
        0.95,
        description="Confidence level for intervals",
        ge=0.0,
        le=1.0
    )
    random_seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility"
    )
    normalize: bool = Field(
        True,
        description="Whether to normalize numeric features"
    )


# ============ XGBOOST MODELS ============

class XGBoostTrainingRequest(BaseModel):
    """
    Request model for XGBoost training.
    """
    target_column: str = Field(
        ...,
        description="Target column to predict"
    )
    dataset_id: Optional[int] = Field(
        None,
        description="Dataset ID from database (optional)"
    )
    test_size: float = Field(
        0.2,
        description="Proportion of data for testing",
        ge=0.1,
        le=0.5
    )
    random_state: int = Field(
        42,
        description="Random seed for reproducibility"
    )
    tune_hyperparameters: bool = Field(
        False,
        description="Whether to perform hyperparameter tuning"
    )
    model_name: Optional[str] = Field(
        None,
        description="Optional name for the model"
    )


class XGBoostPredictionRequest(BaseModel):
    """
    Request model for XGBoost predictions.
    """
    input_data: Union[List[Dict[str, Any]], List[List[float]]] = Field(
        ...,
        description="Input data for prediction"
    )
    return_probabilities: bool = Field(
        False,
        description="Whether to return class probabilities (classification only)"
    )


class XGBoostModelResponse(BaseModel):
    """
    Response model for XGBoost training.
    """
    status: str = Field(..., description="Training status")
    model_id: str = Field(..., description="Model identifier")
    model_path: str = Field(..., description="Path to saved model")
    model_type: str = Field(..., description="Type of model (regression/classification)")
    target_column: str = Field(..., description="Target column name")
    training_time_seconds: float = Field(..., description="Training duration in seconds")
    data_info: Dict[str, Any] = Field(..., description="Data information")
    metrics: Dict[str, Any] = Field(..., description="Model performance metrics")
    feature_importance: Dict[str, Any] = Field(..., description="Feature importance scores")
    model_parameters: Dict[str, Any] = Field(..., description="Model parameters")
    training_date: str = Field(..., description="Training timestamp")


class XGBoostPredictionResponse(BaseModel):
    """
    Response model for XGBoost predictions.
    """
    status: str = Field(..., description="Prediction status")
    model_type: str = Field(..., description="Type of model used")
    predictions: List[Union[float, str, int]] = Field(..., description="Predicted values")
    probabilities: Optional[List[List[float]]] = Field(None, description="Class probabilities")