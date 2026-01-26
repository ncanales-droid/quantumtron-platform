"""XGBoost API endpoints (updated for new structure)."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io

from app.api.dependencies import get_database_session, get_optional_user
from app.models.pydantic_models import (
    XGBoostTrainingRequest,
    XGBoostPredictionRequest,
    XGBoostModelResponse
)
from app.services.ml_models.model_registry import model_registry
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/xgboost", tags=["xgboost"])


@router.post("/train/regression", response_model=XGBoostModelResponse)
async def train_xgboost_regression(
    request: XGBoostTrainingRequest,
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_database_session),
    current_user: dict = Depends(get_optional_user),
) -> XGBoostModelResponse:
    """
    Train an XGBoost regression model.
    
    Either provide a dataset_id or upload a CSV file.
    """
    try:
        df = None
        
        # Load data from uploaded file
        if file:
            if file.content_type not in ['text/csv', 'application/vnd.ms-excel']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only CSV files are supported"
                )
            
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data provided. Please upload a file"
            )
        
        # Check if target column exists
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Target column '{request.target_column}' not found in data"
            )
        
        # Create XGBoost service using registry
        service = model_registry.create_service('xgboost')
        
        # Train model
        result = await service.train(
            df=df,
            target_column=request.target_column,
            task_type="regression",
            test_size=request.test_size,
            random_state=request.random_state,
            tune_hyperparameters=request.tune_hyperparameters,
            model_name=request.model_name
        )
        
        if result['status'] == 'error':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result['error']
            )
        
        return XGBoostModelResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training XGBoost model: {str(e)}"
        )


@router.post("/train/classification", response_model=XGBoostModelResponse)
async def train_xgboost_classification(
    request: XGBoostTrainingRequest,
    file: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_database_session),
    current_user: dict = Depends(get_optional_user),
) -> XGBoostModelResponse:
    """
    Train an XGBoost classification model.
    """
    try:
        df = None
        
        if file:
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No data provided"
            )
        
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Target column '{request.target_column}' not found"
            )
        
        # Create XGBoost service using registry
        service = model_registry.create_service('xgboost')
        
        result = await service.train(
            df=df,
            target_column=request.target_column,
            task_type="classification",
            test_size=request.test_size,
            random_state=request.random_state,
            tune_hyperparameters=request.tune_hyperparameters,
            model_name=request.model_name
        )
        
        if result['status'] == 'error':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result['error']
            )
        
        return XGBoostModelResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training XGBoost model: {str(e)}"
        )


@router.post("/predict/{model_id}")
async def make_xgboost_prediction(
    model_id: str,
    request: XGBoostPredictionRequest,
) -> JSONResponse:
    """
    Make predictions using a trained XGBoost model.
    """
    try:
        model_path = f"./data/models/{model_id}.joblib"
        
        # Create XGBoost service using registry
        service = model_registry.create_service('xgboost')
        
        result = await service.predict(
            model_path=model_path,
            input_data=request.input_data,
            return_probabilities=request.return_probabilities
        )
        
        if result['status'] == 'error':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error making predictions: {str(e)}"
        )


@router.get("/models")
async def list_xgboost_models() -> List[Dict[str, Any]]:
    """
    List all trained XGBoost models.
    """
    try:
        service = model_registry.create_service('xgboost')
        
        # Check if service has list_models method
        if hasattr(service, 'list_models'):
            models = await service.list_models()
        else:
            # Fallback: list files in models directory
            import os
            models = []
            models_dir = "./data/models"
            for filename in os.listdir(models_dir):
                if filename.endswith('.joblib') and 'xgboost' in filename:
                    models.append({
                        'filename': filename,
                        'path': os.path.join(models_dir, filename)
                    })
        
        return models
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )


@router.get("/models/{model_id}")
async def get_xgboost_model_info(
    model_id: str
) -> Dict[str, Any]:
    """
    Get information about a specific XGBoost model.
    """
    try:
        model_path = f"./data/models/{model_id}.joblib"
        service = model_registry.create_service('xgboost')
        
        # Check if service has get_model_info method
        if hasattr(service, 'get_model_info'):
            info = await service.get_model_info(model_path)
        else:
            # Fallback: basic file info
            import os
            from datetime import datetime
            if os.path.exists(model_path):
                info = {
                    'status': 'success',
                    'model_path': model_path,
                    'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat()
                }
            else:
                info = {'status': 'error', 'error': 'Model file not found'}
        
        if info.get('status') == 'error':
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=info['error']
            )
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}"
        )


@router.delete("/models/{model_id}")
async def delete_xgboost_model(
    model_id: str
) -> Dict[str, Any]:
    """
    Delete a trained XGBoost model.
    """
    try:
        model_path = f"./data/models/{model_id}.joblib"
        service = model_registry.create_service('xgboost')
        
        # Check if service has delete_model method
        if hasattr(service, 'delete_model'):
            result = await service.delete_model(model_path)
        else:
            # Fallback: delete file directly
            import os
            if os.path.exists(model_path):
                os.remove(model_path)
                result = {'status': 'success', 'message': f'Model deleted: {model_path}'}
            else:
                result = {'status': 'error', 'error': 'Model file not found'}
        
        if result.get('status') == 'error':
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result['error']
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting model: {str(e)}"
        )


@router.post("/quick-test")
async def quick_xgboost_test(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_optional_user),
) -> Dict[str, Any]:
    """
    Quick test endpoint to verify XGBoost is working.
    
    Upload a CSV, automatically detects if regression or classification.
    """
    try:
        # Read file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Auto-detect target (last column)
        target_column = df.columns[-1]
        
        # Determine task type
        if df[target_column].dtype in ['int64', 'float64']:
            unique_values = df[target_column].nunique()
            if unique_values <= 10 and unique_values < len(df) * 0.1:
                task = "classification"
            else:
                task = "regression"
        else:
            task = "classification"
        
        # Create XGBoost service using registry
        service = model_registry.create_service('xgboost')
        
        # Train model
        result = await service.train(
            df=df,
            target_column=target_column,
            task_type=task,
            test_size=0.2,
            random_state=42,
            tune_hyperparameters=False,
            model_name="quick_test"
        )
        
        return {
            "status": "success",
            "task_type": task,
            "target_column": target_column,
            "data_shape": df.shape,
            "model_result": result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quick test failed: {str(e)}"
        )