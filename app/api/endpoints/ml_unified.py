"""Unified ML API endpoints for all models."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
import pandas as pd
import io
from datetime import datetime

from app.api.dependencies import get_database_session, get_optional_user
from app.services.ml_models.model_registry import model_registry
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/ml", tags=["machine-learning"])


@router.get("/models")
async def list_available_models(
    task_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all available ML models.
    
    Args:
        task_type: Filter by task type (regression, classification, etc.)
    """
    try:
        available_models = model_registry.get_available_models(task_type)
        
        model_info = {}
        for model_name in available_models:
            info = model_registry.get_model_info(model_name)
            model_info[model_name] = info
        
        return {
            'status': 'success',
            'available_models': model_info,
            'total_models': len(available_models),
            'task_types': ['regression', 'classification']
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing models: {str(e)}"
        )


@router.get("/models/compare")
async def compare_models(
    task_type: str
) -> Dict[str, Any]:
    """
    Compare available models for a specific task.
    
    Args:
        task_type: Type of task (regression, classification)
    """
    try:
        comparison = model_registry.get_model_comparison(task_type)
        
        return {
            'status': 'success',
            'task_type': task_type,
            'comparison': comparison,
            'recommendation': 'Use XGBoost for best accuracy, Random Forest for interpretability'
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error comparing models: {str(e)}"
        )


@router.post("/train/{model_name}")
async def train_ml_model(
    model_name: str,
    file: UploadFile = File(...),
    target_column: Optional[str] = None,
    task_type: str = "auto",
    test_size: float = 0.2,
    random_state: int = 42,
    db: AsyncSession = Depends(get_database_session),
    current_user: dict = Depends(get_optional_user),
) -> Dict[str, Any]:
    """
    Train any ML model available in the registry.
    
    Args:
        model_name: Name of the model (xgboost, random_forest, etc.)
        file: CSV file with training data
        target_column: Column to predict (for supervised learning)
        task_type: Type of task (regression, classification, clustering)
        test_size: Proportion for test set
        random_state: Random seed
    """
    try:
        # Read file
        if file.content_type not in ['text/csv', 'application/vnd.ms-excel']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only CSV files are supported"
            )
        
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )
        
        # Auto-detect task type if not specified
        if task_type == "auto" and target_column:
            if df[target_column].dtype in ['int64', 'float64']:
                unique_values = df[target_column].nunique()
                if unique_values <= 10 and unique_values < len(df) * 0.1:
                    task_type = "classification"
                else:
                    task_type = "regression"
            else:
                task_type = "classification"
        elif not target_column:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="target_column is required for supervised learning"
            )
        
        # Validate model exists
        if model_name not in model_registry.get_available_models():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found. Available: {model_registry.get_available_models()}"
            )
        
        # Validate target column exists
        if target_column not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Target column '{target_column}' not found in data"
            )
        
        # Get model service
        service = model_registry.create_service(model_name)
        
        # Prepare training parameters
        params = {
            'df': df,
            'target_column': target_column,
            'task_type': task_type,
            'test_size': test_size,
            'random_state': random_state,
            'model_name': f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Add model-specific parameters
        if model_name == 'random_forest':
            params['n_estimators'] = 100
            params['max_depth'] = None
        
        # Train model
        result = await service.train(**params)
        
        if result['status'] == 'error':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result['error']
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training {model_name}: {str(e)}"
        )


@router.post("/predict/{model_name}/{model_id}")
async def predict_with_model(
    model_name: str,
    model_id: str,
    input_data: List[Dict[str, Any]],
    return_probabilities: bool = False,
    db: AsyncSession = Depends(get_database_session),
) -> Dict[str, Any]:
    """
    Make predictions with any trained model.
    """
    try:
        # Validate model exists
        if model_name not in model_registry.get_available_models():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        # Get model service
        service = model_registry.create_service(model_name)
        
        # Construct model path
        model_path = f"./data/models/{model_id}.joblib"
        
        result = await service.predict(
            model_path=model_path,
            input_data=input_data,
            return_probabilities=return_probabilities
        )
        
        if result['status'] == 'error':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result['error']
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error predicting with {model_name}: {str(e)}"
        )


@router.get("/recommend")
async def recommend_model(
    task_type: str,
    n_samples: Optional[int] = None,
    n_features: Optional[int] = None,
    interpretability: bool = False
) -> Dict[str, Any]:
    """
    Recommend the best model for your needs.
    
    Args:
        task_type: Type of task (regression, classification)
        n_samples: Number of samples in dataset
        n_features: Number of features
        interpretability: Whether model interpretability is important
    """
    try:
        data_info = {
            'n_samples': n_samples or 1000,
            'n_features': n_features or 10,
            'interpretability': interpretability
        }
        
        recommended = model_registry.get_recommended_model(task_type, data_info)
        info = model_registry.get_model_info(recommended)
        
        return {
            'status': 'success',
            'recommended_model': recommended,
            'model_info': info,
            'reasoning': 'XGBoost for best accuracy, Random Forest for interpretability' if interpretability else 'XGBoost recommended for best overall performance'
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recommending model: {str(e)}"
        )


@router.post("/quick-test-all")
async def quick_test_all_models(
    file: UploadFile = File(...),
    target_column: Optional[str] = None,
    db: AsyncSession = Depends(get_database_session),
    current_user: dict = Depends(get_optional_user),
) -> Dict[str, Any]:
    """
    Quick test of all available models on the same dataset.
    """
    try:
        # Read file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        if not target_column:
            target_column = df.columns[-1]
        
        if target_column not in df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Target column '{target_column}' not found"
            )
        
        # Auto-detect task type
        if df[target_column].dtype in ['int64', 'float64']:
            unique_values = df[target_column].nunique()
            if unique_values <= 10 and unique_values < len(df) * 0.1:
                task_type = "classification"
            else:
                task_type = "regression"
        else:
            task_type = "classification"
        
        # Get available models for this task
        available_models = model_registry.get_available_models(task_type)
        
        results = {}
        for model_name in available_models:
            try:
                service = model_registry.create_service(model_name)
                
                result = await service.train(
                    df=df,
                    target_column=target_column,
                    task_type=task_type,
                    test_size=0.2,
                    random_state=42,
                    model_name=f"quick_test_{model_name}"
                )
                
                results[model_name] = {
                    'status': result['status'],
                    'metrics': result.get('metrics', {}).get('test', {}),
                    'training_time': result.get('training_time_seconds', 0)
                }
                
            except Exception as e:
                results[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Determine best model
        best_model = None
        best_score = -float('inf')
        
        for model_name, result in results.items():
            if result['status'] == 'success':
                metrics = result['metrics']
                if task_type == "regression":
                    score = metrics.get('r2', 0)
                else:
                    score = metrics.get('accuracy', 0)
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return {
            'status': 'success',
            'task_type': task_type,
            'target_column': target_column,
            'data_shape': df.shape,
            'results': results,
            'best_model': best_model,
            'best_score': best_score,
            'recommendation': f"Based on quick test, use {best_model} for this task"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quick test failed: {str(e)}"
        )