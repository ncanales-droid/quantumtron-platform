"""Base service for all ML models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseMLService(ABC):
    """Abstract base class for all ML services."""
    
    def __init__(self, models_dir: str = "./data/models"):
        """
        Initialize ML service.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
    
    @abstractmethod
    async def train(self, df: pd.DataFrame, target_column: str, **kwargs) -> Dict[str, Any]:
        """
        Train a model.
        
        Args:
            df: DataFrame with training data
            target_column: Column to predict
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training results
        """
        pass
    
    @abstractmethod
    async def predict(self, model_path: str, input_data: Any, **kwargs) -> Dict[str, Any]:
        """
        Make predictions with a trained model.
        
        Args:
            model_path: Path to saved model file
            input_data: Input data for prediction
            **kwargs: Additional prediction parameters
            
        Returns:
            Dictionary with predictions
        """
        pass
    
    def _save_model(self, model: Any, model_id: str, metadata: Dict[str, Any]) -> str:
        """
        Save model to disk.
        
        Args:
            model: Trained model object
            model_id: Unique identifier for the model
            metadata: Additional information about the model
            
        Returns:
            Path to saved model file
        """
        try:
            model_path = os.path.join(self.models_dir, f"{model_id}.joblib")
            
            model_data = {
                'model': model,
                'metadata': metadata,
                'save_date': datetime.now().isoformat(),
                'model_class': self.__class__.__name__
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Model saved to: {model_path}")
            
            return model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def _load_model(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model from disk.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model_data = joblib.load(model_path)
            
            model = model_data.get('model')
            metadata = model_data.get('metadata', {})
            
            if model is None:
                raise ValueError("Model data is corrupted or invalid")
            
            logger.info(f"Model loaded from: {model_path}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _prepare_data_basic(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None,
        task_type: str = "regression"
    ) -> Tuple[np.ndarray, Optional[np.ndarray], list, Dict[str, Any]]:
        """
        Basic data preparation for ML training.
        
        Args:
            df: Input DataFrame
            target_column: Column to predict (None for unsupervised)
            task_type: Type of task (regression, classification, clustering)
            
        Returns:
            Tuple of (X_features, y_target, feature_names, preprocessing_info)
        """
        preprocessing_info = {
            'task_type': task_type,
            'preparation_date': datetime.now().isoformat()
        }
        
        if target_column and target_column in df.columns:
            # Supervised learning
            X = df.drop(columns=[target_column]).copy()
            y = df[target_column].copy()
            
            preprocessing_info['target_column'] = target_column
            preprocessing_info['target_dtype'] = str(y.dtype)
            preprocessing_info['has_target'] = True
        else:
            # Unsupervised learning (clustering)
            X = df.copy()
            y = None
            preprocessing_info['has_target'] = False
        
        # Store original info
        preprocessing_info['original_columns'] = X.columns.tolist()
        preprocessing_info['original_shape'] = X.shape
        preprocessing_info['original_dtypes'] = X.dtypes.astype(str).to_dict()
        
        # Handle missing values
        missing_counts = X.isnull().sum()
        if missing_counts.sum() > 0:
            preprocessing_info['missing_counts'] = missing_counts[missing_counts > 0].to_dict()
            
            # Simple imputation for numeric columns
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
                preprocessing_info['imputer_strategy'] = 'median'
        
        # Convert to numpy arrays
        X_array = X.values.astype(np.float32)
        
        if y is not None:
            # Handle target based on task type
            if task_type == "regression":
                y_array = y.values.astype(np.float32)
                # Handle missing target values for regression
                if pd.isna(y_array).any():
                    y_array = np.nan_to_num(y_array, nan=np.nanmedian(y_array))
                    preprocessing_info['target_imputed'] = True
            elif task_type == "classification":
                from sklearn.preprocessing import LabelEncoder
                # Handle categorical target
                if y.dtype in ['object', 'category']:
                    label_encoder = LabelEncoder()
                    y_array = label_encoder.fit_transform(y)
                    preprocessing_info['label_encoder'] = {
                        'classes': label_encoder.classes_.tolist()
                    }
                else:
                    y_array = y.values.astype(np.int32)
                preprocessing_info['unique_classes'] = len(np.unique(y_array))
            else:
                y_array = y.values
        else:
            y_array = None
        
        return X_array, y_array, X.columns.tolist(), preprocessing_info
    
    async def validate_model(self, model_path: str) -> Dict[str, Any]:
        """
        Validate a trained model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Validation results
        """
        try:
            model, metadata = self._load_model(model_path)
            
            validation_result = {
                'status': 'valid',
                'model_path': model_path,
                'model_type': metadata.get('model_type', 'unknown'),
                'file_size_mb': os.path.getsize(model_path) / (1024 * 1024),
                'last_modified': datetime.fromtimestamp(os.path.getmtime(model_path)).isoformat(),
                'metadata_keys': list(metadata.keys()),
                'has_model': model is not None
            }
            
            return validation_result
            
        except Exception as e:
            return {
                'status': 'invalid',
                'error': str(e),
                'model_path': model_path
            }
    
    async def delete_model(self, model_path: str) -> Dict[str, Any]:
        """
        Delete a trained model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Deletion result
        """
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
                return {
                    'status': 'success',
                    'message': f'Model deleted: {model_path}'
                }
            else:
                return {
                    'status': 'error',
                    'error': f'Model file not found: {model_path}'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


# Create global models directory
MODELS_DIR = "./data/models"
os.makedirs(MODELS_DIR, exist_ok=True)
logger.info(f"ML models directory: {MODELS_DIR}")