"""Random Forest service for QuantumTron."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

from .base_service import BaseMLService

logger = logging.getLogger(__name__)


class RandomForestService(BaseMLService):
    """Random Forest model service."""
    
    async def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "regression",
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Random Forest model.
        
        Args:
            df: DataFrame with features and target
            target_column: Column to predict
            task_type: "regression" or "classification"
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            test_size: Proportion for test set
            random_state: Random seed
            model_name: Optional model name
            
        Returns:
            Training results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting Random Forest {task_type} training for target: {target_column}")
            
            # 1. Prepare data
            X, y, feature_names, preprocessing_info = self._prepare_data_basic(
                df, target_column, task_type
            )
            
            # 2. Split data
            if task_type == "classification":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            
            # 3. Create and train model
            if task_type == "regression":
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:  # classification
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    n_jobs=-1
                )
            
            model.fit(X_train, y_train)
            
            # 4. Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 5. Calculate metrics
            if task_type == "regression":
                train_metrics = self._calculate_regression_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_regression_metrics(y_test, y_pred_test)
                scoring = 'r2'
            else:
                train_metrics = self._calculate_classification_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_classification_metrics(y_test, y_pred_test)
                scoring = 'accuracy'
            
            # 6. Get feature importance
            feature_importance = self._get_feature_importance(model, feature_names)
            
            # 7. Cross-validation scores
            cv_scores = cross_val_score(
                model, X, y, cv=5, scoring=scoring, n_jobs=-1
            )
            
            # 8. Prepare metadata
            model_id = model_name or f"random_forest_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            metadata = {
                'model_type': f'random_forest_{task_type}',
                'target_column': target_column,
                'feature_names': feature_names,
                'task_type': task_type,
                'parameters': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'random_state': random_state
                },
                'preprocessing_info': preprocessing_info,
                'metrics': {
                    'train': train_metrics,
                    'test': test_metrics,
                    'cross_validation': {
                        'mean_score': float(cv_scores.mean()),
                        'std_score': float(cv_scores.std()),
                        'scores': cv_scores.tolist()
                    }
                },
                'feature_importance': feature_importance
            }
            
            # 9. Save model
            model_path = self._save_model(model, model_id, metadata)
            
            # 10. Prepare response
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'model_id': model_id,
                'model_path': model_path,
                'model_type': f'random_forest_{task_type}',
                'target_column': target_column,
                'training_time_seconds': training_time,
                'data_info': {
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'num_features': len(feature_names),
                    'preprocessing': preprocessing_info
                },
                'metrics': metadata['metrics'],
                'feature_importance': feature_importance,
                'model_parameters': model.get_params(),
                'training_date': datetime.now().isoformat()
            }
            
            logger.info(f"Random Forest {task_type} training completed in {training_time:.2f} seconds")
            
            if task_type == "regression":
                logger.info(f"Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
            else:
                logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training Random Forest {task_type} model: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'model_type': f'random_forest_{task_type}'
            }
    
    async def predict(
        self,
        model_path: str,
        input_data: Union[pd.DataFrame, List[Dict], np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained Random Forest model.
        """
        try:
            # Load model
            model, metadata = self._load_model(model_path)
            
            # Convert input data
            if isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            elif isinstance(input_data, np.ndarray):
                feature_names = metadata.get('feature_names', [])
                if feature_names and len(feature_names) == input_data.shape[1]:
                    df = pd.DataFrame(input_data, columns=feature_names)
                else:
                    df = pd.DataFrame(input_data)
            else:
                df = input_data.copy() if hasattr(input_data, 'copy') else input_data
            
            # Prepare data (simple version - same as training)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            
            # Ensure correct feature order
            feature_names = metadata.get('feature_names', [])
            if feature_names:
                missing_cols = set(feature_names) - set(df.columns)
                for col in missing_cols:
                    df[col] = 0
                df = df[feature_names]
            
            # Make predictions
            X = df.values.astype(np.float32)
            predictions = model.predict(X)
            
            # Decode if classification with label encoder
            preprocessing_info = metadata.get('preprocessing_info', {})
            if metadata.get('task_type') == "classification":
                le_info = preprocessing_info.get('label_encoder')
                if le_info and 'classes' in le_info:
                    predictions = [le_info['classes'][int(p)] for p in predictions]
            
            result = {
                'status': 'success',
                'model_type': metadata.get('model_type', 'unknown'),
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
            }
            
            # Add probabilities for classification
            if (metadata.get('task_type') == "classification" and 
                hasattr(model, 'predict_proba') and 
                kwargs.get('return_probabilities', False)):
                probabilities = model.predict_proba(X)
                result['probabilities'] = probabilities.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate regression metrics."""
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)
        
        return {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'explained_variance': float(np.var(y_pred) / np.var(y_true)) if np.var(y_true) > 0 else 0,
            'max_error': float(max(abs(y_true - y_pred)))
        }
    
    def _calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate classification metrics."""
        y_true = y_true.astype(np.int32)
        y_pred = y_pred.astype(np.int32)
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics for small number of classes
        unique_classes = np.unique(y_true)
        if len(unique_classes) <= 10:
            metrics['per_class'] = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
        
        return metrics
    
    def _get_feature_importance(
        self, 
        model: Union[RandomForestRegressor, RandomForestClassifier], 
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Extract feature importance from Random Forest model."""
        importance_dict = {}
        
        try:
            importance_values = model.feature_importances_
            if len(importance_values) == len(feature_names):
                importance_dict['importance'] = dict(zip(feature_names, importance_values.tolist()))
                
                # Top 10 features
                top_features = sorted(
                    importance_dict['importance'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
                importance_dict['top_features'] = [
                    {'feature': feat, 'importance': float(imp)} 
                    for feat, imp in top_features
                ]
                
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            importance_dict['error'] = str(e)
        
        return importance_dict