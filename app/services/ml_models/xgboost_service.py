"""XGBoost service for QuantumTron (refactored for unified ML structure)."""

import pandas as pd
import numpy as np
import json
import joblib
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_absolute_percentage_error,
    roc_auc_score
)
from sklearn.impute import SimpleImputer

from .base_service import BaseMLService

logger = logging.getLogger(__name__)


class XGBoostService(BaseMLService):
    """XGBoost model service inheriting from BaseMLService."""
    
    def __init__(self, models_dir: str = "./data/models"):
        """
        Initialize XGBoost service.
        
        Args:
            models_dir: Directory to save trained models
        """
        super().__init__(models_dir)
        self.label_encoders = {}
    
    async def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "regression",
        test_size: float = 0.2,
        random_state: int = 42,
        tune_hyperparameters: bool = False,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            df: DataFrame with features and target
            target_column: Column to predict
            task_type: "regression" or "classification"
            test_size: Proportion for test set (0-1)
            random_state: Random seed for reproducibility
            tune_hyperparameters: Whether to perform hyperparameter tuning
            model_name: Optional name for the model
            
        Returns:
            Dictionary with model results, metrics, and feature importance
        """
        try:
            logger.info(f"Starting XGBoost {task_type} training for target: {target_column}")
            start_time = datetime.now()
            
            # 1. Prepare data
            X, y, feature_names, preprocessing_info = self._prepare_data(
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
            if tune_hyperparameters:
                model = self._tune_hyperparameters(X_train, y_train, task_type)
            else:
                if task_type == "regression":
                    model = XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        min_child_weight=1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0,
                        reg_lambda=1,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=0
                    )
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                else:  # classification
                    model = XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        min_child_weight=1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0,
                        reg_lambda=1,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=0,
                        use_label_encoder=False,
                        eval_metric='logloss'
                    )
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        verbose=False
                    )
            
            # 4. Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # 5. Calculate metrics
            if task_type == "regression":
                train_metrics = self._calculate_regression_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_regression_metrics(y_test, y_pred_test)
                scoring = 'r2'
            else:
                y_prob_test = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                train_metrics = self._calculate_classification_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_classification_metrics(y_test, y_pred_test, y_prob_test)
                scoring = 'accuracy'
            
            # 6. Get feature importance
            feature_importance = self._get_feature_importance(model, feature_names)
            
            # 7. Cross-validation scores
            cv_scores = cross_val_score(
                model, X, y, cv=5, scoring=scoring, n_jobs=-1
            )
            
            # 8. Prepare metadata
            model_id = model_name or f"xgboost_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            metadata = {
                'model_type': f'xgboost_{task_type}',
                'target_column': target_column,
                'feature_names': feature_names,
                'task_type': task_type,
                'parameters': model.get_params(),
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
            
            # 9. Save model using parent class method
            model_path = self._save_model(model, model_id, metadata)
            
            # 10. Prepare response
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'model_id': model_id,
                'model_path': model_path,
                'model_type': f'xgboost_{task_type}',
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
            
            logger.info(f"XGBoost {task_type} training completed in {training_time:.2f} seconds")
            
            if task_type == "regression":
                logger.info(f"Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
            else:
                logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training XGBoost {task_type} model: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'model_type': f'xgboost_{task_type}'
            }
    
    async def predict(
        self,
        model_path: str,
        input_data: Union[pd.DataFrame, List[Dict], np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained XGBoost model.
        
        Args:
            model_path: Path to saved model file (.joblib)
            input_data: Input data for prediction
            
        Returns:
            Dictionary with predictions
        """
        try:
            # Load model using parent class method
            model, metadata = self._load_model(model_path)
            
            # Convert input data to DataFrame
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
            
            # Apply preprocessing
            preprocessing_info = metadata.get('preprocessing_info', {})
            categorical_cols = preprocessing_info.get('categorical_cols', [])
            
            if categorical_cols:
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dummy_na=True)
            
            # Ensure all training columns are present
            feature_names = metadata.get('feature_names', [])
            if feature_names:
                missing_cols = set(feature_names) - set(df.columns)
                for col in missing_cols:
                    df[col] = 0
                df = df[feature_names]
            
            # Handle missing values
            imputer_strategy = preprocessing_info.get('imputer_strategy')
            if imputer_strategy:
                imputer = SimpleImputer(strategy=imputer_strategy)
                df_imputed = imputer.fit_transform(df)
                df = pd.DataFrame(df_imputed, columns=df.columns)
            
            # Convert to numpy and predict
            X = df.values.astype(np.float32)
            predictions = model.predict(X)
            
            # Decode predictions if classification with label encoder
            task_type = metadata.get('task_type', 'regression')
            if task_type == "classification":
                le_info = preprocessing_info.get('label_encoder')
                if le_info and 'classes' in le_info:
                    predictions = [le_info['classes'][int(p)] for p in predictions]
            
            result = {
                'status': 'success',
                'model_type': metadata.get('model_type', 'unknown'),
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions
            }
            
            # Add probabilities if requested and available
            return_probabilities = kwargs.get('return_probabilities', False)
            if return_probabilities and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                result['probabilities'] = probabilities.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _prepare_data(
        self, 
        df: pd.DataFrame, 
        target_column: str, 
        task_type: str = "regression"
    ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict]:
        """
        Prepare data for XGBoost training (overrides parent method).
        """
        preprocessing_info = {}
        
        # Separate features and target
        X = df.drop(columns=[target_column]).copy()
        y = df[target_column].copy()
        
        # Store original info
        preprocessing_info['original_dtypes'] = X.dtypes.astype(str).to_dict()
        preprocessing_info['target_dtype'] = str(y.dtype)
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        preprocessing_info['categorical_cols'] = categorical_cols
        
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            preprocessing_info['encoded_columns'] = X.columns.tolist()
        
        # Handle missing values in features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty and X[numeric_cols].isnull().any().any():
            imputer = SimpleImputer(strategy='median')
            X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
            preprocessing_info['imputer_strategy'] = 'median'
        
        # Handle missing values in target
        missing_target_count = y.isnull().sum()
        if missing_target_count > 0:
            logger.warning(f"Found {missing_target_count} missing values in target column")
            if task_type == "regression":
                y = y.fillna(y.median())
                preprocessing_info['target_imputed'] = True
            else:
                # For classification, drop rows with missing target
                mask = ~y.isnull()
                X = X[mask]
                y = y[mask]
                preprocessing_info['rows_dropped'] = int(sum(~mask))
        
        # Encode target for classification
        if task_type == "classification" and y.dtype in ['object', 'category']:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            preprocessing_info['label_encoder'] = {
                'classes': label_encoder.classes_.tolist()
            }
            preprocessing_info['unique_classes'] = len(label_encoder.classes_)
            preprocessing_info['class_distribution'] = dict(zip(*np.unique(y, return_counts=True)))
        
        # Convert to numpy arrays
        X_array = X.values.astype(np.float32)
        y_array = y.values.astype(np.float32 if task_type == "regression" else np.int32)
        
        return X_array, y_array, X.columns.tolist(), preprocessing_info
    
    def _calculate_regression_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate comprehensive regression metrics."""
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'max_error': float(max(abs(y_true - y_pred))),
            'explained_variance': float(np.var(y_pred) / np.var(y_true)) if np.var(y_true) > 0 else 0,
            'median_absolute_error': float(np.median(np.abs(y_true - y_pred))),
            'mean_absolute_percentage_error': float(mean_absolute_percentage_error(y_true, y_pred)),
        }
        
        residuals = y_true - y_pred
        metrics['residual_mean'] = float(np.mean(residuals))
        metrics['residual_std'] = float(np.std(residuals))
        
        return metrics
    
    def _calculate_classification_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        y_true = y_true.astype(np.int32)
        y_pred = y_pred.astype(np.int32)
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        
        unique_classes = np.unique(y_true)
        if len(unique_classes) <= 10:
            metrics['per_class'] = classification_report(
                y_true, y_pred, output_dict=True, zero_division=0
            )
        
        if y_prob is not None and len(unique_classes) == 2:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            except:
                pass
        
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def _get_feature_importance(
        self, 
        model: Union[XGBRegressor, XGBClassifier], 
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Extract feature importance from XGBoost model."""
        importance_dict = {}
        
        try:
            gain_importance = model.feature_importances_
            if len(gain_importance) == len(feature_names):
                importance_dict['gain'] = dict(zip(feature_names, gain_importance.tolist()))
            
            # Top 10 features by gain
            if 'gain' in importance_dict:
                top_features = sorted(
                    importance_dict['gain'].items(), 
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
    
    def _tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str = "regression"
    ) -> Union[XGBRegressor, XGBClassifier]:
        """Perform basic hyperparameter tuning."""
        logger.info(f"Starting hyperparameter tuning for {task_type}")
        
        if task_type == "regression":
            base_model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
            scoring = 'neg_mean_squared_error'
        else:  # classification
            base_model = XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
            scoring = 'accuracy'
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_}")
        
        return grid_search.best_estimator_