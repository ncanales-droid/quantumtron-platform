"""Support Vector Machine (SVM) service for QuantumTron."""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# ========== MLFLOW INTEGRATION ==========
import mlflow
import mlflow.sklearn
# =========================================

from .base_service import BaseMLService

logger = logging.getLogger(__name__)


class SVMService(BaseMLService):
    """Support Vector Machine model service with MLflow tracking."""

    def __init__(self, models_dir: str = "./data/models"):
        """Initialize SVMService with MLflow."""
        super().__init__(models_dir)
        
        # Configurar MLflow
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("quantumtron_production")
            logger.info("✓ MLflow configurado para SVM (sqlite:///mlflow.db)")
        except Exception as e:
            logger.warning(f"⚠ MLflow config falló: {e}")

    async def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "classification",
        kernel: str = "rbf",  # "linear", "poly", "rbf", "sigmoid"
        test_size: float = 0.2,
        random_state: int = 42,
        normalize: bool = True,
        tune_hyperparameters: bool = False,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train SVM model with MLflow tracking.

        Args:
            df: DataFrame with features and target
            target_column: Column to predict
            task_type: "classification" or "regression"
            kernel: SVM kernel type
            test_size: Proportion for test set
            random_state: Random seed
            normalize: Whether to standardize features (CRÍTICO para SVM)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            model_name: Optional model name

        Returns:
            Training results
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting SVM {task_type} ({kernel} kernel) for target: {target_column}")

            # 1. Prepare data
            X, y, feature_names, preprocessing_info = self._prepare_data_basic(
                df, target_column, task_type
            )

            # 2. NORMALIZACIÓN CRÍTICA para SVM
            scaler = None
            if normalize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                preprocessing_info['scaled'] = True
                preprocessing_info['scaler'] = 'StandardScaler'
                logger.info("✓ Features normalized for SVM")

            # 3. Split data
            if task_type == "classification":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )

            # 4. MLflow tracking - INICIAR RUN
            mlflow_run_id = None
            try:
                mlflow.sklearn.autolog()  # Autologging para scikit-learn

                run_name = f"svm_{kernel}_{task_type}_{datetime.now().strftime('%H%M%S')}"

                with mlflow.start_run(run_name=run_name):
                    # 5. Create and train model (MLflow autolog registrará todo)
                    if tune_hyperparameters:
                        model = self._tune_hyperparameters(X_train, y_train, task_type, kernel)
                    else:
                        if task_type == "regression":
                            model = SVR(
                                kernel=kernel,
                                C=1.0,
                                epsilon=0.1,
                                random_state=random_state
                            )
                        else:  # classification
                            model = SVC(
                                kernel=kernel,
                                C=1.0,
                                probability=True,  # Para poder obtener probabilidades
                                random_state=random_state
                            )

                    model.fit(X_train, y_train)

                    # 6. Make predictions
                    y_pred_train = model.predict(X_train)
                    y_pred_test = model.predict(X_test)

                    # 7. Calculate metrics (MLflow ya los registró automáticamente)
                    if task_type == "regression":
                        train_metrics = self._calculate_regression_metrics(y_train, y_pred_train)
                        test_metrics = self._calculate_regression_metrics(y_test, y_pred_test)
                        scoring = 'r2'
                    else:
                        y_prob_test = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                        train_metrics = self._calculate_classification_metrics(y_train, y_pred_train)
                        test_metrics = self._calculate_classification_metrics(y_test, y_pred_test, y_prob_test)
                        scoring = 'accuracy'

                    # 8. Loggear parámetros adicionales manualmente
                    mlflow.log_params({
                        "kernel": kernel,
                        "task_type": task_type,
                        "test_size": test_size,
                        "random_state": random_state,
                        "normalize": normalize,
                        "tune_hyperparameters": tune_hyperparameters,
                        "target_column": target_column
                    })

                    # 9. Loggear info del dataset
                    mlflow.log_dict({
                        "dataset_info": {
                            "total_samples": len(X),
                            "train_samples": len(X_train),
                            "test_samples": len(X_test),
                            "features": feature_names,
                            "task_type": task_type,
                            "kernel": kernel
                        }
                    }, "dataset_info.json")

                    # 10. Cross-validation scores
                    cv_scores = cross_val_score(
                        model, X, y, cv=5, scoring=scoring, n_jobs=-1
                    )

                    # 11. Preparar metadata
                    model_id = model_name or f"svm_{kernel}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    metadata = {
                        'model_type': f'svm_{kernel}_{task_type}',
                        'target_column': target_column,
                        'feature_names': feature_names,
                        'task_type': task_type,
                        'kernel': kernel,
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
                        }
                    }

                    # 12. Guardar modelo (MLflow ya lo hizo automáticamente)
                    model_path = self._save_model(model, model_id, metadata)

                    # 13. Guardar scaler si se usó
                    if scaler:
                        scaler_path = model_path.replace('.joblib', '_scaler.joblib')
                        import joblib
                        joblib.dump(scaler, scaler_path)
                        metadata['scaler_path'] = scaler_path

                    # 14. Obtener run ID de MLflow
                    mlflow_run_id = mlflow.active_run().info.run_id

                    logger.info(f"📤 MLflow: Run {mlflow_run_id} registrado exitosamente")

            except Exception as mlflow_error:
                logger.warning(f"⚠ MLflow logging falló (continuando sin MLflow): {mlflow_error}")

                # Fallback: entrenar sin MLflow
                if tune_hyperparameters:
                    model = self._tune_hyperparameters(X_train, y_train, task_type, kernel)
                else:
                    if task_type == "regression":
                        model = SVR(
                            kernel=kernel,
                            C=1.0,
                            epsilon=0.1,
                            random_state=random_state
                        )
                    else:  # classification
                        model = SVC(
                            kernel=kernel,
                            C=1.0,
                            probability=True,
                            random_state=random_state
                        )

                model.fit(X_train, y_train)

                # Calcular métricas manualmente
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                if task_type == "regression":
                    train_metrics = self._calculate_regression_metrics(y_train, y_pred_train)
                    test_metrics = self._calculate_regression_metrics(y_test, y_pred_test)
                    scoring = 'r2'
                else:
                    y_prob_test = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                    train_metrics = self._calculate_classification_metrics(y_train, y_pred_train)
                    test_metrics = self._calculate_classification_metrics(y_test, y_pred_test, y_prob_test)
                    scoring = 'accuracy'

                cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring, n_jobs=-1)
                model_id = model_name or f"svm_{kernel}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                metadata = {
                    'model_type': f'svm_{kernel}_{task_type}',
                    'target_column': target_column,
                    'feature_names': feature_names,
                    'task_type': task_type,
                    'kernel': kernel,
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
                    }
                }

                model_path = self._save_model(model, model_id, metadata)

                # Guardar scaler si se usó
                if scaler:
                    scaler_path = model_path.replace('.joblib', '_scaler.joblib')
                    import joblib
                    joblib.dump(scaler, scaler_path)
                    metadata['scaler_path'] = scaler_path

            # 15. Preparar respuesta
            training_time = (datetime.now() - start_time).total_seconds()

            result = {
                'status': 'success',
                'model_id': model_id,
                'model_path': model_path,
                'model_type': f'svm_{kernel}_{task_type}',
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
                'model_parameters': model.get_params() if 'model' in locals() else {},
                'training_date': datetime.now().isoformat()
            }

            # Añadir info de MLflow si está disponible
            if mlflow_run_id:
                result['mlflow'] = {
                    'run_id': mlflow_run_id,
                    'ui_url': 'http://localhost:5001'
                }

            logger.info(f"SVM {task_type} ({kernel}) training completed in {training_time:.2f} seconds")

            if task_type == "regression":
                logger.info(f"Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
            else:
                logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error training SVM {task_type} model: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'model_type': f'svm_{task_type}'
            }

    async def predict(
        self,
        model_path: str,
        input_data: Union[pd.DataFrame, List[Dict], np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained SVM model.
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

            # Aplicar scaler si existe (CRÍTICO para SVM)
            preprocessing_info = metadata.get('preprocessing_info', {})
            if preprocessing_info.get('scaled', False) and 'scaler_path' in metadata:
                try:
                    import joblib
                    scaler = joblib.load(metadata['scaler_path'])
                    X = scaler.transform(df.values.astype(np.float32))
                except:
                    X = df.values.astype(np.float32)
            else:
                X = df.values.astype(np.float32)

            # Make predictions
            predictions = model.predict(X)

            # Decode if classification with label encoder
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
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
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

        if y_prob is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            except:
                pass

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

    def _tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        task_type: str,
        kernel: str
    ):
        """Perform hyperparameter tuning for SVM."""
        logger.info(f"Starting hyperparameter tuning for SVM {task_type} ({kernel} kernel)")

        if task_type == "regression":
            base_model = SVR(kernel=kernel)
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'epsilon': [0.01, 0.1, 0.5],
                'gamma': ['scale', 'auto']
            }
            scoring = 'neg_mean_squared_error'
        else:  # classification
            base_model = SVC(kernel=kernel, probability=True, random_state=42)
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto'],
                'degree': [2, 3] if kernel == 'poly' else [3]  # Solo para poly kernel
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
