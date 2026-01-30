"""Linear and Logistic Regression service for QuantumTron."""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
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


class LinearRegressionService(BaseMLService):
    """Linear/Logistic Regression model service with MLflow tracking."""

    def __init__(self, models_dir: str = "./data/models"):
        """Initialize LinearRegressionService with MLflow."""
        super().__init__(models_dir)
        
        # Configurar MLflow
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("quantumtron_production")
            logger.info("✓ MLflow configurado para Linear Regression (sqlite:///mlflow.db)")
        except Exception as e:
            logger.warning(f"⚠ MLflow config falló: {e}")

    async def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        task_type: str = "regression",
        model_type: str = "linear",  # "linear", "ridge", "lasso", "elasticnet"
        test_size: float = 0.2,
        random_state: int = 42,
        normalize: bool = True,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train Linear/Logistic Regression model with MLflow tracking.

        Args:
            df: DataFrame with features and target
            target_column: Column to predict
            task_type: "regression" or "classification"
            model_type: "linear", "ridge", "lasso", "elasticnet"
            test_size: Proportion for test set
            random_state: Random seed
            normalize: Whether to standardize features
            model_name: Optional model name

        Returns:
            Training results
        """
        start_time = datetime.now()

        try:
            logger.info(f"Starting {model_type} {task_type} training for target: {target_column}")

            # 1. Prepare data
            X, y, feature_names, preprocessing_info = self._prepare_data_basic(
                df, target_column, task_type
            )

            # 2. Normalize features if requested
            scaler = None
            if normalize:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                preprocessing_info['scaled'] = True
                preprocessing_info['scaler'] = 'StandardScaler'

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

                run_name = f"lr_{model_type}_{task_type}_{datetime.now().strftime('%H%M%S')}"

                with mlflow.start_run(run_name=run_name):
                    # 5. Create and train model (MLflow autolog registrará todo)
                    if task_type == "regression":
                        if model_type == "ridge":
                            model = Ridge(alpha=1.0, random_state=random_state)
                        elif model_type == "lasso":
                            model = Lasso(alpha=1.0, random_state=random_state)
                        elif model_type == "elasticnet":
                            model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state)
                        else:  # linear
                            model = LinearRegression()
                    else:  # classification
                        model = LogisticRegression(
                            random_state=random_state,
                            max_iter=1000,
                            solver='lbfgs'
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
                        "model_type": model_type,
                        "task_type": task_type,
                        "test_size": test_size,
                        "random_state": random_state,
                        "normalize": normalize,
                        "target_column": target_column
                    })

                    # 9. Loggear coeficientes si es regresión lineal
                    if hasattr(model, 'coef_') and task_type == "regression":
                        coefficients = dict(zip(feature_names, model.coef_.flatten().tolist()))
                        mlflow.log_dict(coefficients, "coefficients.json")
                        
                        # Loggear top coeficientes
                        top_coeffs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                        for feat, coef in top_coeffs:
                            mlflow.log_metric(f"coef_{feat}", coef)

                    # 10. Loggear info del dataset
                    mlflow.log_dict({
                        "dataset_info": {
                            "total_samples": len(X),
                            "train_samples": len(X_train),
                            "test_samples": len(X_test),
                            "features": feature_names,
                            "task_type": task_type,
                            "model_type": model_type
                        }
                    }, "dataset_info.json")

                    # 11. Cross-validation scores
                    cv_scores = cross_val_score(
                        model, X, y, cv=5, scoring=scoring, n_jobs=-1
                    )

                    # 12. Preparar metadata
                    model_id = model_name or f"{model_type}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    metadata = {
                        'model_type': f'{model_type}_{task_type}',
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
                        }
                    }

                    # Añadir coeficientes a metadata si existen
                    if hasattr(model, 'coef_'):
                        if task_type == "regression" and len(model.coef_.shape) == 1:
                            metadata['coefficients'] = dict(zip(feature_names, model.coef_.tolist()))
                        elif task_type == "classification":
                            metadata['coefficients'] = model.coef_.tolist()
                    
                    if hasattr(model, 'intercept_'):
                        metadata['intercept'] = float(model.intercept_)

                    # 13. Guardar modelo (MLflow ya lo hizo automáticamente)
                    model_path = self._save_model(model, model_id, metadata)

                    # 14. Guardar scaler si se usó
                    if scaler:
                        scaler_path = model_path.replace('.joblib', '_scaler.joblib')
                        import joblib
                        joblib.dump(scaler, scaler_path)
                        metadata['scaler_path'] = scaler_path

                    # 15. Obtener run ID de MLflow
                    mlflow_run_id = mlflow.active_run().info.run_id

                    logger.info(f"📤 MLflow: Run {mlflow_run_id} registrado exitosamente")

            except Exception as mlflow_error:
                logger.warning(f"⚠ MLflow logging falló (continuando sin MLflow): {mlflow_error}")

                # Fallback: entrenar sin MLflow
                if task_type == "regression":
                    if model_type == "ridge":
                        model = Ridge(alpha=1.0, random_state=random_state)
                    elif model_type == "lasso":
                        model = Lasso(alpha=1.0, random_state=random_state)
                    elif model_type == "elasticnet":
                        model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=random_state)
                    else:  # linear
                        model = LinearRegression()
                else:  # classification
                    model = LogisticRegression(
                        random_state=random_state,
                        max_iter=1000,
                        solver='lbfgs'
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
                model_id = model_name or f"{model_type}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                metadata = {
                    'model_type': f'{model_type}_{task_type}',
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
                    }
                }

                # Añadir coeficientes
                if hasattr(model, 'coef_'):
                    if task_type == "regression" and len(model.coef_.shape) == 1:
                        metadata['coefficients'] = dict(zip(feature_names, model.coef_.tolist()))
                    elif task_type == "classification":
                        metadata['coefficients'] = model.coef_.tolist()
                
                if hasattr(model, 'intercept_'):
                    metadata['intercept'] = float(model.intercept_)

                model_path = self._save_model(model, model_id, metadata)

                # Guardar scaler si se usó
                if scaler:
                    scaler_path = model_path.replace('.joblib', '_scaler.joblib')
                    import joblib
                    joblib.dump(scaler, scaler_path)
                    metadata['scaler_path'] = scaler_path

            # 16. Preparar respuesta
            training_time = (datetime.now() - start_time).total_seconds()

            result = {
                'status': 'success',
                'model_id': model_id,
                'model_path': model_path,
                'model_type': f'{model_type}_{task_type}',
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

            # Añadir coeficientes si existen
            if 'coefficients' in metadata:
                result['coefficients'] = metadata['coefficients']
            if 'intercept' in metadata:
                result['intercept'] = metadata['intercept']

            # Añadir info de MLflow si está disponible
            if mlflow_run_id:
                result['mlflow'] = {
                    'run_id': mlflow_run_id,
                    'ui_url': 'http://localhost:5001'
                }

            logger.info(f"{model_type} {task_type} training completed in {training_time:.2f} seconds")

            if task_type == "regression":
                logger.info(f"Test R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
                if 'coefficients' in result:
                    logger.info(f"Model intercept: {result.get('intercept', 'N/A')}")
            else:
                logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

            return result

        except Exception as e:
            logger.error(f"Error training {model_type} {task_type} model: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'model_type': f'{model_type}_{task_type}'
            }

    async def predict(
        self,
        model_path: str,
        input_data: Union[pd.DataFrame, List[Dict], np.ndarray],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make predictions using a trained Linear/Logistic Regression model.
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

            # Aplicar scaler si existe
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

            # Add feature importance for regression
            if (metadata.get('task_type') == "regression" and
                hasattr(model, 'coef_') and
                kwargs.get('return_coefficients', False)):
                result['coefficients'] = metadata.get('coefficients', {})
                result['intercept'] = metadata.get('intercept', 0)

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
