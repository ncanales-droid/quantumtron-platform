"""Central registry for all ML models in QuantumTron."""

from typing import Dict, Any, Type, Optional, List
import logging

# ========== MLFLOW INTEGRATION ==========
import mlflow
# =========================================

from .base_service import BaseMLService

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing all available ML models."""

    # Registry of available models: model_name -> service_class
    _registry: Dict[str, Type[BaseMLService]] = {}

    # MLflow tracking
    _mlflow_tracking_uri = "sqlite:///mlflow.db"
    _mlflow_experiment = "quantumtron_production"

    # Model descriptions and capabilities
    _model_info: Dict[str, Dict[str, Any]] = {
        'xgboost': {
            'name': 'XGBoost',
            'description': 'Extreme Gradient Boosting for classification and regression',
            'tasks': ['regression', 'classification'],
            'package': 'xgboost',
            'strengths': ['High accuracy', 'Handles missing values', 'Feature importance'],
            'limitations': ['Can overfit on small datasets', 'Slower training than some models'],
            'parameters': {
                'n_estimators': {'type': 'int', 'default': 100, 'description': 'Number of trees'},
                'learning_rate': {'type': 'float', 'default': 0.1, 'description': 'Step size shrinkage'},
                'max_depth': {'type': 'int', 'default': 6, 'description': 'Maximum tree depth'}
            }
        },
        'random_forest': {
            'name': 'Random Forest',
            'description': 'Ensemble of decision trees for classification and regression',
            'tasks': ['regression', 'classification'],
            'package': 'scikit-learn',
            'strengths': ['Robust to overfitting', 'Handles non-linear relationships', 'Feature importance'],
            'limitations': ['Can be slow with many trees', 'Less accurate than boosting methods'],
            'parameters': {
                'n_estimators': {'type': 'int', 'default': 100, 'description': 'Number of trees'},
                'max_depth': {'type': 'int', 'default': None, 'description': 'Maximum tree depth'},
                'min_samples_split': {'type': 'int', 'default': 2, 'description': 'Minimum samples to split'}
            }
        }
    }

    @classmethod
    def register_model(cls, model_name: str, service_class: Type[BaseMLService],
                      model_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new ML model in the registry.

        Args:
            model_name: Unique name for the model
            service_class: Class that implements the model service
            model_info: Additional information about the model
        """
        if model_name in cls._registry:
            logger.warning(f"Model '{model_name}' is already registered. Overwriting.")

        cls._registry[model_name] = service_class

        if model_info:
            cls._model_info[model_name] = model_info

        # Registrar en MLflow si está disponible
        cls._register_in_mlflow(model_name, model_info)

        logger.info(f"Registered model: {model_name}")

    @classmethod
    def _register_in_mlflow(cls, model_name: str, model_info: Optional[Dict[str, Any]]) -> None:
        """Register model metadata in MLflow."""
        try:
            mlflow.set_tracking_uri(cls._mlflow_tracking_uri)
            mlflow.set_experiment(cls._mlflow_experiment)
            logger.info(f"✓ Model '{model_name}' registrado en MLflow")
        except Exception as e:
            logger.warning(f"⚠ No se pudo registrar en MLflow: {e}")

    @classmethod
    def get_service(cls, model_name: str) -> Optional[Type[BaseMLService]]:
        """
        Get service class for a model.

        Args:
            model_name: Name of the model

        Returns:
            Service class or None if not found
        """
        return cls._registry.get(model_name.lower())

    @classmethod
    def create_service(cls, model_name: str, **kwargs) -> BaseMLService:
        """
        Create an instance of a model service.

        Args:
            model_name: Name of the model
            **kwargs: Arguments to pass to service constructor

        Returns:
            Instance of the model service

        Raises:
            ValueError: If model is not registered
        """
        service_class = cls.get_service(model_name)
        if not service_class:
            raise ValueError(f"Model '{model_name}' is not registered. "
                           f"Available models: {list(cls._registry.keys())}")

        return service_class(**kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered models."""
        return list(cls._registry.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        return cls._model_info.get(model_name.lower())

    @classmethod
    def get_all_model_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered models."""
        return cls._model_info.copy()
