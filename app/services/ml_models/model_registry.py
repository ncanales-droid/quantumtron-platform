"""Central registry for all ML models in QuantumTron."""

from typing import Dict, Any, Type, Optional, List
import logging

from .base_service import BaseMLService

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing all available ML models."""
    
    # Registry of available models: model_name -> service_class
    _registry: Dict[str, Type[BaseMLService]] = {}
    
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
        
        logger.info(f"Registered model: {model_name}")
    
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
    def get_available_models(cls, task_type: Optional[str] = None) -> List[str]:
        """
        Get list of available models, optionally filtered by task type.
        
        Args:
            task_type: Filter by task type (regression, classification, clustering)
            
        Returns:
            List of model names
        """
        if task_type:
            return [
                model_name for model_name, info in cls._model_info.items()
                if task_type in info.get('tasks', [])
            ]
        return list(cls._registry.keys())
    
    @classmethod
    def get_model_info(cls, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model information
        """
        info = cls._model_info.get(model_name, {}).copy()
        info['registered'] = model_name in cls._registry
        
        # Return class name instead of class object for serialization
        service_class = cls._registry.get(model_name)
        if service_class:
            info['service_class_name'] = service_class.__name__
        else:
            info['service_class_name'] = None
            
        return info
    
    @classmethod
    def get_all_models_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered models.
        
        Returns:
            Dictionary mapping model names to their information
        """
        return {
            model_name: cls.get_model_info(model_name)
            for model_name in cls._registry.keys()
        }
    
    @classmethod
    def get_recommended_model(cls, task_type: str, data_info: Dict[str, Any]) -> str:
        """
        Recommend a model based on task and data characteristics.
        
        Args:
            task_type: Type of task (regression, classification, clustering)
            data_info: Information about the data (size, features, etc.)
            
        Returns:
            Recommended model name
        """
        recommendations = {
            'regression': {
                'small_dataset': 'random_forest',
                'large_dataset': 'xgboost',
                'many_features': 'xgboost',
                'interpretability': 'random_forest',
                'default': 'xgboost'
            },
            'classification': {
                'small_dataset': 'random_forest',
                'large_dataset': 'xgboost',
                'imbalanced': 'xgboost',
                'binary': 'xgboost',
                'multiclass': 'random_forest',
                'default': 'xgboost'
            },
            'clustering': {
                'default': 'kmeans'
            }
        }
        
        task_recs = recommendations.get(task_type, {})
        
        # Simple recommendation logic
        n_samples = data_info.get('n_samples', 0)
        n_features = data_info.get('n_features', 0)
        
        if task_type == 'regression':
            if n_samples < 1000 and n_features < 10:
                return task_recs.get('small_dataset', 'random_forest')
            elif n_features > 50:
                return task_recs.get('many_features', 'xgboost')
            elif data_info.get('interpretability', False):
                return task_recs.get('interpretability', 'random_forest')
        
        elif task_type == 'classification':
            unique_classes = data_info.get('unique_classes', 2)
            if unique_classes > 10:
                return task_recs.get('multiclass', 'random_forest')
            if data_info.get('imbalanced', False):
                return task_recs.get('imbalanced', 'xgboost')
        
        return task_recs.get('default', 'xgboost')
    
    @classmethod
    def get_default_parameters(cls, model_name: str, task_type: str) -> Dict[str, Any]:
        """
        Get default parameters for a model and task.
        
        Args:
            model_name: Name of the model
            task_type: Type of task
            
        Returns:
            Dictionary of default parameters
        """
        info = cls.get_model_info(model_name)
        params = info.get('parameters', {}).copy()
        
        # Add task-specific defaults
        if model_name == 'xgboost':
            if task_type == 'regression':
                params.update({
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse'
                })
            elif task_type == 'classification':
                params.update({
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss'
                })
        
        elif model_name == 'random_forest':
            if task_type == 'regression':
                params.update({
                    'criterion': 'squared_error'
                })
            elif task_type == 'classification':
                params.update({
                    'criterion': 'gini'
                })
        
        return params
    
    @classmethod
    def get_model_comparison(cls, task_type: str) -> Dict[str, Any]:
        """
        Get comparison of available models for a task type.
        
        Args:
            task_type: Type of task
            
        Returns:
            Model comparison information
        """
        models = cls.get_available_models(task_type)
        
        comparison = {}
        for model_name in models:
            info = cls.get_model_info(model_name)
            comparison[model_name] = {
                'name': info.get('name', model_name),
                'description': info.get('description', ''),
                'strengths': info.get('strengths', []),
                'limitations': info.get('limitations', []),
                'best_for': [],
                'training_speed': 'medium',
                'prediction_speed': 'fast',
                'interpretability': 'medium'
            }
            
            # Customize based on model
            if model_name == 'xgboost':
                comparison[model_name]['best_for'] = ['Large datasets', 'High accuracy', 'Missing data']
                comparison[model_name]['training_speed'] = 'slow'
                comparison[model_name]['interpretability'] = 'low'
            elif model_name == 'random_forest':
                comparison[model_name]['best_for'] = ['Small datasets', 'Feature importance', 'Robustness']
                comparison[model_name]['training_speed'] = 'medium'
                comparison[model_name]['interpretability'] = 'medium'
        
        return comparison


# Initialize the registry
def initialize_registry():
    """Initialize the model registry with available models."""
    try:
        # Try to import and register XGBoost
        from .xgboost_service import XGBoostService
        ModelRegistry.register_model('xgboost', XGBoostService)
        logger.info("XGBoost registered in model registry")
        
    except ImportError as e:
        logger.warning(f"Could not register XGBoost: {str(e)}")
    
    try:
        # Try to import and register Random Forest
        from .random_forest_service import RandomForestService
        ModelRegistry.register_model('random_forest', RandomForestService)
        logger.info("Random Forest registered in model registry")
        
    except ImportError as e:
        logger.warning(f"Could not register Random Forest: {str(e)}")
    
    logger.info(f"Model registry initialized with {len(ModelRegistry._registry)} models")


# Global registry instance
model_registry = ModelRegistry()

# Auto-initialize when module is imported
initialize_registry()