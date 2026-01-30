from .base_service import BaseMLService
from .model_registry import ModelRegistry
from .model_registry_manager import ModelRegistryManager, model_registry_manager
from .random_forest_service import RandomForestService
from .xgboost_service import XGBoostService
from .linear_regression_service import LinearRegressionService
from .svm_service import SVMService
from .gradient_boosting_service import GradientBoostingService

# Create singleton instances
model_registry = ModelRegistry()
model_registry_manager = ModelRegistryManager()

# Auto-register all available models
model_registry.register_model('random_forest', RandomForestService)
model_registry.register_model('xgboost', XGBoostService)
model_registry.register_model('linear_regression', LinearRegressionService)
model_registry.register_model('svm', SVMService)
model_registry.register_model('gradient_boosting', GradientBoostingService)

__all__ = [
    'BaseMLService',
    'ModelRegistry',
    'ModelRegistryManager',
    'model_registry',
    'model_registry_manager',
    'RandomForestService',
    'XGBoostService',
    'LinearRegressionService',
    'SVMService',
    'GradientBoostingService'
]
