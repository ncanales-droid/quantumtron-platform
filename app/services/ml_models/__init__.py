"""ML Models sub-package for QuantumTron."""

from .base_service import BaseMLService
from .model_registry import ModelRegistry, model_registry

# Importamos los servicios para que se registren automáticamente
try:
    from .xgboost_service import XGBoostService
except ImportError:
    pass

try:
    from .random_forest_service import RandomForestService
except ImportError:
    pass

# Export for easy access
__all__ = ['BaseMLService', 'ModelRegistry', 'model_registry']