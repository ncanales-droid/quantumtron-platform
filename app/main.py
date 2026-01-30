"""FastAPI application for QuantumTron Platform."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="QuantumTron Platform API",
    description="Machine Learning and Diagnostic Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint (si existe)
try:
    from app.api.endpoints.health import router as health_router
    app.include_router(health_router)
    logger.info("✓ Health endpoints loaded")
except ImportError as e:
    logger.warning(f"⚠ Health endpoints not available: {e}")

# ML Models endpoints - IMPORTANTE: Asegurar que se registra
try:
    logger.info("Attempting to load ML Models endpoints...")
    from app.api.endpoints.ml_models.predictions import router as ml_models_router
    app.include_router(ml_models_router, prefix="/api/v1")
    logger.info("✓ ML Models endpoints loaded successfully")
    logger.info(f"  Router prefix: {ml_models_router.prefix}")
    logger.info(f"  Router tags: {ml_models_router.tags}")
except ImportError as e:
    logger.error(f"✗ Failed to import ML Models endpoints: {e}")
    logger.error("  Make sure predictions.py exists and has a router")
except Exception as e:
    logger.error(f"✗ Error loading ML Models endpoints: {e}")

# Diagnostic endpoints (si existen)
try:
    from app.api.endpoints.diagnostic import router as diagnostic_router
    app.include_router(diagnostic_router, prefix="/api/v1")
    logger.info("✓ Diagnostic endpoints loaded")
except ImportError:
    pass  # No problem if not available

# Florence endpoints (si existen)
try:
    from app.api.endpoints.florence import router as florence_router
    app.include_router(florence_router, prefix="/api/v1")
    logger.info("✓ Florence endpoints loaded")
except ImportError:
    pass  # No problem if not available

@app.get("/")
async def root():
    """Root endpoint."""
    endpoints_available = {
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "mlflow_ui": "http://localhost:5001"
    }
    
    # Check which endpoints are actually loaded
    try:
        from app.api.endpoints.ml_models.predictions import router
        endpoints_available["ml_models"] = "/api/v1/ml/models"
    except:
        endpoints_available["ml_models"] = "Not loaded"
    
    return {
        "message": "Welcome to QuantumTron Platform API",
        "version": "1.0.0",
        "endpoints": endpoints_available,
        "mlflow": {
            "ui": "http://localhost:5001",
            "model_registry": "http://localhost:5001/#/models"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    status = {
        "status": "healthy",
        "service": "QuantumTron Platform API",
        "version": "1.0.0"
    }
    
    # Check ML Models endpoints
    try:
        from app.api.endpoints.ml_models.predictions import router
        status["ml_models"] = "loaded"
    except:
        status["ml_models"] = "not_loaded"
    
    return status

@app.on_event("startup")
async def startup_event():
    """Actions to perform on startup."""
    logger.info("Starting QuantumTron Platform API")
    logger.info("MLflow UI available at: http://localhost:5001")
    logger.info("API Documentation available at: /docs")
    logger.info("Model Registry available at: http://localhost:5001/#/models")

@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on shutdown."""
    logger.info("Shutting down QuantumTron Platform API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ========== LOVABLE COMPATIBILITY ENDPOINTS ==========
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

lovable_router = APIRouter(prefix="/api/models", tags=["lovable-compatibility"])
logger = logging.getLogger(__name__)

# Modelo para predicciones de Lovable
class LovablePredictionRequest(BaseModel):
    data: Dict[str, Any]
    model_id: str = None

# MAPEO COMPLETO de modelos Lovable -> QuantumTron
LOVABLE_MODEL_MAPPING = {
    # Regresión
    "gradientboostingregressor": "gradient_boosting",
    "gradientboosting": "gradient_boosting",
    "gbm": "gradient_boosting",
    
    "randomforestregressor": "random_forest", 
    "randomforest": "random_forest",
    "rf": "random_forest",
    
    "linearregression": "linear_regression",
    "linear": "linear_regression",
    "lr": "linear_regression",
    
    "xgboostregressor": "xgboost",
    "xgboost": "xgboost",
    "xgb": "xgboost",
    
    "svmregressor": "svm",
    "supportvectormachine": "svm",
    "svr": "svm",
    
    # Clasificación
    "logisticregression": "logistic_regression",
    "logistic": "logistic_regression",
    "logit": "logistic_regression",
    
    "randomforestclassifier": "random_forest_classifier",
    "rfc": "random_forest_classifier",
    
    "gradientboostingclassifier": "gradient_boosting_classifier",
    "gbc": "gradient_boosting_classifier",
    
    "svmclassifier": "svm_classifier",
    "svc": "svm_classifier"
}

# Información detallada de cada modelo
MODEL_INFO = {
    "gradient_boosting": {
        "name": "Gradient Boosting Regressor",
        "type": "regression",
        "description": "Ensemble de árboles de decisión con boosting",
        "features": ["feature1", "feature2", "feature3", "feature4"],
        "hyperparameters": {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3
        }
    },
    "random_forest": {
        "name": "Random Forest Regressor",
        "type": "regression", 
        "description": "Ensemble de múltiples árboles de decisión",
        "features": ["feature1", "feature2", "feature3"],
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10
        }
    },
    "linear_regression": {
        "name": "Linear Regression",
        "type": "regression",
        "description": "Modelo lineal para regresión",
        "features": ["feature1", "feature2", "feature3"],
        "hyperparameters": {
            "fit_intercept": True,
            "normalize": False
        }
    },
    "svm": {
        "name": "Support Vector Machine Regressor",
        "type": "regression",
        "description": "SVM para problemas de regresión",
        "features": ["feature1", "feature2", "feature3"],
        "hyperparameters": {
            "kernel": "rbf",
            "C": 1.0
        }
    },
    "xgboost": {
        "name": "XGBoost Regressor",
        "type": "regression",
        "description": "Optimized gradient boosting library",
        "features": ["feature1", "feature2", "feature3", "feature4"],
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.3
        }
    },
    "logistic_regression": {
        "name": "Logistic Regression Classifier",
        "type": "classification",
        "description": "Modelo lineal para clasificación binaria",
        "features": ["feature1", "feature2", "feature3"],
        "hyperparameters": {
            "C": 1.0,
            "penalty": "l2"
        }
    }
}

@lovable_router.post("/{model_id}/predict")
async def lovable_predict(model_id: str, request: LovablePredictionRequest):
    """
    Endpoint compatible con Lovable para predicciones.
    Lovable envía: POST /api/models/{model_id}/predict
    """
    
    # Log para debugging
    logger.info(f"Lovable request - Model: {model_id}, Data keys: {list(request.data.keys())}")
    
    # Mapear el model_id de Lovable a tu sistema
    model_id_lower = model_id.lower().replace(" ", "").replace("-", "")
    quantumtron_model = LOVABLE_MODEL_MAPPING.get(model_id_lower, "gradient_boosting")
    
    # Obtener info del modelo
    model_info = MODEL_INFO.get(quantumtron_model, MODEL_INFO["gradient_boosting"])
    
    # Generar predicción basada en el modelo (simulación por ahora)
    import random
    if model_info["type"] == "regression":
        prediction_value = round(random.uniform(0.5, 1.0), 3)
    else:  # classification
        prediction_value = random.choice([0, 1])
    
    # Preparar respuesta compatible
    return {
        "success": True,
        "model": {
            "id": model_id,
            "name": model_info["name"],
            "type": model_info["type"],
            "mapped_to": quantumtron_model
        },
        "prediction": {
            "value": prediction_value,
            "confidence": round(random.uniform(0.85, 0.98), 3),
            "type": model_info["type"]
        },
        "input_features": {
            "received": request.data,
            "expected": model_info["features"]
        },
        "metadata": {
            "api_version": "1.0.0",
            "timestamp": "2024-01-30T00:00:00Z",
            "model_info": model_info
        }
    }

@lovable_router.get("/{model_id}/info")
async def model_info(model_id: str):
    """Información del modelo para Lovable."""
    
    model_id_lower = model_id.lower().replace(" ", "").replace("-", "")
    quantumtron_model = LOVABLE_MODEL_MAPPING.get(model_id_lower, "gradient_boosting")
    info = MODEL_INFO.get(quantumtron_model, MODEL_INFO["gradient_boosting"])
    
    return {
        "model_id": model_id,
        "quantumtron_id": quantumtron_model,
        "name": info["name"],
        "type": info["type"],
        "description": info["description"],
        "status": "active",
        "features": info["features"],
        "hyperparameters": info["hyperparameters"],
        "api_compatible": True,
        "endpoints": {
            "predict": f"/api/models/{model_id}/predict",
            "info": f"/api/models/{model_id}/info"
        }
    }

@lovable_router.get("/")
async def list_lovable_models():
    """Listar TODOS los modelos disponibles para Lovable."""
    
    models_list = []
    
    # Agregar todos los modelos del mapeo
    for lovable_id, quantumtron_id in LOVABLE_MODEL_MAPPING.items():
        if quantumtron_id in MODEL_INFO:
            info = MODEL_INFO[quantumtron_id]
            
            # Solo agregar una vez por quantumtron_id (evitar duplicados)
            if not any(m["quantumtron_id"] == quantumtron_id for m in models_list):
                models_list.append({
                    "id": lovable_id,
                    "quantumtron_id": quantumtron_id,
                    "name": info["name"],
                    "type": info["type"],
                    "description": info["description"],
                    "status": "ready",
                    "endpoint": f"/api/models/{lovable_id}/predict"
                })
    
    # Ordenar por tipo y nombre
    models_list = sorted(models_list, key=lambda x: (x["type"], x["name"]))
    
    return {
        "models": models_list,
        "count": len(models_list),
        "types": {
            "regression": len([m for m in models_list if m["type"] == "regression"]),
            "classification": len([m for m in models_list if m["type"] == "classification"])
        },
        "api_version": "1.0.0"
    }

# Incluir el router en la app
app.include_router(lovable_router)

