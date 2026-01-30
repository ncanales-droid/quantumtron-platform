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
