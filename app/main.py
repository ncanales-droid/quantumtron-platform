"""FastAPI application entry point."""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.database import init_db, close_db
from app.api.endpoints import health, datasets, diagnostics, ml_unified
from app.core.exceptions import QuantumTronException

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper()),
    format=settings.LOG_FORMAT,
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for startup and shutdown events.

    Args:
        app: FastAPI application instance

    Yields:
        None
    """
    # Startup
    logger.info("Starting QuantumTron Intelligence Platform...")
    logger.info(f"Environment: {'DEBUG' if settings.DEBUG else 'PRODUCTION'}")
    
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down QuantumTron Intelligence Platform...")
    await close_db()
    logger.info("Database connections closed")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="QuantumTron Intelligence Platform - Statistical Diagnostic Engine",
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# Configure CORS - Solución completa
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todos los orígenes
    allow_methods=["*"],  # Permitir todos los métodos
    allow_headers=["*"],  # Permitir todos los headers
    expose_headers=["*"],  # Exponer todos los headers
    allow_credentials=False,  # Temporalmente false para debug
    max_age=600,  # Cache de preflight por 10 minutos
)


# Audit Logging Middleware
@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next):
    """
    Middleware for audit logging of all requests.

    Args:
        request: Incoming request
        call_next: Next middleware/endpoint handler

    Returns:
        Response: HTTP response
    """
    start_time = time.time()

    # Log request
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        f"Request: {request.method} {request.url.path} | "
        f"IP: {client_ip} | "
        f"User-Agent: {request.headers.get('user-agent', 'unknown')}"
    )

    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Asegurar headers CORS en cada respuesta
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        
        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Time: {process_time:.3f}s"
        )

        # Add process time header
        response.headers["X-Process-Time"] = str(process_time)

        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Error: {request.method} {request.url.path} | "
            f"Exception: {str(e)} | "
            f"Time: {process_time:.3f}s"
        )
        raise


# Exception handlers
@app.exception_handler(QuantumTronException)
async def quantumtron_exception_handler(
    request: Request, exc: QuantumTronException
) -> JSONResponse:
    """
    Handler for QuantumTron custom exceptions.

    Args:
        request: Request object
        exc: Exception instance

    Returns:
        JSONResponse: Error response
    """
    logger.error(f"QuantumTronException: {exc.detail}")
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
    # Asegurar CORS en errores también
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """
    Handler for general exceptions.

    Args:
        request: Request object
        exc: Exception instance

    Returns:
        JSONResponse: Error response
    """
    logger.exception(f"Unhandled exception: {str(exc)}")
    response = JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
    # Asegurar CORS en errores también
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


# Include routers
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(datasets.router, prefix=settings.API_V1_PREFIX)
app.include_router(diagnostics.router, prefix=settings.API_V1_PREFIX)
app.include_router(ml_unified.router, prefix=settings.API_V1_PREFIX)


@app.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    """Simple test endpoint."""
    try:
        contents = await file.read()
        return {
            "filename": file.filename,
            "size": len(contents),
            "content_type": file.content_type,
            "success": True
        }
    except Exception as e:
        logger.error(f"Test upload error: {str(e)}", exc_info=True)
        raise


@app.get("/lovable-health")
async def lovable_health():
    """Health check specifically for Lovable dashboard."""
    return {
        "status": "healthy",
        "service": "QuantumTron API",
        "version": settings.APP_VERSION,
        "timestamp": time.time()
    }


@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint.

    Returns:
        dict: API information
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "api_prefix": settings.API_V1_PREFIX,
    }