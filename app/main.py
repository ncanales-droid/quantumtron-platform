"""FastAPI application entry point."""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
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

# Configure CORS - Habilitar Lovable y local development
allowed_origins = [
    "https://app.lovable.dev",
    "https://*.lovable.dev", 
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    # También mantener los orígenes existentes de settings
    *settings.CORS_ORIGINS
]

# Remover duplicados y "*" si existe
if "*" in allowed_origins:
    allowed_origins = [
        origin for origin in allowed_origins 
        if origin != "*"
    ]
allowed_origins = list(set(allowed_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


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
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# Include routers
app.include_router(health.router, prefix=settings.API_V1_PREFIX)
app.include_router(datasets.router, prefix=settings.API_V1_PREFIX)
app.include_router(diagnostics.router, prefix=settings.API_V1_PREFIX)
app.include_router(ml_unified.router, prefix=settings.API_V1_PREFIX)


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