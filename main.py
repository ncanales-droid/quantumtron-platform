"""
QuantumTron Platform - Production Main File
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

# Crear app
app = FastAPI(
    title="QuantumTron Platform",
    description="Advanced Statistical AI Platform with Florence PhD Agent",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para producción
origins = [
    "https://*.lovable.app",  # Lovable frontend
    "http://localhost:3000",  # Local development
    "http://localhost:5173",  # Vite dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Importar routers
try:
    from app.api.endpoints import florence, health
    
    # Florence endpoints
    app.include_router(florence.router, prefix="/florence", tags=["florence"])
    
    # Health endpoint
    app.include_router(health.router, prefix="/api", tags=["health"])
    
    print("✅ Routers cargados correctamente")
    
except ImportError as e:
    print(f"⚠️  Error importando routers: {e}")
    print("💡 Creando endpoints básicos...")
    
    # Fallback básico si hay error
    from fastapi import APIRouter
    
    basic_router = APIRouter()
    
    @basic_router.get("/health")
    async def health_check():
        return {"status": "degraded", "message": "Basic health check", "version": "2.0.0"}
    
    app.include_router(basic_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "QuantumTron Platform",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "endpoints": {
            "florence": "/florence/health",
            "api_health": "/api/health"
        }
    }

# Health endpoint simple
@app.get("/health")
async def health():
    return {"status": "healthy", "service": "quantumtron-platform"}

# Inicio de la aplicación
if __name__ == "__main__":
    # Obtener puerto de variable de entorno o usar 8000 por defecto
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"🚀 Iniciando QuantumTron Platform en {host}:{port}")
    uvicorn.run(app, host=host, port=port)
