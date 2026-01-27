"""
QuantumTron Platform - Production Main File
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
