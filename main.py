"""
QuantumTron Platform - Production Main File
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import sys

# Agregar ruta para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Crear app
app = FastAPI(
    title="QuantumTron Platform",
    description="Advanced Statistical AI Platform with Florence PhD Agent",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.lovable.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "*"  # Temporal para debug
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoints (SIEMPRE deben funcionar)
@app.get("/")
async def root():
    return {"status": "ok", "service": "quantumtron", "version": "2.0.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/lovable-health")
async def lovable_health():
    return {"status": "healthy", "for": "lovable", "version": "2.0.0"}

# INTENTAR CARGAR FLORENCE (pero si falla, no rompe todo)
try:
    print("🚀 Intentando cargar Florence...")
    
    # Intentar importar Florence
    from app.api.endpoints import florence
    from app.api.endpoints import health as api_health
    
    # Incluir routers si se importaron
    app.include_router(florence.router, prefix="/florence", tags=["florence"])
    app.include_router(api_health.router, prefix="/api", tags=["api"])
    
    print("✅ Florence cargado correctamente")
    
    # Endpoint adicional si Florence está cargado
    @app.get("/florence-status")
    async def florence_status():
        return {"florence": "loaded", "status": "ready"}
        
except ImportError as e:
    print(f"⚠️  Florence no se pudo cargar: {e}")
    print("💡 La app seguirá funcionando sin Florence")
    
    @app.get("/florence-status")
    async def florence_status():
        return {"florence": "not_loaded", "status": "degraded", "error": "Import failed"}
    
except Exception as e:
    print(f"❌ Error cargando Florence: {e}")
    print("💡 La app básica seguirá funcionando")

# Inicio
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting QuantumTron Platform on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
