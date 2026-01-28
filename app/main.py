from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers (según tu estructura)
from app.api.endpoints.health import router as health_router
from app.api.endpoints.diagnostics import router as diagnostics_router
from app.api.endpoints.ml_unified import router as ml_router
from app.api.endpoints.florence import router as florence_router

app = FastAPI(title="QuantumTron Intelligence Platform API")

# CORS for Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(health_router, prefix="/api/v1/diagnostics", tags=["diagnostics"])
app.include_router(diagnostics_router, prefix="/api/v1/diagnostics", tags=["diagnostics"])
app.include_router(ml_router, prefix="/api/v1/ml", tags=["ml"])
app.include_router(florence_router, prefix="/api/v1/florence", tags=["florence"])

# Convenience endpoints
@app.get("/health")
def health():
    return {"status": "healthy", "service": "quantumtron"}

@app.get("/lovable-health")
def lovable_health():
    return {"ok": True, "service": "quantumtron"}
