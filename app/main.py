from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints.health import router as health_router
from app.api.endpoints.diagnostics import router as diagnostics_router
from app.api.endpoints.ml_unified import router as ml_router

app = FastAPI(title="QuantumTron Intelligence Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API v1
app.include_router(health_router, prefix="/api/v1/diagnostics", tags=["diagnostics"])
app.include_router(diagnostics_router, prefix="/api/v1/diagnostics", tags=["diagnostics"])
app.include_router(ml_router, prefix="/api/v1/ml", tags=["ml"])

# Lovable health (raíz)
@app.get("/lovable-health")
def lovable_health():
    return {"ok": True, "service": "quantumtron"}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "quantumtron"}
