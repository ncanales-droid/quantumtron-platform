from app.main import app  # ✅ QT IP real (routers /api/v1/*)

# ✅ CORS (por si app.main no lo trae o para asegurar Lovable)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Health endpoints para Lovable
@app.get("/health")
def health():
    return {"status": "healthy", "service": "quantumtron"}

@app.get("/lovable-health")
def lovable_health():
    return {"ok": True, "service": "quantumtron"}
