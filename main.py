from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "service": "quantumtron"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/lovable-health")
def lovable_health():
    return {"ok": True, "service": "quantumtron"}

# ✅ Endpoint esperado por Lovable
@app.post("/api/v1/diagnostics/upload")
async def diagnostics_upload(file: UploadFile = File(...)):
    content = await file.read()
    return {
        "ok": True,
        "filename": file.filename,
        "content_type": file.content_type,
        "bytes": len(content)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
