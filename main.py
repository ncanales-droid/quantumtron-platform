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
from io import StringIO
import pandas as pd
from fastapi import HTTPException

@app.post("/api/v1/diagnostics/upload")
async def diagnostics_upload(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw or len(raw) < 5:
        raise HTTPException(status_code=400, detail="Empty file received")

    text = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            pass
    if text is None:
        raise HTTPException(status_code=400, detail="Could not decode file")

    last_err = None
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(StringIO(text), sep=sep)
            if df.shape[1] == 1 and sep == "," and ";" in df.columns[0]:
                continue
            return {
                "ok": True,
                "filename": file.filename,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": [str(c) for c in df.columns],
                "separator_detected": sep
            }
        except Exception as e:
            last_err = str(e)

    raise HTTPException(status_code=400, detail=f"CSV parse failed: {last_err}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

