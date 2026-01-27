from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok", "service": "quantumtron"}

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/lovable-health")
async def lovable_health():
    return {"status": "healthy", "for": "lovable"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
