#!/usr/bin/env python3
"""
QuantumTron Platform - DEBUG RUTAS
"""
import os
import sys

print("=" * 60)
print("🚀 DEBUG - INICIANDO")
print("=" * 60)

# Información de rutas
print(f"📁 Directorio actual: {os.getcwd()}")
print(f"📄 Archivos en directorio actual:")
for f in os.listdir('.'):
    print(f"   - {f}")

print(f"🐍 Python path: {sys.path}")
print(f"📦 Python version: {sys.version}")

# Verificar si estamos en /app
if os.getcwd() == '/app':
    print("✅ Estamos en /app")
else:
    print(f"📍 Estamos en: {os.getcwd()}")
    print("💡 Intentando cambiar a /app...")
    try:
        os.chdir('/app')
        print(f"✅ Cambiado a: {os.getcwd()}")
    except:
        print("⚠️  No se pudo cambiar a /app")

try:
    from fastapi import FastAPI
    import uvicorn
    
    app = FastAPI()
    
    @app.get("/")
    def root():
        return {
            "status": "ok", 
            "app": "quantumtron",
            "cwd": os.getcwd(),
            "port": os.getenv("PORT", "8000")
        }
    
    @app.get("/health")
    def health():
        return {"status": "healthy", "debug": True}
    
    if __name__ == "__main__":
        port = int(os.getenv("PORT", "8000"))
        print(f"🎯 Iniciando uvicorn en 0.0.0.0:{port}")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
