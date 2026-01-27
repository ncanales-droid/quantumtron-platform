#!/usr/bin/env python3
"""
QuantumTron Platform - VERSION DEFINITIVA
"""
import os
import sys
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Función principal que Railway ejecutará"""
    try:
        # OBTENER PORT DE VARIABLES DE ENTORNO
        port_str = os.getenv("PORT", "8000")
        port = int(port_str)
        
        logger.info(f"🚀 INICIANDO QUANTUMTRON PLATFORM")
        logger.info(f"📊 PORT de entorno: {port_str}")
        logger.info(f"📊 PORT como int: {port}")
        logger.info(f"📁 Directorio actual: {os.getcwd()}")
        logger.info(f"🐍 Python version: {sys.version}")
        
        # Importar FastAPI
        from fastapi import FastAPI
        import uvicorn
        
        # Crear app
        app = FastAPI(
            title="QuantumTron Platform",
            version="2.0.0"
        )
        
        # Health endpoint (OBLIGATORIO para Railway)
        @app.get("/")
        async def root():
            return {"status": "ok", "app": "quantumtron", "port": port}
        
        @app.get("/health")
        async def health():
            return {
                "status": "healthy", 
                "port": port,
                "variables": {
                    "PORT_set": "PORT" in os.environ,
                    "DEEPSEEK_set": "DEEPSEEK_API_KEY" in os.environ
                }
            }
        
        # Intentar cargar Florence (opcional)
        try:
            logger.info("🔍 Intentando cargar Florence...")
            from app.api.endpoints import florence
            app.include_router(florence.router, prefix="/florence")
            logger.info("✅ Florence cargado")
            
            @app.get("/florence-status")
            async def florence_status():
                return {"florence": "loaded", "status": "ready"}
                
        except ImportError as e:
            logger.warning(f"⚠️  Florence no cargado: {e}")
            
            @app.get("/florence-status")
            async def florence_status():
                return {"florence": "not_loaded", "status": "degraded"}
        
        # INICIAR SERVER
        logger.info(f"🎯 Iniciando uvicorn en 0.0.0.0:{port}")
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ ERROR FATAL: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
