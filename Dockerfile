# Dockerfile corregido para Render
FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements PRIMERO (para cache de Docker)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar el resto de la aplicación
COPY . /app

# Variables de entorno
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://localhost:5001
ENV FASTAPI_HOST=0.0.0.0
ENV FASTAPI_PORT=${PORT:-8000}

# Exponer puerto (Render usa $PORT)
EXPOSE ${PORT:-8000}

# Health check simplificado
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Comando de inicio para Render
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
