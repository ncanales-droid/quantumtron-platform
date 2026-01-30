# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY . .

# Variables de entorno
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://localhost:5001
ENV FASTAPI_HOST=0.0.0.0
ENV FASTAPI_PORT=8000

# Exponer puertos
EXPOSE 8000 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicio
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --serve-artifacts & sleep 5 && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
