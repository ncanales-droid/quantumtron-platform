# QuantumTron Intelligence Platform

API backend para análisis estadístico y modelos de machine learning.

## Características

- ✅ Análisis estadístico automatizado
- ✅ Diagnóstico de datasets
- ✅ Modelos de ML (XGBoost, Random Forest)
- ✅ API REST con FastAPI
- ✅ Documentación automática Swagger/OpenAPI
- ✅ Base de datos SQLite/PostgreSQL
- ✅ Logging de auditoría

## Instalación

```bash
# Clonar repositorio
git clone <url-del-repositorio>
cd quantumtron-platform/backend

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar servidor
python -m uvicorn app.main:app --reload