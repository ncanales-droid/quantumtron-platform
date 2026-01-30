"""
MLflow Dashboard Endpoint
Permite ver MLflow UI desde Lovable
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
import os
from typing import Dict, Any
import mlflow

router = APIRouter()

@router.get("/dashboard")
async def get_mlflow_dashboard():
    """
    Redirige a MLflow UI si está corriendo localmente.
    """
    try:
        # Verificar si MLflow está corriendo
        import requests
        response = requests.get("http://localhost:5001", timeout=2)
        
        if response.status_code == 200:
            # MLflow está corriendo, redirigir
            return RedirectResponse(url="http://localhost:5001")
        else:
            # MLflow no responde
            return HTMLResponse(content=f"""
            <html>
                <head><title>MLflow Dashboard</title></head>
                <body style="font-family: Arial, sans-serif; padding: 20px;">
                    <h2>🚀 MLflow Dashboard</h2>
                    <p>MLflow UI no está corriendo en este momento.</p>
                    
                    <h3>📋 Para iniciar MLflow:</h3>
                    <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                        <code>python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001</code>
                    </div>
                    
                    <h3>📊 Tu configuración actual:</h3>
                    <ul>
                        <li>Tracking URI: {mlflow.get_tracking_uri()}</li>
                        <li>Base de datos: mlflow.db</li>
                    </ul>
                </body>
            </html>
            """)
            
    except Exception:
        # Error de conexión
        return HTMLResponse(content=f"""
        <html>
            <head><title>MLflow Dashboard</title></head>
            <body style="font-family: Arial, sans-serif; padding: 20px;">
                <h2>🚀 MLflow Dashboard</h2>
                <p>MLflow UI no está corriendo en localhost:5001</p>
                
                <h3>📋 Para iniciar MLflow en Windows:</h3>
                <div style="background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0;">
                    <p><strong>1. En una nueva terminal PowerShell:</strong></p>
                    <code>cd C:\\Users\\hello\\Desktop\\quantumtron-platform\\backend</code><br><br>
                    <code>python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001</code>
                </div>
            </body>
        </html>
        """)

@router.get("/status")
async def get_mlflow_status():
    """Devuelve estado de MLflow"""
    try:
        tracking_uri = mlflow.get_tracking_uri()
        
        # Intentar conectar a MLflow
        import requests
        ui_status = "stopped"
        try:
            response = requests.get("http://localhost:5001", timeout=2)
            ui_status = "running" if response.status_code == 200 else "stopped"
        except:
            ui_status = "stopped"
        
        return {
            "status": "success",
            "mlflow": {
                "tracking_uri": tracking_uri,
                "ui_status": ui_status,
                "ui_url": "http://localhost:5001" if ui_status == "running" else None
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@router.get("/latest-runs")
async def get_latest_runs(limit: int = 5):
    """Obtiene los últimos runs de MLflow"""
    try:
        # Configurar MLflow primero
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        # Buscar todos los experimentos
        experiments = client.search_experiments()
        
        all_runs = []
        for exp in experiments[:2]:  # Solo primeros 2 experimentos
            runs = client.search_runs(
                experiment_ids=[exp.experiment_id],
                max_results=limit,
                order_by=["start_time DESC"]
            )
            
            for run in runs:
                all_runs.append({
                    "run_id": run.info.run_id,
                    "run_name": run.info.run_name,
                    "status": run.info.status,
                    "experiment": exp.name,
                    "metrics": {k: float(v) for k, v in run.data.metrics.items()} if run.data.metrics else {}
                })
        
        # Ordenar por fecha (más recientes primero)
        all_runs.sort(key=lambda x: x.get('run_name', ''), reverse=True)
        
        return {
            "status": "success",
            "total_runs": len(all_runs),
            "runs": all_runs[:limit]
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "runs": []
        }