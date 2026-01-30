# ========== LOVABLE TRAINING ENDPOINT (REAL) ==========
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import pandas as pd
import numpy as np
import joblib
import os
import time  # (puede quedarse, pero NO dependemos de este import)

@lovable_router.post("/train")
async def lovable_train(request: dict):
    """
    Endpoint REAL de training para Lovable.
    Entrena un modelo con datos proporcionados.

    ✅ Respuesta Lovable-friendly:
    - run_id estable (run_<timestamp>_<rand>)
    - métricas planas (accuracy/precision/recall/f1_score)
    - para regresión: rmse/r2_score/mse (y accuracy/precision/recall/f1_score en 0.0)
    - model_saved + mlflow_registered

    🔥 Hotfix: import time dentro de la función (evita NameError en Render)
    """
    logger.info("Lovable REAL training request received")

    try:
        # 🔥 HOTFIX DEFINITIVO: no dependemos del import global
        import time as _time

        # 1) Leer payload
        algorithm = request.get("algorithm", "gradientboostingregressor")
        training_data = request.get("data", [])
        target_column = request.get("target", "target")

        if not training_data:
            return {
                "success": False,
                "error": "No training data provided",
                "required_format": {
                    "algorithm": "gradientboostingregressor|randomforestregressor|linearregression|logisticregression|svm|xgboost",
                    "data": "[{feature1: value1, feature2: value2, ... target: value}]",
                    "target": "name_of_target_column",
                },
            }

        # 2) DataFrame
        df = pd.DataFrame(training_data)

        # 3) Validar target
        if target_column not in df.columns:
            return {
                "success": False,
                "error": f"Target column '{target_column}' not found in data",
                "available_columns": list(df.columns),
            }

        # 4) Limpieza básica: quitar filas con NaN en target
        df = df.dropna(subset=[target_column])
        if len(df) < 5:
            return {
                "success": False,
                "error": "Not enough valid rows after cleaning. Need at least 5 rows.",
                "rows_after_cleaning": int(len(df)),
            }

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 5) Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 6) Seleccionar modelo
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.svm import SVR
        import xgboost as xgb

        algo = str(algorithm).lower().replace(" ", "").replace("-", "")
        model = None
        model_type = "regression"

        if algo in ["gradientboostingregressor", "gbm", "gradientboosting"]:
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            model_type = "regression"
        elif algo in ["randomforestregressor", "randomforest", "rf"]:
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model_type = "regression"
        elif algo in ["linearregression", "linear", "lr"]:
            model = LinearRegression()
            model_type = "regression"
        elif algo in ["logisticregression", "logistic", "logit"]:
            model = LogisticRegression(random_state=42, max_iter=2000)
            model_type = "classification"
        elif algo in ["svm", "svr", "supportvectormachine"]:
            model = SVR()
            model_type = "regression"
        elif algo in ["xgboost", "xgb", "xgboostregressor"]:
            model = xgb.XGBRegressor(n_estimators=200, random_state=42)
            model_type = "regression"
        else:
            return {
                "success": False,
                "error": f"Algorithm '{algorithm}' not supported",
                "supported_algorithms": [
                    "gradientboostingregressor",
                    "randomforestregressor",
                    "linearregression",
                    "logisticregression",
                    "svm",
                    "xgboost",
                ],
            }

        # 7) Entrenar
        logger.info(f"Training {algorithm} with {len(X_train)} samples...")
        model.fit(X_train, y_train)

        # 8) Evaluar
        y_pred = model.predict(X_test)

        # Defaults "planos" (Lovable los renderiza directo)
        flat_accuracy = 0.0
        flat_precision = 0.0
        flat_recall = 0.0
        flat_f1 = 0.0

        metrics = {}

        if model_type == "regression":
            mse = mean_squared_error(y_test, y_pred)
            rmse = float(np.sqrt(mse))
            r2 = float(model.score(X_test, y_test))

            metrics = {"mse": float(mse), "rmse": rmse, "r2_score": r2}

        else:
            # Asegurar predicciones como clases 0/1
            try:
                y_pred_cls = np.array(y_pred, dtype=int)
            except Exception:
                y_pred_cls = np.array(
                    [1 if float(v) >= 0.5 else 0 for v in y_pred], dtype=int
                )

            try:
                y_true_cls = np.array(y_test, dtype=int)
            except Exception:
                y_true_cls = np.array([int(v) for v in y_test], dtype=int)

            flat_accuracy = float(accuracy_score(y_true_cls, y_pred_cls))

            average_mode = "binary"
            unique_vals = np.unique(y_true_cls)
            if len(unique_vals) > 2:
                average_mode = "macro"

            flat_precision = float(
                precision_score(
                    y_true_cls, y_pred_cls, average=average_mode, zero_division=0
                )
            )
            flat_recall = float(
                recall_score(
                    y_true_cls, y_pred_cls, average=average_mode, zero_division=0
                )
            )
            flat_f1 = float(
                f1_score(y_true_cls, y_pred_cls, average=average_mode, zero_division=0)
            )

            metrics = {
                "accuracy": flat_accuracy,
                "precision": flat_precision,
                "recall": flat_recall,
                "f1_score": flat_f1,
            }

        # 9) Guardar modelo
        run_id = f"run_{int(_time.time())}_{np.random.randint(1000,9999)}"
        os.makedirs("data/models", exist_ok=True)
        model_filename = f"data/models/{run_id}.joblib"
        joblib.dump(model, model_filename)

        # 10) Registrar en MLflow (opcional)
        mlflow_registered = False
        try:
            import mlflow

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001"))

            with mlflow.start_run(run_name=run_id):
                mlflow.log_param("algorithm", algorithm)
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("dataset_size", int(len(df)))
                mlflow.log_param("features", list(X.columns))
                mlflow.log_param("target", target_column)

                for k, v in metrics.items():
                    if v is not None:
                        mlflow.log_metric(k, float(v))

                mlflow.sklearn.log_model(model, "model")
                mlflow_registered = True

        except Exception as mlflow_error:
            logger.warning(f"MLflow registration failed: {mlflow_error}")

        # ✅ 11) RESPUESTA LOVABLE-FRIENDLY (MÉTRICAS PLANAS + IDs)
        return {
            "success": True,

            # Identificadores
            "run_id": run_id,
            "model_id": run_id,  # compat
            "algorithm": algorithm,
            "model_type": model_type,

            # Métricas planas
            "accuracy": float(flat_accuracy),
            "precision": float(flat_precision),
            "recall": float(flat_recall),
            "f1_score": float(flat_f1),

            # Regresión (si aplica)
            "rmse": metrics.get("rmse"),
            "r2_score": metrics.get("r2_score"),
            "mse": metrics.get("mse"),

            # Dataset info
            "dataset_size": int(len(df)),
            "training_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "features": list(X.columns),
            "target": target_column,

            # Debug/compat
            "metrics": metrics,
            "model_saved": model_filename,
            "mlflow_registered": mlflow_registered,
        }

    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        return {"success": False, "error": str(e), "error_type": type(e).__name__}
