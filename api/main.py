
from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

APP_VERSION = os.getenv("APP_VERSION", "1.0.2-fixed")
MODEL_NAME = os.getenv("MODEL_NAME", "Bank Marketing Final Model")
DEFAULT_THRESHOLD = float(os.getenv("BUSINESS_THRESHOLD", "0.05"))
AWS_API_DOCS_URL = os.getenv("AWS_API_DOCS_URL", "http://23.22.41.61:8000/docs")
ENVIRONMENT = os.getenv("ENVIRONMENT", "local")
API_KEY_ENV = os.getenv("API_KEY", "")

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "api" else Path.cwd()

MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "final_model.pkl"))
THRESHOLD_PATH = Path(os.getenv("THRESHOLD_PATH", PROJECT_ROOT / "models" / "bagging_rf_threshold.pkl"))
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

FEATURES = [
    "age", "education", "campaign", "pdays", "previous", "emp.var.rate",
    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
    "job_blue-collar", "job_entrepreneur", "job_housemaid", "job_management",
    "job_retired", "job_self-employed", "job_services", "job_student",
    "job_technician", "job_unemployed", "marital_married", "marital_single",
    "default_yes", "housing_yes", "loan_yes", "contact_telephone",
    "month_aug", "month_dec", "month_jul", "month_jun", "month_mar",
    "month_may", "month_nov", "month_oct", "month_sep", "day_of_week_mon",
    "day_of_week_thu", "day_of_week_tue", "day_of_week_wed",
    "poutcome_nonexistent", "poutcome_success", "contacted_before",
    "campaign_intensity", "has_loan_or_housing",
]


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Bank Marketing API",
    description=(
        "API REST para predicción de suscripción bancaria. "
        "Incluye endpoints de diagnóstico, Business Value, MLflow y dashboard incremental."
    ),
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CARGA ROBUSTA DEL MODELO
# ============================================================

def unwrap_model_artifact(artifact: Any) -> tuple[Any, Optional[float], Dict[str, Any]]:
    """
    Corrige el caso detectado:
    final_model.pkl fue guardado como dict, no directamente como estimador sklearn.

    Esta función busca dentro del dict la llave que contiene el modelo real.
    """
    metadata: Dict[str, Any] = {
        "raw_artifact_type": type(artifact).__name__,
        "unwrapped": False,
        "selected_key": None,
        "available_keys": None,
    }

    artifact_threshold = None

    if not isinstance(artifact, dict):
        return artifact, artifact_threshold, metadata

    metadata["available_keys"] = list(artifact.keys())

    # Posibles nombres usados normalmente para guardar modelos
    candidate_keys = [
        "model",
        "estimator",
        "final_model",
        "best_model",
        "best_estimator",
        "best_estimator_",
        "pipeline",
        "clf",
        "classifier",
        "bagging_model",
        "trained_model",
    ]

    # Recuperar threshold si vino dentro del dict
    for threshold_key in ["threshold", "best_threshold", "optimal_threshold", "business_threshold"]:
        if threshold_key in artifact:
            try:
                artifact_threshold = float(artifact[threshold_key])
                break
            except Exception:
                pass

    # Buscar por llaves conocidas
    for key in candidate_keys:
        if key in artifact:
            candidate = artifact[key]
            if hasattr(candidate, "predict") or hasattr(candidate, "predict_proba"):
                metadata["unwrapped"] = True
                metadata["selected_key"] = key
                return candidate, artifact_threshold, metadata

    # Buscar cualquier valor dentro del dict que parezca estimador
    for key, value in artifact.items():
        if hasattr(value, "predict") or hasattr(value, "predict_proba"):
            metadata["unwrapped"] = True
            metadata["selected_key"] = key
            return value, artifact_threshold, metadata

    raise TypeError(
        "El archivo final_model.pkl es un dict, pero no contiene ningún objeto con predict/predict_proba. "
        f"Keys disponibles: {list(artifact.keys())}"
    )


def load_model_and_metadata() -> tuple[Any, Optional[float], Dict[str, Any]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Verifica models/final_model.pkl o configura MODEL_PATH."
        )

    raw_artifact = joblib.load(MODEL_PATH)
    estimator, artifact_threshold, metadata = unwrap_model_artifact(raw_artifact)

    if not (hasattr(estimator, "predict") or hasattr(estimator, "predict_proba")):
        raise TypeError(
            f"El objeto cargado desde {MODEL_PATH} no parece un modelo sklearn válido. "
            f"Tipo: {type(estimator).__name__}"
        )

    return estimator, artifact_threshold, metadata


def load_threshold(artifact_threshold: Optional[float]) -> float:
    if artifact_threshold is not None:
        return float(artifact_threshold)

    if THRESHOLD_PATH.exists():
        try:
            value = joblib.load(THRESHOLD_PATH)
            return float(value)
        except Exception as e:
            print(f"[WARN] No se pudo cargar threshold desde {THRESHOLD_PATH}: {e}")
            return DEFAULT_THRESHOLD

    print(f"[WARN] No existe {THRESHOLD_PATH}. Se usará threshold por defecto: {DEFAULT_THRESHOLD}")
    return DEFAULT_THRESHOLD


model, threshold_from_artifact, MODEL_METADATA = load_model_and_metadata()
threshold = load_threshold(threshold_from_artifact)


# ============================================================
# SCHEMAS
# ============================================================

class ClientInput(BaseModel):
    age: float
    education: float
    campaign: float
    pdays: float
    previous: float
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float
    job_blue_collar: int = 0
    job_entrepreneur: int = 0
    job_housemaid: int = 0
    job_management: int = 0
    job_retired: int = 0
    job_self_employed: int = 0
    job_services: int = 0
    job_student: int = 0
    job_technician: int = 0
    job_unemployed: int = 0
    marital_married: int = 0
    marital_single: int = 0
    default_yes: int = 0
    housing_yes: int = 0
    loan_yes: int = 0
    contact_telephone: int = 0
    month_aug: int = 0
    month_dec: int = 0
    month_jul: int = 0
    month_jun: int = 0
    month_mar: int = 0
    month_may: int = 0
    month_nov: int = 0
    month_oct: int = 0
    month_sep: int = 0
    day_of_week_mon: int = 0
    day_of_week_thu: int = 0
    day_of_week_tue: int = 0
    day_of_week_wed: int = 0
    poutcome_nonexistent: int = 0
    poutcome_success: int = 0
    contacted_before: int = 0
    campaign_intensity: float = 0.0
    has_loan_or_housing: int = 0


class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    label: str
    threshold_used: float
    decision: str


class BatchInput(BaseModel):
    records: List[ClientInput]


# ============================================================
# UTILIDADES
# ============================================================

def validate_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY_ENV and x_api_key != API_KEY_ENV:
        raise HTTPException(status_code=401, detail="API Key inválida")


def get_model_expected_features() -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    if hasattr(model, "estimators_") and len(getattr(model, "estimators_", [])) > 0:
        first_estimator = model.estimators_[0]
        if hasattr(first_estimator, "feature_names_in_"):
            return list(first_estimator.feature_names_in_)

    return None


def client_to_model_dataframe(client: ClientInput) -> pd.DataFrame:
    row = {
        "age": client.age,
        "education": client.education,
        "campaign": client.campaign,
        "pdays": client.pdays,
        "previous": client.previous,
        "emp.var.rate": client.emp_var_rate,
        "cons.price.idx": client.cons_price_idx,
        "cons.conf.idx": client.cons_conf_idx,
        "euribor3m": client.euribor3m,
        "nr.employed": client.nr_employed,
        "job_blue-collar": client.job_blue_collar,
        "job_entrepreneur": client.job_entrepreneur,
        "job_housemaid": client.job_housemaid,
        "job_management": client.job_management,
        "job_retired": client.job_retired,
        "job_self-employed": client.job_self_employed,
        "job_services": client.job_services,
        "job_student": client.job_student,
        "job_technician": client.job_technician,
        "job_unemployed": client.job_unemployed,
        "marital_married": client.marital_married,
        "marital_single": client.marital_single,
        "default_yes": client.default_yes,
        "housing_yes": client.housing_yes,
        "loan_yes": client.loan_yes,
        "contact_telephone": client.contact_telephone,
        "month_aug": client.month_aug,
        "month_dec": client.month_dec,
        "month_jul": client.month_jul,
        "month_jun": client.month_jun,
        "month_mar": client.month_mar,
        "month_may": client.month_may,
        "month_nov": client.month_nov,
        "month_oct": client.month_oct,
        "month_sep": client.month_sep,
        "day_of_week_mon": client.day_of_week_mon,
        "day_of_week_thu": client.day_of_week_thu,
        "day_of_week_tue": client.day_of_week_tue,
        "day_of_week_wed": client.day_of_week_wed,
        "poutcome_nonexistent": client.poutcome_nonexistent,
        "poutcome_success": client.poutcome_success,
        "contacted_before": client.contacted_before,
        "campaign_intensity": client.campaign_intensity,
        "has_loan_or_housing": client.has_loan_or_housing,
    }

    expected = get_model_expected_features()
    columns = expected if expected else FEATURES

    df = pd.DataFrame([row])
    df = df.reindex(columns=columns, fill_value=0)

    # Seguridad adicional: todo numérico.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def predict_one(client: ClientInput) -> Dict[str, Any]:
    data = client_to_model_dataframe(client)

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(data)[0][1])
    elif hasattr(model, "decision_function"):
        raw = float(model.decision_function(data)[0])
        prob = 1.0 / (1.0 + pow(2.718281828, -raw))
    else:
        pred_raw = int(model.predict(data)[0])
        prob = float(pred_raw)

    pred = int(prob >= threshold)
    label = "Suscribe" if pred == 1 else "No suscribe"
    decision = "Priorizar llamada" if pred == 1 else "No priorizar inicialmente"

    return {
        "prediction": pred,
        "probability": prob,
        "label": label,
        "threshold_used": threshold,
        "decision": decision,
    }


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def list_files(directory: Path, patterns: tuple[str, ...]) -> List[Dict[str, Any]]:
    files = []
    if not directory.exists():
        return files

    for pattern in patterns:
        for file in directory.glob(pattern):
            if file.is_file():
                files.append({
                    "name": file.name,
                    "relative_path": str(file.relative_to(PROJECT_ROOT)) if PROJECT_ROOT in file.parents else str(file),
                    "size_kb": round(file.stat().st_size / 1024, 2),
                })

    return sorted(files, key=lambda x: x["name"])


# ============================================================
# ENDPOINTS BASE
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Bank Marketing API - MVP incremental Sprint 1 a Sprint 6",
        "docs": "/docs",
        "health": "/health",
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": ENVIRONMENT,
        "project_root": str(PROJECT_ROOT),
        "model_loaded": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "model_class": type(model).__name__,
        "model_metadata": MODEL_METADATA,
        "threshold": threshold,
        "threshold_path": str(THRESHOLD_PATH),
        "threshold_file_exists": THRESHOLD_PATH.exists(),
        "mlruns_exists": MLRUNS_DIR.exists(),
        "reports_exists": REPORTS_DIR.exists(),
    }


@app.get("/version")
def version():
    return {
        "model": MODEL_NAME,
        "version": APP_VERSION,
        "threshold": threshold,
        "model_path": str(MODEL_PATH),
        "model_class": type(model).__name__,
    }


@app.get("/model-info")
def model_info():
    expected = get_model_expected_features()

    return {
        "model_name": MODEL_NAME,
        "model_class": type(model).__name__,
        "model_file": str(MODEL_PATH),
        "model_metadata": MODEL_METADATA,
        "threshold_file": str(THRESHOLD_PATH),
        "threshold_used": threshold,
        "api_features_count": len(FEATURES),
        "api_features": FEATURES,
        "model_expected_features_count": len(expected) if expected else None,
        "model_expected_features": expected,
        "deployment_ready": True,
        "stage": "Sprint 6 MVP",
    }


@app.get("/features")
def get_features():
    expected = get_model_expected_features()
    return {
        "api_features_count": len(FEATURES),
        "api_features": FEATURES,
        "model_expected_features_count": len(expected) if expected else None,
        "model_expected_features": expected,
    }


@app.get("/debug-model")
def debug_model():
    expected = get_model_expected_features()
    columns = expected if expected else FEATURES
    sample_df = pd.DataFrame([{col: 0 for col in columns}])

    response = {
        "model_class": type(model).__name__,
        "model_path": str(MODEL_PATH),
        "model_metadata": MODEL_METADATA,
        "threshold": threshold,
        "expected_features": expected,
        "sample_shape": sample_df.shape,
    }

    try:
        if hasattr(model, "predict_proba"):
            response["sample_predict_proba"] = model.predict_proba(sample_df).tolist()
        response["sample_predict"] = model.predict(sample_df).tolist()
        response["sample_test_status"] = "ok"
    except Exception as e:
        response["sample_test_status"] = "error"
        response["sample_error"] = str(e)
        response["traceback"] = traceback.format_exc()

    return response


@app.get("/aws-status")
def aws_status():
    return {
        "environment": ENVIRONMENT,
        "aws_docs_url": AWS_API_DOCS_URL,
        "local_docs_url": "http://localhost:8000/docs",
        "can_run_local": True,
        "can_run_aws": True,
        "note": "Configure API_URL en el dashboard para apuntar a local o AWS.",
    }


# ============================================================
# BUSINESS VALUE / REPORTES / MLFLOW
# ============================================================

@app.get("/business-value")
def business_value():
    summary_path = REPORTS_DIR / "business_value_summary.csv"
    summary = read_csv_if_exists(summary_path)

    if summary is not None and not summary.empty:
        return {
            "source": str(summary_path),
            "rows": summary.to_dict(orient="records"),
        }

    return {
        "source": "fallback",
        "threshold_optimo": threshold,
        "recomendacion": "Avanzar como MVP controlado y validar supuestos con negocio.",
        "note": "No se encontró reports/business_value_summary.csv.",
    }


@app.get("/reports")
def reports():
    return {
        "reports_dir": str(REPORTS_DIR),
        "files": list_files(REPORTS_DIR, ("*.csv", "*.md", "*.png")),
    }


@app.get("/experiments-log")
def experiments_log():
    candidates = [
        MODELS_DIR / "experiments_log.csv",
        REPORTS_DIR / "experiments_log.csv",
    ]

    for path in candidates:
        df = read_csv_if_exists(path)
        if df is not None:
            return {
                "source": str(path),
                "n_rows": int(len(df)),
                "columns": df.columns.tolist(),
                "preview": df.head(20).to_dict(orient="records"),
            }

    return {
        "source": None,
        "n_rows": 0,
        "columns": [],
        "preview": [],
        "note": "No se encontró experiments_log.csv.",
    }


@app.get("/mlflow-status")
def mlflow_status():
    experiments = []
    if MLRUNS_DIR.exists():
        for child in MLRUNS_DIR.iterdir():
            if child.is_dir():
                experiments.append(child.name)

    return {
        "mlflow_local_enabled": MLRUNS_DIR.exists(),
        "mlruns_dir": str(MLRUNS_DIR),
        "experiments_detected": experiments,
        "ui_command": "python -m mlflow ui --backend-store-uri mlruns",
        "ui_url": "http://127.0.0.1:5000",
        "expected_experiment": "bank_marketing_mvp",
    }


@app.get("/sprint-deliverables")
def sprint_deliverables():
    return {
        "Sprint 1": {
            "name": "Data Understanding + widgets",
            "evidence": ["notebooks/04_prototype.ipynb"],
        },
        "Sprint 2": {
            "name": "Data Preparation",
            "evidence": [
                "notebooks/05_data_cleaning.ipynb",
                "notebooks/06_feature_eng.ipynb",
                "notebooks/07_class_balance.ipynb",
                "notebooks/08_pipeline.ipynb",
            ],
        },
        "Sprint 3": {
            "name": "Baseline Models",
            "evidence": [
                "notebooks/09_baseline_models.ipynb",
                "notebooks/10_metrics_evaluator.ipynb",
                "notebooks/11_model_comparator.ipynb",
                "models/evaluation_cv_results.csv",
            ],
        },
        "Sprint 4": {
            "name": "Tuning + Final Model + Experiment Tracker",
            "evidence": [
                "notebooks/12_hyperparam_tuning.ipynb",
                "notebooks/13_ensembles.ipynb",
                "notebooks/14_final_validation.ipynb",
                "notebooks/14.5_experiment_tracker_sprint4.ipynb",
                "notebooks/14.75_mlflow_tracking.ipynb",
                "models/final_model.pkl",
                "models/experiments_log.csv",
            ],
        },
        "Sprint 5": {
            "name": "Business Value",
            "evidence": [
                "notebooks/15_business_value.ipynb",
                "reports/business_value_summary.csv",
                "reports/business_value_sensitivity.csv",
                "reports/gain_curve.csv",
                "reports/lift_curve.csv",
                "reports/recommendations.md",
            ],
        },
        "Sprint 6": {
            "name": "API + Dashboard + AWS",
            "evidence": [
                "api/main.py",
                "dashboard/app.py",
                "http://23.22.41.61:8000/docs",
            ],
        },
    }


# ============================================================
# PREDICCIÓN
# ============================================================

@app.post("/predict", response_model=PredictionOutput)
def predict(client: ClientInput, x_api_key: Optional[str] = Header(default=None)):
    validate_api_key(x_api_key)

    try:
        return PredictionOutput(**predict_one(client))
    except Exception as e:
        print("\n================ ERROR EN /predict ================")
        print(traceback.format_exc())
        print("====================================================\n")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-debug")
def predict_debug(client: ClientInput, x_api_key: Optional[str] = Header(default=None)):
    validate_api_key(x_api_key)

    try:
        data = client_to_model_dataframe(client)
        result = predict_one(client)

        return {
            "status": "ok",
            "input_shape": data.shape,
            "input_columns": data.columns.tolist(),
            "input_preview": data.head(1).to_dict(orient="records"),
            "result": result,
        }

    except Exception as e:
        data_preview = None
        columns = None

        try:
            data = client_to_model_dataframe(client)
            data_preview = data.head(1).to_dict(orient="records")
            columns = data.columns.tolist()
        except Exception:
            pass

        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "input_columns": columns,
            "input_preview": data_preview,
            "model_expected_features": get_model_expected_features(),
            "model_metadata": MODEL_METADATA,
        }


@app.post("/predict-batch")
def predict_batch(batch: BatchInput, x_api_key: Optional[str] = Header(default=None)):
    validate_api_key(x_api_key)

    try:
        results = [predict_one(record) for record in batch.records]
        return {
            "n_records": len(results),
            "threshold_used": threshold,
            "results": results,
        }
    except Exception as e:
        print("\n================ ERROR EN /predict-batch ================")
        print(traceback.format_exc())
        print("==========================================================\n")
        raise HTTPException(status_code=500, detail=str(e))
