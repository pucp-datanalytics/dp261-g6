from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

LOCAL_API_URL = "http://localhost:8000"
AWS_API_URL = "http://23.22.41.61:8000"
DEFAULT_API_URL = os.getenv("API_URL", LOCAL_API_URL)
API_KEY = os.getenv("API_KEY", "")
MAX_ROWS = int(os.getenv("MAX_ROWS", "500"))

THIS_FILE = Path(__file__).resolve() if "__file__" in globals() else Path.cwd() / "app.py"
PROJECT_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "dashboard" else Path.cwd()

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

st.set_page_config(page_title="Bank Marketing MVP Dashboard", layout="wide")
st.title("Bank Marketing — Dashboard Incremental MVP")
st.caption("Sprint 1 a Sprint 6 | Widgets + Modelos + MLflow + Business Value + API/AWS")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Configuración de ejecución")

api_mode = st.sidebar.radio(
    "Modo de API",
    ["Local", "AWS", "Personalizado"],
    horizontal=True,
)

if api_mode == "Local":
    API_URL = LOCAL_API_URL
elif api_mode == "AWS":
    API_URL = AWS_API_URL
else:
    API_URL = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)

st.sidebar.code(API_URL)

section = st.sidebar.radio(
    "Navegación incremental",
    [
        "Resumen ejecutivo",
        "Sprint 1 — Widgets exploratorios",
        "Sprint 2 — Data Preparation",
        "Sprint 3 — Baselines y métricas",
        "Sprint 4 — Experiment Tracker + MLflow",
        "Sprint 5 — Business Value",
        "Sprint 6 — API/AWS",
        "Predicción en vivo",
    ],
)


# ============================================================
# HELPERS
# ============================================================

def get_headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["x-api-key"] = API_KEY
    return h


def api_get(endpoint: str, timeout: int = 8) -> Optional[dict]:
    try:
        r = requests.get(f"{API_URL}{endpoint}", headers=get_headers(), timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def check_api_health() -> bool:
    data = api_get("/health", timeout=5)
    return bool(data and data.get("status") == "ok")


def safe_int(val, default=0):
    try:
        if pd.isna(val):
            return default
        return int(val)
    except Exception:
        return default


def safe_float(val, default=0.0):
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


def read_csv_smart(path: Path) -> pd.DataFrame:
    """
    Lee CSV de forma robusta:
    - data/raw/03-bank_marketing.csv usa separador ';'
    - data/processed/*.csv usa separador ','
    """
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path)

    # Si quedó una sola columna con ';', relanzar con sep=';'
    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")

    return df


def load_csv_candidates(candidates: list[Path]) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    for path in candidates:
        try:
            if path.exists():
                return read_csv_smart(path), path
        except Exception:
            continue
    return None, None


@st.cache_data
def load_csv_cached(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def show_file_status(label: str, path: Path):
    if path.exists():
        st.success(f"✅ {label}: `{path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in path.parents else path}`")
    else:
        st.warning(f"⚠️ {label}: no encontrado en `{path}`")


def call_predict(row: dict) -> dict:
    payload = {
        "age": safe_float(row.get("age")),
        "education": safe_float(row.get("education")),
        "campaign": safe_float(row.get("campaign")),
        "pdays": safe_float(row.get("pdays")),
        "previous": safe_float(row.get("previous")),
        "emp_var_rate": safe_float(row.get("emp.var.rate")),
        "cons_price_idx": safe_float(row.get("cons.price.idx")),
        "cons_conf_idx": safe_float(row.get("cons.conf.idx")),
        "euribor3m": safe_float(row.get("euribor3m")),
        "nr_employed": safe_float(row.get("nr.employed")),
        "job_blue_collar": safe_int(row.get("job_blue-collar", 0)),
        "job_entrepreneur": safe_int(row.get("job_entrepreneur", 0)),
        "job_housemaid": safe_int(row.get("job_housemaid", 0)),
        "job_management": safe_int(row.get("job_management", 0)),
        "job_retired": safe_int(row.get("job_retired", 0)),
        "job_self_employed": safe_int(row.get("job_self-employed", 0)),
        "job_services": safe_int(row.get("job_services", 0)),
        "job_student": safe_int(row.get("job_student", 0)),
        "job_technician": safe_int(row.get("job_technician", 0)),
        "job_unemployed": safe_int(row.get("job_unemployed", 0)),
        "marital_married": safe_int(row.get("marital_married", 0)),
        "marital_single": safe_int(row.get("marital_single", 0)),
        "default_yes": safe_int(row.get("default_yes", 0)),
        "housing_yes": safe_int(row.get("housing_yes", 0)),
        "loan_yes": safe_int(row.get("loan_yes", 0)),
        "contact_telephone": safe_int(row.get("contact_telephone", 0)),
        "month_aug": safe_int(row.get("month_aug", 0)),
        "month_dec": safe_int(row.get("month_dec", 0)),
        "month_jul": safe_int(row.get("month_jul", 0)),
        "month_jun": safe_int(row.get("month_jun", 0)),
        "month_mar": safe_int(row.get("month_mar", 0)),
        "month_may": safe_int(row.get("month_may", 0)),
        "month_nov": safe_int(row.get("month_nov", 0)),
        "month_oct": safe_int(row.get("month_oct", 0)),
        "month_sep": safe_int(row.get("month_sep", 0)),
        "day_of_week_mon": safe_int(row.get("day_of_week_mon", 0)),
        "day_of_week_thu": safe_int(row.get("day_of_week_thu", 0)),
        "day_of_week_tue": safe_int(row.get("day_of_week_tue", 0)),
        "day_of_week_wed": safe_int(row.get("day_of_week_wed", 0)),
        "poutcome_nonexistent": safe_int(row.get("poutcome_nonexistent", 0)),
        "poutcome_success": safe_int(row.get("poutcome_success", 0)),
        "contacted_before": safe_int(row.get("contacted_before", 0)),
        "campaign_intensity": safe_float(row.get("campaign_intensity", 0.0)),
        "has_loan_or_housing": safe_int(row.get("has_loan_or_housing", 0)),
    }

    r = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers=get_headers(),
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def run_predictions(df, n_filas):
    results = []
    errores = 0
    prog = st.progress(0)

    for i, (_, row) in enumerate(df.head(n_filas).iterrows()):
        try:
            res = call_predict(row.to_dict())
            results.append(res)
        except Exception:
            errores += 1
            results.append({
                "prediction": 0,
                "probability": 0.0,
                "label": "Error",
                "threshold_used": None,
                "decision": "Error",
            })
        prog.progress((i + 1) / n_filas)

    return results, errores


def show_metrics_and_charts(y_true, y_pred, y_proba):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
    col3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
    col4.metric("F1-Score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")

    try:
        col5.metric("AUC-ROC", f"{roc_auc_score(y_true, y_proba):.3f}")
    except Exception:
        col5.metric("AUC-ROC", "N/A")

    st.subheader("Confusion Matrix y Curva ROC")
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            display_labels=["No", "Yes"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        st.pyplot(fig)

    with col_b:
        fig, ax = plt.subplots()
        try:
            RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="Final Model API")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
            st.pyplot(fig)
        except Exception:
            st.info("No se pudo calcular ROC-AUC para la muestra cargada.")


def plot_curve(df: pd.DataFrame, x_col: str, y_col: str, title: str, baseline: bool = False):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[x_col], df[y_col], label="Modelo")
    if baseline:
        ax.plot([0, 1], [0, 1], "--", label="Aleatorio")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


# ============================================================
# API STATUS
# ============================================================

api_ok = check_api_health()
if api_ok:
    st.sidebar.success("API conectada")
else:
    st.sidebar.warning("API no conectada")


# ============================================================
# SECCIONES
# ============================================================

def show_summary():
    st.header("Resumen ejecutivo del MVP incremental")

    st.markdown(
        """
        Este dashboard consolida el proyecto de telemarketing bancario desde Sprint 1 hasta Sprint 6.
        Mantiene el flujo acumulable: exploración interactiva, preparación de datos, modelos baseline,
        modelo final, Business Value, MLflow y API local/AWS.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fase", "Sprint 6")
    col2.metric("API", "Conectada" if api_ok else "No conectada")
    col3.metric("MLflow", "Detectado" if MLRUNS_DIR.exists() else "Pendiente")
    col4.metric("Modelo final", "OK" if (MODELS_DIR / "final_model.pkl").exists() else "No encontrado")

    st.subheader("Evidencia acumulada por sprint")
    api_deliverables = api_get("/sprint-deliverables") if api_ok else None

    if api_deliverables:
        rows = []
        for sprint, info in api_deliverables.items():
            rows.append({
                "Sprint": sprint,
                "Entregable": info.get("name"),
                "Evidencias": ", ".join(info.get("evidence", [])),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        rows = [
            ["Sprint 1", "Widgets exploratorios", "notebooks/04_prototype.ipynb"],
            ["Sprint 2", "Data Preparation", "notebooks/05-08"],
            ["Sprint 3", "Baselines y métricas", "notebooks/09-11, models/evaluation_cv_results.csv"],
            ["Sprint 4", "Tuning + Experiment Tracker + MLflow", "notebooks/12-14.75, models/final_model.pkl"],
            ["Sprint 5", "Business Value", "notebooks/15_business_value.ipynb, reports/"],
            ["Sprint 6", "API/AWS + Dashboard", "api/main.py, dashboard/app.py"],
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Sprint", "Entregable", "Evidencia"]), use_container_width=True)


def show_sprint1_widgets():
    st.header("Sprint 1 — Widgets exploratorios")

    st.markdown(
        """
        Esta sección replica en Streamlit el notebook `04_prototype.ipynb`.
        El objetivo es que el stakeholder explore el dataset original de telemarketing bancario
        con los mismos controles del prototipo: categoría, rango de edad, filtro de vivienda y tipo de gráfico.
        """
    )

    df, path = load_csv_candidates([
        DATA_DIR / "raw" / "03-bank_marketing.csv",
        DATA_DIR / "raw" / "bank_marketing.csv",
        DATA_DIR / "processed" / "test_original.csv",
    ])

    if df is None:
        st.warning("No se encontró dataset para exploración. Coloca `03-bank_marketing.csv` en `data/raw/`.")
        return

    # Normalizar nombres por seguridad
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = {"age", "housing", "y"}
    missing = sorted(list(required_cols - set(df.columns)))
    if missing:
        st.error(
            "El archivo cargado no parece ser el dataset original del notebook 04. "
            f"Faltan columnas: {missing}. Revisa que `03-bank_marketing.csv` esté separado por `;`."
        )
        st.dataframe(df.head(10), use_container_width=True)
        return

    st.success(f"Dataset cargado correctamente: `{path}`")
    st.caption(f"Registros: {len(df):,} | Columnas: {len(df.columns)}")

    with st.expander("Vista previa del dataset original", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # Widgets alineados con el notebook 04
    st.subheader("Controles interactivos del prototipo")

    col1, col2, col3, col4 = st.columns([1.2, 1.8, 1.2, 1.2])

    available_categories = [c for c in ["job", "education", "month"] if c in df.columns]
    if not available_categories:
        available_categories = df.select_dtypes(include="object").columns.tolist()

    with col1:
        categoria = st.selectbox(
            "Categoría",
            options=available_categories,
            index=0,
            help="Mismo Dropdown del notebook: job, education o month.",
        )

    with col2:
        age_min = int(pd.to_numeric(df["age"], errors="coerce").min())
        age_max = int(pd.to_numeric(df["age"], errors="coerce").max())
        rango_edad = st.slider(
            "Edad",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            step=1,
            help="Mismo IntRangeSlider del notebook.",
        )

    with col3:
        solo_vivienda = st.checkbox(
            "Solo con Vivienda",
            value=False,
            help="Replica el checkbox del notebook: housing == yes.",
        )

    with col4:
        tipo_grafico = st.radio(
            "Estilo",
            options=["Barras", "Conteo"],
            horizontal=False,
            help="Replica los RadioButtons del notebook.",
        )

    # Filtrado
    datos_filtrados = df.copy()
    datos_filtrados["age"] = pd.to_numeric(datos_filtrados["age"], errors="coerce")
    datos_filtrados = datos_filtrados[
        (datos_filtrados["age"] >= rango_edad[0]) &
        (datos_filtrados["age"] <= rango_edad[1])
    ]

    if solo_vivienda:
        datos_filtrados = datos_filtrados[
            datos_filtrados["housing"].astype(str).str.lower().eq("yes")
        ]

    st.metric("Registros filtrados", f"{len(datos_filtrados):,}")

    if datos_filtrados.empty:
        st.warning("No hay registros con los filtros seleccionados.")
        return

    # KPIs rápidos
    c1, c2, c3 = st.columns(3)
    tasa_yes = (
        datos_filtrados["y"].astype(str).str.lower().eq("yes").mean()
        if "y" in datos_filtrados.columns else 0
    )
    c1.metric("Clientes filtrados", f"{len(datos_filtrados):,}")
    c2.metric("Tasa de suscripción", f"{tasa_yes:.2%}")
    c3.metric("Categoría analizada", categoria)

    st.subheader(f"Análisis de `{categoria}` — clientes filtrados")

    # Preparar datos para gráfico
    plot_df = datos_filtrados.copy()
    plot_df[categoria] = plot_df[categoria].astype(str)
    plot_df["y"] = plot_df["y"].astype(str)

    top_categories = plot_df[categoria].value_counts().head(15).index
    plot_df = plot_df[plot_df[categoria].isin(top_categories)]

    fig, ax = plt.subplots(figsize=(11, 5))

    if tipo_grafico == "Barras":
        # Barras agrupadas por suscripción, como el notebook corregido
        counts = (
            plot_df
            .groupby([categoria, "y"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=top_categories)
        )
        counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Cantidad de clientes")
        ax.set_xlabel(categoria)
        ax.set_title(f"Distribución de {categoria} por resultado de suscripción")
        ax.legend(title="Suscripción")
    else:
        # Conteo total por categoría
        counts = plot_df[categoria].value_counts().reindex(top_categories)
        counts.plot(kind="bar", ax=ax)
        ax.set_ylabel("Cantidad de clientes")
        ax.set_xlabel(categoria)
        ax.set_title(f"Conteo de clientes por {categoria}")

    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Tabla resumen")
    if "y" in plot_df.columns:
        summary = (
            plot_df
            .groupby(categoria)
            .agg(
                total=("y", "size"),
                suscribe=("y", lambda s: (s.astype(str).str.lower() == "yes").sum()),
            )
            .reset_index()
        )
        summary["no_suscribe"] = summary["total"] - summary["suscribe"]
        summary["tasa_suscripcion"] = summary["suscribe"] / summary["total"]
        summary = summary.sort_values("total", ascending=False)
        st.dataframe(summary, use_container_width=True)
    else:
        st.dataframe(plot_df[categoria].value_counts().reset_index(), use_container_width=True)


def show_sprint2_data_prep():
    st.header("Sprint 2 — Data Preparation")

    st.markdown("Evidencia: limpieza, feature engineering, balanceo y pipeline reproducible.")

    files = [
        ("Raw dataset", DATA_DIR / "raw" / "03-bank_marketing.csv"),
        ("Feature dataset", DATA_DIR / "interim" / "04_bank_marketing_feature.csv"),
        ("Train balanceado", DATA_DIR / "processed" / "train_balanced.csv"),
        ("Test original", DATA_DIR / "processed" / "test_original.csv"),
        ("Preprocessor", MODELS_DIR / "preprocessor.pkl"),
    ]

    for label, path in files:
        show_file_status(label, path)

    df, path = load_csv_candidates([DATA_DIR / "processed" / "train_balanced.csv"])
    if df is not None:
        st.subheader("Vista train_balanced.csv")
        st.write("Dimensión:", df.shape)
        st.dataframe(df.head(50), use_container_width=True)

        if "y" in df.columns:
            st.subheader("Distribución del target")
            st.bar_chart(df["y"].value_counts().sort_index())


def show_sprint3_baselines():
    st.header("Sprint 3 — Baselines y métricas")

    st.markdown(
        """
        En esta sección se comparan los modelos baseline usando métricas de validación cruzada.
        La gráfica de F1 se ordena de mayor a menor para evitar interpretaciones visuales incorrectas.
        """
    )

    eval_path = MODELS_DIR / "evaluation_cv_results.csv"
    log_path = MODELS_DIR / "experiments_log.csv"

    def extract_metric_mean(value):
        """
        Convierte valores tipo '0.7432 ± 0.0029' o valores numéricos a float.
        """
        try:
            if pd.isna(value):
                return np.nan
            value_str = str(value)
            if "±" in value_str:
                value_str = value_str.split("±")[0].strip()
            return float(value_str)
        except Exception:
            return np.nan

    if eval_path.exists():
        st.subheader("Cross-validation — evaluation_cv_results.csv")
        df_eval = pd.read_csv(eval_path, index_col=0)
        st.dataframe(df_eval, use_container_width=True)

        if "f1" in df_eval.columns:
            df_plot = df_eval.copy()
            df_plot["f1_numeric"] = df_plot["f1"].apply(extract_metric_mean)
            df_plot = df_plot.dropna(subset=["f1_numeric"])
            df_plot = df_plot.sort_values("f1_numeric", ascending=False)

            if not df_plot.empty:
                st.subheader("Ranking F1 Cross-Validation — mayor a menor")

                fig, ax = plt.subplots(figsize=(9, 4.8))
                bars = ax.bar(df_plot.index.astype(str), df_plot["f1_numeric"])

                ax.set_xlabel("Modelo")
                ax.set_ylabel("F1-score promedio CV")
                ax.set_title("Comparación de modelos baseline por F1-score")
                ax.grid(axis="y", alpha=0.3)

                for bar in bars:
                    value = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                plt.xticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)

                best_model = str(df_plot.index[0])
                best_f1 = float(df_plot["f1_numeric"].iloc[0])

                col1, col2, col3 = st.columns(3)
                col1.metric("Mejor modelo por F1", best_model)
                col2.metric("F1 promedio CV", f"{best_f1:.3f}")
                col3.metric("Modelos comparados", len(df_plot))

                st.caption(
                    "El ranking se calcula extrayendo el promedio antes del símbolo ±. "
                    "El orden mostrado es de mayor a menor F1-score."
                )
        else:
            st.info("El archivo evaluation_cv_results.csv no contiene columna `f1`.")
    else:
        st.warning("No se encontró models/evaluation_cv_results.csv.")

    if log_path.exists():
        st.subheader("Registro acumulado — experiments_log.csv")
        df_log = pd.read_csv(log_path)
        st.dataframe(df_log, use_container_width=True)

        if "model" in df_log.columns and "f1" in df_log.columns:
            plot_data = df_log.copy()
            plot_data["f1_numeric"] = pd.to_numeric(plot_data["f1"], errors="coerce")
            plot_data = plot_data.dropna(subset=["f1_numeric"])
            plot_data = plot_data.sort_values("f1_numeric", ascending=False)

            if not plot_data.empty:
                st.subheader("F1 por experimento registrado — mayor a menor")

                fig, ax = plt.subplots(figsize=(9, 4.8))
                labels = plot_data["model"].astype(str)
                values = plot_data["f1_numeric"]

                bars = ax.bar(labels, values)
                ax.set_xlabel("Modelo / experimento")
                ax.set_ylabel("F1-score")
                ax.set_title("Experimentos registrados ordenados por F1")
                ax.grid(axis="y", alpha=0.3)

                for bar in bars:
                    value = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                plt.xticks(rotation=35, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

                st.caption(
                    "Esta segunda gráfica usa `experiments_log.csv` y también se ordena de mayor a menor F1."
                )
    else:
        st.info("No se encontró models/experiments_log.csv.")


def show_sprint4_mlflow():
    st.header("Sprint 4 — Experiment Tracker + MLflow")

    st.markdown(
        """
        Evidencia: `models/experiments_log.csv`, `models/final_model.pkl`,
        `notebooks/14.75_mlflow_tracking.ipynb` y carpeta `mlruns/`.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("final_model.pkl", "OK" if (MODELS_DIR / "final_model.pkl").exists() else "Pendiente")
    col2.metric("experiments_log.csv", "OK" if (MODELS_DIR / "experiments_log.csv").exists() else "Pendiente")
    col3.metric("mlruns/", "OK" if MLRUNS_DIR.exists() else "Pendiente")

    st.subheader("MLflow local")
    st.code("python -m mlflow ui --backend-store-uri mlruns")
    st.markdown("Abrir: http://127.0.0.1:5000")

    if api_ok:
        st.subheader("Estado MLflow vía API")
        st.json(api_get("/mlflow-status"))

    log_path = MODELS_DIR / "experiments_log.csv"
    if log_path.exists():
        df_log = pd.read_csv(log_path)
        st.subheader("Preview experiments_log.csv")
        st.dataframe(df_log.head(30), use_container_width=True)


def show_sprint5_business_value():
    st.header("Sprint 5 — Business Value")

    st.markdown(
        """
        Esta sección presenta la decisión de negocio del MVP usando el **umbral operativo 0.204**.
        El umbral 0.05 se mantiene únicamente como referencia teórica de máximo valor económico, pero
        **no se recomienda para operación diaria** porque genera demasiadas llamadas no convertidas.
        """
    )

    summary_path = REPORTS_DIR / "business_value_summary.csv"
    sensitivity_path = REPORTS_DIR / "business_value_sensitivity.csv"
    threshold_path = REPORTS_DIR / "threshold_business_value.csv"
    gain_path = REPORTS_DIR / "gain_curve.csv"
    lift_path = REPORTS_DIR / "lift_curve.csv"
    rec_path = REPORTS_DIR / "recommendations.md"

    # ------------------------------------------------------------
    # 1. Decisión ejecutiva
    # ------------------------------------------------------------
    st.subheader("1. Decisión ejecutiva")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Umbral operativo recomendado", "0.204")
    col_b.metric("Precisión operativa", "0.230")
    col_c.metric("Recall operativo", "0.758")
    col_d.metric("Valor test operativo", "S/ 27,343")

    st.success(
        "Recomendación: usar el umbral 0.204 para el piloto comercial. "
        "Este umbral sacrifica parte del valor teórico máximo, pero mejora la eficiencia del call center "
        "y mantiene un recall suficiente para capturar clientes potenciales."
    )

    # ------------------------------------------------------------
    # 2. Comparativa de umbrales candidatos
    # ------------------------------------------------------------
    st.subheader("2. Comparativa de umbrales candidatos")

    threshold_comparison = pd.DataFrame(
        {
            "Indicador": [
                "Precisión",
                "Recall",
                "F1-score",
                "Valor económico total (test)",
                "Valor por cliente",
                "Impacto mensual estimado",
                "Llamadas por conversión",
                "Lectura operativa",
            ],
            "Umbral 0.05 (teórico)": [
                "0.115",
                "0.986",
                "0.206",
                "S/ 36,745",
                "S/ 4.46",
                "S/ 446,259",
                "~9 llamadas por conversión",
                "Maximiza valor bajo supuestos ideales, pero sobrecarga al equipo comercial.",
            ],
            "Umbral 0.204 (operativo recomendado)": [
                "0.230",
                "0.758",
                "0.358",
                "S/ 27,343",
                "S/ 3.26",
                "S/ 326,087",
                "~4 llamadas por conversión",
                "Balancea conversión, esfuerzo comercial y sostenibilidad operativa.",
            ],
        }
    )

    st.dataframe(threshold_comparison, use_container_width=True, hide_index=True)

    # ------------------------------------------------------------
    # 3. Por qué se elige 0.204
    # ------------------------------------------------------------
    st.subheader("3. Por qué se elige el umbral 0.204 y no el 0.05")

    st.markdown(
        """
        El umbral **0.05** logra el mayor valor económico esperado en el test, pero su precisión de **0.115**
        implica que, en promedio, solo **11 o 12 de cada 100 llamadas** terminarían en una conversión real.
        Para un call center bancario, esa tasa es operativamente débil: consume tiempo de agentes,
        incrementa carga de supervisión y puede deteriorar la experiencia del cliente.

        El umbral **0.204** es más defendible para negocio porque:

        - **Duplica aproximadamente la precisión** frente al umbral 0.05: pasa de 0.115 a 0.230.
        - Mantiene un **recall de 0.758**, capturando cerca del 76% de los clientes realmente interesados.
        - Mejora el **F1-score**, pasando de 0.206 a 0.358.
        - Reduce el esfuerzo comercial de aproximadamente **9 llamadas por conversión** a cerca de **4 llamadas por conversión**.
        - Mantiene un valor económico relevante para piloto: aproximadamente **S/ 27.3 mil en test** y **S/ 326 mil mensual estimado**.

        Por tanto, el umbral **0.204** no maximiza la fórmula económica pura, pero sí maximiza la
        **viabilidad operativa** del MVP.
        """
    )

    st.info(
        "Mensaje para stakeholders: el modelo no se despliega con el umbral matemáticamente más agresivo, "
        "sino con el umbral que permite operar la campaña sin saturar al equipo comercial."
    )

    # ------------------------------------------------------------
    # 4. Resumen oficial usando umbral operativo
    # ------------------------------------------------------------
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        st.subheader("4. Resumen de valor de negocio — foco en umbral operativo")

        st.dataframe(summary, use_container_width=True)

        # Seleccionar explícitamente la fila operativa.
        summary_tmp = summary.copy()
        summary_tmp["umbral_numeric"] = pd.to_numeric(summary_tmp.get("umbral"), errors="coerce")

        operational_rows = summary_tmp[
            summary_tmp.get("criterio", "").astype(str).str.lower().str.contains("operativo", na=False)
        ]

        if operational_rows.empty:
            operational_rows = summary_tmp[
                (summary_tmp["umbral_numeric"] - 0.204).abs() < 0.005
            ]

        if operational_rows.empty:
            row = summary_tmp.iloc[-1]
            st.warning(
                "No se detectó explícitamente la fila operativa; se está usando la última fila del resumen."
            )
        else:
            row = operational_rows.iloc[0]

        col1, col2, col3, col4, col5 = st.columns(5)

        if "umbral" in summary.columns:
            col1.metric("Umbral operativo", f"{float(row['umbral']):.3f}")
        if "precision" in summary.columns:
            col2.metric("Precisión", f"{float(row['precision']):.3f}")
        if "recall" in summary.columns:
            col3.metric("Recall", f"{float(row['recall']):.3f}")
        if "f1" in summary.columns:
            col4.metric("F1-score", f"{float(row['f1']):.3f}")
        if "valor_total_test" in summary.columns:
            col5.metric("Valor test", f"S/ {float(row['valor_total_test']):,.0f}")

        if "impacto_mensual_estimado" in summary.columns:
            st.metric("Impacto mensual estimado con umbral operativo", f"S/ {float(row['impacto_mensual_estimado']):,.0f}")

        st.caption(
            "Las métricas destacadas usan la fila operativa del business_value_summary.csv, no la fila teórica de umbral 0.05."
        )

    else:
        st.warning("No se encontró reports/business_value_summary.csv.")
        if api_ok:
            st.subheader("Business Value vía API")
            st.json(api_get("/business-value"))

    # ------------------------------------------------------------
    # 5. Sensibilidad
    # ------------------------------------------------------------
    if sensitivity_path.exists():
        st.subheader("5. Escenarios de sensibilidad")
        st.markdown(
            "Los escenarios de sensibilidad sirven para validar si la recomendación sigue siendo razonable "
            "cuando cambian beneficios, costos o supuestos comerciales."
        )
        st.dataframe(pd.read_csv(sensitivity_path), use_container_width=True)

    # ------------------------------------------------------------
    # 6. Valor económico por umbral
    # ------------------------------------------------------------
    if threshold_path.exists():
        st.subheader("6. Valor económico por umbral")
        df_th = pd.read_csv(threshold_path)
        st.dataframe(df_th.head(20), use_container_width=True)

        possible_x = "threshold" if "threshold" in df_th.columns else "umbral"
        possible_y = "valor_total" if "valor_total" in df_th.columns else None

        if possible_x in df_th.columns and possible_y in df_th.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df_th[possible_x], df_th[possible_y], label="Valor esperado")
            ax.axvline(0.05, linestyle="--", label="0.05 teórico")
            ax.axvline(0.204, linestyle="--", label="0.204 operativo")
            ax.set_xlabel(possible_x)
            ax.set_ylabel(possible_y)
            ax.set_title("Valor económico esperado por umbral")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            st.caption(
                "La curva muestra que 0.05 maximiza el valor teórico; sin embargo, 0.204 se adopta "
                "como umbral operativo por su mejor balance entre precisión, recall y carga comercial."
            )

    # ------------------------------------------------------------
    # 7. Gain / Lift
    # ------------------------------------------------------------
    if gain_path.exists():
        st.subheader("7. Gain Curve")
        gain = pd.read_csv(gain_path)
        if {"population_pct", "gain"}.issubset(gain.columns):
            plot_curve(gain, "population_pct", "gain", "Gain Curve — Modelo final", baseline=True)
        st.dataframe(gain.head(10), use_container_width=True)

    if lift_path.exists():
        st.subheader("8. Lift Curve")
        lift = pd.read_csv(lift_path)
        if {"population_pct", "lift"}.issubset(lift.columns):
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(lift["population_pct"], lift["lift"], label="Lift")
            ax.axhline(1, linestyle="--", label="Baseline aleatoria")
            ax.set_xlabel("% población contactada")
            ax.set_ylabel("Lift")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
        st.dataframe(lift.head(10), use_container_width=True)

    # ------------------------------------------------------------
    # 9. Recomendación final reescrita
    # ------------------------------------------------------------
    st.subheader("9. Recomendación final")

    st.markdown(
        """
        Se recomienda pasar a un piloto controlado usando el **umbral operativo 0.204**.
        El piloto debe medir no solo precisión, recall y conversiones, sino también capacidad diaria del call center,
        tasa de contacto efectivo, duración promedio de llamada y conversión real por agente.

        El umbral 0.05 queda documentado como frontera económica teórica. No se recomienda usarlo como regla de
        operación inicial porque exige demasiado volumen de llamadas para obtener conversiones reales.
        """
    )

    if rec_path.exists():
        with st.expander("Ver archivo recommendations.md original"):
            st.markdown(rec_path.read_text(encoding="utf-8", errors="ignore"))


def show_sprint6_api():
    st.header("Sprint 6 — API local/AWS + Dashboard")

    st.markdown("El dashboard puede consumir la API local o la API desplegada en AWS.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Local")
        st.code("uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")
        st.markdown("Docs: http://localhost:8000/docs")

    with col2:
        st.subheader("AWS")
        st.code(AWS_API_URL)
        st.markdown("Docs AWS: http://23.22.41.61:8000/docs")

    if api_ok:
        st.success(f"API conectada en {API_URL}")

        for endpoint in ["/health", "/version", "/model-info", "/features", "/aws-status", "/reports", "/mlflow-status"]:
            with st.expander(endpoint):
                st.json(api_get(endpoint))
    else:
        st.error(f"No se pudo conectar con la API en {API_URL}")


def show_live_prediction():
    st.header("Predicción en vivo vía API")

    if not api_ok:
        st.error(f"No se pudo conectar con la API en `{API_URL}`.")
        st.stop()

    try:
        ver = api_get("/version")
        st.caption(f"Modelo: `{ver.get('model')}` — versión `{ver.get('version')}` — threshold `{ver.get('threshold')}`")
    except Exception:
        pass

    @st.cache_data
    def load_test_data():
        candidates = [
            DATA_DIR / "processed" / "test_original.csv",
            PROJECT_ROOT / "test_original.csv",
        ]
        return load_csv_candidates(candidates)

    df_local, ruta_encontrada = load_test_data()

    st.subheader("Fuente de datos")

    if df_local is not None:
        st.success(f"CSV local encontrado: `{ruta_encontrada}` ({len(df_local)} filas)")
        usar_local = st.radio(
            "¿Qué datos quieres usar?",
            ["Usar CSV local automáticamente", "Subir CSV manualmente"],
            horizontal=True,
        )
    else:
        st.info("No se encontró CSV local. Sube el archivo manualmente.")
        usar_local = "Subir CSV manualmente"

    if usar_local == "Usar CSV local automáticamente" and df_local is not None:
        df_datos = df_local
    else:
        uploaded = st.file_uploader(
            "Sube el CSV con las columnas procesadas del modelo",
            type="csv",
        )
        if not uploaded:
            st.info("Sube `test_original.csv` o un CSV compatible para continuar.")
            st.stop()
        df_datos = pd.read_csv(uploaded)

    st.dataframe(df_datos.head(), use_container_width=True)

    tiene_target = "y" in df_datos.columns
    total_filas = len(df_datos)

    st.subheader("Configuración de predicción")
    col_opt1, col_opt2 = st.columns(2)

    with col_opt1:
        limitar = st.checkbox("Limitar filas", value=True)

    with col_opt2:
        if limitar:
            n_filas = st.slider(
                "¿Cuántas filas procesar?",
                min_value=1,
                max_value=min(MAX_ROWS, total_filas),
                value=min(200, total_filas),
                step=1,
            )
        else:
            n_filas = total_filas
            st.warning(f"Se procesarán {total_filas} filas.")

    if st.button("Obtener predicciones"):
        if tiene_target:
            X_datos = df_datos.drop(columns="y")
            y_true = df_datos["y"].head(n_filas).values
        else:
            X_datos = df_datos
            y_true = None

        with st.spinner(f"Consultando API para {n_filas} filas..."):
            results, errores = run_predictions(X_datos, n_filas)

        st.session_state["results"] = results
        st.session_state["errores"] = errores
        st.session_state["y_true"] = y_true
        st.session_state["X_datos"] = X_datos.head(n_filas)
        st.session_state["n_filas"] = n_filas

    if "results" in st.session_state:
        results = st.session_state["results"]
        errores = st.session_state["errores"]
        y_true = st.session_state["y_true"]
        X_datos = st.session_state["X_datos"]
        n_filas = st.session_state["n_filas"]

        if errores > 0:
            st.warning(f"{errores} filas tuvieron errores.")
        else:
            st.success(f"{n_filas} predicciones completadas sin errores.")

        y_pred = np.array([r["prediction"] for r in results])
        y_proba = np.array([r["probability"] for r in results])

        if y_true is not None:
            st.subheader("Métricas generales")
            show_metrics_and_charts(y_true, y_pred, y_proba)
        else:
            st.info("El CSV no tiene columna `y`; solo se muestran predicciones.")

        st.subheader("Tabla de predicciones")
        df_result = X_datos.copy()

        if y_true is not None:
            df_result["y_real"] = y_true

        df_result["y_pred"] = y_pred
        df_result["probabilidad"] = y_proba
        df_result["label"] = [r["label"] for r in results]
        df_result["decision"] = [r.get("decision", "") for r in results]
        st.dataframe(df_result, use_container_width=True)

        st.subheader("Simulador de predicción por instancia")
        idx = st.slider("Selecciona un cliente", 0, n_filas - 1, 0, key="slider_instancia")
        st.write(
            f"Predicción: **{df_result['label'].iloc[idx]}** — "
            f"Probabilidad: **{df_result['probabilidad'].iloc[idx]:.3f}**"
        )

        csv_out = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar resultados",
            data=csv_out,
            file_name="predicciones.csv",
            mime="text/csv",
        )


# ============================================================
# ROUTER
# ============================================================

if section == "Resumen ejecutivo":
    show_summary()
elif section == "Sprint 1 — Widgets exploratorios":
    show_sprint1_widgets()
elif section == "Sprint 2 — Data Preparation":
    show_sprint2_data_prep()
elif section == "Sprint 3 — Baselines y métricas":
    show_sprint3_baselines()
elif section == "Sprint 4 — Experiment Tracker + MLflow":
    show_sprint4_mlflow()
elif section == "Sprint 5 — Business Value":
    show_sprint5_business_value()
elif section == "Sprint 6 — API/AWS":
    show_sprint6_api()
elif section == "Predicción en vivo":
    show_live_prediction()
