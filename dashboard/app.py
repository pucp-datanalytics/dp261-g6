import os
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
MAX_ROWS = 500

st.set_page_config(page_title="Bank Marketing Dashboard", layout="wide")
st.title("Bank Marketing — Model Evaluation Dashboard")

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def get_headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["x-api-key"] = API_KEY
    return h

def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", headers=get_headers(), timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def safe_int(val, default=0):
    """Convierte float 0.0/1.0 a int sin perder valores negativos."""
    try:
        return int(val)
    except Exception:
        return default

def safe_float(val, default=0.0):
    """Convierte a float de forma segura."""
    try:
        return float(val)
    except Exception:
        return default

def call_predict(row: dict) -> dict:
    payload = {
        "age":               safe_float(row.get("age")),
        "education":         safe_float(row.get("education")),
        "campaign":          safe_float(row.get("campaign")),
        "pdays":             safe_float(row.get("pdays")),
        "previous":          safe_float(row.get("previous")),
        "emp_var_rate":      safe_float(row.get("emp.var.rate")),
        "cons_price_idx":    safe_float(row.get("cons.price.idx")),
        "cons_conf_idx":     safe_float(row.get("cons.conf.idx")),
        "euribor3m":         safe_float(row.get("euribor3m")),
        "nr_employed":       safe_float(row.get("nr.employed")),
        "job_blue_collar":   safe_int(row.get("job_blue-collar", 0)),
        "job_entrepreneur":  safe_int(row.get("job_entrepreneur", 0)),
        "job_housemaid":     safe_int(row.get("job_housemaid", 0)),
        "job_management":    safe_int(row.get("job_management", 0)),
        "job_retired":       safe_int(row.get("job_retired", 0)),
        "job_self_employed": safe_int(row.get("job_self-employed", 0)),
        "job_services":      safe_int(row.get("job_services", 0)),
        "job_student":       safe_int(row.get("job_student", 0)),
        "job_technician":    safe_int(row.get("job_technician", 0)),
        "job_unemployed":    safe_int(row.get("job_unemployed", 0)),
        "marital_married":   safe_int(row.get("marital_married", 0)),
        "marital_single":    safe_int(row.get("marital_single", 0)),
        "default_yes":       safe_int(row.get("default_yes", 0)),
        "housing_yes":       safe_int(row.get("housing_yes", 0)),
        "loan_yes":          safe_int(row.get("loan_yes", 0)),
        "contact_telephone": safe_int(row.get("contact_telephone", 0)),
        "month_aug":         safe_int(row.get("month_aug", 0)),
        "month_dec":         safe_int(row.get("month_dec", 0)),
        "month_jul":         safe_int(row.get("month_jul", 0)),
        "month_jun":         safe_int(row.get("month_jun", 0)),
        "month_mar":         safe_int(row.get("month_mar", 0)),
        "month_may":         safe_int(row.get("month_may", 0)),
        "month_nov":         safe_int(row.get("month_nov", 0)),
        "month_oct":         safe_int(row.get("month_oct", 0)),
        "month_sep":         safe_int(row.get("month_sep", 0)),
        "day_of_week_mon":   safe_int(row.get("day_of_week_mon", 0)),
        "day_of_week_thu":   safe_int(row.get("day_of_week_thu", 0)),
        "day_of_week_tue":   safe_int(row.get("day_of_week_tue", 0)),
        "day_of_week_wed":   safe_int(row.get("day_of_week_wed", 0)),
        "poutcome_nonexistent": safe_int(row.get("poutcome_nonexistent", 0)),
        "poutcome_success":  safe_int(row.get("poutcome_success", 0)),
        "contacted_before":  safe_int(row.get("contacted_before", 0)),
        "campaign_intensity": safe_float(row.get("campaign_intensity", 0.0)),
        "has_loan_or_housing": safe_int(row.get("has_loan_or_housing", 0)),
    }
    r = requests.post(
        f"{API_URL}/predict",
        json=payload,
        headers=get_headers(),
        timeout=30
    )
    r.raise_for_status()
    return r.json()

def run_predictions(df, n_filas):
    results = []
    errores = 0
    primer_error = None
    prog = st.progress(0)
    for i, (_, row) in enumerate(df.head(n_filas).iterrows()):
        try:
            res = call_predict(row.to_dict())
            results.append(res)
        except Exception as e:
            errores += 1
            if primer_error is None:
                primer_error = str(e)
            results.append({"prediction": 0, "probability": 0.0, "label": "Error"})
        prog.progress((i + 1) / n_filas)
    if primer_error:
        st.error(f"🔍 Primer error encontrado: `{primer_error}`")
    return results, errores

def show_metrics_and_charts(y_true, y_pred, y_proba):
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy",  f"{accuracy_score(y_true, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
    col3.metric("Recall",    f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
    col4.metric("F1-Score",  f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
    col5.metric("AUC-ROC",   f"{roc_auc_score(y_true, y_proba):.3f}")

    st.subheader("Confusion Matrix y Curva ROC")
    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=['No', 'Yes'],
            cmap='Blues', colorbar=False, ax=ax
        )
        st.pyplot(fig)
    with col_b:
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name='Final Model (API)')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        st.pyplot(fig)

# ──────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────
with st.spinner("Verificando conexión con la API..."):
    api_ok = check_api_health()

if not api_ok:
    st.error(
        f"❌ No se pudo conectar con la API en `{API_URL}`.\n\n"
        "Asegúrate de que la API esté corriendo con:\n"
        "`uvicorn api.main:app --reload --port 8000`"
    )
    st.stop()

st.success(f"✅ API conectada en `{API_URL}`")

try:
    ver = requests.get(f"{API_URL}/version", headers=get_headers(), timeout=5).json()
    st.caption(f"Modelo: `{ver.get('model')}` — versión `{ver.get('version')}`")
except Exception:
    pass

# ──────────────────────────────────────────────
# FUENTE DE DATOS
# ──────────────────────────────────────────────
@st.cache_data
def load_test_data():
    rutas = [
        'data/processed/test_original.csv',
        '../data/processed/test_original.csv',
        'test_original.csv',
    ]
    for ruta in rutas:
        try:
            test = pd.read_csv(ruta)
            return test, ruta
        except FileNotFoundError:
            continue
    return None, None

df_local, ruta_encontrada = load_test_data()

st.subheader("Fuente de datos")

if df_local is not None:
    st.success(f"✅ CSV local encontrado: `{ruta_encontrada}` ({len(df_local)} filas)")
    usar_local = st.radio(
        "¿Qué datos quieres usar?",
        ["📁 Usar CSV local automáticamente", "⬆️ Subir CSV manualmente"],
        horizontal=True
    )
else:
    st.info("ℹ️ No se encontró CSV local. Sube el archivo manualmente.")
    usar_local = "⬆️ Subir CSV manualmente"

if usar_local == "📁 Usar CSV local automáticamente" and df_local is not None:
    df_datos = df_local
    st.write(f"Usando CSV local: **{len(df_datos)} filas**")
else:
    uploaded = st.file_uploader(
        "Sube el CSV (con columna 'y' para ver métricas y gráficos)",
        type="csv"
    )
    if not uploaded:
        st.info("👆 Sube el `test_original.csv` para continuar.")
        st.stop()
    df_datos = pd.read_csv(uploaded)
    st.write(f"CSV cargado: **{len(df_datos)} filas**")

# ──────────────────────────────────────────────
# CONTROL DE FILAS
# ──────────────────────────────────────────────
tiene_target = 'y' in df_datos.columns
total_filas  = len(df_datos)

st.subheader("Configuración de predicción")
col_opt1, col_opt2 = st.columns(2)

with col_opt1:
    limitar = st.checkbox(
        "⚡ Limitar filas (recomendado para laptops lentas)",
        value=True
    )

with col_opt2:
    if limitar:
        n_filas = st.slider(
            "¿Cuántas filas procesar?",
            min_value=50,
            max_value=min(MAX_ROWS, total_filas),
            value=min(200, total_filas),
            step=50
        )
        st.caption(f"~{n_filas // 10}-{n_filas // 5} segundos estimados")
    else:
        n_filas = total_filas
        tiempo_est = total_filas // 10
        st.warning(f"⚠️ Se procesarán **{total_filas} filas** (~{tiempo_est//60} min {tiempo_est%60} seg). Solo recomendado en laptops potentes.")

st.write(f"Se procesarán: **{n_filas} filas**")

# ──────────────────────────────────────────────
# PREDICCIONES
# ──────────────────────────────────────────────
if st.button("🚀 Obtener predicciones"):

    if tiene_target:
        X_datos = df_datos.drop(columns='y')
        y_true  = df_datos['y'].head(n_filas).values
    else:
        X_datos = df_datos
        y_true  = None

    with st.spinner(f"Consultando API para {n_filas} filas..."):
        results, errores = run_predictions(X_datos, n_filas)

    if errores > 0:
        st.warning(f"⚠️ {errores} filas tuvieron errores.")
    else:
        st.success(f"✅ {n_filas} predicciones completadas sin errores.")

    y_pred  = np.array([r["prediction"] for r in results])
    y_proba = np.array([r["probability"] for r in results])

    # Métricas y gráficos
    if tiene_target:
        st.subheader("Métricas generales")
        show_metrics_and_charts(y_true, y_pred, y_proba)
    else:
        st.info("ℹ️ El CSV no tiene columna 'y' — solo se muestran predicciones.")

    # Tabla
    st.subheader("Tabla de predicciones")
    df_result = X_datos.head(n_filas).copy()
    if tiene_target:
        df_result['y_real'] = y_true
    df_result['y_pred']       = y_pred
    df_result['probabilidad'] = y_proba
    df_result['label']        = [r["label"] for r in results]
    st.dataframe(df_result)

    # Simulador
    st.subheader("Simulador de predicción por instancia")
    idx = st.slider("Selecciona un cliente", 0, n_filas - 1, 0)
    st.write(f"**Predicción:** {df_result['label'].iloc[idx]} — **Probabilidad:** {df_result['probabilidad'].iloc[idx]:.3f}")

    # Descarga
    csv_out = df_result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar resultados",
        data=csv_out,
        file_name="predicciones.csv",
        mime="text/csv"
    )
