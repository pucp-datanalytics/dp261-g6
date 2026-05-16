import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

st.set_page_config(page_title="Bank Marketing Dashboard", layout="wide")
st.title("Bank Marketing — Model Evaluation Dashboard")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('../models/preprocessor.pkl')
    model = joblib.load('../models/final_model.pkl')
    return preprocessor, model

@st.cache_data
def load_data():
    test = pd.read_csv('../data/processed/test_original.csv')
    X = test.drop(columns='y')
    y = test['y']
    return X, y

preprocessor, model = load_artifacts()
X_test, y_test = load_data()

# Transformación de datos de prueba
X_test_transformed = preprocessor.transform(X_test)
y_pred = model.predict(X_test_transformed)
y_proba = model.predict_proba(X_test_transformed)[:, 1]

# KPIs
st.subheader("Métricas generales")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.3f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
col3.metric("Recall",    f"{recall_score(y_test, y_pred):.3f}")
col4.metric("F1-Score",  f"{f1_score(y_test, y_pred):.3f}")
col5.metric("AUC-ROC",   f"{roc_auc_score(y_test, y_proba):.3f}")

# Confusion Matrix + ROC
st.subheader("Confusion Matrix y Curva ROC")
col_a, col_b = st.columns(2)

with col_a:
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=['No', 'Yes'],
        cmap='Blues', colorbar=False, ax=ax
    )
    st.pyplot(fig)

with col_b:
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name='Final Model (Bagging)')
    ax.plot([0,1],[0,1],'k--', lw=0.8)
    st.pyplot(fig)

# Simulador
st.subheader("Simulador de predicción por instancia")
idx = st.slider("Selecciona un cliente", 0, len(X_test)-1, 0)
instancia = X_test.iloc[[idx]]
instancia_t = preprocessor.transform(instancia)
pred = model.predict(instancia_t)[0]
prob = model.predict_proba(instancia_t)[0][1]

st.write(f"Predicción: {'Suscribe' if pred == 1 else 'No suscribe'} — Probabilidad: {prob:.3f}")

# SHAP (Módulo Blindado y Adaptado para Bagging)
st.subheader("Explicación SHAP")

@st.cache_resource
def get_explainer(_model, _X_sparse):
    # Convertimos la muestra de background a un array denso para evitar conflictos de Sparse Matrix
    if hasattr(_X_sparse, "toarray"):
        dense_bg = _X_sparse.toarray()
    else:
        dense_bg = np.array(_X_sparse)
        
    background_data = shap.sample(dense_bg, 50, random_state=42)
    # Usamos predict_proba para evaluar el impacto en la probabilidad final
    return shap.Explainer(_model.predict_proba, background_data)

try:
    # 1. Obtener explainer denso optimizado
    explainer = get_explainer(model, X_test_transformed)
    
    # 2. Asegurar que la instancia evaluada sea un array denso
    if hasattr(instancia_t, "toarray"):
        instancia_densa = instancia_t.toarray()
    else:
        instancia_densa = np.array(instancia_t)
        
    # 3. Calcular los valores SHAP
    shap_values = explainer(instancia_densa)
    
    # 4. Extraer los valores correspondientes a la clase 1 (Probabilidad de suscripción)
    if len(shap_values.shape) == 3:  
        shap_object = shap_values[0, :, 1]
    else:
        shap_object = shap_values[0]

    # Asignar nombres de las columnas para que el gráfico sea legible
    if hasattr(preprocessor, "get_feature_names_out"):
        features = preprocessor.get_feature_names_out()
    else:
        features = X_test.columns.tolist()
        
    shap_object.feature_names = features

    # 5. Dibujar e imprimir el gráfico de cascada
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_object, show=False)
    st.pyplot(fig)

except Exception as e:
    st.warning(f"SHAP está procesando la estructura de variables o no se encuentra disponible momentáneamente.")

# Tabla de predicciones (Siempre se ejecutará al estar fuera del try/except)
st.subheader("Tabla de predicciones")
df_pred = X_test.copy()
df_pred['y_real'] = y_test.values
df_pred['y_pred'] = y_pred
df_pred['probabilidad'] = y_proba
st.dataframe(df_pred.head(100))