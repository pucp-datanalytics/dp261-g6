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
    model = joblib.load('../models/tuned_rf.pkl')
    return preprocessor, model

@st.cache_data
def load_data():
    test = pd.read_csv('../data/processed/test_original.csv')
    X = test.drop(columns='y')
    y = test['y']
    return X, y

preprocessor, model = load_artifacts()
X_test, y_test = load_data()

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
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name='Tuned RF')
    ax.plot([0,1],[0,1],'k--', lw=0.8)
    st.pyplot(fig)

# Simulador
st.subheader("Simulador de prediccion por instancia")
idx = st.slider("Selecciona un cliente", 0, len(X_test)-1, 0)
instancia = X_test.iloc[[idx]]
instancia_t = preprocessor.transform(instancia)
pred = model.predict(instancia_t)[0]
prob = model.predict_proba(instancia_t)[0][1]

st.write(f"Prediccion: {'Suscribe' if pred == 1 else 'No suscribe'} — Probabilidad: {prob:.3f}")

# SHAP
st.subheader("Explicacion SHAP")

@st.cache_resource
def get_explainer(_model, _X):
    return shap.TreeExplainer(_model)

try:
    explainer = get_explainer(model, X_test_transformed)
    shap_values = explainer.shap_values(instancia_t)
    
    fig, ax = plt.subplots()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0][0],
base_values=explainer.expected_value[0],
            data=instancia.values[0],
            feature_names=X_test.columns.tolist()
        ),
        show=False
    )
    st.pyplot(fig)
except Exception as e:
    st.warning(f"SHAP no disponible: {e}")

# Tabla
st.subheader("Tabla de predicciones")
df_pred = X_test.copy()
df_pred['y_real'] = y_test.values
df_pred['y_pred'] = y_pred
df_pred['probabilidad'] = y_proba
st.dataframe(df_pred.head(100))