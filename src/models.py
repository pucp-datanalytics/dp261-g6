"""
src/models.py

Rol 5: Experiment Tracker — Sprint 3

Este módulo contiene funciones reutilizables para:
1. Entrenar modelos baseline.
2. Persistir modelos entrenados con joblib.
3. Evaluar modelos con cross-validation.
4. Evaluar modelos sobre test.
5. Registrar experimentos en experiments_log.csv.
6. Cargar modelos persistidos.
7. Validar que un modelo guardado puede predecir.
8. Preparar el registro para Sprint 4.

Uso esperado:
- 09_baseline_models.ipynb
- 10_metrics_evaluator.ipynb
- 11_model_comparator.ipynb
"""

import os
import json
import joblib
import pandas as pd

from datetime import datetime
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. ENTRENAMIENTO Y PERSISTENCIA DE MODELOS BASELINE
# ============================================================

def train_baseline(pipe, X, y, name, models_dir="../models"):
    """
    Entrena y persiste un modelo baseline.

    Parameters
    ----------
    pipe : sklearn Pipeline o estimator
        Pipeline completo o modelo a entrenar.
        Idealmente debe incluir preprocesamiento + clasificador.

    X : pandas.DataFrame
        Variables predictoras.

    y : pandas.Series
        Variable objetivo.

    name : str
        Nombre corto del modelo.
        Ejemplos: "lr", "dt", "rf", "svm", "knn".

    models_dir : str
        Carpeta donde se guardará el modelo.

    Returns
    -------
    pipe : sklearn Pipeline o estimator
        Modelo entrenado.

    model_path : str
        Ruta del modelo guardado.
    """

    os.makedirs(models_dir, exist_ok=True)

    pipe.fit(X, y)

    model_path = os.path.join(models_dir, f"baseline_{name}.pkl")
    joblib.dump(pipe, model_path)

    return pipe, model_path


# ============================================================
# 2. EVALUACIÓN CON CROSS-VALIDATION
# ============================================================

def evaluate_model(pipe, X, y, cv, scoring):
    """
    Evalúa un modelo con cross-validation.

    Parameters
    ----------
    pipe : sklearn Pipeline o estimator
        Modelo o pipeline a evaluar.

    X : pandas.DataFrame
        Variables predictoras.

    y : pandas.Series
        Variable objetivo.

    cv : cross-validation splitter
        Ejemplo:
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring : dict or list
        Métricas de evaluación.
        Ejemplo:
        {
            "accuracy": "accuracy",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "roc_auc": "roc_auc"
        }

    Returns
    -------
    results : dict
        Resultado completo de cross_validate.

    metrics_summary : dict
        Promedio y desviación estándar de cada métrica.
    """

    results = cross_validate(
        pipe,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True
    )

    metrics_summary = {}

    if isinstance(scoring, dict):
        metric_names = scoring.keys()
    else:
        metric_names = scoring

    for metric_name in metric_names:

        test_key = f"test_{metric_name}"
        train_key = f"train_{metric_name}"

        metrics_summary[f"{metric_name}_cv_mean"] = results[test_key].mean()
        metrics_summary[f"{metric_name}_cv_std"] = results[test_key].std()

        if train_key in results:
            metrics_summary[f"{metric_name}_train_mean"] = results[train_key].mean()
            metrics_summary[f"{metric_name}_train_std"] = results[train_key].std()

    return results, metrics_summary


# ============================================================
# 3. EVALUACIÓN EN TEST SET
# ============================================================

def evaluate_on_test(model, X_test, y_test, positive_label=1):
    """
    Evalúa un modelo ya entrenado sobre el conjunto de test.

    Parameters
    ----------
    model : sklearn Pipeline o estimator
        Modelo entrenado.

    X_test : pandas.DataFrame
        Variables predictoras del test.

    y_test : pandas.Series
        Target real del test.

    positive_label : int or str
        Etiqueta de la clase positiva.
        Por defecto: 1.

    Returns
    -------
    metrics : dict
        Métricas principales en test.

    report : dict
        Classification report en formato diccionario.

    cm : array
        Matriz de confusión.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            pos_label=positive_label,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            pos_label=positive_label,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            pos_label=positive_label,
            zero_division=0
        )
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["roc_auc"] = None

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    return metrics, report, cm


# ============================================================
# 4. REGISTRO DE EXPERIMENTOS
# ============================================================

def log_experiment(
    name,
    params,
    metrics,
    path="../models/experiments_log.csv",
    model_path=None,
    selected=False,
    notes=""
):
    """
    Registra un experimento en un archivo CSV acumulativo.

    Parameters
    ----------
    name : str
        Nombre del modelo.
        Ejemplo: "Random Forest".

    params : dict
        Parámetros principales del modelo.
        Ejemplo: {"n_estimators": 100, "random_state": 42}

    metrics : dict
        Métricas obtenidas.
        Ejemplo: {"f1": 0.416, "recall": 0.393, "roc_auc": 0.88}

    path : str
        Ruta del CSV de experimentos.

    model_path : str
        Ruta del modelo persistido.

    selected : bool
        Indica si el modelo fue seleccionado para Sprint 4.

    notes : str
        Comentarios o justificación del experimento.

    Returns
    -------
    row : dict
        Registro insertado en el log.
    """

    log_dir = os.path.dirname(path)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": name,
        "params": json.dumps(params, ensure_ascii=False),
        "model_path": model_path,
        "selected": selected,
        "notes": notes,
        **metrics
    }

    df_row = pd.DataFrame([row])

    if os.path.exists(path):
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, index=False)

    return row


# ============================================================
# 5. CARGA DE MODELOS PERSISTIDOS
# ============================================================

def load_model(model_path):
    """
    Carga un modelo persistido con joblib.

    Parameters
    ----------
    model_path : str
        Ruta del archivo .pkl.

    Returns
    -------
    model : sklearn Pipeline o estimator
        Modelo cargado.
    """

    return joblib.load(model_path)


# ============================================================
# 6. VALIDACIÓN DE CARGA Y PREDICCIÓN
# ============================================================

def validate_model_load_and_predict(model_path, X_sample):
    """
    Valida que un modelo guardado puede cargarse y predecir.

    Parameters
    ----------
    model_path : str
        Ruta del modelo guardado.

    X_sample : pandas.DataFrame
        Muestra de datos para probar predicción.

    Returns
    -------
    preds : array
        Predicciones generadas por el modelo cargado.
    """

    model = load_model(model_path)
    preds = model.predict(X_sample)

    return preds


# ============================================================
# 7. ENTRENAR, EVALUAR Y REGISTRAR EN UN SOLO FLUJO
# ============================================================

def train_evaluate_log(
    pipe,
    X,
    y,
    name,
    params,
    cv,
    scoring,
    models_dir="../models",
    log_path="../models/experiments_log.csv",
    selected=False,
    notes=""
):
    """
    Ejecuta el flujo completo del Experiment Tracker:

    1. Entrena el modelo.
    2. Guarda el modelo con joblib.
    3. Evalúa con cross-validation.
    4. Registra el experimento en experiments_log.csv.

    Parameters
    ----------
    pipe : sklearn Pipeline o estimator
        Modelo o pipeline completo.

    X : pandas.DataFrame
        Variables predictoras.

    y : pandas.Series
        Variable objetivo.

    name : str
        Nombre corto del modelo.
        Ejemplo: "rf".

    params : dict
        Parámetros principales del experimento.

    cv : cross-validation splitter
        Objeto de validación cruzada.

    scoring : dict or list
        Métricas a evaluar.

    models_dir : str
        Carpeta donde se guardan los modelos.

    log_path : str
        Ruta del registro de experimentos.

    selected : bool
        Indica si el modelo se marca como candidato para Sprint 4.

    notes : str
        Comentarios o justificación.

    Returns
    -------
    fitted_pipe : estimator
        Modelo entrenado.

    metrics_summary : dict
        Métricas resumidas de cross-validation.

    model_path : str
        Ruta del modelo persistido.
    """

    fitted_pipe, model_path = train_baseline(
        pipe=pipe,
        X=X,
        y=y,
        name=name,
        models_dir=models_dir
    )

    _, metrics_summary = evaluate_model(
        pipe=fitted_pipe,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring
    )

    log_experiment(
        name=name,
        params=params,
        metrics=metrics_summary,
        path=log_path,
        model_path=model_path,
        selected=selected,
        notes=notes
    )

    return fitted_pipe, metrics_summary, model_path


# ============================================================
# 8. CREAR LOG DESDE TABLA COMPARATIVA EXISTENTE
# ============================================================

def create_experiments_log_from_results(
    df_results,
    path="../models/experiments_log.csv",
    selected_model="rf"
):
    """
    Crea experiments_log.csv desde una tabla comparativa existente.

    Esta función sirve si el equipo ya calculó las métricas en
    10_metrics_evaluator.ipynb o 11_model_comparator.ipynb y solo
    necesita formalizar el log.

    Admite columnas en minúsculas o nombres de presentación:
    - modelo / model
    - accuracy / Accuracy
    - precision / Precision
    - recall / Recall
    - f1 / F1-Score
    - roc_auc / AUC-ROC

    Parameters
    ----------
    df_results : pandas.DataFrame
        Tabla comparativa de modelos.

    path : str
        Ruta donde se guardará experiments_log.csv.

    selected_model : str
        Modelo seleccionado para Sprint 4.
        En tu caso puede ser: "rf" o "Random Forest".

    Returns
    -------
    df_log : pandas.DataFrame
        Registro de experimentos generado.
    """

    log_dir = os.path.dirname(path)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    rows = []

    for _, row in df_results.iterrows():

        model_name = (
            row.get("modelo")
            if "modelo" in row.index
            else row.get("model", row.get("Modelo"))
        )

        model_key = str(model_name).lower().replace(" ", "_")

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "params": "baseline_default",
            "model_path": f"../models/baseline_{model_key}.pkl",
            "accuracy": row.get("accuracy", row.get("Accuracy")),
            "precision": row.get("precision", row.get("Precision")),
            "recall": row.get("recall", row.get("Recall")),
            "f1": row.get("f1", row.get("F1-Score")),
            "roc_auc": row.get("roc_auc", row.get("AUC-ROC")),
            "selected": str(model_name).lower() == str(selected_model).lower(),
            "notes": (
                "Modelo seleccionado como candidato para Sprint 4."
                if str(model_name).lower() == str(selected_model).lower()
                else "Baseline evaluado en Sprint 3."
            )
        }

        rows.append(record)

    df_log = pd.DataFrame(rows)
    df_log.to_csv(path, index=False)

    return df_log


# ============================================================
# 9. VALIDAR VARIOS MODELOS GUARDADOS
# ============================================================

def validate_saved_models(model_paths, X_sample):
    """
    Valida que varios modelos guardados puedan cargarse y predecir.

    Parameters
    ----------
    model_paths : dict
        Diccionario con nombre y ruta de modelo.
        Ejemplo:
        {
            "lr": "../models/baseline_lr.pkl",
            "rf": "../models/baseline_rf.pkl"
        }

    X_sample : pandas.DataFrame
        Muestra para predicción.

    Returns
    -------
    validation_results : dict
        Diccionario con predicciones por modelo.
    """

    validation_results = {}

    for name, path in model_paths.items():
        model = load_model(path)
        preds = model.predict(X_sample)
        validation_results[name] = preds

    return validation_results