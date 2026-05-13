"""
src/evaluation.py

Sprint 5 — Business Value Evaluation

Funciones reutilizables para:
1. Calcular matriz costo-beneficio.
2. Buscar umbral óptimo de decisión.
3. Calcular gain curve.
4. Calcular lift curve.
5. Estimar impacto económico mensual/anual.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def business_value_at_threshold(
    y_true,
    y_proba,
    threshold,
    benefit_tp=100,
    cost_fp=-10,
    cost_fn=-80,
    benefit_tn=0
):
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    valor_total = (
        tp * benefit_tp +
        fp * cost_fp +
        fn * cost_fn +
        tn * benefit_tn
    )

    return {
        "threshold": threshold,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "valor_total": valor_total,
        "valor_por_cliente": valor_total / len(y_true),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }


def find_optimal_threshold(
    y_true,
    y_proba,
    thresholds=None,
    benefit_tp=100,
    cost_fp=-10,
    cost_fn=-80,
    benefit_tn=0
):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.01)

    results = []

    for threshold in thresholds:
        result = business_value_at_threshold(
            y_true=y_true,
            y_proba=y_proba,
            threshold=threshold,
            benefit_tp=benefit_tp,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
            benefit_tn=benefit_tn
        )
        results.append(result)

    df_results = pd.DataFrame(results)
    best_row = df_results.loc[df_results["valor_total"].idxmax()]

    return df_results, best_row


def gain_curve_data(y_true, y_proba):
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)

    order = np.argsort(-y_proba)
    y_sorted = y_true[order]

    cum_pos = np.cumsum(y_sorted)
    total_pos = y_true.sum()

    population_pct = np.arange(1, len(y_true) + 1) / len(y_true)
    gain = cum_pos / total_pos

    return pd.DataFrame({
        "population_pct": population_pct,
        "gain": gain
    })


def lift_curve_data(y_true, y_proba):
    gain_df = gain_curve_data(y_true, y_proba)
    gain_df["lift"] = gain_df["gain"] / gain_df["population_pct"]

    return gain_df


def estimate_annual_impact(valor_por_cliente, clientes_mes=100000, meses=12):
    impacto_mensual = valor_por_cliente * clientes_mes
    impacto_anual = impacto_mensual * meses

    return {
        "valor_por_cliente": valor_por_cliente,
        "clientes_mes": clientes_mes,
        "meses": meses,
        "impacto_mensual_estimado": impacto_mensual,
        "impacto_anual_estimado": impacto_anual
    }


def technical_metrics_at_threshold(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba)
    }