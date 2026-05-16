# Paquete de Handoff Integral — Grupo 6

Este repositorio y este directorio formalizan el contrato de entrega técnica para el pase a producción del modelo de Bank Marketing de la PUCP.

## 🔄 Fase 1: Transición Sprint 5 → Sprint 6 (Modelo y Negocio)
* **Objetivo:** Iniciar la etapa de despliegue sin bloqueos técnicos analíticos.
* **Artefactos analíticos iniciales incorporados:**
  * `models/final_model.pkl`: Modelo final validado.
  * `notebooks/15_business_value.ipynb`: Evaluación del valor de negocio.
  * `reports/business_value_summary.csv`: Resumen económico del proyecto.
  * `reports/threshold_business_value.csv`: Evaluación de umbrales comerciales.
  * `reports/gain_curve.csv`: Datos calculados para la curva de ganancia.
  * `reports/lift_curve.csv`: Datos calculados para la curva lift.
  * `reports/business_value_sensitivity.csv`: Análisis de escenarios de sensibilidad financiera.

## 🚀 Fase 2: Cierre Sprint 6 (Contrato de Despliegue y MLOps)
Cumpliendo rigurosamente las pautas de ingeniería de software exigidas por el jurado, este paquete habilita el consumo robusto y tipado de los artefactos predictivos.

### 🛠️ Instrucciones de Inicialización Local
1. Asegúrese de activar su entorno virtual e instalar las dependencias requeridas en producción:
   ```bash
   pip install -r requirements.txt
