# KPIs, Targets y SLA de Despliegue — Sprint 6

## 📈 KPIs Analíticos (Performance Real del Modelo Final)
El modelo final implementado corresponde a un ensamble de **BaggingClassifier (Estimador Base: Random Forest)**, validado formalmente con las métricas reales del informe escrito para evitar discrepancias:

* **Recall (KPI Principal):** 0.5254 (Intervalo de Confianza al 95%: [0.4921 - 0.5569])
* **Precisión:** 0.4929
* **F1-Score de Referencia:** 0.5086
* **Capacidad:** Clasificación binaria con estimación probabilística para la optimización de estrategias comerciales bajo umbrales de negocio.

## ⏱️ Acuerdos de Nivel de Servicio (SLA Técnicos del MVP)
* **Tiempo de Respuesta (Latencia de Predicción):** < 1.0 segundo por solicitud individual (Single Request).
* **Formatos Soportados:** Carga por lotes mediante archivos tabulares (CSV) e inferencia individual en tiempo real (JSON).
* **Campos del Contrato de Salida:** Predicción binaria [0, 1], probabilidad calculada y recomendación de priorización de contacto.
* **Uso sugerido:** Fase piloto controlada para la validación del impacto económico frente al baseline histórico (no apto para producción masiva de alta concurrencia aún).
