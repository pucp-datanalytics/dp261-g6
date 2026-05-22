# KPIs y SLA objetivo — Sprint 6

## KPIs técnicos mínimos

- ROC-AUC mínimo esperado: 0.75
- Recall objetivo con umbral de negocio: mayor a 0.70
- F1-score de referencia: mayor a 0.30 con umbral optimizado
- Modelo capaz de generar probabilidad de suscripción

## KPIs de negocio

- Priorizar clientes con mayor probabilidad de suscripción.
- Reducir llamadas improductivas frente a selección aleatoria.
- Aumentar captura de clientes interesados en los primeros segmentos contactados.
- Validar impacto económico mediante campaña piloto.

## SLA inicial del MVP

- Tiempo de respuesta por predicción: menor a 1 segundo.
- Entrada: CSV o JSON.
- Salida: predicción, probabilidad y recomendación de contacto.
- Uso inicial: piloto controlado, no producción masiva.