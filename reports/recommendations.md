# Recomendaciones de negocio — Sprint 5

## 1. Contexto

El modelo final del Sprint 4 fue evaluado desde una perspectiva de negocio para campañas de telemarketing bancario. El objetivo fue traducir las métricas técnicas a impacto económico, usando una matriz costo–beneficio ajustada al contexto peruano.

## 2. Supuestos económicos

Se usaron supuestos en soles para aproximar el valor de una campaña local:

| Evento | Descripción | Valor estimado |
|---|---|---:|
| True Positive (TP) | Cliente interesado correctamente priorizado | S/ 60 |
| False Positive (FP) | Cliente no interesado contactado innecesariamente | -S/ 2.50 |
| False Negative (FN) | Cliente interesado no priorizado | -S/ 40 |
| True Negative (TN) | Cliente no interesado correctamente no contactado | S/ 0 |

Estos valores deben ser validados con el sponsor del proyecto antes de producción.

## 3. Resultados principales

Con el umbral estándar de 0.50, el modelo genera valor económico positivo. Sin embargo, el análisis de umbrales mostró que el valor esperado se maximiza con un umbral aproximado de 0.05.

Con el umbral óptimo se obtuvo:

| Indicador | Resultado |
|---|---:|
| Umbral óptimo | 0.05 |
| Valor económico total en test | S/ 36,550 |
| Valor por cliente | S/ 4.44 |
| Precision | 0.1149 |
| Recall | 0.9838 |
| F1-score | 0.2057 |

## 4. Interpretación

El umbral óptimo reduce la precisión, pero aumenta significativamente el recall. Esto es coherente con el negocio, porque en campañas de telemarketing bancario puede ser preferible contactar más clientes potenciales si el costo de llamada es bajo frente al costo de oportunidad de perder un cliente interesado.

La Gain Curve y la Lift Curve muestran que el modelo supera a una estrategia aleatoria, especialmente en los primeros segmentos priorizados.

## 5. Impacto económico estimado

Bajo el escenario base Perú y asumiendo 100,000 clientes evaluados por mes, el impacto estimado fue:

| Indicador | Resultado |
|---|---:|
| Impacto mensual estimado | S/ 443,891.18 |
| Impacto anual estimado | S/ 5,326,694.19 |

Estos valores son referenciales y dependen de la validación real de costos, ticket promedio y tasa de conversión efectiva.

## 6. Recomendación

Se recomienda avanzar al Sprint 6 como MVP controlado.

El modelo no debe reemplazar la decisión comercial humana en una primera etapa. Debe utilizarse como herramienta de priorización para ordenar clientes por probabilidad de suscripción y apoyar al equipo de telemarketing.

## 7. Condiciones para pasar a Sprint 6

Antes del despliegue productivo, se recomienda:

1. Validar los supuestos económicos con el sponsor.
2. Ejecutar una campaña piloto con grupo de control.
3. Medir conversión real frente a estrategia tradicional.
4. Monitorear precision, recall, lift y valor económico real.
5. Ajustar el umbral según resultados de negocio.