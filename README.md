# Taller 2 – Clasificación de Éxito en Cafeterías

## Descripción del Proyecto

Este proyecto tiene como objetivo aplicar técnicas de aprendizaje automático para predecir el éxito financiero de cafeterías, clasificándolas en dos categorías: **exitosas** y **no exitosas**, en función de sus ingresos diarios.

Se implementan distintos algoritmos de clasificación supervisada con el fin de comparar su desempeño e identificar el modelo más adecuado para este problema.

---

## Dataset

Se utiliza el dataset **Coffee Shop Daily Revenue**, que contiene información operativa de cafeterías.

### Características

| Variable | Descripción |
|---------|------------|
| Number_of_Customers_Per_Day | Número de clientes diarios |
| Average_Order_Value | Valor promedio de orden |
| Operating_Hours_Per_Day | Horas de operación |
| Number_of_Employees | Número de empleados |
| Marketing_Spend_Per_Day | Gasto en marketing |
| Location_Foot_Traffic | Tráfico de ubicación |
| Daily_Revenue | Ingresos diarios (target) |

### Clasificación

| Categoría | Condición |
|----------|----------|
| Exitosa | Ingresos ≥ 2000 |
| No exitosa | Ingresos < 2000 |

---

## Metodología

El desarrollo del proyecto se llevó a cabo en las siguientes etapas:

1. Exploración de datos  
2. Preprocesamiento (limpieza y escalado)  
3. División de datos (80% entrenamiento, 20% prueba)  
4. Entrenamiento de modelos  
5. Evaluación y comparación  

---

## Modelos Implementados

- Regresión Logística  
- Máquinas de Soporte Vectorial (SVM)  
- Árbol de Decisión  
- Random Forest  
- Red Neuronal (MLP)  

---

## Métricas de Evaluación

Se utilizaron las siguientes métricas:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- AUC-ROC  

---

## Resultados

| Modelo | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| Red Neuronal (MLP) | 95.25% | 95.30% | 95.20% | 95.25% |
| Random Forest | 95.00% | 95.10% | 94.90% | 95.00% |
| SVM | 94.75% | 94.80% | 94.70% | 94.75% |
| Regresión Logística | 93.00% | 93.10% | 92.90% | 93.00% |
| Árbol de Decisión | 85.50% | 86.00% | 85.00% | 85.40% |

---

## Conclusiones

- Los modelos no lineales presentan mejor desempeño  
- La Red Neuronal (MLP) fue el modelo más preciso  
- El número de clientes es la variable más influyente  
- El problema presenta relaciones no lineales  

---

## Estructura del Proyecto
