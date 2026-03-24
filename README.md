Taller 2 – Clasificación de Éxito en Cafeterías
Descripción del Proyecto

Este proyecto tiene como objetivo aplicar técnicas de aprendizaje automático para predecir el éxito financiero de cafeterías, clasificándolas en dos categorías: exitosas y no exitosas, en función de sus ingresos diarios.

Se utilizaron distintos algoritmos de clasificación supervisada con el fin de comparar su desempeño e identificar el modelo más adecuado para este problema.

Dataset

El proyecto utiliza el dataset Coffee Shop Daily Revenue, el cual contiene información operativa de cafeterías.

Características principales:
Número de clientes por día
Valor promedio de orden
Horas de operación
Número de empleados
Gasto en marketing
Tráfico de la ubicación
Ingresos diarios (variable objetivo)
Clasificación:
Exitosa: ingresos ≥ 2000
No exitosa: ingresos < 2000
Metodología

El desarrollo del proyecto se realizó en las siguientes etapas:

Exploración de datos: análisis de estructura, distribución y correlaciones
Preprocesamiento: limpieza, escalado de variables y transformación del target
División de datos: entrenamiento (80%) y prueba (20%)
Entrenamiento de modelos: implementación de algoritmos de clasificación
Evaluación: uso de métricas estándar para comparar desempeño
Modelos Implementados

Se entrenaron y evaluaron los siguientes modelos:

Regresión Logística
Máquinas de Soporte Vectorial (SVM)
Árbol de Decisión
Random Forest
Red Neuronal (MLP)
Métricas de Evaluación

Para evaluar el rendimiento de los modelos se utilizaron:

Accuracy
Precision
Recall
F1-score
AUC-ROC
Resultados
Modelo	Accuracy
Red Neuronal (MLP)	95.25%
Random Forest	95.00%
SVM	94.75%
Regresión Logística	93.00%
Árbol de Decisión	85.50%

La Red Neuronal (MLP) presentó el mejor desempeño, evidenciando su capacidad para modelar relaciones no lineales complejas.

Conclusiones
Los modelos no lineales superan a los modelos lineales en este problema
La Red Neuronal fue el modelo más preciso
El número de clientes es la variable más influyente
El problema presenta relaciones no lineales entre variables
Estructura del Proyecto
ML-Algorithms/
│
├── data/
│   └── coffee_shop_revenue.csv
│
├── src/
│   ├── models/
│   ├── data_processing/
│   ├── evaluation/
│   └── utils/
│
├── results/
│   ├── svm/
│   ├── randomforest/
│   ├── neuralnetwork/
│   ├── logisticregression/
│   └── decisiontree/
│
├── ml_analysis.py
├── requirements.txt
└── README.md
Requisitos

El proyecto fue desarrollado en Python 3.11+.

Dependencias principales:

pandas
numpy
scikit-learn
matplotlib
seaborn
scipy
joblib
Ejecución
Clonar el repositorio:
git clone https://github.com/RyuWilliam/taller-2.git
Instalar dependencias:
pip install -r requirements.txt
Ejecutar el script principal:
python ml_analysis.py
