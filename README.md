# 🤖 ML-Algorithms – Análisis Completo de Machine Learning

## 📋 Descripción breve

Sistema completo de análisis de Machine Learning para predecir el éxito de cafeterías usando múltiples algoritmos. Implementa los 6 pasos principales del desarrollo de modelos ML y compara el rendimiento de diferentes algoritmos automáticamente.

## 🎯 Objetivo

Predecir si una cafetería es "exitosa" (Daily_Revenue ≥ $2,000) o "no exitosa" (Daily_Revenue < $2,000) basándose en sus métricas operacionales (clientes, ingresos, empleados, marketing, etc.) comparando 5 algoritmos diferentes de ML:

- **Regresión Logística**
- **Máquinas de Vector de Soporte (SVM)**
- **Árboles de Decisión**
- **Random Forest**
- **Redes Neuronales Artificiales (MLP)**

## 🚀 Levantamiento rápido

### 1. Prerrequisitos

- Python 3.8+
- pip actualizado

### 2. Instalación

```bash
cd ML-Algorithms
python -m venv ml_env

# Activar entorno virtual
# Linux/Mac
source ml_env/bin/activate
# Windows (PowerShell)
ml_env\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Ejecución

**Análisis completo (todos los algoritmos):**

```bash
python ml_analysis.py
```

**Análisis rápido (sin optimización de hiperparámetros):**

```bash
python ml_analysis.py --quick-mode
```

**Algoritmos específicos:**

```bash
python ml_analysis.py --algorithms LogisticRegression SVM DecisionTree RandomForest NeuralNetwork
```

**Opciones adicionales:**

```bash
python ml_analysis.py --help
```

### 4. Parámetros de Ejecución Completos

#### Parámetros Principales

| Parámetro       | Descripción                                     | Valores por Defecto            | Ejemplo                       |
| --------------- | ----------------------------------------------- | ------------------------------ | ----------------------------- |
| `--data-path`   | Ruta al archivo CSV de datos                    | `data/coffee_shop_revenue.csv` | `--data-path mi_datos.csv`    |
| `--sample-size` | Tamaño de muestra para análisis rápido          | `None` (usar todo el dataset)  | `--sample-size 1000`          |
| `--output-dir`  | Directorio donde guardar resultados             | `results`                      | `--output-dir mis_resultados` |
| `--skip-viz`    | Saltar generación de visualizaciones            | `False`                        | `--skip-viz`                  |
| `--quick-mode`  | Modo rápido sin optimización de hiperparámetros | `False`                        | `--quick-mode`                |
| `--retrain`     | Forzar reentrenamiento (ignorar caché)          | `False`                        | `--retrain`                   |

#### Selección de Algoritmos

| Parámetro      | Descripción                    | Algoritmos Disponibles                                                       |
| -------------- | ------------------------------ | ---------------------------------------------------------------------------- |
| `--algorithms` | Lista de algoritmos a ejecutar | `LogisticRegression`, `SVM`, `DecisionTree`, `RandomForest`, `NeuralNetwork` |

**Ejemplos de uso:**

```bash
# Solo algoritmos específicos
python ml_analysis.py --algorithms NeuralNetwork RandomForest

# Análisis rápido con muestra pequeña
python ml_analysis.py --quick-mode --sample-size 500

# Reentrenar todos los modelos
python ml_analysis.py --retrain

# Sin visualizaciones (más rápido)
python ml_analysis.py --skip-viz

# Directorio personalizado
python ml_analysis.py --output-dir experimento_1
```

#### Configuración Avanzada

El sistema utiliza configuración por defecto optimizada:

- **División de datos**: 80% entrenamiento, 20% prueba (estratificada)
- **Validación cruzada**: 5 folds estratificados
- **Semilla aleatoria**: 42 (reproducibilidad)
- **Métrica de optimización**: Accuracy
- **Escalado**: StandardScaler para normalización
- **Estratificación**: Mantiene proporción de clases

#### Modos de Ejecución

1. **Modo Completo** (`python ml_analysis.py`):

   - Optimización de hiperparámetros con Grid Search
   - Generación completa de visualizaciones
   - Tiempo estimado: 15-20 minutos

2. **Modo Rápido** (`--quick-mode`):

   - Sin optimización de hiperparámetros
   - Hiperparámetros por defecto
   - Tiempo estimado: 5-10 minutos

3. **Modo de Muestra** (`--sample-size N`):
   - Usa solo N registros del dataset
   - Ideal para pruebas rápidas
   - Tiempo estimado: 2-5 minutos

Tiempo estimado: 5–20 minutos (según configuración y algoritmos seleccionados).

## 📊 Resultados Actuales

### Ranking de Algoritmos (Última Ejecución)

| Posición | Algoritmo               | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Tiempo |
| -------- | ----------------------- | -------- | --------- | ------ | -------- | ------- | ------ |
| 🥇 1     | **Neural Network**      | 95.25%   | 95.34%    | 95.25% | 95.26%   | 99.31%  | 0.36s  |
| 🥈 2     | **Random Forest**       | 95.00%   | 95.04%    | 95.00% | 95.01%   | 99.09%  | 3.48s  |
| 🥉 3     | **SVM**                 | 94.75%   | 94.78%    | 94.75% | 94.76%   | 99.21%  | 1.77s  |
| 4        | **Logistic Regression** | 93.00%   | 93.24%    | 93.00% | 93.03%   | 98.84%  | 0.14s  |
| 5        | **Decision Tree**       | 85.50%   | 86.24%    | 85.50% | 85.61%   | 88.60%  | 0.06s  |

### Características del Dataset

- **Tamaño**: 2,000 registros de cafeterías
- **Características**: 6 variables operativas
- **División**: 80% entrenamiento (1,600) / 20% prueba (400)
- **Balance**: 59.5% No Exitosas / 40.5% Exitosas
- **Umbral de éxito**: Daily_Revenue ≥ $2,000

### Métricas de Rendimiento

- **Mejor modelo**: Neural Network con 95.25% de accuracy
- **Más rápido**: Decision Tree (0.06s) pero menor rendimiento
- **Balance óptimo**: SVM con buen rendimiento (94.75%) y tiempo moderado
- **Todos los modelos superan el 85% de accuracy**

## 📂 Resultados al ejecutar

### Estructura de resultados en `results/`

```
results/
├── algorithm_comparison_report.json     # Comparación completa de todos los algoritmos
├── correlation_matrix.png               # Análisis de correlaciones del dataset
├── comparisons/                         # Visualizaciones comparativas
│   ├── metrics_comparison.png          # Comparación de métricas por algoritmo
│   ├── rankings_heatmap.png            # Heatmap de rankings
│   ├── radar_comparison.png            # Gráfico radar multidimensional
│   └── time_vs_accuracy.png            # Tiempo vs precisión
├── logisticregression/                  # Resultados de Regresión Logística
│   ├── logisticregression_model.pkl
│   ├── coefficients.png                # Visualización de coeficientes
│   └── ...
├── svm/                                 # Resultados de SVM
│   ├── svm_model.pkl
│   ├── decision_boundary.png           # Frontera de decisión (2D)
│   └── ...
├── decisiontree/                        # Resultados de Árbol de Decisión
│   ├── decisiontree_model.pkl
│   ├── tree_visualization.png          # Visualización del árbol
│   ├── feature_importance.png
│   └── ...
├── randomforest/                        # Resultados de Random Forest
│   ├── randomforest_model.pkl
│   ├── feature_importance.png
│   ├── trees_depth_distribution.png
│   └── ...
└── neuralnetwork/                       # Resultados de Red Neuronal
    ├── neuralnetwork_model.pkl
    ├── training_curves.png             # Curvas de pérdida y validación
    ├── network_architecture.png        # Visualización de la arquitectura
    └── ...

## 📚 Documentación

- Análisis e interpretación de resultados: `docs/ANALISIS_RESULTADOS.md`

## 🧭 Estructura mínima del proyecto

```

ML-Algorithms/
├── src/
├── data/
├── results/
└── docs/

```

## 📝 Notas

- Ejecuta `python ml_analysis.py --help` para ver opciones de ejecución (algoritmos, modo rápido, rutas).
```
