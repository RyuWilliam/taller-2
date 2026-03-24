# 📖 Índice Completo de Refactorización - ML-Algorithms

## 🎯 Contenido

Este documento proporciona una guía completa de todos los cambios realizados durante la refactorización del proyecto ML-Algorithms.

---

## 📚 Documentos Principales

### 1. **REFACTORING.md** - Guía de Cambios Principales
- Resumen de todos los nuevos módulos
- Mejoras realizadas
- Beneficios de la refactorización
- Estructura mejorada del proyecto

**Leer cuando:** Necesitas entender qué cambió y por qué

---

### 2. **STYLE_GUIDE.md** - Guía de Estilo y Convenciones
- Convenciones de código Python
- Formato de docstrings
- Manejo de errores
- Logging y debugging
- Validación de entrada

**Leer cuando:** Vas a escribir o modificar código

---

## 🏗️ Módulos Nuevos

### Utilidades (`src/utils/`)

#### `constants.py` ⭐ NEW
**Propósito:** Centralizar todos los valores mágicos

```python
from src.utils.constants import (
    DEFAULT_RANDOM_STATE,       # 42
    DEFAULT_CV_FOLDS,           # 5
    DEFAULT_TEST_SIZE,          # 0.2
    CORRELATION_THRESHOLD,      # 0.3
    DEFAULT_SUCCESS_THRESHOLD,  # 2000.0
)
```

**Contiene:**
- Constantes numéricas (DEFAULT_RANDOM_STATE, DEFAULT_CV_FOLDS, etc.)
- Configuración de rutas y archivos
- Nombres de columnas y clases
- Validaciones y umbrales
- Enums seguros (AlgorithmName, MetricName, etc.)

**Uso:**
```python
from src.utils.constants import AlgorithmName
if algo == AlgorithmName.LOGISTIC_REGRESSION:
    ...
```

---

#### `exceptions.py` ⭐ NEW
**Propósito:** Excepciones personalizadas y consistentes

```python
from src.utils.exceptions import (
    DataError,
    ModelError,
    ConfigError,
    InvalidModelStateError,
)
```

**Jerarquía:**
```
MLProjectError (base)
├── DataError
│   ├── DataValidationError
│   ├── DataNotFoundError
│   └── PreprocessingError
├── ModelError
│   ├── ModelTrainingError
│   ├── InvalidModelStateError
│   └── HyperparameterError
├── ConfigError
│   └── ConfigValidationError
└── ...
```

**Uso:**
```python
try:
    model.predict(X)
except InvalidModelStateError as e:
    logger.error(f"Modelo no entrenado: {e}")
```

---

#### `validators.py` ⭐ NEW
**Propósito:** Validación reutilizable de datos y configuración

**Clases:**
- `DataValidator` - Valida DataFrames, arrays, features
- `ConfigValidator` - Valida configuración
- `HyperparameterValidator` - Valida parámetros de modelos
- `PathValidator` - Valida archivos y directorios

```python
from src.utils import DataValidator

# Validar datos
DataValidator.validate_dataframe(df, min_rows=100)
DataValidator.validate_features(X)
DataValidator.validate_target(y)
DataValidator.validate_feature_names(names, n_features=10)
```

**Beneficio:** Validación centralizada, no repetida en múltiples lugares

---

#### `helpers.py` ⭐ NEW
**Propósito:** Utilidades comunes en un solo lugar

**Clases:**
- `FileUtils` - Operaciones con archivos JSON, CSV
- `StringUtils` - Formateo de strings
- `ArrayUtils` - Operaciones NumPy comunes
- `TimingUtils` - Timing y benchmarking
- `DictUtils` - Operaciones con diccionarios

```python
from src.utils.helpers import FileUtils, StringUtils, TimingUtils

# Guardar/cargar JSON
FileUtils.save_json(data, "results/report.json")
data = FileUtils.load_json("results/report.json")

# Formatear valores
StringUtils.format_metric(0.9545, decimals=4)
TimingUtils.format_duration(125)  # "2m 5s"

# Operaciones NumPy
ArrayUtils.get_class_distribution(y)
ArrayUtils.get_class_weights(y)
```

---

### Core Modules

#### `types.py` ⭐ NEW
**Propósito:** Definiciones de tipos y protocolos

```python
from src.types import (
    FeatureMatrix,    # NDArray[np.floating]
    TargetVector,     # Union[NDArray, pd.Series]
    MetricsDict,      # Dict[str, Union[int, float]]
    Classifier,       # Protocolo
    Pipeline,         # Protocolo
)

def train(X: FeatureMatrix, y: TargetVector) -> MetricsDict:
    pass
```

**Beneficios:**
- Type hints más claros
- Protocolos para estructuras comunes
- Documentación implícita

---

#### `decorators.py` ⭐ NEW
**Propósito:** Decoradores reutilizables

```python
from src.decorators import (
    timer,                  # Mide tiempo de ejecución
    log_execution,          # Registra ejecución
    requires_trained,       # Verifica modelo entrenado
    memoize,                # Cachea resultados
    retry,                  # Reintentos automáticos
    deprecated,             # Marca como deprecado
    logged_and_timed,       # Combinación común
)

@timer
@log_execution("INFO")
def expensive_function():
    pass

class Classifier:
    @requires_trained
    def predict(self, X):
        return self.model.predict(X)
```

---

#### `pipeline.py` ⭐ NEW
**Propósito:** Orquestar el pipeline completo de ML

```python
from src.pipeline import MLPipeline, PipelineConfig

# Crear pipeline
pipeline = MLPipeline(config)

# Configurar ejecución
config = PipelineConfig(
    data_path="data/coffee_shop_revenue.csv",
    sample_size=1000,
    output_dir="results",
    skip_viz=False,
    quick_mode=False,
    algorithms=["LogisticRegression", "SVM"]
)

# Ejecutar
results = pipeline.run(config)
```

**Pasos orchestrados:**
1. Cargar datos
2. Elegir métricas
3. Establecer protocolo
4. Preparar datos
5. Análisis exploratorio
6. Entrenar modelos
7. Comparar modelos

---

## 🔧 Módulos Refactorizados

### `src/models/base_classifier.py`

**Mejoras:**
- ✅ Validación mejorada de inputs
- ✅ Mejor manejo de errores con excepciones específicas
- ✅ Type hints completos
- ✅ Docstrings detallados
- ✅ Métodos mejorados: `train()`, `predict()`, `save_model()`, `load_model()`

**Ejemplo de mejora:**

```python
# ANTES
def predict(self, X):
    if not self.is_trained:
        raise ValueError("El modelo debe ser entrenado")
    return self.model.predict(X)

# DESPUÉS
def predict(self, X: np.ndarray) -> np.ndarray:
    """
    Hacer predicciones.
    
    Args:
        X: Características a predecir.
        
    Returns:
        Array de predicciones.
        
    Raises:
        InvalidModelStateError: Si modelo no está entrenado.
        ModelPredictionError: Si ocurre error.
    """
    if not self.is_trained:
        raise InvalidModelStateError(
            "El modelo debe estar entrenado antes de predecir"
        )
    
    try:
        DataValidator.validate_features(X)
        assert self.model is not None
        return self.model.predict(X)
    except Exception as e:
        raise ModelPredictionError(f"Error prediciendo: {str(e)}")
```

---

### `src/utils/config.py`

**Mejoras:**
- ✅ Validación en `__post_init__`
- ✅ Métodos auxiliares mejor organizados
- ✅ Documentación mejorada

```python
# Validación automática
config = Config(
    TEST_SIZE=0.2,      # Validado automáticamente
    CV_FOLDS=5,         # Validado automáticamente
    RANDOM_STATE=42,    # Validado automáticamente
)

# Acceso a secciones específicas
preprocessing = config.get_preprocessing_config()
hyperparameters = config.get_hyperparameter_config()
dataset_info = config.get_dataset_info()
```

---

## 📊 Cómo Usar las Nuevas Herramientas

### Ejemplo 1: Validar Datos
```python
from src.utils import DataValidator, DataValidationError

try:
    DataValidator.validate_dataframe(df)
    DataValidator.validate_features(X)
    DataValidator.validate_target(y)
except DataValidationError as e:
    print(f"Datos inválidos: {e}")
```

### Ejemplo 2: Usar Constantes
```python
from src.utils.constants import (
    DEFAULT_RANDOM_STATE,
    CORRELATION_THRESHOLD,
    AlgorithmName
)

# Evitar magic numbers
if correlation > CORRELATION_THRESHOLD:
    print("Correlación significativa")

# Usar Enums
if algorithm == AlgorithmName.RANDOM_FOREST:
    print("Usando Random Forest")
```

### Ejemplo 3: Operaciones con Archivos
```python
from src.utils.helpers import FileUtils

# Guardar resultados
results = {"accuracy": 0.95, "f1": 0.93}
FileUtils.save_json(results, "results/metrics.json")

# Cargar datos
df = FileUtils.load_csv("data/dataset.csv")
```

### Ejemplo 4: Decoradores
```python
from src.decorators import timer, log_execution

@timer
@log_execution("INFO")
def train_model(X, y):
    # Se registra automáticamente
    # Se mide tiempo automáticamente
    return model.fit(X, y)
```

### Ejemplo 5: Pipeline Completo
```python
from src.pipeline import MLPipeline, PipelineConfig

pipeline = MLPipeline()

config = PipelineConfig(
    data_path="data/coffee_shop_revenue.csv",
    quick_mode=False,
    algorithms=["LogisticRegression", "SVM", "RandomForest"]
)

results = pipeline.run(config)
print(f"Modelos comparados: {results['models'].keys()}")
```

---

## 🎓 Patrones Comunes

### Patrón 1: Validar y Procesar
```python
from src.utils import DataValidator

def process_data(df):
    # Paso 1: Validar
    DataValidator.validate_dataframe(df)
    
    # Paso 2: Procesar
    df_clean = df.dropna()
    
    # Paso 3: Retornar
    return df_clean
```

### Patrón 2: Excepciones Informativas
```python
from src.utils.exceptions import ModelTrainingError

try:
    model.train(X, y)
except Exception as e:
    raise ModelTrainingError(
        f"Entrenamiento falló: {str(e)}",
        context=f"Algorithm: {algorithm}, Dataset: {X.shape}"
    )
```

### Patrón 3: Timing y Logging
```python
from src.decorators import logged_and_timed

@logged_and_timed
def expensive_operation():
    # Automáticamente:
    # 1. Registra inicio y fin
    # 2. Mide tiempo de ejecución
    pass
```

---

## 🚀 Próximos Pasos Sugeridos

### Corto Plazo
1. [ ] Refactorizar `ml_analysis.py` usando `MLPipeline`
2. [ ] Agregar type hints a todos los módulos existentes
3. [ ] Crear tests unitarios para validadores
4. [ ] Documentar todos los módulos clave

### Mediano Plazo
5. [ ] Crear test suite completa
6. [ ] Agregar CI/CD (GitHub Actions)
7. [ ] Optimizar perfor performance crítica
8. [ ] Crear API REST con FastAPI

### Largo Plazo
9. [ ] Agregar soporte para MLflow
10. [ ] Crear dashboard con Streamlit
11. [ ] Packaging para PyPI
12. [ ] Documentación completa con Sphinx

---

## 📝 Mapeo de Cambios

| Aspecto | Antes | Después | Beneficio |
|---------|-------|---------|-----------|
| Constantes | Hardcoded | `constants.py` | DRY, fácil mantenimiento |
| Errores | `ValueError` genérico | Excepciones específicas | Mejor debugging |
| Validación | Dispersa | Módulo `validators.py` | Reutilizable |
| Helpers | Duplicados | Módulo `helpers.py` | Consistente |
| Tipos | Ninguno | `types.py` | Seguridad |
| Pipeline | Manual | Clase `MLPipeline` | Automatizado |

---

## 🔗 Referencias Rápidas

### Imports Comunes
```python
# Utilidades
from src.utils import Config, DataValidator, FileUtils, TimingUtils
from src.utils.constants import DEFAULT_RANDOM_STATE, AlgorithmName
from src.utils.exceptions import DataError, ModelError

# Core
from src.types import FeatureMatrix, TargetVector, MetricsDict
from src.decorators import timer, logged_and_timed
from src.pipeline import MLPipeline, PipelineConfig

# Modelos
from src.models import (
    LogisticRegressionClassifier,
    SVMClassifier,
    RandomForestClassifierCustom,
)
```

---

## 📞 Soporte

### Para problemas de:
- **Validación:** Ver `STYLE_GUIDE.md` sección 8
- **Errores:** Ver `exceptions.py` para jerarquía
- **Tipos:** Ver `types.py` para definiciones
- **Estilo:** Ver `STYLE_GUIDE.md` para convenciones

---

**Última actualización:** March 2024  
**Versión:** 1.0.0 Refactorizado  
**Estado:** ✅ Refactorización Completa
