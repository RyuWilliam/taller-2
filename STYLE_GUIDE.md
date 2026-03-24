# 📚 Guía de Estilo y Convenciones - ML-Algorithms

## Objetivo

Mantener consistencia en código, mejorar legibilidad y facilitar colaboración.

---

## 1. 🐍 Convenciones Python

### Nombres

```python
# ✅ BIEN
class ModelEvaluator:
    def calculate_metrics(self):
        pass

def validate_input_data(df):
    pass

config_path = "data/config.json"
MAX_ITERATIONS = 1000

# ❌ MAL
class ModelEval:
    def calc_metrics(self):
        pass

def validate(d):
    pass

configPath = "data/config.json"
max_iterations = 1000
```

### Espaciado y Formato

```python
# ✅ BIEN
def train_model(X: np.ndarray, y: np.ndarray, learning_rate: float = 0.01) -> None:
    """Entrenar el modelo."""
    result = process_data(X) + learning_rate
    return result

# ❌ MAL
def train_model(X,y,learning_rate=0.01):
    result=process_data(X)+learning_rate
    return result
```

### Imports

```python
# ✅ BIEN
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from ..utils import Config, DataValidator
from ..models import BaseClassifier

# ❌ MAL
from *
import numpy as np, pandas as pd
from ..utils import *
```

---

## 2. 📝 Docstrings

### Formato Google Style

```python
def evaluate_model(
    model: Classifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluar rendimiento del modelo en datos de prueba.
    
    Realiza evaluación completa incluyendo métricas de clasificación,
    matriz de confusión y análisis detallado por clase.
    
    Args:
        model: Modelo entrenado a evaluar.
        X_test: Características de prueba (n_samples, n_features).
        y_test: Etiquetas verdaderas.
        metrics: Lista de métricas a calcular. Si None, usa todas.
        
    Returns:
        Diccionario con resultados:
            - accuracy: Exactitud general
            - precision: Precisión global
            - recall: Recall global
            - f1_score: F1-Score global
            - por_clase: Dict con métricas por clase
            
    Raises:
        InvalidModelStateError: Si modelo no está entrenado.
        DataValidationError: Si datos no son válidos.
        
    Examples:
        ```python
        metrics = evaluate_model(model, X_test, y_test)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        ```
    """
    pass
```

### Docstrings Mínimos

```python
def get_feature_names(self) -> List[str]:
    """Obtener nombres de características."""
    return self.feature_names

def set_verbose(self, verbose: bool) -> None:
    """Establecer modo verbose."""
    self.verbose = verbose
```

---

## 3. 🏗️ Estructura de Archivos

### Organización

```
src/
├── utils/               # Utilidades comunes
│   ├── config.py       # Configuración
│   ├── logger.py       # Logging
│   ├── constants.py    # Constantes
│   ├── exceptions.py   # Excepciones
│   ├── validators.py   # Validadores
│   └── helpers.py      # Funciones auxiliares
├── models/             # Modelos de ML
│   ├── base_classifier.py
│   ├── decision_tree_classifier.py
│   └── ...
├── data_processing/    # Carga y preprocesamiento
├── evaluation/         # Evaluación de modelos
├── visualization/      # Visualizaciones
├── decorators.py       # Decoradores
├── types.py            # Definiciones de tipos
└── pipeline.py         # Orquestación principal
```

---

## 4. 🔍 Type Hints

### Excepciones

```python
# ✅ BIEN
from typing import Dict, List, Optional, Union
from numpy.typing import NDArray
import numpy as np

def process_data(
    X: NDArray[np.floating],
    y: Optional[np.ndarray] = None
) -> Dict[str, NDArray]:
    pass

# ❌ MAL
def process_data(X, y=None):
    pass
```

### Tipos Complejos

```python
# ✅ BIEN - usar alias definidos
from src.types import FeatureMatrix, TargetVector, MetricsDict

def train(X: FeatureMatrix, y: TargetVector) -> MetricsDict:
    pass

# ✅ BIEN - O definir inline
from typing import Dict, Union

MetricsDict = Dict[str, Union[int, float]]
```

---

## 5. ⚠️ Manejo de Errores

### Excepciones Personalizadas

```python
# ✅ BIEN
from src.utils.exceptions import InvalidModelStateError, DataValidationError

def predict(self, X):
    if not self.is_trained:
        raise InvalidModelStateError(
            "Modelo debe estar entrenado antes de predecir",
            context=f"Algorithm: {self.name}, X_shape: {X.shape}"
        )
    
    try:
        return self.model.predict(X)
    except Exception as e:
        raise ModelPredictionError(f"Error prediciendo: {str(e)}")

# ❌ MAL
def predict(self, X):
    if not self.is_trained:
        raise ValueError("Entrenar primero")
    return self.model.predict(X)
```

### Try-Except

```python
# ✅ BIEN
try:
    data = load_data(path)
    validated = validate_data(data)
except FileNotFoundError:
    logger.error(f"Archivo no encontrado: {path}")
    raise
except DataValidationError as e:
    logger.warning(f"Validación falló: {e}")
    handle_invalid_data()
except Exception as e:
    logger.error(f"Error inesperado: {e}", exc_info=True)
    raise

# ❌ MAL
try:
    data = load_data(path)
except:
    pass
```

---

## 6. 🎯 Logging

### Mensajes

```python
# ✅ BIEN
self.logger.info(f"Entrenando modelo {self.name} con {X.shape[0]:,} muestras")
self.logger.warning(f"Valor anómalo detectado: {value}")
self.logger.error(f"Error cargando datos: {str(e)}")

# ❌ MAL
self.logger.info("Entrenando")
print(f"Training with {X.shape[0]} samples")
```

### Niveles

```python
# DEBUG: Información muy detallada para debugging
logger.debug(f"Parámetro {param} = {value}")

# INFO: Eventos significativos
logger.info("Modelo entrenado exitosamente")

# WARNING: Algo inesperado pero no fatal
logger.warning(f"Datos faltantes detectados: {count} valores")

# ERROR: Evento serio, función fallió
logger.error(f"Error cargando datos: {str(e)}")

# CRITICAL: Error muy serio
logger.critical("Sistema no puede continuar")
```

---

## 7. ✨ Constantes

### Dónde Definir

```python
# ✅ BIEN - en constants.py
from src.utils.constants import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_CV_FOLDS,
    CORRELATION_THRESHOLD,
    DEFAULT_ALGORITHMS
)

# ❌ MAL - hardcoded
if correlation > 0.3:  # ¿Por qué 0.3?
    ...

if random_state == 42:  # ¿De dónde salió 42?
    ...
```

### Nombres

```python
# ✅ BIEN
MAX_ITERATIONS = 1000
DEFAULT_LEARNING_RATE = 0.01
CORRELATION_THRESHOLD = 0.3
VALID_KERNELS = ["linear", "rbf", "poly"]

# ❌ MAL
max_iter = 1000
lr = 0.01
threshold = 0.3
kernels = ["linear", "rbf", "poly"]
```

---

## 8. 🔐 Validación

### Usar Validadores

```python
# ✅ BIEN
from src.utils import DataValidator, HyperparameterValidator

DataValidator.validate_dataframe(df)
DataValidator.validate_features(X)
DataValidator.validate_target(y)
HyperparameterValidator.validate_svm_params(params)

# ❌ MAL
if df.empty:
    raise ValueError("df está vacío")

if X.ndim != 2:
    raise ValueError("X debe ser 2D")
```

---

## 9. 📊 Visualizaciones

### Convenciones

```python
# ✅ BIEN
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(x, y, label="Data", color="steelblue", linewidth=2)
axes[0].set_xlabel("Característica", fontsize=12)
axes[0].set_ylabel("Valor", fontsize=12)
axes[0].set_title("Análisis de Datos", fontsize=14, fontweight="bold")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output.png", dpi=300, bbox_inches="tight")
plt.close()

# ❌ MAL
plt.plot(x, y)
plt.title("Plot")
plt.savefig("output.png")
```

---

## 10. 🧪 Testing y Calidad

### Antes de Commit

```bash
# Validar tipos
mypy src/ --ignore-missing-imports

# Linter
flake8 src/ --max-line-length=100

# Formatter
black src/

# Tests
pytest tests/
```

### Estructura de Tests

```python
# tests/test_validators.py
import pytest
from src.utils.validators import DataValidator
from src.utils.exceptions import DataValidationError

class TestDataValidator:
    """Tests para DataValidator."""
    
    def test_validate_dataframe_empty(self):
        """Test que valida rechazo de DataFrame vacío."""
        df_empty = pd.DataFrame()
        with pytest.raises(DataValidationError):
            DataValidator.validate_dataframe(df_empty)
    
    def test_validate_features_nan_values(self):
        """Test que valida rechazo de valores NaN."""
        X = np.array([[1, 2, np.nan], [4, 5, 6]])
        with pytest.raises(DataValidationError):
            DataValidator.validate_features(X)
```

---

## 11. 🚀 Performance

### Mejores Prácticas

```python
# ✅ BIEN - usar list comprehension
result = [process(x) for x in data]

# ✅ BIEN - usar vectorización NumPy
result = np.sqrt(data)  # en lugar de np.sqrt a cada elemento

# ✅ BIEN - usar generadores para datos grandes
def data_generator(path):
    for batch in batches:
        yield batch

# ❌ MAL
result = []
for x in data:
    result.append(process(x))

# ❌ MAL - ejecución lenta
result = [math.sqrt(x) for x in data]  # en lugar de NumPy
```

---

## 12. 📦 Dependencias

### Imports Organizados

```python
# Orden correcto:
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 2. Third-party
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 3. Local
from ..utils import Config, DataValidator
from ..models import BaseClassifier
```

---

## 13. ✅ Checklist Pre-Commit

- [ ] Código sigue convenciones de PEP 8
- [ ] Type hints están presentes
- [ ] Docstrings están completos
- [ ] Excepciones son específicas
- [ ] Logging es adecuado
- [ ] Sin valores mágicos (usar constants.py)
- [ ] Validación anticipada de inputs
- [ ] Sin librerias no documentadas
- [ ] Tests pasan
- [ ] Código documentado con ejemplos

---

## 14. 🔗 Referencias

- [PEP 8](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Type Hints](https://docs.python.org/3/library/typing.html)
- [FastAPI Style](https://fastapi.tiangolo.com/deployment/concepts/)

---

**Última actualización:** March 2024
**Versión de referencia:** 1.0.0
