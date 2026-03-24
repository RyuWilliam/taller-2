"""
Definiciones de tipos y protocolos para el proyecto.

Centraliza definiciones de tipos complejos para mejorar
type hints y documentación del código.
"""

from typing import Protocol, Union, Sequence, Callable, Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# ================================================================
# ALIASES DE TIPOS COMUNES
# ================================================================

# Arrays y matrices
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame, list, tuple]
NumericArray = NDArray[np.floating]  # Array numérico
BinaryArray = NDArray[np.integer]  # Array binario 0/1
FeatureMatrix = NDArray[np.floating]  # Matriz de características (n_samples, n_features)
TargetVector = Union[NDArray[np.integer], pd.Series]  # Vector objetivo
Predictions = NDArray[np.integer]  # Predicciones
Probabilities = NDArray[np.floating]  # Probabilidades (0-1)

# Configuración
ConfigDict = Dict[str, Any]
HyperparameterDict = Dict[str, Any]
MetricsDict = Dict[str, Union[int, float, str]]

# Rutas
PathLike = Union[str, "Path"]  # Compatible con Path de pathlib

# ================================================================
# PROTOCOLOS (STRUCTURAL TYPING)
# ================================================================

class Classifier(Protocol):
    """Protocolo para clasificadores."""
    
    def fit(self, X: FeatureMatrix, y: TargetVector) -> "Classifier":
        """Entrenar el clasificador."""
        ...
    
    def predict(self, X: FeatureMatrix) -> Predictions:
        """Hacer predicciones."""
        ...
    
    def predict_proba(self, X: FeatureMatrix) -> Probabilities:
        """Obtener probabilidades de clase."""
        ...
    
    def score(self, X: FeatureMatrix, y: TargetVector) -> float:
        """Calcular score en datos."""
        ...


class Pipeline(Protocol):
    """Protocolo para pipelines de datos."""
    
    def fit(self, X: FeatureMatrix, y: Optional[TargetVector] = None) -> "Pipeline":
        """Ajustar pipeline."""
        ...
    
    def transform(self, X: FeatureMatrix) -> FeatureMatrix:
        """Transformar datos."""
        ...
    
    def fit_transform(self, X: FeatureMatrix, y: Optional[TargetVector] = None) -> FeatureMatrix:
        """Ajustar y transformar."""
        ...


class Evaluator(Protocol):
    """Protocolo para evaluadores de modelos."""
    
    def evaluate(self, y_true: TargetVector, y_pred: Predictions) -> MetricsDict:
        """Evaluar predicciones."""
        ...
    
    def get_report(self) -> str:
        """Obtener reporte en formato string."""
        ...


class Visualizer(Protocol):
    """Protocolo para visualizadores."""
    
    def plot(self, *args, **kwargs) -> None:
        """Crear visualización."""
        ...
    
    def save(self, filepath: PathLike) -> None:
        """Guardar visualización."""
        ...


# ================================================================
# TIPOS COMPLEJOS
# ================================================================

class TrainTestSplit:
    """Resultado de train-test split."""
    
    X_train: FeatureMatrix
    X_test: FeatureMatrix
    y_train: TargetVector
    y_test: TargetVector


class EvaluationResult:
    """Resultado completo de evaluación."""
    
    train_metrics: MetricsDict
    test_metrics: MetricsDict
    cv_scores: List[float]
    confusion_matrix: np.ndarray
    feature_importance: Optional[np.ndarray]


class ModelComparison:
    """Comparación de múltiples modelos."""
    
    models: Dict[str, Any]
    metrics: Dict[str, MetricsDict]
    best_model: str
    ranking: List[Tuple[str, float]]


# ================================================================
# CALLBACKS Y FUNCIONES
# ================================================================

MetricFunction = Callable[[TargetVector, Predictions], float]
TransformFunction = Callable[[FeatureMatrix], FeatureMatrix]
CallbackFunction = Callable[..., None]
LoggerFunction = Callable[[str], None]

# ================================================================
# UTILIDADES DE TIPOS
# ================================================================

def is_array_like(obj: Any) -> bool:
    """Verificar si objeto es similar a array."""
    return isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame, list, tuple))


def to_numpy_array(obj: ArrayLike) -> np.ndarray:
    """Convertir objeto array-like a numpy array."""
    if isinstance(obj, pd.DataFrame):
        return obj.values
    elif isinstance(obj, pd.Series):
        return obj.values
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        return np.asarray(obj)


def validate_classification_target(y: TargetVector) -> int:
    """
    Validar que objetivo sea de clasificación y retornar n_classes.
    
    Args:
        y: Vector objetivo.
        
    Returns:
        Número de clases.
    """
    y_array = to_numpy_array(y).flatten()
    classes = np.unique(y_array)
    
    if len(classes) < 2:
        raise ValueError(f"Se necesitan al menos 2 clases, got {len(classes)}")
    
    return len(classes)
