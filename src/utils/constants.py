"""
Constantes y valores predefinidos para el proyecto de ML.

Este módulo centraliza todos los valores mágicos, umbrales y configuraciones
que se repiten en todo el proyecto.
"""

from enum import Enum
from typing import Final

# ================================================================
# VALORES NUMÉRICOS
# ================================================================
DEFAULT_RANDOM_STATE: Final[int] = 42
DEFAULT_CV_FOLDS: Final[int] = 5
DEFAULT_TEST_SIZE: Final[float] = 0.2
DEFAULT_N_JOBS: Final[int] = -1
DEFAULT_N_ITER_SEARCH: Final[int] = 50
DEFAULT_SUCCESS_THRESHOLD: Final[float] = 2000.0
DEFAULT_FIGURE_DPI: Final[int] = 300
DEFAULT_IMG_QUALITY: Final[int] = 95

# ================================================================
# STRINGS Y RUTAS
# ================================================================
DEFAULT_DATA_PATH: Final[str] = "data/coffee_shop_revenue.csv"
DEFAULT_RESULTS_PATH: Final[str] = "results"
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
DEFAULT_LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_CACHE_FILENAME: Final[str] = "preprocessed_data.pkl"
DEFAULT_COMPARISON_REPORT: Final[str] = "algorithm_comparison_report.json"
DEFAULT_COMPARISON_CSV: Final[str] = "algorithm_comparison_metrics.csv"

# ================================================================
# NOMBRES DE COLUMNAS Y CLASES
# ================================================================
DEFAULT_TARGET_COLUMN: Final[str] = "Successful"
DEFAULT_DAILY_REVENUE_COLUMN: Final[str] = "Daily_Revenue"
DEFAULT_CLASS_NAMES: Final[list] = ["No Exitosa", "Exitosa"]

DEFAULT_COFFEE_FEATURES: Final[list] = [
    "Number_of_Customers_Per_Day",
    "Average_Order_Value",
    "Operating_Hours_Per_Day",
    "Number_of_Employees",
    "Marketing_Spend_Per_Day",
    "Location_Foot_Traffic",
]

# ================================================================
# ALGORITMOS Y PARÁMETROS
# ================================================================
DEFAULT_ALGORITHMS: Final[list] = [
    "LogisticRegression",
    "SVM",
    "DecisionTree",
    "RandomForest",
    "NeuralNetwork",
]

DEFAULT_SCORING_METRIC: Final[str] = "accuracy"

# ================================================================
# UMBRALES Y LÍMITES
# ================================================================
CORRELATION_THRESHOLD: Final[float] = 0.3
MIN_SAMPLES_FOR_PROCESSING: Final[int] = 100
MAX_ITERATIONS_ML: Final[int] = 2000
VISUALIZATION_MAX_FEATURES_CORRELATION: Final[int] = 20

# ================================================================
# PATRONES Y VALIDACIÓN
# ================================================================
VALID_SCORING_METRICS: Final[list] = [
    "accuracy",
    "f1",
    "precision",
    "recall",
    "roc_auc",
]

VALID_KERNELS_SVM: Final[list] = ["linear", "rbf", "poly", "sigmoid"]
VALID_CRITERIA_TREES: Final[list] = ["gini", "entropy"]

# ================================================================
# ENUMS PARA TIPOS
# ================================================================
class AlgorithmName(str, Enum):
    """Nombres de algoritmos disponibles."""
    
    LOGISTIC_REGRESSION = "LogisticRegression"
    SVM = "SVM"
    DECISION_TREE = "DecisionTree"
    RANDOM_FOREST = "RandomForest"
    NEURAL_NETWORK = "NeuralNetwork"


class MetricName(str, Enum):
    """Nombres de métricas disponibles."""
    
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    AVERAGE_PRECISION = "average_precision"


class DataSplit(str, Enum):
    """Tipos de división de datos."""
    
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


# ================================================================
# MENSAJES Y FORMATOS
# ================================================================
SEPARATOR_MAIN: Final[str] = "=" * 100
SEPARATOR_SECTION: Final[str] = "-" * 60
SEPARATOR_SUBSECTION: Final[str] = "-" * 40

# ================================================================
# CONFIGURACIÓN DE MATPLOTLIB
# ================================================================
DEFAULT_FIGURE_SIZE: Final[tuple] = (12, 8)
DEFAULT_FIGURE_STYLE: Final[str] = "seaborn-v0_8"
PALETTE_COLORS: Final[str] = "husl"
