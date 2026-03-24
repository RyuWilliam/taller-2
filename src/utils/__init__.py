"""Objetivo: Reunir utilidades comunes como configuración, logging, validación y manejo de errores.

Módulos:
- config: Configuración centralizada
- logger: Logging estructurado
- constants: Valores y constantes
- exceptions: Excepciones personalizadas
- validators: Validadores de datos y configuración
- helpers: Utilidades generales (archivos, strings, arrays, etc.)
"""

from .config import Config
from .logger import setup_logger, LoggerMixin
from .constants import (
    AlgorithmName,
    MetricName,
    DataSplit,
    DEFAULT_RANDOM_STATE,
    DEFAULT_CV_FOLDS,
    DEFAULT_TEST_SIZE,
    DEFAULT_ALGORITHMS,
)
from .exceptions import (
    MLProjectError,
    DataError,
    ModelError,
    ConfigError,
)
from .validators import (
    DataValidator,
    ConfigValidator,
    HyperparameterValidator,
    PathValidator,
)
from .helpers import (
    FileUtils,
    StringUtils,
    ArrayUtils,
    TimingUtils,
    DictUtils,
)

__all__ = [
    "Config",
    "setup_logger",
    "LoggerMixin",
    "AlgorithmName",
    "MetricName",
    "DataSplit",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_CV_FOLDS",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_ALGORITHMS",
    "MLProjectError",
    "DataError",
    "ModelError",
    "ConfigError",
    "DataValidator",
    "ConfigValidator",
    "HyperparameterValidator",
    "PathValidator",
    "FileUtils",
    "StringUtils",
    "ArrayUtils",
    "TimingUtils",
    "DictUtils",
]
