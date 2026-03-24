"""
Validadores para datos, configuraciones y parámetros.

Este módulo proporciona funciones de validación reutilizables para
asegurar la integridad de datos y configuraciones.
"""

from typing import Any, List, Dict, Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path

from .constants import (
    VALID_SCORING_METRICS,
    VALID_KERNELS_SVM,
    VALID_CRITERIA_TREES,
    MIN_SAMPLES_FOR_PROCESSING,
)
from .exceptions import (
    ConfigValidationError,
    DataValidationError,
    HyperparameterError,
)


class DataValidator:
    """Validador de datos."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_rows: int = MIN_SAMPLES_FOR_PROCESSING) -> None:
        """
        Validar que un DataFrame cumpla requisitos mínimos.
        
        Args:
            df: DataFrame a validar.
            min_rows: Número mínimo de filas requeridas.
            
        Raises:
            DataValidationError: Si el DataFrame no es válido.
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input no es un DataFrame")
        
        if df.empty:
            raise DataValidationError("DataFrame está vacío")
        
        if len(df) < min_rows:
            raise DataValidationError(
                f"DataFrame tiene {len(df)} filas pero requiere mínimo {min_rows}"
            )
    
    @staticmethod
    def validate_features(X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Validar matriz de características.
        
        Args:
            X: Matriz de características a validar.
            
        Raises:
            DataValidationError: Si X no es válido.
        """
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)
        
        if X_array.ndim != 2:
            raise DataValidationError(f"Features debe ser 2D, pero tiene {X_array.ndim} dimensiones")
        
        if X_array.shape[0] == 0:
            raise DataValidationError("Features tiene 0 muestras")
        
        if np.any(np.isnan(X_array)):
            raise DataValidationError("Features contiene valores NaN")
        
        if np.any(np.isinf(X_array)):
            raise DataValidationError("Features contiene valores infinitos")
    
    @staticmethod
    def validate_target(y: Union[np.ndarray, pd.Series]) -> None:
        """
        Validar vector objetivo.
        
        Args:
            y: Vector objetivo a validar.
            
        Raises:
            DataValidationError: Si y no es válido.
        """
        y_array = np.asarray(y).flatten()
        
        if y_array.ndim != 1:
            raise DataValidationError(f"Target debe ser 1D, pero tiene {y_array.ndim} dimensiones")
        
        if len(y_array) == 0:
            raise DataValidationError("Target tiene 0 muestras")
        
        if np.any(np.isnan(y_array)):
            raise DataValidationError("Target contiene valores NaN")
    
    @staticmethod
    def validate_feature_names(feature_names: Optional[List[str]], n_features: int) -> None:
        """
        Validar que los nombres de características sean válidos.
        
        Args:
            feature_names: Lista de nombres de características.
            n_features: Número esperado de características.
            
        Raises:
            DataValidationError: Si feature_names no es válido.
        """
        if feature_names is None:
            return
        
        if not isinstance(feature_names, (list, tuple)):
            raise DataValidationError("feature_names debe ser lista o tupla")
        
        if len(feature_names) != n_features:
            raise DataValidationError(
                f"feature_names tiene {len(feature_names)} elementos "
                f"pero se esperaban {n_features}"
            )
        
        if not all(isinstance(name, str) for name in feature_names):
            raise DataValidationError("Todos los nombres de características deben ser strings")


class ConfigValidator:
    """Validador de configuración."""
    
    @staticmethod
    def validate_scoring_metric(metric: str) -> None:
        """Validar que la métrica de scoring sea válida."""
        if metric not in VALID_SCORING_METRICS:
            raise ConfigValidationError(
                f"Métrica '{metric}' no válida. "
                f"Opciones: {VALID_SCORING_METRICS}"
            )
    
    @staticmethod
    def validate_test_size(test_size: float) -> None:
        """Validar que test_size esté entre 0 y 1."""
        if not (0 < test_size < 1):
            raise ConfigValidationError(f"test_size debe estar entre 0 y 1, got {test_size}")
    
    @staticmethod
    def validate_cv_folds(cv_folds: int) -> None:
        """Validar que cv_folds sea un entero positivo."""
        if not isinstance(cv_folds, int) or cv_folds < 2:
            raise ConfigValidationError(f"cv_folds debe ser entero >= 2, got {cv_folds}")
    
    @staticmethod
    def validate_n_jobs(n_jobs: int) -> None:
        """Validar que n_jobs sea válido."""
        if not isinstance(n_jobs, int) or (n_jobs != -1 and n_jobs < 1):
            raise ConfigValidationError(f"n_jobs debe ser -1 o entero positivo, got {n_jobs}")
    
    @staticmethod
    def validate_random_state(random_state: int) -> None:
        """Validar que random_state sea un entero no negativo."""
        if not isinstance(random_state, int) or random_state < 0:
            raise ConfigValidationError(f"random_state debe ser entero >= 0, got {random_state}")


class HyperparameterValidator:
    """Validador de hiperparámetros."""
    
    @staticmethod
    def validate_svm_params(params: Dict[str, Any]) -> None:
        """Validar parámetros de SVM."""
        if "kernel" in params and params["kernel"] not in VALID_KERNELS_SVM:
            raise HyperparameterError(
                f"SVM kernel '{params['kernel']}' no válido. "
                f"Opciones: {VALID_KERNELS_SVM}"
            )
        
        if "C" in params and params["C"] <= 0:
            raise HyperparameterError(f"SVM C debe ser positivo, got {params['C']}")
        
        if "gamma" in params:
            gamma = params["gamma"]
            if isinstance(gamma, (int, float)) and gamma <= 0:
                raise HyperparameterError(f"SVM gamma debe ser positivo, got {gamma}")
    
    @staticmethod
    def validate_tree_params(params: Dict[str, Any]) -> None:
        """Validar parámetros de árbol de decisión."""
        if "criterion" in params and params["criterion"] not in VALID_CRITERIA_TREES:
            raise HyperparameterError(
                f"Tree criterion '{params['criterion']}' no válido. "
                f"Opciones: {VALID_CRITERIA_TREES}"
            )
        
        if "max_depth" in params:
            max_depth = params["max_depth"]
            if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1):
                raise HyperparameterError(f"max_depth debe ser None o entero >= 1, got {max_depth}")
        
        if "min_samples_split" in params and params["min_samples_split"] < 2:
            raise HyperparameterError(f"min_samples_split debe ser >= 2")
        
        if "min_samples_leaf" in params and params["min_samples_leaf"] < 1:
            raise HyperparameterError(f"min_samples_leaf debe ser >= 1")


class PathValidator:
    """Validador de rutas y archivos."""
    
    @staticmethod
    def validate_file_exists(file_path: Union[str, Path]) -> Path:
        """
        Validar que un archivo existe.
        
        Args:
            file_path: Ruta del archivo.
            
        Returns:
            Path del archivo si existe.
            
        Raises:
            DataValidationError: Si el archivo no existe.
        """
        path = Path(file_path)
        if not path.exists():
            raise DataValidationError(f"Archivo no encontrado: {file_path}")
        if not path.is_file():
            raise DataValidationError(f"Ruta no es un archivo: {file_path}")
        return path
    
    @staticmethod
    def validate_directory_exists(dir_path: Union[str, Path]) -> Path:
        """
        Validar que un directorio existe (o crear si es necesario).
        
        Args:
            dir_path: Ruta del directorio.
            
        Returns:
            Path del directorio.
        """
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def validate_csv_file(file_path: Union[str, Path]) -> Path:
        """
        Validar que sea un archivo CSV válido.
        
        Args:
            file_path: Ruta del archivo.
            
        Returns:
            Path del archivo si es válido.
            
        Raises:
            DataValidationError: Si no es un CSV válido.
        """
        path = PathValidator.validate_file_exists(file_path)
        if path.suffix.lower() != ".csv":
            raise DataValidationError(f"Archivo debe ser .csv, got {path.suffix}")
        return path
