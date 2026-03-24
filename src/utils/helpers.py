"""
Utilidades del proyecto: funciones y clases reutilizables.

Proporciona helpers comunes para operaciones de I/O, formateo de datos,
y utilidades generales utilizadas en múltiples módulos.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np
import pandas as pd
from datetime import datetime

from .exceptions import IOError as ProjectIOError


class FileUtils:
    """Utilidades para operaciones con archivos."""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Asegurar que un directorio existe, crearlo si es necesario.
        
        Args:
            path: Ruta del directorio.
            
        Returns:
            Path del directorio.
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path], 
                  indent: int = 2, pretty: bool = True) -> None:
        """
        Guardar datos como JSON.
        
        Args:
            data: Datos a guardar.
            file_path: Ruta del archivo.
            indent: Indentación para formato legible.
            pretty: Si formatear para legibilidad.
        """
        try:
            path = Path(file_path)
            FileUtils.ensure_directory(path.parent)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent if pretty else None, ensure_ascii=False)
        except Exception as e:
            raise ProjectIOError(f"Error guardando JSON en {file_path}: {str(e)}")
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Cargar datos desde JSON.
        
        Args:
            file_path: Ruta del archivo JSON.
            
        Returns:
            Datos cargados desde JSON.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ProjectIOError(f"Error cargando JSON desde {file_path}: {str(e)}")
    
    @staticmethod
    def save_csv(df: pd.DataFrame, file_path: Union[str, Path], 
                 index: bool = False) -> None:
        """
        Guardar DataFrame como CSV.
        
        Args:
            df: DataFrame a guardar.
            file_path: Ruta del archivo.
            index: Si guardar index.
        """
        try:
            path = Path(file_path)
            FileUtils.ensure_directory(path.parent)
            df.to_csv(path, index=index, encoding='utf-8')
        except Exception as e:
            raise ProjectIOError(f"Error guardando CSV en {file_path}: {str(e)}")
    
    @staticmethod
    def load_csv(file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Cargar CSV como DataFrame.
        
        Args:
            file_path: Ruta del archivo CSV.
            
        Returns:
            DataFrame cargado.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            
            return pd.read_csv(path)
        except Exception as e:
            raise ProjectIOError(f"Error cargando CSV desde {file_path}: {str(e)}")


class StringUtils:
    """Utilidades para manipulación de strings."""
    
    @staticmethod
    def format_metric(value: Union[int, float], decimals: int = 4) -> str:
        """
        Formatear métrica numérica para display.
        
        Args:
            value: Valor a formatear.
            decimals: Número de decimales.
            
        Returns:
            String formateado.
        """
        if isinstance(value, int):
            return f"{value:,}"
        return f"{value:.{decimals}f}"
    
    @staticmethod
    def format_size(size_bytes: int) -> str:
        """
        Formatear tamaño en bytes a formato legible.
        
        Args:
            size_bytes: Tamaño en bytes.
            
        Returns:
            String formateado (ej: "1.2 MB").
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    @staticmethod
    def pluralize(word: str, count: int) -> str:
        """
        Pluralizar palabra según conteo.
        
        Args:
            word: Palabra base (singular).
            count: Conteo.
            
        Returns:
            Palabra singularizada o pluralizada.
        """
        if count == 1:
            return word
        return word + "s"


class ArrayUtils:
    """Utilidades para operaciones con arreglos."""
    
    @staticmethod
    def get_class_distribution(y: np.ndarray) -> Dict[str, int]:
        """
        Obtener distribución de clases.
        
        Args:
            y: Array objetivo.
            
        Returns:
            Dict con conteo por clase.
        """
        unique, counts = np.unique(y, return_counts=True)
        return {f"class_{int(u)}": int(c) for u, c in zip(unique, counts)}
    
    @staticmethod
    def get_class_weights(y: np.ndarray) -> Dict[int, float]:
        """
        Calcular pesos de clase (inverso de frecuencia).
        
        Args:
            y: Array objetivo.
            
        Returns:
            Dict con peso para cada clase.
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return {int(c): float(w) for c, w in zip(classes, weights)}
    
    @staticmethod
    def normalize_array(X: np.ndarray, axis: Optional[int] = 0) -> np.ndarray:
        """
        Normalizar arreglo a rango [0, 1].
        
        Args:
            X: Array a normalizar.
            axis: Eje a lo largo del cual normalizar.
            
        Returns:
            Array normalizado.
        """
        X_min = np.min(X, axis=axis, keepdims=True)
        X_max = np.max(X, axis=axis, keepdims=True)
        
        # Evitar división por cero
        denominator = np.where(X_max - X_min == 0, 1, X_max - X_min)
        return (X - X_min) / denominator


class TimingUtils:
    """Utilidades para timing y benchmarking."""
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Formatear duración en segundos a string legible.
        
        Args:
            seconds: Duración en segundos.
            
        Returns:
            String formateado (ej: "1m 23s").
        """
        if seconds < 60:
            return f"{seconds:.2f}s"
        
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        
        if minutes < 60:
            return f"{minutes}m {secs}s"
        
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h {minutes}m {secs}s"
    
    @staticmethod
    def get_timestamp() -> str:
        """
        Obtener timestamp actual en formato ISO.
        
        Returns:
            String con timestamp.
        """
        return datetime.now().isoformat()


class DictUtils:
    """Utilidades para operaciones con diccionarios."""
    
    @staticmethod
    def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict:
        """
        Aplanar diccionario anidado.
        
        Args:
            d: Diccionario a aplanar.
            parent_key: Clave padre para recursión.
            sep: Separador para claves.
            
        Returns:
            Diccionario aplanado.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(DictUtils.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    @staticmethod
    def merge_dicts(*dicts: Dict) -> Dict:
        """
        Fusionar múltiples diccionarios.
        
        Args:
            dicts: Diccionarios a fusionar.
            
        Returns:
            Diccionario fusionado.
        """
        result = {}
        for d in dicts:
            result.update(d)
        return result
    
    @staticmethod
    def subset_dict(d: Dict, keys: List[str]) -> Dict:
        """
        Obtener subconjunto de diccionario.
        
        Args:
            d: Diccionario original.
            keys: Claves a incluir.
            
        Returns:
            Subconjunto del diccionario.
        """
        return {k: v for k, v in d.items() if k in keys}
