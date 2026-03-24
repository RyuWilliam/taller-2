"""
Decoradores útiles para el proyecto.

Proporciona decoradores para logging, timing, validation, caching, etc.
"""

import functools
import time
import logging
from typing import Callable, Any, Optional, TypeVar, cast
from pathlib import Path

F = TypeVar('F', bound=Callable[..., Any])


def timer(func: F) -> F:
    """
    Decorador que mide tiempo de ejecución de una función.
    
    Ejemplo:
        @timer
        def process_data():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start
            logger = logging.getLogger(func.__module__)
            logger.info(f"{func.__name__} ejecutado en {elapsed:.2f}s")
    
    return cast(F, wrapper)


def log_execution(level: str = "INFO") -> Callable[[F], F]:
    """
    Decorador que registra ejecución de función.
    
    Ejemplo:
        @log_execution("DEBUG")
        def my_function(x):
            return x * 2
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            log_func = getattr(logger, level.lower(), logger.info)
            
            log_func(f"Iniciando {func.__name__}")
            try:
                result = func(*args, **kwargs)
                log_func(f"✅ {func.__name__} completado")
                return result
            except Exception as e:
                log_func(f"❌ {func.__name__} falló: {str(e)}")
                raise
        
        return cast(F, wrapper)
    return decorator


def validate_args(**validators: Callable[[Any], bool]) -> Callable[[F], F]:
    """
    Decorador que valida argumentos de función.
    
    Ejemplo:
        @validate_args(
            X=lambda x: x is not None,
            y=lambda y: len(y) > 0
        )
        def train(X, y):
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validar argumentos nombrados
            for arg_name, validator in validators.items():
                if arg_name in kwargs:
                    value = kwargs[arg_name]
                    if not validator(value):
                        raise ValueError(f"Validación falló para argumento '{arg_name}'")
            
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    return decorator


def requires_trained(func: F) -> F:
    """
    Decorador que verifica que modelo esté entrenado.
    
    Solo funciona para métodos de clases con atributo 'is_trained'.
    
    Ejemplo:
        class Classifier:
            @requires_trained
            def predict(self, X):
                pass
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'is_trained') or not self.is_trained:
            raise ValueError(
                f"Modelo debe estar entrenado antes de llamar {func.__name__}"
            )
        return func(self, *args, **kwargs)
    
    return cast(F, wrapper)


def memoize(func: F) -> F:
    """
    Decorador que cachea resultados de función.
    
    Útil para funciones puras (sin side effects).
    
    Ejemplo:
        @memoize
        def expensive_computation(x):
            return x ** 2
    """
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Crear clave del cache (solo funciona con args hashables)
        key = (args, tuple(sorted(kwargs.items())))
        
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        
        return cache[key]
    
    # Agregar métodos para inspeccionar cache
    wrapper.cache_clear = lambda: cache.clear()  # type: ignore
    wrapper.cache_info = lambda: f"Cache size: {len(cache)}"  # type: ignore
    
    return cast(F, wrapper)


def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0) -> Callable[[F], F]:
    """
    Decorador que reinenta función si falla.
    
    Ejemplo:
        @retry(max_attempts=3, delay=0.5)
        def unreliable_operation():
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay
            
            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    
                    logger = logging.getLogger(func.__module__)
                    logger.warning(
                        f"Intento {attempt} falló, reintentando en {current_delay}s: {str(e)}"
                    )
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
                    attempt += 1
        
        return cast(F, wrapper)
    return decorator


def deprecated(message: str = "") -> Callable[[F], F]:
    """
    Decorador que marca función como deprecada.
    
    Ejemplo:
        @deprecated("Use new_function instead")
        def old_function():
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            warning_msg = f"⚠️  {func.__name__} está deprecada"
            if message:
                warning_msg += f": {message}"
            logger.warning(warning_msg)
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    return decorator


def ensure_path_exists(path_param: str = "file_path") -> Callable[[F], F]:
    """
    Decorador que asegura que ruta existe antes de ejecutar función.
    
    Ejemplo:
        @ensure_path_exists("output_path")
        def save_report(output_path):
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if path_param in kwargs:
                path = Path(kwargs[path_param])
                path.parent.mkdir(parents=True, exist_ok=True)
            
            return func(*args, **kwargs)
        
        return cast(F, wrapper)
    return decorator


def profile(func: F) -> F:
    """
    Decorador que genera profiling de uso de memoria.
    
    Útil para identificar memory leaks.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import tracemalloc
            
            tracemalloc.start()
            start_memory = tracemalloc.get_traced_memory()[0]
            
            result = func(*args, **kwargs)
            
            current_memory = tracemalloc.get_traced_memory()[0]
            memory_used = (current_memory - start_memory) / 1024 / 1024  # MB
            
            logger = logging.getLogger(func.__module__)
            logger.info(f"{func.__name__} usó {memory_used:.2f} MB de memoria")
            
            tracemalloc.stop()
            return result
        except ImportError:
            # tracemalloc no disponible
            return func(*args, **kwargs)
    
    return cast(F, wrapper)


# ================================================================
# DECORADORES COMBINADOS COMUNES
# ================================================================

def logged_and_timed(func: F) -> F:
    """Combinación de logging y timing."""
    @log_execution("INFO")
    @timer
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return cast(F, wrapper)


def safe_execution(func: F) -> F:
    """
    Ejecución segura con manejo de errores.
    
    Captura excepciones y registra contexto.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(func.__module__)
            logger.error(f"Error en {func.__name__}: {str(e)}", exc_info=True)
            raise
    
    return cast(F, wrapper)
