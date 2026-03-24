"""
Excepciones personalizadas para el proyecto de ML.

Cada excepción está diseñada para un caso específico permitiendo
mejor manejo de errores y mensajes más informativos.
"""


class MLProjectError(Exception):
    """Clase base para todas las excepciones del proyecto."""
    
    def __init__(self, message: str, context: str = ""):
        self.message = message
        self.context = context
        super().__init__(self.format_error())
    
    def format_error(self) -> str:
        """Formato consistente para mensajes de error."""
        if self.context:
            return f"{self.__class__.__name__}: {self.message}\nContext: {self.context}"
        return f"{self.__class__.__name__}: {self.message}"


class DataError(MLProjectError):
    """Error relacionado con carga o procesamiento de datos."""
    pass


class DataValidationError(DataError):
    """Error de validación en los datos."""
    pass


class DataNotFoundError(DataError):
    """Archivo de datos no encontrado."""
    pass


class PreprocessingError(DataError):
    """Error durante el preprocesamiento de datos."""
    pass


class ModelError(MLProjectError):
    """Error relacionado con modelos de ML."""
    pass


class ModelTrainingError(ModelError):
    """Error durante el entrenamiento del modelo."""
    pass


class ModelPredictionError(ModelError):
    """Error durante la predicción del modelo."""
    pass


class InvalidModelStateError(ModelError):
    """Modelo en estado inválido (ej: no entrenado)."""
    pass


class HyperparameterError(ModelError):
    """Error en hiperparámetros o su optimización."""
    pass


class EvaluationError(MLProjectError):
    """Error durante la evaluación del modelo."""
    pass


class VisualizationError(MLProjectError):
    """Error durante la generación de visualizaciones."""
    pass


class ConfigError(MLProjectError):
    """Error relacionado con configuración."""
    pass


class ConfigValidationError(ConfigError):
    """Error de validación en la configuración."""
    pass


class IOError(MLProjectError):
    """Error de entrada/salida de archivos."""
    pass


class ReportGenerationError(MLProjectError):
    """Error al generar reportes."""
    pass


def handle_error(error: Exception, context: str = "") -> None:
    """
    Manejar errores de forma consistente.
    
    Args:
        error: La excepción a manejar.
        context: Contexto adicional del error.
    """
    if isinstance(error, MLProjectError):
        print(f"❌ {error}")
    else:
        wrapped_error = MLProjectError(str(error), context)
        print(f"❌ {wrapped_error}")
