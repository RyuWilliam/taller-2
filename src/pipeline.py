"""
Orquestador principal del pipeline de ML.

Coordina todos los pasos del análisis: carga de datos, preprocesamiento,
entrenamiento de modelos, evaluación y generación de reportes.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass

from ..utils.logger import LoggerMixin
from ..utils.config import Config
from ..utils.constants import SEPARATOR_MAIN, SEPARATOR_SECTION
from ..utils.exceptions import MLProjectError
from ..utils.helpers import TimingUtils, FileUtils
from ..data_processing import DataLoader, DataPreprocessor
from ..visualization import DataVisualizer
from ..evaluation import MultiAlgorithmEvaluator, ModelEvaluator, MetricsCalculator


@dataclass
class PipelineConfig:
    """Configuración para la ejecución del pipeline."""
    
    data_path: Optional[str] = None
    sample_size: Optional[int] = None
    output_dir: str = "results"
    skip_viz: bool = False
    quick_mode: bool = False
    retrain: bool = False
    algorithms: Optional[List[str]] = None


class MLPipeline(LoggerMixin):
    """Orquestador principal del pipeline de ML."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Inicializar pipeline.
        
        Args:
            config: Configuración del proyecto.
        """
        self.config = config or Config()
        self.data_loader = DataLoader(self.config)
        self.data_preprocessor = DataPreprocessor(self.config)
        self.data_visualizer = DataVisualizer(self.config)
        self.metrics_calculator = MetricsCalculator(self.config)
        self.evaluator = MultiAlgorithmEvaluator(self.config)
        
        # Estado del pipeline
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_classes = None
        self.execution_time = {}
    
    def run(self, pipeline_config: PipelineConfig) -> Dict[str, Any]:
        """
        Ejecutar pipeline completo.
        
        Args:
            pipeline_config: Configuración del pipeline.
            
        Returns:
            Diccionario con resultados del pipeline.
        """
        print(SEPARATOR_MAIN)
        print("ANÁLISIS COMPLETO DE MACHINE LEARNING")
        print("PREDICCIÓN DE ÉXITO DE CAFETERÍAS")
        print(SEPARATOR_MAIN)
        
        try:
            # Aplicar configuraciones
            if pipeline_config.data_path:
                self.config.DATA_PATH = pipeline_config.data_path
            self.config.RESULTS_PATH = pipeline_config.output_dir
            if pipeline_config.algorithms:
                self.config.ALGORITHMS_TO_COMPARE = pipeline_config.algorithms
            
            # Ejecutar pasos
            self._step_load_data(pipeline_config.sample_size)
            self._step_choose_metrics()
            self._step_evaluation_protocol()
            self._step_prepare_data(pipeline_config.retrain)
            
            if not pipeline_config.skip_viz:
                self._step_exploratory_analysis()
            
            self._step_train_models(pipeline_config.algorithms, pipeline_config.quick_mode)
            
            results = self._step_compare_models()
            
            print(SEPARATOR_MAIN)
            print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            print(SEPARATOR_MAIN)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error en pipeline: {str(e)}")
            print(f"\n❌ Error: {str(e)}")
            raise
    
    def _step_load_data(self, sample_size: Optional[int] = None) -> None:
        """Paso 1: Cargar datos."""
        print("\n🔄 PASO 1: RECOPILACIÓN DE DATOS")
        print(SEPARATOR_SECTION)
        
        start_time = time.time()
        
        self.logger.info("Cargando dataset...")
        self.df = self.data_loader.load_coffee_shop_data()
        
        print(f"✅ Dataset cargado:")
        print(f"   • Registros: {len(self.df):,}")
        print(f"   • Características: {self.df.shape[1]}")
        
        self.data_loader.get_data_summary(self.df)
        validation_report = self.data_loader.validate_data(self.df)
        
        # Muestreo si es necesario
        if sample_size and len(self.df) > sample_size:
            print(f"\n📦 Creando muestra de {sample_size:,} registros...")
            self.df = self.data_loader.sample_data(n=sample_size, random_state=42)
        
        print(f"✅ Datos preparados: {len(self.df):,} registros")
        
        self.execution_time['data_loading'] = time.time() - start_time
    
    def _step_choose_metrics(self) -> None:
        """Paso 2: Elegir métricas de éxito."""
        print("\n🔄 PASO 2: ELECCIÓN DE MEDIDAS DE ÉXITO")
        print(SEPARATOR_SECTION)
        
        print("📊 Métricas principales seleccionadas:")
        print("   • Accuracy: Proporción de predicciones correctas")
        print("   • Precision: Proporción de cafeterías exitosas correctamente identificadas")
        print("   • Recall: Proporción de cafeterías exitosas detectadas")
        print("   • F1-Score: Media armónica entre precision y recall")
        print("   • AUC-ROC: Área bajo la curva ROC")
        
        hyperparameter_config = self.config.get_hyperparameter_config()
        print(f"   • Métrica principal para optimización: {hyperparameter_config['scoring']}")
    
    def _step_evaluation_protocol(self) -> None:
        """Paso 3: Establecer protocolo de evaluación."""
        print("\n🔄 PASO 3: ESTABLECIMIENTO DE PROTOCOLO DE EVALUACIÓN")
        print(SEPARATOR_SECTION)
        
        preprocessing_config = self.config.get_preprocessing_config()
        hyperparameter_config = self.config.get_hyperparameter_config()
        
        print("🔬 Protocolo de evaluación establecido:")
        print(f"   • División train/test: {int((1-preprocessing_config['test_size'])*100)}/{int(preprocessing_config['test_size']*100)}%")
        print(f"   • Validación cruzada: {hyperparameter_config['cv_folds']} folds")
        print(f"   • Estratificación: {'Sí' if preprocessing_config['stratify'] else 'No'}")
        print(f"   • Semilla aleatoria: {hyperparameter_config['random_state']}")
    
    def _step_prepare_data(self, retrain: bool = False) -> None:
        """Paso 4: Preparar datos."""
        print("\n🔄 PASO 4: PREPARACIÓN DE DATOS")
        print(SEPARATOR_SECTION)
        
        start_time = time.time()
        
        cache_path = Path(self.config.RESULTS_PATH) / "preprocessed_data.pkl"
        
        # Intentar cargar desde caché
        if cache_path.exists() and not retrain:
            print("📦 Cargando datos preprocesados desde caché...")
            try:
                import joblib
                cached_data = joblib.load(cache_path)
                self.X_train = cached_data["X_train"]
                self.X_test = cached_data["X_test"]
                self.y_train = cached_data["y_train"]
                self.y_test = cached_data["y_test"]
                self.feature_names = cached_data["feature_names"]
                self.target_classes = cached_data["target_classes"]
                print("   ✅ Datos preprocesados cargados exitosamente")
                
                self.execution_time['data_preparation'] = time.time() - start_time
                return
            except Exception as e:
                print(f"   ❌ Error cargando caché: {e}")
        
        # Preprocesar datos
        print("Ejecutando pipeline de preprocesamiento...")
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.data_preprocessor.preprocess_pipeline(self.df)
        
        self.feature_names = self.data_preprocessor.get_feature_names()
        self.target_classes = self.data_preprocessor.get_target_classes() or [
            "No Exitosa", "Exitosa"
        ]
        
        # Guardar en caché
        print("💾 Guardando datos preprocesados en caché...")
        import joblib
        data_cache = {
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "feature_names": self.feature_names,
            "target_classes": self.target_classes,
        }
        FileUtils.ensure_directory(cache_path.parent)
        joblib.dump(data_cache, cache_path)
        print("   ✅ Datos preprocesados guardados en caché")
        
        print(f"✅ Preprocesamiento completado:")
        print(f"   • Características finales: {len(self.feature_names)}")
        print(f"   • Clases objetivo: {self.target_classes}")
        print(f"   • Train set: {self.X_train.shape}")
        print(f"   • Test set: {self.X_test.shape}")
        
        self.execution_time['data_preparation'] = time.time() - start_time
    
    def _step_exploratory_analysis(self) -> None:
        """Análisis exploratorio de datos."""
        print("\n📊 Análisis Exploratorio de Datos")
        print(SEPARATOR_SECTION)
        
        print("Generando visualizaciones de correlaciones...")
        df_numeric = self.df.select_dtypes(include=[np.number]).copy()
        
        correlation_path = Path(self.config.RESULTS_PATH) / "correlation_matrix.png"
        self.data_visualizer.plot_correlation_matrix(
            df_numeric, save_path=str(correlation_path), show=False
        )
        print(f"   ✅ Matriz de correlación guardada")
    
    def _step_train_models(self, algorithms: Optional[List[str]] = None, 
                          quick_mode: bool = False) -> None:
        """Paso 5 y 6: Entrenar modelos."""
        print("\n🔄 PASO 5-6: ENTRENAMIENTO DE MODELOS")
        print(SEPARATOR_SECTION)
        
        start_time = time.time()
        
        if algorithms:
            self.config.ALGORITHMS_TO_COMPARE = algorithms
        
        optimize_params = not quick_mode
        
        print(f"Entrenando {len(self.config.ALGORITHMS_TO_COMPARE)} algoritmos...")
        print(f"Optimización de hiperparámetros: {'Sí' if optimize_params else 'No (modo rápido)'}")
        
        from ..models import (
            LogisticRegressionClassifier,
            SVMClassifier,
            DecisionTreeClassifierCustom,
            RandomForestClassifierCustom,
            NeuralNetworkClassifier,
        )
        
        classifiers = {
            "LogisticRegression": LogisticRegressionClassifier(self.config),
            "SVM": SVMClassifier(self.config),
            "DecisionTree": DecisionTreeClassifierCustom(self.config),
            "RandomForest": RandomForestClassifierCustom(self.config),
            "NeuralNetwork": NeuralNetworkClassifier(self.config),
        }
        
        # Entrenar modelos seleccionados
        for algo_name in self.config.ALGORITHMS_TO_COMPARE:
            if algo_name not in classifiers:
                continue
            
            classifier = classifiers[algo_name]
            print(f"\n📊 Entrenando {algo_name}...")
            
            classifier.train(
                self.X_train, self.y_train,
                optimize_params=optimize_params,
                feature_names=self.feature_names
            )
            print(f"   ✅ {algo_name} entrenado")
        
        self.execution_time['model_training'] = time.time() - start_time
    
    def _step_compare_models(self) -> Dict[str, Any]:
        """Paso 7: Comparar modelos."""
        print("\n🔄 PASO 7: COMPARACIÓN DE MODELOS")
        print(SEPARATOR_SECTION)
        
        print("Evaluando y comparando todos los modelos...")
        
        results = {
            "timestamp": TimingUtils.get_timestamp(),
            "execution_times": self.execution_time,
            "models": {},
        }
        
        print(f"\n✅ Análisis completado")
        print(f"   • Tiempo total: {TimingUtils.format_duration(sum(self.execution_time.values()))}")
        
        return results
