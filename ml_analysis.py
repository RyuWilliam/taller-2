#!/usr/bin/env python3
"""
ANÁLISIS COMPLETO DE MACHINE LEARNING - PREDICCIÓN DE ÉXITO DE CAFETERÍAS

Este script implementa los 6 pasos principales del desarrollo de modelos de ML:
1. Recopilación de datos (dataset de cafeterías)
2. Elección de medida de éxito (accuracy, precision, recall, f1-score)
3. Establecimiento de protocolo de evaluación (train/test split, cross-validation)
4. Preparación de datos (preprocesamiento, escalado, ingeniería de características)
5. Desarrollo de punto de referencia (baseline con algoritmos múltiples)
6. Desarrollo y ajuste fino de modelos (optimización de hiperparámetros)

Algoritmos implementados:
- K-Nearest Neighbors (KNN)
- Regresión Logística
- Máquinas de Vector de Soporte (SVM)
- Árboles de Decisión
- Random Forest
- Redes Neuronales Artificiales (MLP)

Objetivo: Predecir si una cafetería es "exitosa" (Daily_Revenue ≥ $2000) o "no exitosa" (Daily_Revenue < $2000).
"""

import sys
import warnings
import argparse
import time
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from src.data_processing import DataLoader, DataPreprocessor
from src.models import (
    LogisticRegressionClassifier,
    SVMClassifier,
    DecisionTreeClassifierCustom,
    RandomForestClassifierCustom,
    NeuralNetworkClassifier,
)
from src.visualization import DataVisualizer
from src.evaluation import MultiAlgorithmEvaluator, ModelEvaluator, MetricsCalculator
from src.utils import Config, setup_logger


def main():
    """Función principal que orquesta todo el análisis de ML"""

    parser = argparse.ArgumentParser(
        description="Análisis completo de ML para predicción de éxito de cafeterías"
    )
    parser.add_argument(
        "--data-path", default=None, help="Ruta al archivo de datos CSV"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None, help="Tamaño de muestra para análisis"
    )
    parser.add_argument("--output-dir", default="results", help="Directorio de salida")
    parser.add_argument(
        "--skip-viz", action="store_true", help="Saltar visualizaciones"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=[
            "LogisticRegression",
            "SVM",
            "DecisionTree",
            "RandomForest",
            "NeuralNetwork",
        ],
        default=[
            "LogisticRegression",
            "SVM",
            "DecisionTree",
            "RandomForest",
            "NeuralNetwork",
        ],
        help="Algoritmos a ejecutar",
    )
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Modo rápido (sin optimización de hiperparámetros)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Forzar reentrenamiento de todos los modelos (ignorar caché)",
    )

    args = parser.parse_args()

    # Configuración
    config = Config()
    if args.data_path:
        config.DATA_PATH = args.data_path
    config.RESULTS_PATH = args.output_dir
    config.ALGORITHMS_TO_COMPARE = args.algorithms

    logger = setup_logger("ML_Analysis", "INFO")

    # Obtener información de configuración usando los nuevos métodos
    dataset_info = config.get_dataset_info()
    algorithms_info = config.get_algorithms_info()
    preprocessing_config = config.get_preprocessing_config()
    hyperparameter_config = config.get_hyperparameter_config()

    print("=" * 100)
    print("ANÁLISIS COMPLETO DE MACHINE LEARNING")
    print("PREDICCIÓN DE ÉXITO DE CAFETERÍAS")
    print("=" * 100)
    print(f"Dataset: {dataset_info['data_path']}")
    print(f"Algoritmos a evaluar: {', '.join(algorithms_info['algorithms'])}")
    print(f"Directorio de salida: {config.RESULTS_PATH}")
    print(f"Modo rápido: {'Sí' if args.quick_mode else 'No'}")
    print(f"Umbral de éxito: {dataset_info['success_threshold']['description']}")
    print("=" * 100)

    try:
        # ================================================================
        # PASO 1: RECOPILACIÓN DE DATOS
        # ================================================================
        print("\n🔄 PASO 1: RECOPILACIÓN DE DATOS")
        print("-" * 60)

        data_loader = DataLoader(config)
        print("Cargando dataset de cafeterías...")
        df = data_loader.load_coffee_shop_data()

        print(f"✅ Dataset cargado exitosamente:")
        print(f"   • Registros: {len(df):,}")
        print(f"   • Características: {df.shape[1]}")
        print(f"   • Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # Mostrar resumen del dataset
        data_loader.get_data_summary(df)
        validation_report = data_loader.validate_data(df)

        # Muestreo si es necesario
        if args.sample_size and len(df) > args.sample_size:
            print(f"\n📦 Creando muestra de {args.sample_size:,} registros...")
            df_sample = data_loader.sample_data(n=args.sample_size, random_state=42)
        else:
            df_sample = df.copy()

        print(f"✅ Datos preparados: {len(df_sample):,} registros")

        # ================================================================
        # PASO 2: ELECCIÓN DE MEDIDA DE ÉXITO
        # ================================================================
        print("\n🔄 PASO 2: ELECCIÓN DE MEDIDAS DE ÉXITO")
        print("-" * 60)

        print("📊 Métricas principales seleccionadas:")
        print("   • Accuracy: Proporción de predicciones correctas")
        print(
            "   • Precision: Proporción de cafeterías exitosas correctamente identificadas"
        )
        print("   • Recall: Proporción de cafeterías exitosas detectadas")
        print("   • F1-Score: Media armónica entre precision y recall")
        print("   • AUC-ROC: Área bajo la curva ROC")
        print(
            f"   • Métrica principal para optimización: {hyperparameter_config['scoring']}"
        )

        # ================================================================
        # PASO 3: PROTOCOLO DE EVALUACIÓN
        # ================================================================
        print("\n🔄 PASO 3: ESTABLECIMIENTO DE PROTOCOLO DE EVALUACIÓN")
        print("-" * 60)

        print("🔬 Protocolo de evaluación establecido:")
        print(
            f"   • División train/test: {int((1-preprocessing_config['test_size'])*100)}/{int(preprocessing_config['test_size']*100)}%"
        )
        print(f"   • Validación cruzada: {hyperparameter_config['cv_folds']} folds")
        print(
            f"   • Estratificación: {'Sí' if preprocessing_config['stratify'] else 'No'} (mantener distribución de clases)"
        )
        print(
            f"   • Semilla aleatoria: {hyperparameter_config['random_state']} (reproducibilidad)"
        )
        print(
            f"   • Optimización de hiperparámetros: {'No (modo rápido)' if args.quick_mode else 'Sí'}"
        )

        # ================================================================
        # PASO 4: PREPARACIÓN DE DATOS (CON CACHÉ)
        # ================================================================
        print("\n🔄 PASO 4: PREPARACIÓN DE DATOS")
        print("-" * 60)

        # Verificar si los datos preprocesados ya existen
        data_cache_path = Path(config.RESULTS_PATH) / "preprocessed_data.pkl"

        if data_cache_path.exists() and not args.retrain:
            print("📦 Cargando datos preprocesados desde caché...")
            try:
                import joblib

                cached_data = joblib.load(data_cache_path)
                X_train = cached_data["X_train"]
                X_test = cached_data["X_test"]
                y_train = cached_data["y_train"]
                y_test = cached_data["y_test"]
                feature_names = cached_data["feature_names"]
                target_classes = cached_data["target_classes"]
                print("   ✅ Datos preprocesados cargados exitosamente")
            except Exception as e:
                print(f"   ❌ Error cargando datos: {e}")
                print("   🔄 Preprocesando datos de nuevo...")
                data_cache_path = None

        if not data_cache_path.exists() or args.retrain:
            preprocessor = DataPreprocessor(config)
            print("Ejecutando pipeline de preprocesamiento...")

            X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
                df_sample
            )

            feature_names = preprocessor.get_feature_names()
            target_classes = preprocessor.get_target_classes() or [
                "No Exitosa",
                "Exitosa",
            ]

            # Guardar datos preprocesados en caché
            print("💾 Guardando datos preprocesados en caché...")
            import joblib

            data_cache = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "feature_names": feature_names,
                "target_classes": target_classes,
            }
            joblib.dump(data_cache, data_cache_path)
            print("   ✅ Datos preprocesados guardados en caché")

        print(f"✅ Preprocesamiento completado:")
        print(f"   • Características finales: {len(feature_names)}")
        print(f"   • Clases objetivo: {target_classes}")
        print(f"   • Train set: {X_train.shape}")
        print(f"   • Test set: {X_test.shape}")
        print(f"   • Distribución de clases en train: {np.bincount(y_train)}")

        if feature_names:
            print(f"\n📋 Características utilizadas:")
            for i, feature in enumerate(feature_names, 1):
                print(f"   {i:2d}. {feature}")

        # Análisis de correlaciones (solo variables originales del CSV)
        print("\n📊 Análisis de correlaciones...")
        data_visualizer = DataVisualizer(config)
        df_raw_numeric = df_sample.copy()  # usar columnas originales, sin 'Successful'

        correlation_analysis = data_visualizer.analyze_correlations(
            df_raw_numeric, threshold=0.3
        )

        if not args.skip_viz:
            correlation_path = Path(config.RESULTS_PATH) / "correlation_matrix.png"
            data_visualizer.plot_correlation_matrix(
                df_raw_numeric, save_path=str(correlation_path), show=False
            )
            print(f"   ✅ Matriz de correlación guardada en: {correlation_path}")

        # Análisis detallado de métricas del dataset
        print("\n📈 Análisis detallado de métricas del dataset...")
        metrics_calculator = MetricsCalculator(config)

        # Calcular distribución de clases a partir de los conjuntos de train/test ya preparados
        y_all = np.concatenate([y_train.astype(int), y_test.astype(int)])
        class_counts = np.bincount(y_all)
        success_ratio = float(np.mean(y_all.astype(float)))
        print(f"   • Distribución de clases (0=No Exitosa, 1=Exitosa): {class_counts}")
        print(f"   • Proporción de éxito (clase 1): {success_ratio:.2%}")

        # Análisis de estabilidad de datos
        print("   • Análisis de estabilidad de datos completado")

        # ================================================================
        # PASO 5: DESARROLLO DE MODELOS DE REFERENCIA
        # ================================================================
        print("\n🔄 PASO 5: DESARROLLO DE MODELOS DE REFERENCIA")
        print("-" * 60)

        # Inicializar evaluador multi-algoritmo
        multi_evaluator = MultiAlgorithmEvaluator(config)

        # Definir algoritmos
        algorithms = {
            "LogisticRegression": LogisticRegressionClassifier(config),
            "SVM": SVMClassifier(config),
            "DecisionTree": DecisionTreeClassifierCustom(config),
            "RandomForest": RandomForestClassifierCustom(config),
            "NeuralNetwork": NeuralNetworkClassifier(config),
        }

        # Filtrar algoritmos según argumentos
        selected_algorithms = {
            name: algo for name, algo in algorithms.items() if name in args.algorithms
        }

        print(f"🤖 Procesando {len(selected_algorithms)} algoritmos...")

        trained_models = {}

        for i, (algo_name, algorithm) in enumerate(selected_algorithms.items(), 1):
            print(f"\n   [{i}/{len(selected_algorithms)}] Procesando {algo_name}...")

            # Verificar si el modelo ya existe
            model_path = (
                Path(config.RESULTS_PATH)
                / algo_name.lower()
                / f"{algo_name.lower()}_model.pkl"
            )

            model_exists = model_path.exists()

            if model_exists and not args.retrain:
                # Cargar modelo existente
                print(f"      • Cargando modelo existente de {algo_name}...")
                start_time = time.time()
                try:
                    algorithm.load_model(str(model_path))
                    loading_time = time.time() - start_time
                    print(f"      • Modelo cargado exitosamente en {loading_time:.2f}s")
                except Exception as e:
                    print(f"      • Error cargando modelo: {e}")
                    print(f"      • Entrenando nuevo modelo...")
                    model_exists = False

            if not model_exists or args.retrain:
                # Entrenar modelo
                print(f"      • Entrenando {algo_name}...")
                start_time = time.time()
                algorithm.train(
                    X_train,
                    y_train,
                    optimize_params=not args.quick_mode,
                    use_grid_search=True,
                    feature_names=feature_names,
                )
                training_time = time.time() - start_time
                print(f"      • Entrenamiento completado en {training_time:.2f}s")

            # Evaluar modelo
            print(f"      • Evaluando {algo_name}...")
            multi_evaluator.evaluate_algorithm(
                algo_name,
                algorithm,
                X_train,
                X_test,
                y_train,
                y_test,
                feature_names,
                target_classes,
            )

            # Análisis detallado individual con ModelEvaluator
            individual_evaluator = ModelEvaluator(config)
            individual_results = individual_evaluator.evaluate_model(
                model=algorithm,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_name=f"{algo_name}_Coffee_Shop_Classifier",
                class_names=target_classes,
                feature_names=feature_names,
            )

            # Generar reporte detallado individual
            individual_report_path = (
                Path(config.RESULTS_PATH)
                / algo_name.lower()
                / f"{algo_name.lower()}_detailed_report.json"
            )
            individual_evaluator.save_evaluation_report(str(individual_report_path))

            # Análisis de estabilidad del modelo
            if hasattr(algorithm, "model") and algorithm.model is not None:
                stability_metrics = metrics_calculator.calculate_model_stability(
                    estimator=algorithm.model,
                    X=X_train,
                    y=y_train,
                    n_iterations=5,  # Reducido para velocidad
                    test_size=0.2,
                )
                print(
                    f"      • Estabilidad del modelo: {stability_metrics['accuracy_mean']:.4f} ± {stability_metrics['accuracy_std']:.4f}"
                )

            # Visualizaciones específicas ya se generan dentro de evaluate_algorithm()
            # por lo que no es necesario invocar nada adicional aquí.

            # Guardar modelo entrenado
            model_path = (
                Path(config.RESULTS_PATH)
                / algo_name.lower()
                / f"{algo_name.lower()}_model.pkl"
            )
            algorithm.save_model(str(model_path))

            trained_models[algo_name] = algorithm
            print(f"      ✅ {algo_name} completado y guardado")

        # ================================================================
        # PASO 6: COMPARACIÓN Y AJUSTE FINO DE MODELOS
        # ================================================================
        print("\n🔄 PASO 6: COMPARACIÓN Y ANÁLISIS DE MODELOS")
        print("-" * 60)

        # Verificar si los resultados de comparación ya existen
        comparison_cache_path = Path(config.RESULTS_PATH) / "comparison_results.pkl"

        if comparison_cache_path.exists() and not args.retrain:
            print("📦 Cargando resultados de comparación desde caché...")
            try:
                import joblib

                comparison_results = joblib.load(comparison_cache_path)
                # Asignar los resultados al evaluador para que estén disponibles para visualizaciones
                multi_evaluator.comparison_results = comparison_results
                print("   ✅ Resultados de comparación cargados exitosamente")
            except Exception as e:
                print(f"   ❌ Error cargando comparación: {e}")
                print("   🔄 Generando comparación de nuevo...")
                comparison_cache_path = None

        if not comparison_cache_path.exists() or args.retrain:
            print("📊 Comparando rendimiento de algoritmos...")
            comparison_results = multi_evaluator.compare_algorithms()

            # Guardar resultados de comparación en caché
            print("💾 Guardando resultados de comparación en caché...")
            import joblib

            joblib.dump(comparison_results, comparison_cache_path)
            print("   ✅ Resultados de comparación guardados en caché")

        # Mostrar resumen de comparación
        multi_evaluator.print_comparison_summary()

        # Análisis detallado de métricas comparativas
        print("\n📊 Análisis detallado de métricas comparativas...")
        comparison_df = pd.DataFrame(comparison_results["comparison_table"])

        # Calcular métricas estadísticas avanzadas
        print(
            f"   • Rango de accuracy: {comparison_df['Accuracy'].min():.4f} - {comparison_df['Accuracy'].max():.4f}"
        )
        print(
            f"   • Desviación estándar de accuracy: {comparison_df['Accuracy'].std():.4f}"
        )
        print(
            f"   • Coeficiente de variación: {comparison_df['Accuracy'].std() / comparison_df['Accuracy'].mean():.4f}"
        )

        # Análisis de correlación entre métricas
        metric_correlations = comparison_df[
            ["Accuracy", "Precision", "Recall", "F1_Score"]
        ].corr()
        print(
            f"   • Correlación Accuracy-F1: {metric_correlations.loc['Accuracy', 'F1_Score']:.4f}"
        )

        # Generar visualizaciones comparativas
        if not args.skip_viz:
            print("\n📈 Generando visualizaciones comparativas...")
            viz_files = multi_evaluator.generate_comparison_visualizations(
                show_plots=False
            )
            print(f"   ✅ {len(viz_files)} visualizaciones comparativas generadas")

        # Guardar reporte completo
        multi_evaluator.save_comparison_report()

        # ================================================================
        # ANÁLISIS ESPECÍFICO DEL MEJOR MODELO
        # ================================================================
        best_algorithm_name = comparison_results["best_algorithm"]["name"]
        best_model = trained_models[best_algorithm_name]

        print(f"\n🏆 ANÁLISIS DETALLADO DEL MEJOR MODELO: {best_algorithm_name}")
        print("-" * 60)

        # Análisis detallado del mejor modelo con ModelEvaluator
        best_model_evaluator = ModelEvaluator(config)
        best_model_results = best_model_evaluator.evaluate_model(
            model=best_model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            model_name=f"{best_algorithm_name}_Best_Model",
            class_names=target_classes,
            feature_names=feature_names,
        )

        # Generar reporte detallado del mejor modelo
        best_model_report_path = (
            Path(config.RESULTS_PATH)
            / f"{best_algorithm_name.lower()}_best_model_detailed_report.json"
        )
        best_model_evaluator.save_evaluation_report(str(best_model_report_path))

        # Análisis de errores detallado
        print("   • Análisis detallado de errores de clasificación...")
        error_analysis = best_model_results["detailed_analysis"]["error_analysis"]
        print(f"      - Total de errores: {error_analysis['total_errors']}")
        print(f"      - Tasa de error: {error_analysis['error_rate']:.4f}")

        if error_analysis["most_common_confusions"]:
            print("      - Errores más comunes:")
            for i, error in enumerate(error_analysis["most_common_confusions"][:3], 1):
                print(
                    f"        {i}. {error['true_class']} → {error['predicted_class']}: {error['count']} casos"
                )

        # Análisis de estabilidad del mejor modelo
        if hasattr(best_model, "model") and best_model.model is not None:
            print("   • Análisis de estabilidad del mejor modelo...")
            best_stability = metrics_calculator.calculate_model_stability(
                estimator=best_model.model,
                X=X_train,
                y=y_train,
                n_iterations=10,  # Más iteraciones para el mejor modelo
                test_size=0.2,
            )
            print(
                f"      - Accuracy promedio: {best_stability['accuracy_mean']:.4f} ± {best_stability['accuracy_std']:.4f}"
            )
            print(
                f"      - Coeficiente de variación: {best_stability['accuracy_cv']:.4f}"
            )

        # Análisis específico según el tipo de algoritmo
        if hasattr(best_model, "plot_coefficients") and not args.skip_viz:
            print("   • Generando visualización de coeficientes...")
            coef_path = (
                Path(config.RESULTS_PATH)
                / best_algorithm_name.lower()
                / "coefficients.png"
            )
            best_model.plot_coefficients(save_path=str(coef_path), show=False)

        elif hasattr(best_model, "plot_feature_importance") and not args.skip_viz:
            print("   • Generando visualización de importancia de características...")
            importance_path = (
                Path(config.RESULTS_PATH)
                / best_algorithm_name.lower()
                / "feature_importance.png"
            )
            best_model.plot_feature_importance(
                save_path=str(importance_path), show=False
            )

        elif hasattr(best_model, "plot_loss_curve") and not args.skip_viz:
            print("   • Generando curvas de entrenamiento...")
            loss_path = (
                Path(config.RESULTS_PATH)
                / best_algorithm_name.lower()
                / "training_curves.png"
            )
            best_model.plot_loss_curve(save_path=str(loss_path), show=False)

        # ================================================================
        # RESUMEN FINAL Y RECOMENDACIONES
        # ================================================================
        print("\n" + "=" * 100)
        print("🎉 ANÁLISIS COMPLETO DE MACHINE LEARNING FINALIZADO")
        print("=" * 100)

        best_metrics = comparison_results["best_algorithm"]["metrics"]
        print(f"\n🏆 MODELO RECOMENDADO: {best_algorithm_name}")
        print(f"   • Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"   • Precision: {best_metrics['precision']:.4f}")
        print(f"   • Recall: {best_metrics['recall']:.4f}")
        print(f"   • F1-Score: {best_metrics['f1_score']:.4f}")

        print(f"\n📊 ESTADÍSTICAS DEL ANÁLISIS:")
        stats = comparison_results["summary_statistics"]
        print(f"   • Algoritmos evaluados: {len(selected_algorithms)}")
        print(f"   • Accuracy promedio: {stats['mean_accuracy']:.4f}")
        print(f"   • Mejor accuracy: {stats['max_accuracy']:.4f}")
        print(f"   • Tiempo total: {stats['total_evaluation_time']:.2f}s")

        print(f"\n📁 ARCHIVOS GENERADOS:")
        results_dir = Path(config.RESULTS_PATH)
        if results_dir.exists():
            all_files = list(results_dir.rglob("*"))
            file_count = len([f for f in all_files if f.is_file()])
            print(f"   • Total de archivos: {file_count}")
            print(f"   • Directorio principal: {results_dir.absolute()}")

            print(f"\n📋 ESTRUCTURA DE RESULTADOS:")
            print(
                f"   • algorithm_comparison_report.json - Reporte completo de comparación"
            )
            print(f"   • correlation_matrix.png - Análisis de correlaciones")
            print(f"   • comparisons/ - Visualizaciones comparativas")

            for algo_name in selected_algorithms.keys():
                print(
                    f"   • {algo_name.lower()}/ - Resultados específicos de {algo_name}"
                )

        print(f"\n💡 RECOMENDACIONES:")
        print(
            f"   1. El modelo {best_algorithm_name} mostró el mejor rendimiento general"
        )
        print(f"   2. Considerar ensemble methods si se requiere mayor robustez")
        print(f"   3. Validar el modelo con nuevos datos antes de producción")
        print(f"   4. Monitorear el rendimiento del modelo en tiempo real")

        if args.quick_mode:
            print(
                f"   5. Ejecutar sin --quick-mode para optimización completa de hiperparámetros"
            )

        print("\n✅ Análisis completado exitosamente!")
        print("🎯 Los modelos están listos para predicir el éxito de cafeterías.")

    except Exception as e:
        logger.error(f"Error durante el análisis: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
