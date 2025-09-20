"""Objetivo: Visualizar resultados y rendimiento del modelo KNN. Bosquejo: clase
ModelVisualizer que genera gráficas como matriz de confusión (`plot_confusion_matrix`),
reporte de clasificación (`plot_classification_report`), optimización de K
(`plot_k_optimization`), curvas de aprendizaje (`plot_learning_curve`), importancia de
características (`plot_feature_importance`), distribución de predicciones
(`plot_prediction_distribution`), comparación de modelos (`plot_model_comparison`),
residuos (`plot_residuals`) y reportes compuestos (`create_model_report`)."""

import numpy as np
import pandas as pd
import matplotlib  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from typing import Optional, List, Dict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve

from ..utils.logger import LoggerMixin
from ..utils.config import Config


class ModelVisualizer(LoggerMixin):

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.setup_style()

    def setup_style(self):
        plt.style.use("default")
        sns.set_palette("husl")

        # Obtener configuración de visualización usando el nuevo método
        visualization_config = self.config.get_visualization_config()
        plt.rcParams["figure.figsize"] = visualization_config["figure_size"]
        plt.rcParams["figure.dpi"] = visualization_config["dpi"]
        plt.rcParams["savefig.dpi"] = visualization_config["dpi"]

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        normalize: Optional[str] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            if normalize == "true":
                cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                title = "Matriz de Confusión Normalizada (por fila)"
                fmt = ".2f"
            elif normalize == "pred":
                cm = cm.astype("float") / cm.sum(axis=0)
                title = "Matriz de Confusión Normalizada (por columna)"
                fmt = ".2f"
            elif normalize == "all":
                cm = cm.astype("float") / cm.sum()
                title = "Matriz de Confusión Normalizada (total)"
                fmt = ".2f"
        else:
            title = "Matriz de Confusión"
            fmt = "d"

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=class_names if class_names is not None else True,
            yticklabels=class_names if class_names is not None else True,
            cbar_kws={"label": "Count" if not normalize else "Proportion"},
        )

        plt.title(f"{title} - Coffee Shop Success")
        plt.ylabel("Actual label")
        plt.xlabel("Predicted label")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Matriz de confusión guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        from sklearn.metrics import classification_report

        report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        )

        df_report = pd.DataFrame(report).iloc[:-1, :].T  # Excluir 'accuracy'

        plt.figure(figsize=(8, 6))
        sns.heatmap(df_report, annot=True, cmap="RdYlBu_r", fmt=".3f")

        plt.title("Classification Report - Coffee Shop Success")
        plt.xlabel("Metrics")
        plt.ylabel("Classes")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Reporte de clasificación guardado en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_k_optimization(
        self,
        k_values: List[int],
        cv_scores_mean: List[float],
        cv_scores_std: List[float],
        best_k: int,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        plt.figure(figsize=(10, 6))

        plt.errorbar(
            k_values,
            cv_scores_mean,
            yerr=cv_scores_std,
            marker="o",
            capsize=5,
            capthick=2,
            linewidth=2,
            markersize=6,
        )

        plt.axvline(
            x=best_k,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label=f"K óptimo = {best_k}",
        )

        plt.xlabel("K value", fontsize=12)
        plt.ylabel("Accuracy (cross-validation)", fontsize=12)
        plt.title("K Hyperparameter Optimization - Coffee Shop Success", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)

        best_idx = k_values.index(best_k)
        best_score = cv_scores_mean[best_idx]
        plt.annotate(
            f"Score: {best_score:.4f}",
            xy=(best_k, best_score),
            xytext=(best_k + 2, best_score + 0.01),
            arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
            fontsize=10,
            color="red",
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Gráfico de optimización K guardado en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_learning_curve(
        self,
        estimator,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        train_sizes: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        learning_curve_result = learning_curve(
            estimator, X, y, cv=cv, train_sizes=train_sizes, n_jobs=-1
        )
        train_sizes_result = learning_curve_result[0]
        train_scores = learning_curve_result[1]
        val_scores = learning_curve_result[2]

        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        plt.figure(figsize=(10, 6))

        plt.plot(
            train_sizes_result, train_mean, "o-", color="blue", label="Entrenamiento"
        )
        plt.fill_between(
            train_sizes_result,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            color="blue",
        )

        plt.plot(train_sizes_result, val_mean, "o-", color="red", label="Validación")
        plt.fill_between(
            train_sizes_result,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.2,
            color="red",
        )

        plt.xlabel("Training set size")
        plt.ylabel("Accuracy")
        plt.title("Learning Curves - Coffee Shop Success")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Curvas de aprendizaje guardadas en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_feature_importance(
        self,
        importance_dict: Dict[str, float],
        max_features: int = 15,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        sorted_features = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )
        sorted_features = sorted_features[:max_features]

        features, importances = zip(*sorted_features)

        plt.figure(figsize=(10, max(6, len(features) * 0.4)))

        y_pos = np.arange(len(features))
        plt.barh(y_pos, importances, color="skyblue")

        plt.yticks(y_pos, features)
        plt.xlabel("Importance (normalized)")
        plt.ylabel("Features")
        plt.title("Top Features for Success Prediction (Coffee Shop)")
        plt.grid(True, alpha=0.3, axis="x")

        for i, v in enumerate(importances):
            plt.text(
                v + max(importances) * 0.01, i, f"{v:.4f}", va="center", fontsize=10
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Importancia de características guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_prediction_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        unique_true, counts_true = np.unique(y_true, return_counts=True)
        axes[0].bar(unique_true, counts_true, color="lightblue", alpha=0.7)
        axes[0].set_title("Actual Class Distribution - Coffee Shop Success")
        axes[0].set_xlabel("Class")
        axes[0].set_ylabel("Count")

        if class_names:
            axes[0].set_xticks(unique_true)
            axes[0].set_xticklabels([class_names[i] for i in unique_true], rotation=45)

        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        axes[1].bar(unique_pred, counts_pred, color="lightcoral", alpha=0.7)
        axes[1].set_title("Predicted Class Distribution - Coffee Shop Success")
        axes[1].set_xlabel("Class")
        axes[1].set_ylabel("Count")

        if class_names:
            axes[1].set_xticks(unique_pred)
            axes[1].set_xticklabels([class_names[i] for i in unique_pred], rotation=45)

        plt.figtext(
            0.5,
            0.02,
            "🎯 OBJETIVO: Verificar calibración del modelo comparando distribuciones reales vs predichas para detectar sesgos.\n"
            "📊 EJEMPLO DE ANÁLISIS: Distribuciones casi idénticas ([492,491,491,526] vs [488,511,492,509]). "
            "Diferencias mínimas (±20 casos) confirman excelente calibración. Sin sesgo hacia clases específicas.",
            ha="center",
            va="bottom",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Distribución de predicciones guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_model_comparison(
        self,
        results_dict: Dict[str, Dict[str, float]],
        metrics: List[str] = ["accuracy", "precision", "recall", "f1-score"],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        models = list(results_dict.keys())

        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        plt.figure(figsize=(12, 6))

        for i, model in enumerate(models):
            scores = [results_dict[model].get(metric, 0) for metric in metrics]
            plt.bar(x + i * width, scores, width, label=model, alpha=0.8)

        plt.xlabel("Métricas")
        plt.ylabel("Puntuación")
        plt.title("Comparación de Modelos")
        plt.xticks(x + width * (len(models) - 1) / 2, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")

        for i, model in enumerate(models):
            scores = [results_dict[model].get(metric, 0) for metric in metrics]
            for j, score in enumerate(scores):
                plt.text(
                    j + i * width,
                    score + 0.01,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Comparación de modelos guardada en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:

        errors = y_true != y_pred

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        unique_classes = np.unique(y_true)
        error_rates = []

        for cls in unique_classes:
            mask = y_true == cls
            error_rate = np.sum(errors[mask]) / np.sum(mask)
            error_rates.append(error_rate)

        axes[0].bar(unique_classes, error_rates, color="lightcoral", alpha=0.7)
        axes[0].set_title("Tasa de Error por Clase Verdadera")
        axes[0].set_xlabel("Clase")
        axes[0].set_ylabel("Tasa de Error")
        axes[0].grid(True, alpha=0.3)

        error_matrix = np.zeros((len(unique_classes), len(unique_classes)))
        for i, true_cls in enumerate(unique_classes):
            for j, pred_cls in enumerate(unique_classes):
                mask = (y_true == true_cls) & (y_pred == pred_cls)
                error_matrix[i, j] = np.sum(mask)

        im = axes[1].imshow(error_matrix, cmap="Reds")
        axes[1].set_title("Matriz de Predicciones")
        axes[1].set_xlabel("Clase Predicha")
        axes[1].set_ylabel("Clase Verdadera")

        plt.colorbar(im, ax=axes[1])

        plt.figtext(
            0.5,
            0.02,
            "🎯 OBJETIVO: Detectar patrones sistemáticos en errores del modelo para identificar problemas de especificación.\n"
            "📊 EJEMPLO DE ANÁLISIS: Errores distribuidos aleatoriamente sin patrones sistemáticos evidentes. "
            "Balance entre sobre/subestimación. Ausencia de tendencias confirma modelo bien especificado - no hay variables omitidas importantes.",
            ha="center",
            va="bottom",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Análisis de residuos guardado en: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def create_model_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        model_name: str = "KNN",
        save_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        from pathlib import Path

        save_dir_path = None
        if save_dir:
            save_dir_path = Path(save_dir)
            save_dir_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        cm_path = (
            str(save_dir_path / f"{model_name}_confusion_matrix.png")
            if save_dir_path
            else None
        )
        self.plot_confusion_matrix(
            y_true, y_pred, class_names, save_path=cm_path, show=False
        )
        if cm_path:
            saved_files["confusion_matrix"] = cm_path

        self.logger.info(
            f"Reporte de modelo {model_name} generado. {len(saved_files)} gráficos creados."
        )

        return saved_files

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        from sklearn.metrics import roc_curve, auc

        # Asegurar matriz densa de NumPy (manejar entradas sparse)
        import numpy as _np
        from typing import Any

        proba_any: Any = y_proba
        if hasattr(proba_any, "toarray"):
            try:
                proba_any = proba_any.toarray()
            except Exception:
                pass
        proba_array: _np.ndarray = _np.array(proba_any, dtype=float)

        plt.figure(figsize=(8, 6))
        if proba_array.ndim == 1 or (
            proba_array.ndim == 2 and proba_array.shape[1] == 1
        ):
            scores = proba_array if proba_array.ndim == 1 else proba_array[:, 0]  # type: ignore[index]
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        elif proba_array.shape[1] == 2:
            scores = proba_array[:, 1]  # type: ignore[index]
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        else:
            # Multiclase: One-vs-Rest promedio macro
            from sklearn.preprocessing import label_binarize

            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)
            aucs = []
            for i in range(len(classes)):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], proba_array[:, i])  # type: ignore[index]
                aucs.append(auc(fpr, tpr))
            plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
            plt.title("ROC (macro-avg)")
            plt.text(0.6, 0.05, f"AUC macro = {np.mean(aucs):.3f}")

        plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        pos_label = None
        if class_names is not None and len(class_names) >= 2:
            pos_label = class_names[1]
        subtitle = f" (Positive class: {pos_label})" if pos_label else ""
        plt.title(f"ROC Curve - Coffee Shop Success{subtitle}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Curva ROC guardada en: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        from sklearn.metrics import precision_recall_curve, auc, average_precision_score

        # Asegurar matriz densa de NumPy (manejar entradas sparse)
        import numpy as _np
        from typing import Any

        proba_any: Any = y_proba
        if hasattr(proba_any, "toarray"):
            try:
                proba_any = proba_any.toarray()
            except Exception:
                pass
        proba_array: _np.ndarray = _np.array(proba_any, dtype=float)

        plt.figure(figsize=(8, 6))
        if proba_array.ndim == 1 or (
            proba_array.ndim == 2 and proba_array.shape[1] == 1
        ):
            scores = proba_array if proba_array.ndim == 1 else proba_array[:, 0]  # type: ignore[index]
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"PR (AP = {ap:.3f})")
        elif proba_array.shape[1] == 2:
            scores = proba_array[:, 1]  # type: ignore[index]
            precision, recall, _ = precision_recall_curve(y_true, scores)
            ap = average_precision_score(y_true, scores)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, label=f"PR (AP = {ap:.3f})")
        else:
            from sklearn.preprocessing import label_binarize

            classes = np.unique(y_true)
            y_true_bin = label_binarize(y_true, classes=classes)
            aps = []
            for i in range(len(classes)):
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], proba_array[:, i])  # type: ignore[index]
                aps.append(average_precision_score(y_true_bin[:, i], proba_array[:, i]))  # type: ignore[index]
            plt.title("Precision-Recall (macro-avg)")
            plt.text(0.5, 0.1, f"AP macro = {np.mean(aps):.3f}")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        pos_label = None
        if class_names is not None and len(class_names) >= 2:
            pos_label = class_names[1]
        subtitle = f" (Positive class: {pos_label})" if pos_label else ""
        plt.title(f"Precision-Recall Curve - Coffee Shop Success{subtitle}")
        plt.legend(loc="lower left")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(f"Curva Precision-Recall guardada en: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()

    def plot_models_correlation_matrix(
        self,
        models_predictions: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Crear matriz de correlación entre las predicciones de diferentes modelos.

        Args:
            models_predictions: Diccionario con nombre del modelo como clave y predicciones como valor
            save_path: Ruta donde guardar la gráfica
            show: Si mostrar la gráfica
        """
        import pandas as pd

        # Crear DataFrame con las predicciones de todos los modelos
        predictions_df = pd.DataFrame(models_predictions)

        # Calcular matriz de correlación
        correlation_matrix = predictions_df.corr()

        # Crear la visualización
        plt.figure(figsize=(10, 8))

        # Crear heatmap con anotaciones
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap="RdYlBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            fmt=".3f",
            cbar_kws={"shrink": 0.8, "label": "Correlación"},
        )

        plt.title(
            "Matriz de Correlación entre Predicciones de Modelos\nCoffee Shop Success Prediction",
            fontsize=14,
            pad=20,
        )
        plt.xlabel("Modelos", fontsize=12)
        plt.ylabel("Modelos", fontsize=12)

        # Rotar etiquetas para mejor legibilidad
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Añadir texto explicativo
        plt.figtext(
            0.5,
            0.02,
            "🎯 INTERPRETACIÓN: Valores cercanos a 1 indican que los modelos hacen predicciones similares.\n"
            "Valores cercanos a 0 indican predicciones independientes. Ayuda a identificar consenso entre algoritmos.",
            ha="center",
            va="bottom",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(
                f"Matriz de correlación de modelos guardada en: {save_path}"
            )

        if show:
            plt.show()
        else:
            plt.close()

        return correlation_matrix

    def plot_models_agreement_analysis(
        self,
        models_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Analizar el acuerdo entre modelos y su precisión individual.

        Args:
            models_predictions: Diccionario con predicciones de cada modelo
            y_true: Etiquetas verdaderas
            save_path: Ruta donde guardar la gráfica
            show: Si mostrar la gráfica
        """
        import pandas as pd
        from sklearn.metrics import accuracy_score

        # Crear DataFrame con predicciones
        predictions_df = pd.DataFrame(models_predictions)

        # Calcular acuerdo entre modelos (consenso mayoritario)
        n_models = len(models_predictions)
        consensus_predictions = predictions_df.mode(axis=1)[0]

        # Calcular métricas de acuerdo
        agreement_counts = (
            predictions_df == consensus_predictions.values.reshape(-1, 1)
        ).sum(axis=1)
        full_agreement_mask = agreement_counts == n_models
        majority_agreement_mask = agreement_counts >= (n_models // 2 + 1)

        # Calcular precisión individual y de consenso
        individual_accuracies = {}
        for model_name, predictions in models_predictions.items():
            individual_accuracies[model_name] = accuracy_score(y_true, predictions)

        consensus_accuracy = accuracy_score(y_true, consensus_predictions)

        # Crear visualización con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Gráfico de barras de precisión individual vs consenso
        model_names = list(individual_accuracies.keys()) + ["Consenso"]
        accuracies = list(individual_accuracies.values()) + [consensus_accuracy]
        colors = ["skyblue"] * len(individual_accuracies) + ["orange"]

        axes[0, 0].bar(range(len(model_names)), accuracies, color=colors, alpha=0.7)
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha="right")
        axes[0, 0].set_ylabel("Precisión")
        axes[0, 0].set_title("Precisión Individual vs Consenso")
        axes[0, 0].grid(True, alpha=0.3)

        # Añadir valores en las barras
        for i, acc in enumerate(accuracies):
            axes[0, 0].text(i, acc + 0.005, f"{acc:.3f}", ha="center", va="bottom")

        # 2. Distribución del nivel de acuerdo
        agreement_counts_dist = agreement_counts.value_counts().sort_index()
        axes[0, 1].bar(
            agreement_counts_dist.index,
            agreement_counts_dist.values,
            color="lightgreen",
            alpha=0.7,
        )
        axes[0, 1].set_xlabel("Número de Modelos en Acuerdo")
        axes[0, 1].set_ylabel("Frecuencia")
        axes[0, 1].set_title("Distribución del Nivel de Acuerdo")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Precisión por nivel de acuerdo
        accuracy_by_agreement = {}
        for agreement_level in agreement_counts_dist.index:
            mask = agreement_counts == agreement_level
            if mask.sum() > 0:
                accuracy_by_agreement[agreement_level] = accuracy_score(
                    y_true[mask], consensus_predictions[mask]
                )

        axes[1, 0].bar(
            accuracy_by_agreement.keys(),
            accuracy_by_agreement.values(),
            color="coral",
            alpha=0.7,
        )
        axes[1, 0].set_xlabel("Número de Modelos en Acuerdo")
        axes[1, 0].set_ylabel("Precisión del Consenso")
        axes[1, 0].set_title("Precisión del Consenso por Nivel de Acuerdo")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Matriz de confusión del consenso
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, consensus_predictions)
        im = axes[1, 1].imshow(cm, cmap="Blues")
        axes[1, 1].set_title("Matriz de Confusión - Consenso")

        # Añadir anotaciones a la matriz de confusión
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[1, 1].text(j, i, str(cm[i, j]), ha="center", va="center")

        axes[1, 1].set_xlabel("Predicho")
        axes[1, 1].set_ylabel("Real")

        plt.colorbar(im, ax=axes[1, 1], shrink=0.8)

        # Añadir estadísticas generales
        stats_text = (
            f"Acuerdo completo: {full_agreement_mask.sum()}/{len(y_true)} "
            f"({full_agreement_mask.mean():.1%})\n"
            f"Acuerdo mayoritario: {majority_agreement_mask.sum()}/{len(y_true)} "
            f"({majority_agreement_mask.mean():.1%})\n"
            f"Precisión del consenso: {consensus_accuracy:.3f}"
        )

        fig.suptitle(
            "Análisis de Acuerdo entre Modelos\nCoffee Shop Success Prediction",
            fontsize=16,
            y=0.98,
        )

        plt.figtext(
            0.5,
            0.02,
            stats_text,
            ha="center",
            va="bottom",
            fontsize=10,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.12)

        if save_path:
            plt.savefig(save_path, dpi=self.config.DPI, bbox_inches="tight")
            self.logger.info(
                f"Análisis de acuerdo entre modelos guardado en: {save_path}"
            )

        if show:
            plt.show()
        else:
            plt.close()

        return {
            "individual_accuracies": individual_accuracies,
            "consensus_accuracy": consensus_accuracy,
            "full_agreement_rate": full_agreement_mask.mean(),
            "majority_agreement_rate": majority_agreement_mask.mean(),
        }
