import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve


# ============================================================
# 1. Reliability diagram
# ============================================================

def plot_reliability_diagram(y_true, y_prob, n_bins=10,
                             title="Reliability diagram",
                             filename="reliability_diagram.png"):
    """
    Визуализация reliability diagram для оценки калибровки.

    Args:
        y_true: np.array, истинные метки (0 или 1)
        y_prob: np.array, предсказанные вероятности
        n_bins: int, число бинов
        title: str, заголовок графика
        filename: str, имя файла для сохранения
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Reliability diagram ---
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    axes[0].plot(prob_pred, prob_true, 'b-', marker='o', linewidth=2, label='Model')
    axes[0].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect calibration')
    axes[0].fill_between([0, 1], [0, 1], [0, 1], alpha=0.1, color='red')

    axes[0].set_xlabel('Predicted probability', fontsize=11)
    axes[0].set_ylabel('Empirical frequency', fontsize=11)
    axes[0].set_title(title, fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # --- Гистограмма предсказанных вероятностей ---
    axes[1].hist(y_prob, bins=n_bins, color='#1976D2', alpha=0.7, edgecolor='white')
    axes[1].set_xlabel('Predicted probability', fontsize=11)
    axes[1].set_ylabel('Number of samples', fontsize=11)
    axes[1].set_title('Distribution of predicted probabilities', fontsize=12)
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ============================================================
# 2. PR-curve
# ============================================================

def plot_pr_curve(y_true, y_prob, thresholds=None,
                  title="PR curve",
                  filename="pr_curve.png"):
    """
    Визуализация PR-curve для выбора operating point.

    Args:
        y_true: np.array, истинные метки
        y_prob: np.array, предсказанные вероятности
        thresholds: list of float, пороги для отметки
        title: str, заголовок
        filename: str, имя файла
    """
    precision, recall, thresh = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(recall, precision, 'b-', linewidth=2, label='PR curve')

    if thresholds is not None:
        for t in thresholds:
            idx = np.argmin(np.abs(thresh - t))
            ax.scatter(recall[idx], precision[idx], color='red', s=80, zorder=5)
            ax.annotate(
                f'τ={t:.2f}',
                xy=(recall[idx], precision[idx]),
                xytext=(recall[idx] + 0.02, precision[idx] - 0.05),
                fontsize=9,
                color='red'
            )

    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ============================================================
# 3. SHAP summary
# ============================================================

def plot_shap_summary(shap_values, feature_names,
                      X=None,
                      title="SHAP summary",
                      filename="shap_summary.png"):
    """
    Визуализация SHAP values.

    Args:
        shap_values: np.array (n_samples, n_features)
        feature_names: list of str
        X: np.array (n_samples, n_features), значения признаков
           (нужны для окраски точек). Если None — используется shap_values.
        title: str, заголовок
        filename: str, имя файла
    """
    import shap

    # Создаём фигуру явно и передаём её в shap
    fig = plt.figure(figsize=(10, 8))

    # show=False — чтобы shap не вызывал plt.show() сам
    shap.summary_plot(
        shap_values,
        features=X if X is not None else shap_values,
        feature_names=feature_names,
        show=False
    )

    plt.title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# ============================================================
# Демонстрация
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    n_samples = 10000
    y_true = np.random.binomial(1, 0.05, n_samples)  # 5% illicit

    # --- Плохо калиброванные вероятности (overconfident) ---
    # Сдвигаем вверх: модель говорит "риск высокий" там, где его нет
    y_prob_bad = np.clip(
        y_true * 0.3 + np.random.randn(n_samples) * 0.15 + 0.35,
        0, 1
    )

    # --- Хорошо калиброванные вероятности ---
    # Совпадают с эмпирической частотой
    y_prob_good = np.clip(
        y_true * 0.85 + np.random.randn(n_samples) * 0.05 + 0.02,
        0, 1
    )

    # Разные имена файлов — чтобы не перезаписывали друг друга
    plot_reliability_diagram(
        y_true, y_prob_bad,
        title="Before calibration (overconfident)",
        filename="reliability_before.png"
    )

    plot_reliability_diagram(
        y_true, y_prob_good,
        title="After calibration",
        filename="reliability_after.png"
    )

    plot_pr_curve(
        y_true, y_prob_good,
        thresholds=[0.3, 0.5, 0.7, 0.9],
        filename="pr_curve.png"
    )

    # --- SHAP (синтетический пример) ---
    # В реальности shap_values получаются из shap.TreeExplainer(model)
    try:
        import shap  # noqa: F401
        n_features = 8
        feature_names = [
            "distance_to_nearest_OFAC",
            "causal_filter_pass_rate",
            "e_value_max",
            "anchor_source_diversity",
            "pagerank",
            "hawkes_lambda",
            "news_co_mention_count",
            "mixer_proximity",
        ]
        X_demo = np.random.randn(2000, n_features)
        shap_values_demo = np.random.randn(2000, n_features) * 0.1

        plot_shap_summary(
            shap_values_demo,
            feature_names=feature_names,
            X=X_demo,
            title="SHAP summary (demo)",
            filename="shap_summary.png"
        )
    except ImportError:
        print("SHAP не установлен — пропускаем визуализацию SHAP.")