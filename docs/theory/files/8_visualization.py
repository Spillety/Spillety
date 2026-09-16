import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_temporal_validation(dates, median_distances, threshold,
                              title="Temporal validation: median distance to old anchors",
                              filename="temporal_validation.png"):
    """
    Визуализация temporal validation: median distance нового anchor
    до старых anchors во времени.
    
    Args:
        dates: list of datetime
        median_distances: list of float
        threshold: float, порог для drift
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(dates, median_distances, 'b-', marker='o', linewidth=2, label='Median distance')
    ax.axhline(y=threshold, color='red', linestyle='--', 
               label=f'Drift threshold: {threshold:.3f}')
    
    # Отметить точки выше порога
    above = [i for i, d in enumerate(median_distances) if d > threshold]
    if above:
        ax.scatter([dates[i] for i in above], 
                   [median_distances[i] for i in above],
                   color='red', s=100, zorder=5, label='Drift detected')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Median distance', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_ks_drift(embeddings_old, embeddings_new, 
                  title="KS drift: distribution of embeddings",
                  filename="ks_drift.png"):
    """
    Визуализация KS-теста: сравнение распределений embeddings.
    
    Args:
        embeddings_old: np.array (N, D)
        embeddings_new: np.array (M, D)
    """
    # Проекция на первую главную компоненту для визуализации
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=1)
    combined = np.vstack([embeddings_old, embeddings_new])
    pca.fit(combined)
    
    proj_old = pca.transform(embeddings_old).flatten()
    proj_new = pca.transform(embeddings_new).flatten()
    
    # KS-тест
    ks_stat, p_value = stats.ks_2samp(proj_old, proj_new)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Гистограммы
    axes[0].hist(proj_old, bins=50, alpha=0.6, color='#1976D2', 
                 label='Old', density=True, edgecolor='white')
    axes[0].hist(proj_new, bins=50, alpha=0.6, color='#D32F2F', 
                 label='New', density=True, edgecolor='white')
    axes[0].set_xlabel('First principal component', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title(f'KS stat = {ks_stat:.4f}, p = {p_value:.4f}', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.2)
    
    # CDF
    axes[1].hist(proj_old, bins=50, alpha=1.0, cumulative=True, 
                 density=True, histtype='step', color='#1976D2', 
                 linewidth=2, label='Old CDF')
    axes[1].hist(proj_new, bins=50, alpha=1.0, cumulative=True, 
                 density=True, histtype='step', color='#D32F2F', 
                 linewidth=2, label='New CDF')
    axes[1].set_xlabel('First principal component', fontsize=11)
    axes[1].set_ylabel('Cumulative density', fontsize=11)
    axes[1].set_title('CDF comparison', fontsize=12)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    return ks_stat, p_value


def plot_drift_dashboard(metrics_dict, title="Drift monitoring dashboard",
                         filename="drift_dashboard.png"):
    """
    Визуализация dashboard с несколькими drift-метриками.
    
    Args:
        metrics_dict: dict, {metric_name: list of values over time}
    """
    n_metrics = len(metrics_dict)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics))
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, metrics_dict.items()):
        ax.plot(values, 'b-', marker='o', linewidth=2)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.set_title(f'{metric_name} over time', fontsize=12)
    
    axes[-1].set_xlabel('Time step', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_walk_forward_validation(fold_metrics, title="Walk-forward validation",
                                  filename="walk_forward.png"):
    """
    Визуализация метрик по folds walk-forward validation.
    
    Args:
        fold_metrics: dict, {metric_name: list of values per fold}
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    folds = range(1, len(list(fold_metrics.values())[0]) + 1)
    
    for metric_name, values in fold_metrics.items():
        ax.plot(folds, values, marker='o', linewidth=2, label=metric_name)
    
    ax.set_xlabel('Fold', fontsize=11)
    ax.set_ylabel('Metric value', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    ax.set_xticks(list(folds))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Пример вызова
if __name__ == "__main__":
    np.random.seed(42)
    
    # --- Temporal validation ---
    import datetime
    start_date = datetime.date(2024, 1, 1)
    n_days = 180
    
    dates = [start_date + datetime.timedelta(days=i) for i in range(n_days)]
    
    # Стабильный период + drift в конце
    median_distances = []
    for i in range(n_days):
        if i < 120:
            median_distances.append(0.25 + np.random.randn() * 0.02)
        else:
            median_distances.append(0.25 + (i - 120) * 0.005 + np.random.randn() * 0.02)
    
    threshold = 0.35
    plot_temporal_validation(dates, median_distances, threshold)
    
    # --- KS drift ---
    embeddings_old = np.random.randn(5000, 128)
    embeddings_new = np.random.randn(5000, 128) * 1.2 + 0.3
    plot_ks_drift(embeddings_old, embeddings_new)
    
    # --- Drift dashboard ---
    metrics = {
        'KS statistic': np.abs(np.random.randn(180) * 0.02 + 0.05),
        'Brier score': np.abs(np.random.randn(180) * 0.01 + 0.15),
        'ECE': np.abs(np.random.randn(180) * 0.008 + 0.08),
        'Alert rate': np.abs(np.random.randn(180) * 0.005 + 0.03),
    }
    plot_drift_dashboard(metrics)
    
    # --- Walk-forward validation ---
    fold_metrics = {
        'PR-AUC': [0.85, 0.84, 0.82, 0.80, 0.78],
        'Precision@K': [0.90, 0.88, 0.86, 0.83, 0.81],
        'Recall@K': [0.75, 0.73, 0.70, 0.68, 0.65],
    }
    plot_walk_forward_validation(fold_metrics)