import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cluster_size_distribution(cluster_sizes, title="Cluster size distribution"):
    """
    Визуализация распределения размеров кластеров.
    
    Args:
        cluster_sizes: list of int, размеры кластеров
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Линейная шкала
    axes[0].hist(cluster_sizes, bins=50, color='#1976D2', alpha=0.7, edgecolor='white')
    axes[0].set_xlabel('Cluster size', fontsize=11)
    axes[0].set_ylabel('Number of clusters', fontsize=11)
    axes[0].set_title('Linear scale', fontsize=12)
    axes[0].grid(True, alpha=0.2)
    
    # Логарифмическая шкала
    axes[1].hist(cluster_sizes, bins=50, color='#D32F2F', alpha=0.7, edgecolor='white')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Cluster size (log)', fontsize=11)
    axes[1].set_ylabel('Number of clusters (log)', fontsize=11)
    axes[1].set_title('Log-log scale', fontsize=12)
    axes[1].grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('cluster_size_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Статистика
    print(f"Mean cluster size: {np.mean(cluster_sizes):.2f}")
    print(f"Median cluster size: {np.median(cluster_sizes):.2f}")
    print(f"Max cluster size: {max(cluster_sizes)}")
    print(f"Clusters with size 1: {sum(1 for s in cluster_sizes if s == 1)}")


def plot_pr_curve_for_clustering(precisions, recalls, thresholds, title="PR curve for entity resolution"):
    """
    Визуализация PR-curve для выбора operating point.
    
    Args:
        precisions: list of float
        recalls: list of float
        thresholds: list of float
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(recalls, precisions, 'b-', linewidth=2)
    
    # Отметить несколько порогов
    n_points = min(5, len(thresholds))
    indices = np.linspace(0, len(thresholds) - 1, n_points, dtype=int)
    
    for idx in indices:
        ax.scatter(recalls[idx], precisions[idx], color='red', s=80, zorder=5)
        ax.annotate(
            f'τ={thresholds[idx]:.2f}',
            xy=(recalls[idx], precisions[idx]),
            xytext=(recalls[idx] + 0.02, precisions[idx] - 0.05),
            fontsize=9,
            color='red'
        )
    
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('pr_curve_entity_resolution.png', dpi=150, bbox_inches='tight')
    plt.show()


# Пример вызова
if __name__ == "__main__":
    np.random.seed(72)
    
    # Синтетические данные: степенное распределение размеров кластеров
    # (типично для blockchain: много мелких, мало крупных)
    cluster_sizes = np.random.pareto(a=1.5, size=10000) * 10 + 1
    cluster_sizes = cluster_sizes.astype(int)
    
    plot_cluster_size_distribution(cluster_sizes)
    
    # Синтетическая PR-curve
    recalls = np.linspace(0.01, 1, 100)
    precisions = 0.95 - 0.3 * recalls**2 + np.random.randn(100) * 0.01
    precisions = np.clip(precisions, 0, 1)
    thresholds = np.linspace(0.9, 0.1, 100)
    
    plot_pr_curve_for_clustering(precisions, recalls, thresholds)