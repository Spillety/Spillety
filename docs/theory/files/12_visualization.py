import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_latency_budget(latency_components, labels, title="Latency budget",
                        filename="latency_budget.png"):
    """Визуализация распределения latency по компонентам."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = sns.color_palette("Blues_d", len(latency_components))

    bars = axes[0].barh(labels, latency_components, color=colors, edgecolor='white')
    for bar, lat in zip(bars, latency_components):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{lat:.1f} ms', va='center', fontsize=10)

    axes[0].set_xlabel('Latency (ms)', fontsize=11)
    axes[0].set_title('Latency per component', fontsize=12)
    axes[0].grid(True, alpha=0.2, axis='x')

    axes[1].pie(latency_components, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Latency distribution', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_memory_budget(memory_components, labels, title="Memory budget",
                       filename="memory_budget.png"):
    """Визуализация распределения памяти по компонентам."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = sns.color_palette("Reds_d", len(memory_components))

    bars = axes[0].barh(labels, memory_components, color=colors, edgecolor='white')
    for bar, mem in zip(bars, memory_components):
        axes[0].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f'{mem:.1f} GB', va='center', fontsize=10)

    axes[0].set_xlabel('Memory (GB)', fontsize=11)
    axes[0].set_title('Memory per component', fontsize=12)
    axes[0].grid(True, alpha=0.2, axis='x')

    axes[1].pie(memory_components, labels=labels, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Memory distribution', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_competitive_comparison(metrics, competitors, spillety_values,
                                title="Competitive comparison",
                                filename="competitive_comparison.png"):
    """Визуализация конкурентного сравнения."""
    fig, ax = plt.subplots(figsize=(12, 7))
    n_metrics = len(metrics)
    n_competitors = len(competitors)
    x = np.arange(n_metrics)
    width = 0.8 / (n_competitors + 1)
    colors = sns.color_palette("husl", n_competitors + 1)

    for i, (name, values) in enumerate(competitors.items()):
        ax.bar(x + i * width, values, width, label=name, color=colors[i], alpha=0.8)

    ax.bar(x + n_competitors * width, spillety_values, width,
           label='Spillety', color=colors[-1], alpha=0.9)

    ax.set_xlabel('Metric', fontsize=11)
    ax.set_ylabel('Value (normalized)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (n_competitors / 2))
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Пример вызова
if __name__ == "__main__":
    # Latency budget
    latency_components = [1, 5, 2, 3, 10, 1, 2]  # ms
    labels = ['Embedding', 'HNSW', 'Causal', 'Features', 'GBDT', 'Calibration', 'Evidence']
    plot_latency_budget(latency_components, labels,
                        filename="12-10-3_latency_budget.png")

    # Memory budget
    memory_components = [40, 40, 100, 0.1, 0.001]  # GB
    labels = ['HNSW', 'Embeddings', 'Graph', 'GBDT', 'Calibration']
    plot_memory_budget(memory_components, labels,
                       filename="12-10-4_memory_budget.png")

    # Competitive comparison
    metrics = ['Labels', 'Embedding', 'Causal', 'Cost', 'FTE', 'Latency']
    competitors = {
        'Chainalysis': [0.9, 0.5, 0.5, 0.1, 0.1, 0.7],
        'TRM Labs': [0.9, 0.7, 0.5, 0.1, 0.1, 0.7],
        'Elliptic': [0.9, 0.7, 0.5, 0.1, 0.2, 0.7],
    }
    spillety_values = [1.0, 0.9, 0.9, 1.0, 1.0, 0.9]
    plot_competitive_comparison(metrics, competitors, spillety_values,
                                filename="12-10-5_competitive_comparison.png")
