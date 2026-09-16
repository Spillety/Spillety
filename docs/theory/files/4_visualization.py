import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Предполагается, что у нас есть:
# embeddings: (N, 128) массив embeddings
# labels: (N,) массив меток (0 = non-anchor, 1 = OFAC, 2 = mixer, 3 = exchange)

def visualize_embeddings(embeddings, labels, title="t-SNE projection of wallet embeddings"):
    """
    Визуализация embedding space через t-SNE.
    
    Args:
        embeddings: np.array (N, 128)
        labels: np.array (N,) с метками классов
        title: заголовок графика
    """
    # t-SNE проекция в 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=72, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Создание DataFrame для seaborn
    import pandas as pd
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': labels
    })
    
    # Маппинг меток на читаемые названия
    label_names = {
        0: 'Non-anchor',
        1: 'OFAC SDN',
        2: 'Mixer',
        3: 'Exchange'
    }
    df['label_name'] = df['label'].map(label_names)
    
    # Палитра цветов
    palette = {
        'Non-anchor': '#B0B0B0',  # серый
        'OFAC SDN': '#D32F2F',     # красный
        'Mixer': '#F57C00',        # оранжевый
        'Exchange': '#1976D2'      # синий
    }
    
    # Визуализация
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for label_name in ['Non-anchor', 'Exchange', 'Mixer', 'OFAC SDN']:
        subset = df[df['label_name'] == label_name]
        ax.scatter(
            subset['x'], subset['y'],
            c=palette[label_name],
            label=label_name,
            alpha=0.6 if label_name == 'Non-anchor' else 0.9,
            s=20 if label_name == 'Non-anchor' else 50,
            edgecolors='white' if label_name != 'Non-anchor' else 'none',
            linewidth=0.5
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE dimension 1', fontsize=11)
    ax.set_ylabel('t-SNE dimension 2', fontsize=11)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('embedding_tsne.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return embeddings_2d


def plot_silhouette_analysis(embeddings, labels, title="Silhouette analysis"):
    """
    Визуализация silhouette score для разных кластеров.
    
    Args:
        embeddings: np.array (N, 128)
        labels: np.array (N,)
    """
    from sklearn.metrics import silhouette_samples, silhouette_score
    
    # Вычисление silhouette scores
    silhouette_avg = silhouette_score(embeddings, labels)
    sample_silhouette_values = silhouette_samples(embeddings, labels)
    
    n_clusters = len(np.unique(labels))
    unique_labels = np.unique(labels)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    y_lower = 10
    colors = ['#B0B0B0', '#D32F2F', '#F57C00', '#1976D2']
    
    for i, label in enumerate(unique_labels):
        # Silhouette scores для объектов этого кластера
        ith_cluster_silhouette_values = sample_silhouette_values[labels == label]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0, ith_cluster_silhouette_values,
            facecolor=colors[i % len(colors)],
            edgecolor=colors[i % len(colors)],
            alpha=0.7
        )
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f'Average: {silhouette_avg:.3f}')
    ax.set_xlabel('Silhouette coefficient', fontsize=11)
    ax.set_ylabel('Cluster label', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('silhouette_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_ntxent_training_curve(train_losses, val_losses, title="NT-Xent training curve"):
    """
    Визуализация кривой обучения NT-Xent loss.
    
    Args:
        train_losses: list of float
        val_losses: list of float
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Train loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('NT-Xent loss', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    
    # Аннотация точки минимума validation loss
    min_val_idx = np.argmin(val_losses)
    min_val_loss = val_losses[min_val_idx]
    ax.annotate(
        f'Min val loss: {min_val_loss:.4f}',
        xy=(epochs[min_val_idx], min_val_loss),
        xytext=(epochs[min_val_idx] + 5, min_val_loss + 1),
        arrowprops=dict(arrowstyle='->', color='black'),
        fontsize=10,
        color='black'
    )
    
    plt.tight_layout()
    plt.savefig('ntxent_training_curve.png', dpi=150, bbox_inches='tight')
    plt.show()


# Пример вызова (с синтетическими данными)
if __name__ == "__main__":
    np.random.seed(72)
    
    # Синтетические данные: 4 кластера по 500 точек в 128-мерном пространстве
    n_per_class = 500
    dim = 128
    
    # Центры кластеров
    centers = np.random.randn(4, dim) * 3
    
    embeddings_list = []
    labels_list = []
    
    for i, center in enumerate(centers):
        # Точки вокруг центра с шумом
        noise = np.random.randn(n_per_class, dim) * 0.5
        embeddings_list.append(center + noise)
        labels_list.append(np.full(n_per_class, i))
    
    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    
    # Визуализации
    visualize_embeddings(embeddings, labels)
    plot_silhouette_analysis(embeddings, labels)
    
    # Синтетическая кривая обучения
    train_losses = [2.5 * np.exp(-0.05 * i) + 0.5 + np.random.randn() * 0.05 for i in range(100)]
    val_losses = [2.5 * np.exp(-0.05 * i) + 0.6 + np.random.randn() * 0.08 for i in range(100)]
    plot_ntxent_training_curve(train_losses, val_losses)