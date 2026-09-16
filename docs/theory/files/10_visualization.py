import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


def plot_materiality_matrix(models, title="SR 26-2 Materiality Matrix",
                            filename="materiality_matrix.png"):
    """
    Визуализация materiality matrix для моделей Spillety.

    Args:
        models: list of dict {name, exposure (0-1), purpose (0-1), tier}
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Background quadrants
    ax.add_patch(Rectangle((0, 0), 0.5, 0.5, facecolor='#E8F5E9', alpha=0.5))
    ax.add_patch(Rectangle((0.5, 0), 0.5, 0.5, facecolor='#FFF3E0', alpha=0.5))
    ax.add_patch(Rectangle((0, 0.5), 0.5, 0.5, facecolor='#FFF3E0', alpha=0.5))
    ax.add_patch(Rectangle((0.5, 0.5), 0.5, 0.5, facecolor='#FFEBEE', alpha=0.5))

    # Models
    colors = {'High': '#D32F2F', 'Medium': '#F57C00', 'Low': '#388E3C'}

    for m in models:
        color = colors.get(m['tier'], '#1976D2')
        ax.scatter(m['exposure'], m['purpose'], s=300, color=color,
                   zorder=5, edgecolors='white', linewidth=2)
        ax.annotate(m['name'], (m['exposure'], m['purpose']),
                    xytext=(m['exposure'] + 0.03, m['purpose'] + 0.03),
                    fontsize=10, fontweight='bold')

    # Labels
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)

    ax.text(0.25, 0.75, 'Low exposure\nHigh purpose', ha='center', fontsize=10, alpha=0.7)
    ax.text(0.75, 0.75, 'High materiality', ha='center', fontsize=10, alpha=0.7)
    ax.text(0.25, 0.25, 'Low materiality', ha='center', fontsize=10, alpha=0.7)
    ax.text(0.75, 0.25, 'High exposure\nLow purpose', ha='center', fontsize=10, alpha=0.7)

    ax.set_xlabel('Model Exposure (significance to decisions)', fontsize=11)
    ax.set_ylabel('Model Purpose (regulatory vs financial)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='High materiality',
               markerfacecolor='#D32F2F', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Medium materiality',
               markerfacecolor='#F57C00', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Low materiality',
               markerfacecolor='#388E3C', markersize=12),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_fairness_audit(groups, tpr_values, fpr_values, precision_values,
                        title="Jurisdiction-level fairness audit",
                        filename="fairness_audit.png"):
    """
    Визуализация fairness audit по юрисдикциям.

    Args:
        groups: list of str, jurisdictions
        tpr_values: list of float
        fpr_values: list of float
        precision_values: list of float
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(groups))
    width = 0.6

    # TPR
    axes[0].bar(x, tpr_values, width, color='#1976D2', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(groups, rotation=45, ha='right')
    axes[0].set_ylabel('True Positive Rate', fontsize=11)
    axes[0].set_title('TPR by jurisdiction', fontsize=12)
    axes[0].grid(True, alpha=0.2, axis='y')

    # FPR
    axes[1].bar(x, fpr_values, width, color='#D32F2F', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(groups, rotation=45, ha='right')
    axes[1].set_ylabel('False Positive Rate', fontsize=11)
    axes[1].set_title('FPR by jurisdiction', fontsize=12)
    axes[1].grid(True, alpha=0.2, axis='y')

    # Precision
    axes[2].bar(x, precision_values, width, color='#388E3C', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(groups, rotation=45, ha='right')
    axes[2].set_ylabel('Precision', fontsize=11)
    axes[2].set_title('Precision by jurisdiction', fontsize=12)
    axes[2].grid(True, alpha=0.2, axis='y')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_sar_timeline(sar_events, title="SAR filing timeline", filename="sar_timeline.png"):
    """
    Визуализация timeline SAR filing.

    Args:
        sar_events: list of dict {date, type, description}
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    dates = [e['date'] for e in sar_events]
    types = [e['type'] for e in sar_events]
    descriptions = [e.get('description', '') for e in sar_events]

    unique_types = list(set(types))
    colors = {t: c for t, c in zip(unique_types,
              ['#1976D2', '#D32F2F', '#F57C00', '#388E3C', '#7B1FA2', '#00ACC1', '#5E35B1', '#F06292'])}

    for i, (date, typ, desc) in enumerate(zip(dates, types, descriptions)):
        color = colors.get(typ, '#1976D2')
        ax.scatter(date, i, s=200, color=color, zorder=5,
                   edgecolors='white', linewidth=2)
        if desc:
            ax.text(date, i + 0.3, desc, ha='center', fontsize=8)

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Event index', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_yticks([])

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=t,
               markerfacecolor=colors[t], markersize=12)
        for t in unique_types
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# Пример вызова
if __name__ == "__main__":
    # Materiality matrix
    models = [
        {'name': 'GBDT (decision)', 'exposure': 0.9, 'purpose': 0.9, 'tier': 'High'},
        {'name': 'Contrastive encoder', 'exposure': 0.7, 'purpose': 0.8, 'tier': 'High'},
        {'name': 'Causal DAG', 'exposure': 0.6, 'purpose': 0.7, 'tier': 'Medium'},
        {'name': 'HNSW retrieval', 'exposure': 0.4, 'purpose': 0.3, 'tier': 'Low'},
        {'name': 'Rule filters', 'exposure': 0.3, 'purpose': 0.4, 'tier': 'Low'},
    ]
    plot_materiality_matrix(models, filename="10-8-1_materiality_matrix.png")

    # Fairness audit
    groups = ['US', 'EU', 'UK', 'Singapore', 'UAE']
    tpr_values = [0.75, 0.73, 0.74, 0.70, 0.68]
    fpr_values = [0.05, 0.05, 0.04, 0.06, 0.07]
    precision_values = [0.85, 0.84, 0.86, 0.82, 0.80]
    plot_fairness_audit(groups, tpr_values, fpr_values, precision_values,
                        filename="10-8-2_fairness_audit.png")

    # SAR timeline
    base = datetime.datetime(2026, 9, 1)
    sar_events = [
        {'date': base, 'type': 'detection', 'description': 'Initial detection'},
        {'date': base + datetime.timedelta(days=5), 'type': 'review', 'description': 'Analyst review'},
        {'date': base + datetime.timedelta(days=15), 'type': 'escalation', 'description': 'Escalated to compliance'},
        {'date': base + datetime.timedelta(days=25), 'type': 'sar_filing', 'description': 'SAR filed (30 days)'},
        {'date': base + datetime.timedelta(days=115), 'type': 'continuing_sar', 'description': 'Continuing SAR (120 days)'},
    ]
    plot_sar_timeline(sar_events, filename="10-8-3_sar_timeline.png")