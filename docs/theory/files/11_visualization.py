import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, brier_score_loss


def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc = y_true[in_bin].mean()
            conf = y_prob[in_bin].mean()
            ece += np.abs(conf - acc) * prop_in_bin
    return ece


def plot_pr_auc_comparison(y_true, y_prob_list, labels_list,
                           title="PR-AUC comparison",
                           filename="pr_auc_comparison.png"):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#1976D2', '#D32F2F', '#F57C00', '#388E3C']
    for i, (y_prob, label) in enumerate(zip(y_prob_list, labels_list)):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color=colors[i % len(colors)],
                linewidth=2, label=f'{label} (PR-AUC = {pr_auc:.3f})')
    base_rate = y_true.mean()
    ax.axhline(y=base_rate, color='gray', linestyle='--',
               label=f'Base rate = {base_rate:.4f}')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_calibration_comparison(y_true, y_prob_list, labels_list, n_bins=10,
                                title="Calibration comparison",
                                filename="calibration_comparison.png"):
    from sklearn.calibration import calibration_curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#1976D2', '#D32F2F', '#F57C00', '#388E3C']
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    for i, (y_prob, label) in enumerate(zip(y_prob_list, labels_list)):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        brier = brier_score_loss(y_true, y_prob)
        axes[0].plot(prob_pred, prob_true, color=colors[i % len(colors)],
                     marker='o', linewidth=2, label=f'{label} (Brier = {brier:.4f})')
    axes[0].set_xlabel('Predicted probability', fontsize=11)
    axes[0].set_ylabel('Empirical frequency', fontsize=11)
    axes[0].set_title('Reliability diagram', fontsize=12)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    ece_values = [compute_ece(y_true, y_prob, n_bins) for y_prob in y_prob_list]
    bars = axes[1].bar(range(len(labels_list)), ece_values,
                       color=[colors[i % len(colors)] for i in range(len(labels_list))],
                       alpha=0.7, edgecolor='white')
    for bar, ece in zip(bars, ece_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                     f'{ece:.4f}', ha='center', va='bottom', fontsize=10)
    axes[1].set_xticks(range(len(labels_list)))
    axes[1].set_xticklabels(labels_list, rotation=15, ha='right')
    axes[1].set_ylabel('ECE', fontsize=11)
    axes[1].set_title('Expected Calibration Error', fontsize=12)
    axes[1].grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_power_analysis(p_values, n_values,
                        title="Power analysis for random sampling",
                        filename="power_analysis.png"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p_values, n_values, 'b-', linewidth=2)
    ax.fill_between(p_values, 0, n_values, alpha=0.2, color='blue')
    for p in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(np.array(p_values) - p))
        ax.scatter(p, n_values[idx], color='red', s=80, zorder=5)
        ax.annotate(f'p={p:.2f}\nn={int(n_values[idx])}',
                    xy=(p, n_values[idx]),
                    xytext=(p + 0.01, n_values[idx] + 100),
                    fontsize=9, color='red')
    ax.set_xlabel('Expected precision (p)', fontsize=11)
    ax.set_ylabel('Required sample size (n)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_drift_monitoring(ks_statistics, timestamps,
                          title="Drift monitoring (KS statistic)",
                          filename="drift_monitoring.png"):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(ks_statistics)), ks_statistics, 'b-', linewidth=2, marker='o')
    threshold = 0.1
    ax.axhline(y=threshold, color='red', linestyle='--',
               label=f'Drift threshold = {threshold}')
    for i, ks in enumerate(ks_statistics):
        if ks > threshold:
            ax.scatter(i, ks, color='red', s=100, zorder=5)
    ax.set_xlabel('Time window', fontsize=11)
    ax.set_ylabel('KS statistic', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 10000
    y_true = np.random.binomial(1, 0.05, n_samples)
    y_prob_baseline = np.clip(y_true * 0.3 + np.random.randn(n_samples) * 0.15 + 0.05, 0, 1)
    y_prob_gbdt = np.clip(y_true * 0.7 + np.random.randn(n_samples) * 0.1 + 0.03, 0, 1)
    y_prob_calibrated = np.clip(y_true * 0.85 + np.random.randn(n_samples) * 0.05 + 0.02, 0, 1)
    plot_pr_auc_comparison(y_true, [y_prob_baseline, y_prob_gbdt, y_prob_calibrated],
                           ['Baseline', 'GBDT', 'GBDT + calibration'],
                           filename="11-10-1_pr_auc_comparison.png")
    plot_calibration_comparison(y_true, [y_prob_baseline, y_prob_gbdt, y_prob_calibrated],
                                ['Baseline', 'GBDT', 'GBDT + calibration'],
                                filename="11-10-2_calibration_comparison.png")
    p_values = np.linspace(0.01, 0.2, 100)
    n_values = (1.96 ** 2 * p_values * (1 - p_values)) / (0.01 ** 2)
    plot_power_analysis(p_values, n_values, filename="11-10-3_power_analysis.png")
    ks_statistics = np.abs(np.random.randn(30) * 0.05 + 0.08)
    timestamps = [f"2026-{i // 4 + 1:02d}" for i in range(30)]
    plot_drift_monitoring(ks_statistics, timestamps, filename="11-10-4_drift_monitoring.png")
