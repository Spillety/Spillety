import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_evalue_sensitivity(rr_values, evalues, title="E-value sensitivity analysis"):
    """
    Визуализация E-value для разных значений risk ratio.
    
    Args:
        rr_values: list of float, risk ratios
        evalues: list of float, соответствующие E-values
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(rr_values, evalues, 'b-', linewidth=2, label='E-value')
    ax.fill_between(rr_values, 0, evalues, alpha=0.2, color='blue')
    
    # Отметить порог
    threshold = 2.0
    ax.axhline(y=threshold, color='red', linestyle='--', 
               label=f'Threshold for Tier 1: {threshold}')
    
    # Отметить точку, где E-value = threshold
    idx = np.argmin(np.abs(np.array(evalues) - threshold))
    ax.scatter(rr_values[idx], evalues[idx], color='red', s=100, zorder=5)
    ax.annotate(
        f'RR={rr_values[idx]:.2f}, E-value={evalues[idx]:.2f}',
        xy=(rr_values[idx], evalues[idx]),
        xytext=(rr_values[idx] + 0.1, evalues[idx] + 0.3),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=10,
        color='red'
    )
    
    ax.set_xlabel('Risk Ratio (RR)', fontsize=11)
    ax.set_ylabel('E-value', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('evalue_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_rosenbaum_gamma(gamma_values, p_values, title="Rosenbaum Γ* sensitivity"):
    """
    Визуализация Rosenbaum Γ*.
    
    Args:
        gamma_values: list of float, значения Γ
        p_values: list of float, соответствующие p-values
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(gamma_values, p_values, 'b-', linewidth=2, label='p-value')
    ax.fill_between(gamma_values, 0, p_values, alpha=0.2, color='blue')
    
    # Отметить порог p = 0.05
    ax.axhline(y=0.05, color='red', linestyle='--', label='p = 0.05')
    
    # Найти Γ*, где p = 0.05
    idx = np.argmin(np.abs(np.array(p_values) - 0.05))
    gamma_star = gamma_values[idx]
    ax.scatter(gamma_star, p_values[idx], color='red', s=100, zorder=5)
    ax.annotate(
        f'Γ* = {gamma_star:.2f}',
        xy=(gamma_star, p_values[idx]),
        xytext=(gamma_star + 0.1, p_values[idx] + 0.05),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=10,
        color='red'
    )
    
    ax.set_xlabel('Γ (odds ratio)', fontsize=11)
    ax.set_ylabel('p-value (upper bound)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('rosenbaum_gamma.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_sensemakr_partial_r2(r2_values, effect_changes, title="Sensemakr partial R² sensitivity"):
    """
    Визуализация sensemakr partial R².
    
    Args:
        r2_values: list of float, partial R² unmeasured confounder
        effect_changes: list of float, изменение эффекта
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(r2_values, effect_changes, 'b-', linewidth=2, label='Effect change')
    ax.fill_between(r2_values, 0, effect_changes, alpha=0.2, color='blue')
    
    # Отметить порог
    threshold = 0.10
    ax.axvline(x=threshold, color='red', linestyle='--', 
               label=f'Threshold for Tier 1: R²={threshold}')
    
    ax.set_xlabel('Partial R² of unmeasured confounder', fontsize=11)
    ax.set_ylabel('Change in effect estimate', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('sensemakr_partial_r2.png', dpi=150, bbox_inches='tight')
    plt.show()


# Пример вызова
if __name__ == "__main__":
    # Синтетические данные для E-value
    rr_values = np.linspace(1.0, 5.0, 100)
    evalues = rr_values + np.sqrt(rr_values * (rr_values - 1))
    plot_evalue_sensitivity(rr_values, evalues)
    
    # Синтетические данные для Rosenbaum
    gamma_values = np.linspace(1.0, 3.0, 100)
    p_values = 0.5 * np.exp(-2 * (gamma_values - 1)) + 0.001
    plot_rosenbaum_gamma(gamma_values, p_values)
    
    # Синтетические данные для sensemakr
    r2_values = np.linspace(0, 0.3, 100)
    effect_changes = 0.5 * r2_values / (1 - r2_values)
    plot_sensemakr_partial_r2(r2_values, effect_changes)