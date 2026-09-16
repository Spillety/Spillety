import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix


def setup() -> None:
    """
    Apply unified visual style for all notebooks
    """
    sns.set_theme(style="whitegrid", palette="colorblind", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (7, 4.2),
            "figure.dpi": 120,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
        }
    )


def plot_pr_curve(y_true, y_score, ax=None, label: str = "model"):
    if ax is None:
        _, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name=label)
    ax.set_title("Precision-Recall curve")
    return ax


def plot_roc_curve(y_true, y_score, ax=None, label: str = "model"):
    if ax is None:
        _, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name=label)
    ax.set_title("ROC curve")
    return ax


def plot_confusion(y_true, y_pred, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    return ax
