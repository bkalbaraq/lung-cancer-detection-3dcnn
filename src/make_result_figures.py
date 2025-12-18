import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from src.config import OUT_DIR 

def main():
    
    y_val = np.load(OUT_DIR / "y_val.npy")

    
    y_prob_path = OUT_DIR / "y_prob_val.npy"
    if y_prob_path.exists():
        y_prob = np.load(y_prob_path)
    else:
        #error if no probability
        raise FileNotFoundError(
            "y_prob_val.npy not found. "
            "Quick fix: modify eval_metrics.py to save y_prob (see instructions below)."
        )

    y_pred = (y_prob >= 0.5).astype(int)

    
    cm = confusion_matrix(y_val, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (Validation Cubes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Healthy (0)", "Cancer (1)"])
    plt.yticks([0, 1], ["Healthy (0)", "Cancer (1)"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title("ROC Curve (Validation Cubes)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curve.png", dpi=300)
    plt.close()

    
    prec, rec, _ = precision_recall_curve(y_val, y_prob)
    pr_auc = average_precision_score(y_val, y_prob)

    plt.figure()
    plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.4f}")
    plt.title("Precision-Recall Curve (Validation Cubes)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pr_curve.png", dpi=300)
    plt.close()

    print("Saved:")
    print(" -", OUT_DIR / "confusion_matrix.png")
    print(" -", OUT_DIR / "roc_curve.png")
    print(" -", OUT_DIR / "pr_curve.png")

if __name__ == "__main__":
    main()
