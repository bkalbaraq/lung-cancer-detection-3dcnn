import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score
import tensorflow as tf
from tensorflow import keras

# currently src.config, it works like this
from src.config import OUT_DIR, BATCH

def make_ds(X, y):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

def main():
    X_val = np.load(OUT_DIR / "X_val.npy")
    y_val = np.load(OUT_DIR / "y_val.npy")

    model = keras.models.load_model("lidc_3dcnn.keras")
    val_ds = make_ds(X_val, y_val)

    y_prob = model.predict(val_ds).ravel()
    np.save(OUT_DIR / "y_prob_val.npy", y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_val, y_pred)
    print("Confusion matrix:\n", cm)
    print(classification_report(y_val, y_pred, digits=4))
    try:
        print("ROC AUC:", roc_auc_score(y_val, y_prob))
        print("PR  AUC:", average_precision_score(y_val, y_prob))
    except Exception as e:
        print("AUCs could not be computed:", e)

if __name__ == "__main__":
    main()
