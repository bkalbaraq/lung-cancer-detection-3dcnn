import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow as tf

# using cpu only cause 1050ti doesnt work anymore, ignore with rtx 4060
"""
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass
"""
# for directml
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


from src.config import OUT_DIR, BATCH, LR, EPOCHS, PATIENCE

def make_ds(X, y, training):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(1024, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

def build_model(input_shape):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv3D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling3D(2),
        layers.Conv3D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling3D(2),
        layers.Conv3D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling3D(2),
        layers.Conv3D(128, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling3D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])


def main():
    X_train = np.load(OUT_DIR / "X_train.npy")
    y_train = np.load(OUT_DIR / "y_train.npy")
    X_val   = np.load(OUT_DIR / "X_val.npy")
    y_val   = np.load(OUT_DIR / "y_val.npy")

    print("Loaded:")
    print("  X_train", X_train.shape, "y_train", y_train.shape)
    print("  X_val  ", X_val.shape,   "y_val",   y_val.shape)

    train_ds = make_ds(X_train, y_train, True)
    val_ds   = make_ds(X_val,   y_val,   False)

    model = build_model(input_shape=X_train.shape[1:])
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )

    ckpt = keras.callbacks.ModelCheckpoint(
        "lidc_3dcnn.keras", monitor="val_auc", mode="max", save_best_only=True
    )
    hist = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ckpt]  # Removed EarlyStopping
)


    print("Best val AUC:", max(hist.history["val_auc"]))
    model.save("lidc_3dcnn_final.keras")

if __name__ == "__main__":
    main()
