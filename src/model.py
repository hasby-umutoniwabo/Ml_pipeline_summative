"""Model architecture and training utilities for malaria classification."""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

IMG_SIZE = 128

def build_model():
    # CNN with conv/pool blocks, batch norm for stable training, dropout for regularization
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_training_callbacks():
    # early stopping avoids overfitting, reduce_lr fine tunes when progress slows
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    return [early_stop, reduce_lr]

def retrain_model(model_path, new_data_dir, epochs=5):
    """
    Loads the existing trained model, fine-tunes it on new uploaded data,
    then evaluates on that same data to demonstrate production evaluation.
    """
    import os
    import glob
    import pandas as pd
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    from src.preprocessing import make_dataset

    existing_model = tf.keras.models.load_model(model_path)

    new_parasitized = glob.glob(os.path.join(new_data_dir, "Parasitized", "*"))
    new_uninfected = glob.glob(os.path.join(new_data_dir, "Uninfected", "*"))
    new_files = new_parasitized + new_uninfected
    new_labels = [1] * len(new_parasitized) + [0] * len(new_uninfected)

    if len(new_files) == 0:
        raise ValueError("No new data found to retrain on.")

    new_df = pd.DataFrame({"filepath": new_files, "label": new_labels})
    new_ds = make_dataset(new_df, augment_data=True, shuffle=True)

    existing_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = existing_model.fit(new_ds, epochs=epochs)

    # evaluate on the same uploaded data post-retrain, demonstrates production evaluation
    eval_ds = make_dataset(new_df, augment_data=False, shuffle=False)
    y_true = new_df["label"].values
    y_pred_probs = existing_model.predict(eval_ds, verbose=0).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    eval_metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }

    existing_model.save(model_path)
    return existing_model, history, eval_metrics