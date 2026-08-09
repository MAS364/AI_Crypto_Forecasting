"""Autoencoder feature extraction: reusable production implementation.

Extracted from 06b_autoencoder_feature_extraction.ipynb, where the
architecture and training were originally defined and tested against real
feature sets. The notebook now imports and calls these functions rather
than redefining them, matching the pattern already used for feature
engineering (Notebook 06) and modelling (Notebook 07).

Architecture: a simple symmetric dense autoencoder, input -> 16 ->
bottleneck -> 16 -> input, trained to reconstruct its own scaled input.
`EarlyStopping` monitors validation reconstruction loss -- validation
data is never used to update the model's weights, only to decide when to
stop training.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


ENCODING_DIM = 8
AE_EPOCHS = 150
AE_PATIENCE = 15
AE_SEED = 42


def build_autoencoder(
    input_dim: int,
    encoding_dim: int = ENCODING_DIM,
):
    """Return (autoencoder, encoder).

    The encoder shares the autoencoder's trained weights and is used
    afterward to encode train/validation/test independently.
    """
    inp = keras.layers.Input(shape=(input_dim,))

    encoded = keras.layers.Dense(
        16,
        activation="relu",
    )(inp)

    encoded = keras.layers.Dense(
        encoding_dim,
        activation="relu",
        name="bottleneck",
    )(encoded)

    decoded = keras.layers.Dense(
        16,
        activation="relu",
    )(encoded)

    decoded = keras.layers.Dense(
        input_dim,
        activation="linear",
    )(decoded)

    autoencoder = keras.Model(
        inp,
        decoded,
    )

    encoder = keras.Model(
        inp,
        encoded,
    )

    autoencoder.compile(
        optimizer="adam",
        loss="mse",
    )

    return autoencoder, encoder


def train_autoencoder(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    encoding_dim: int = ENCODING_DIM,
    epochs: int = AE_EPOCHS,
    patience: int = AE_PATIENCE,
    seed: int = AE_SEED,
):
    """Train the autoencoder on training data only.

    Validation data is used solely for EarlyStopping's monitored loss
    and is never used to update model weights.

    `encoding_dim`, `epochs`, `patience`, and `seed` can be overridden
    by the caller while defaulting to the module-level research settings.
    """
    keras.utils.set_random_seed(seed)

    # Fit scaler on training data only.
    scaler = StandardScaler().fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    autoencoder, encoder = build_autoencoder(
        X_train.shape[1],
        encoding_dim,
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = autoencoder.fit(
        X_train_scaled,
        X_train_scaled,
        validation_data=(
            X_val_scaled,
            X_val_scaled,
        ),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
    )

    return (
        autoencoder,
        encoder,
        scaler,
        history,
    )


def encode(
    encoder,
    scaler,
    X: pd.DataFrame,
) -> np.ndarray:
    """Encode predictors using an already-fitted encoder and scaler."""
    X_scaled = scaler.transform(X)

    return encoder.predict(
        X_scaled,
        verbose=0,
    )


def reconstruction_loss(
    autoencoder,
    scaler,
    X: pd.DataFrame,
) -> float:
    """Compute reconstruction MSE without refitting anything."""
    X_scaled = scaler.transform(X)

    return float(
        autoencoder.evaluate(
            X_scaled,
            X_scaled,
            verbose=0,
        )
    )


def save_autoencoder(
    autoencoder,
    encoder,
    scaler,
    history,
    path_prefix: Path,
    metadata: dict = None,
) -> dict:
    """Save autoencoder, encoder, scaler, history and metadata."""

    path_prefix = Path(path_prefix)

    path_prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "autoencoder": path_prefix.with_name(
            path_prefix.name + "_autoencoder.keras"
        ),
        "encoder": path_prefix.with_name(
            path_prefix.name + "_encoder.keras"
        ),
        "scaler": path_prefix.with_name(
            path_prefix.name + "_scaler.joblib"
        ),
        "history": path_prefix.with_name(
            path_prefix.name + "_training_history.csv"
        ),
        "metadata": path_prefix.with_name(
            path_prefix.name + "_metadata.json"
        ),
    }

    autoencoder.save(
        paths["autoencoder"]
    )

    encoder.save(
        paths["encoder"]
    )

    joblib.dump(
        scaler,
        paths["scaler"],
    )

    pd.DataFrame(
        history.history
    ).to_csv(
        paths["history"],
        index=False,
    )

    full_metadata = {
        "encoding_dim": encoder.output_shape[-1],
        "input_dim": encoder.input_shape[-1],
        "epochs_trained": len(
            history.history["loss"]
        ),
        "final_train_loss": float(
            history.history["loss"][-1]
        ),
        "final_val_loss": float(
            history.history["val_loss"][-1]
        ),
    }

    if metadata:
        full_metadata.update(metadata)

    with open(
        paths["metadata"],
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            full_metadata,
            file,
            indent=2,
        )

    return {
        key: str(value)
        for key, value in paths.items()
    }


@lru_cache(maxsize=None)
def load_autoencoder(
    path_prefix: Path,
) -> dict:
    """Load saved autoencoder artefacts without retraining."""

    path_prefix = Path(path_prefix)

    paths = {
        "autoencoder": path_prefix.with_name(
            path_prefix.name + "_autoencoder.keras"
        ),
        "encoder": path_prefix.with_name(
            path_prefix.name + "_encoder.keras"
        ),
        "scaler": path_prefix.with_name(
            path_prefix.name + "_scaler.joblib"
        ),
        "metadata": path_prefix.with_name(
            path_prefix.name + "_metadata.json"
        ),
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{name} not found at {path} -- "
                "was save_autoencoder() run for this prefix?"
            )

    autoencoder = keras.models.load_model(
        paths["autoencoder"]
    )

    encoder = keras.models.load_model(
        paths["encoder"]
    )

    scaler = joblib.load(
        paths["scaler"]
    )

    with open(
        paths["metadata"],
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    return {
        "autoencoder": autoencoder,
        "encoder": encoder,
        "scaler": scaler,
        "metadata": metadata,
    }


def predict_encode(
    path_prefix: Path,
    X: pd.DataFrame,
) -> np.ndarray:
    """Encode inference-time data using saved artefacts only."""

    loaded = load_autoencoder(
        path_prefix
    )

    expected_dim = loaded[
        "metadata"
    ]["input_dim"]

    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Input has {X.shape[1]} columns, "
            f"but the saved encoder at {path_prefix} "
            f"expects {expected_dim} -- "
            "check the feature configuration matches."
        )

    scaled = loaded[
        "scaler"
    ].transform(X)

    return loaded[
        "encoder"
    ](
        scaled,
        training=False,
    ).numpy()