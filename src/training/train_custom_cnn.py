from pathlib import Path
import sys
import json
import argparse
import os

import tensorflow as tf
from tensorflow import keras


# ============================================================
# PROJECT PATHS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_ROOT)
)

from data.tf_loader import load_mixed_datasets
from models.custom_cnn import build_custom_cnn


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
LEARNING_RATE = 1e-3
BATCH_SIZE = 16


# ============================================================
# MAIN
# ============================================================

def main(
    epochs,
    run_name
):

    print("=" * 75)
    print(run_name)
    print("=" * 75)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    keras.utils.set_random_seed(SEED)

    try:
        tf.config.experimental.enable_op_determinism()
        print(
            "Deterministic TensorFlow operations enabled."
        )
    except Exception:
        print(
            "Deterministic operations could not be explicitly enabled."
        )


    # --------------------------------------------------------
    # Hardware information
    # --------------------------------------------------------

    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        compute_device = "GPU"
        print(
            f"GPU detected: {gpus}"
        )
    else:
        compute_device = "CPU"
        print(
            "No GPU detected. Training will use CPU."
        )


    # --------------------------------------------------------
    # Results directory
    #
    # Local default:
    #   repo/results/
    #
    # Colab example:
    #   /content/drive/MyDrive/AIRE410/results
    # --------------------------------------------------------

    results_root = Path(
        os.environ.get(
            "AIRE410_RESULTS_ROOT",
            str(REPO_ROOT / "results")
        )
    )

    output_dir = (
        results_root
        / run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nResults directory:\n"
        f"{output_dir.resolve()}"
    )


    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print(
        "\nLoading MIXED_RANDOM_V1..."
    )

    train_ds, val_ds, _ = (
        load_mixed_datasets()
    )


    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_custom_cnn()


    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    loss = keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )


    # --------------------------------------------------------
    # Model summary
    # --------------------------------------------------------

    model.summary()


    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    best_model_path = (
        output_dir
        / "best_model.keras"
    )

    callbacks = [

        keras.callbacks.ModelCheckpoint(
            filepath=str(
                best_model_path
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),

        keras.callbacks.CSVLogger(
            str(
                output_dir
                / "training_log.csv"
            )
        ),
    ]


    # --------------------------------------------------------
    # Save experiment configuration BEFORE training
    # --------------------------------------------------------

    configuration = {

        "experiment": run_name,

        "seed": SEED,

        "dataset_split":
            "MIXED_RANDOM_V1",

        "model":
            "CustomCNN_V1",

        "image_size":
            [224, 224],

        "batch_size":
            BATCH_SIZE,

        "optimizer":
            "Adam",

        "initial_learning_rate":
            LEARNING_RATE,

        "loss":
            "SparseCategoricalCrossentropy",

        "maximum_epochs":
            epochs,

        "early_stopping_patience":
            5,

        "reduce_lr_patience":
            2,

        "augmentation": {

            "flip":
                "horizontal_and_vertical",

            "rotation":
                0.08,

            "zoom":
                0.10,

            "contrast":
                0.10,
        },

        "class_mapping": {

            "Desert": 0,

            "Flooded": 1,

            "Neither": 2,
        },

        "compute_device":
            compute_device,

        "tensorflow_version":
            tf.__version__,

        "test_set_used":
            False,
    }

    with open(
        output_dir
        / "config.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            configuration,
            f,
            indent=4
        )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "\nStarting training..."
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )


    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_dict = {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }

    with open(
        output_dir
        / "history.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history_dict,
            f,
            indent=4
        )


    # --------------------------------------------------------
    # Load the BEST checkpoint explicitly
    # --------------------------------------------------------

    print(
        "\nLoading best saved checkpoint..."
    )

    best_model = keras.models.load_model(
        best_model_path
    )


    # --------------------------------------------------------
    # Save final official model
    #
    # This is intentionally the best-validation model,
    # not simply the model from the last epoch.
    # --------------------------------------------------------

    final_model_path = (
        output_dir
        / "final_model.keras"
    )

    best_model.save(
        final_model_path
    )


    # --------------------------------------------------------
    # Final validation result
    # --------------------------------------------------------

    print(
        "\nEvaluating BEST model "
        "on VALIDATION only..."
    )

    val_loss, val_accuracy = (
        best_model.evaluate(
            val_ds,
            verbose=1
        )
    )


    # --------------------------------------------------------
    # Save validation metrics
    # --------------------------------------------------------

    validation_metrics = {

        "validation_loss":
            float(val_loss),

        "validation_accuracy":
            float(val_accuracy),

        "test_set_used":
            False,
    }

    with open(
        output_dir
        / "validation_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            validation_metrics,
            f,
            indent=4
        )


    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print(
        "\n" + "=" * 75
    )

    print(
        f"Validation loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        f"\nBest model saved to:\n"
        f"{best_model_path}"
    )

    print(
        f"\nOfficial final model saved to:\n"
        f"{final_model_path}"
    )

    print(
        "\nTEST SET HAS NOT BEEN USED."
    )

    print(
        "=" * 75
    )


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="E001_CustomCNN_Mixed"
    )

    args = parser.parse_args()

    main(
        epochs=args.epochs,
        run_name=args.run_name
    )