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

from data.tf_loader import (
    load_mixed_datasets,
    load_source_disjoint_fold,
)

from models.custom_cnn import build_custom_cnn


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
LEARNING_RATE = 1e-3
BATCH_SIZE = 16


# ============================================================
# SUPPORTED RESEARCH SPLITS
# ============================================================

SUPPORTED_SPLITS = {
    "MIXED_RANDOM_V1",
    "SD1_FDL_AerialLandscape",
    "SD2_Louisiana_SatelliteClassification",
    "SD3_Disasters_AerialLandscape",
    "SD4_Cyclone_SatelliteClassification_Forest",
}


def load_experiment_split(split_name):
    """
    Load one of the frozen research splits.

    MIXED_RANDOM_V1:
        Conventional source-mixed baseline.

    SD1-SD4:
        Source-disjoint experimental folds.
    """

    if split_name not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported split: {split_name}\n"
            f"Supported splits: {sorted(SUPPORTED_SPLITS)}"
        )

    if split_name == "MIXED_RANDOM_V1":
        return load_mixed_datasets()

    return load_source_disjoint_fold(
        split_name
    )


# ============================================================
# MAIN
# ============================================================

def main(
    epochs,
    run_name,
    split_name
):

    print("=" * 75)
    print(run_name)
    print("=" * 75)

    print(
        f"\nDataset split: {split_name}"
    )


    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    keras.utils.set_random_seed(
        SEED
    )

    try:
        tf.config.experimental.enable_op_determinism()

        print(
            "Deterministic TensorFlow operations enabled."
        )

    except Exception:

        print(
            "Deterministic operations could not be explicitly enabled."
        )


    # ========================================================
    # HARDWARE INFORMATION
    # ========================================================

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

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


    # ========================================================
    # RESULTS DIRECTORY
    #
    # Local default:
    #   repo/results/
    #
    # Colab:
    #   /content/drive/MyDrive/AIRE410/results
    # ========================================================

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


    # ========================================================
    # LOAD DATA
    # ========================================================

    print(
        f"\nLoading split: {split_name}"
    )

    train_ds, val_ds, _ = (
        load_experiment_split(
            split_name
        )
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    print(
        "\nBuilding CustomCNN_V1..."
    )

    model = build_custom_cnn()


    # ========================================================
    # COMPILE
    # ========================================================

    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    loss = (
        keras.losses
        .SparseCategoricalCrossentropy()
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            keras.metrics
            .SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )


    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    model.summary()


    # ========================================================
    # CALLBACKS
    # ========================================================

    best_model_path = (
        output_dir
        / "best_model.keras"
    )

    callbacks = [

        # ----------------------------------------------------
        # Save only the model with the lowest validation loss.
        # ----------------------------------------------------

        keras.callbacks.ModelCheckpoint(
            filepath=str(
                best_model_path
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1
        ),

        # ----------------------------------------------------
        # Stop when validation performance stops improving.
        # ----------------------------------------------------

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        # ----------------------------------------------------
        # Lower learning rate when validation loss plateaus.
        # ----------------------------------------------------

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),

        # ----------------------------------------------------
        # Save epoch-by-epoch training results.
        # ----------------------------------------------------

        keras.callbacks.CSVLogger(
            str(
                output_dir
                / "training_log.csv"
            )
        ),
    ]


    # ========================================================
    # SAVE CONFIGURATION BEFORE TRAINING
    # ========================================================

    configuration = {

        "experiment":
            run_name,

        "seed":
            SEED,

        "dataset_split":
            split_name,

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


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\nStarting training..."
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )


    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

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


    # ========================================================
    # LOAD BEST CHECKPOINT EXPLICITLY
    # ========================================================

    print(
        "\nLoading best saved checkpoint..."
    )

    best_model = keras.models.load_model(
        best_model_path
    )


    # ========================================================
    # SAVE OFFICIAL FINAL MODEL
    #
    # final_model.keras is deliberately the
    # best-validation checkpoint, not the final epoch.
    # ========================================================

    final_model_path = (
        output_dir
        / "final_model.keras"
    )

    best_model.save(
        final_model_path
    )


    # ========================================================
    # FINAL VALIDATION EVALUATION
    # ========================================================

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


    # ========================================================
    # SAVE VALIDATION METRICS
    # ========================================================

    validation_metrics = {

        "experiment":
            run_name,

        "dataset_split":
            split_name,

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


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        f"Experiment: {run_name}"
    )

    print(
        f"Split: {split_name}"
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

    parser = argparse.ArgumentParser(
        description=(
            "Train CustomCNN_V1 on a frozen "
            "AIRE410 research split."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum number of training epochs."
    )

    parser.add_argument(
        "--run-name",
        type=str,

        # Default now points to E002.
        default="E002_CustomCNN_SD1",

        help="Experiment output folder name."
    )

    parser.add_argument(
        "--split",
        type=str,

        # Default now points to SD1.
        default="SD1_FDL_AerialLandscape",

        choices=sorted(
            SUPPORTED_SPLITS
        ),

        help="Frozen research split to use."
    )

    args = parser.parse_args()

    main(
        epochs=args.epochs,
        run_name=args.run_name,
        split_name=args.split
    )