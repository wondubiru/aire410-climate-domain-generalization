"""
E015 — MobileNetV2 + Representation Self-Challenging (RSC)

Controlled comparison against E007.

Protocol
--------
Dataset protocol:
    SD1_FDL_AerialLandscape

Architecture:
    MobileNetV2, ImageNet pretrained

Stage 1:
    Backbone frozen
    Adam, LR = 1e-3
    Max epochs = 10

Stage 2:
    Last 30 backbone layers unfrozen
    BatchNorm layers remain frozen
    Adam, LR = 1e-5
    Max epochs = 20

Model selection:
    Minimum validation loss across Stage 1 and Stage 2

RSC:
    Feature drop fraction = 1/3
    Batch challenge fraction = 1/3

IMPORTANT:
    The test set is NOT used anywhere in this script.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()

SRC_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from data.tf_loader import (
    load_source_disjoint_fold,
)

from models.mobilenet_v2_rsc import (
    MobileNetV2RSC,
)


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224
NUM_CLASSES = 3

BATCH_SIZE = 16

STAGE1_LR = 1e-3
STAGE2_LR = 1e-5

STAGE1_MAX_EPOCHS = 10
STAGE2_MAX_EPOCHS = 20

EARLY_STOPPING_PATIENCE = 5

RSC_FEATURE_DROP = 1.0 / 3.0
RSC_BATCH_DROP = 1.0 / 3.0

FINE_TUNE_LAST_N = 30


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Train MobileNetV2 with "
            "Representation Self-Challenging."
        )
    )

    parser.add_argument(
        "--split",
        type=str,
        default="SD1_FDL_AerialLandscape",
        help="Frozen source-disjoint split name.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="E015_MobileNetV2_RSC_SD1",
        help="Experiment output directory name.",
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def pythonize_history(history_dict):
    """
    Convert Keras history values into JSON-safe Python floats.
    """

    clean = {}

    for key, values in history_dict.items():

        clean[key] = [
            float(v)
            for v in values
        ]

    return clean


def best_epoch_from_history(
    history,
):
    """
    Select the epoch having minimum validation loss.

    Returns
    -------
    index : int
        Zero-based epoch index.

    val_loss : float

    val_accuracy : float
    """

    val_losses = history.history[
        "val_loss"
    ]

    best_idx = int(
        np.argmin(
            val_losses
        )
    )

    best_loss = float(
        val_losses[
            best_idx
        ]
    )

    val_accuracy_history = (
        history.history.get(
            "val_accuracy"
        )
    )

    if val_accuracy_history is None:

        raise RuntimeError(
            "val_accuracy was not found "
            "in the Keras history."
        )

    best_accuracy = float(
        val_accuracy_history[
            best_idx
        ]
    )

    return (
        best_idx,
        best_loss,
        best_accuracy,
    )


def make_callbacks(
    checkpoint_path,
    csv_path,
):
    """
    Fresh callbacks for one training stage.
    """

    checkpoint = (
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                checkpoint_path
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
    )

    early_stopping = (
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=(
                EARLY_STOPPING_PATIENCE
            ),
            restore_best_weights=False,
            verbose=1,
        )
    )

    reduce_lr = (
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        )
    )

    csv_logger = (
        tf.keras.callbacks.CSVLogger(
            filename=str(
                csv_path
            ),
            append=False,
        )
    )

    return [
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger,
    ]


def count_trainable_parameters(
    model,
):
    """
    Count trainable parameters.
    """

    return int(
        np.sum(
            [
                np.prod(
                    variable.shape
                )
                for variable
                in model.trainable_variables
            ]
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    tf.keras.utils.set_random_seed(
        SEED
    )

    try:
        tf.config.experimental.enable_op_determinism()

        print(
            "TensorFlow deterministic "
            "operations enabled."
        )

    except Exception:

        print(
            "TensorFlow deterministic "
            "operations unavailable."
        )


    # --------------------------------------------------------
    # RESULTS DIRECTORY
    # --------------------------------------------------------

    results_root = Path(
        os.environ.get(
            "AIRE410_RESULTS_ROOT",
            str(
                REPO_ROOT
                / "results"
            ),
        )
    )

    run_dir = (
        results_root
        / args.run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "E015 — MOBILENETV2 + "
        "REPRESENTATION SELF-CHALLENGING"
    )

    print("=" * 78)

    print(
        "Experiment:",
        args.run_name
    )

    print(
        "Split:",
        args.split
    )

    print(
        "Seed:",
        SEED
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "RSC feature drop:",
        RSC_FEATURE_DROP
    )

    print(
        "RSC batch challenge:",
        RSC_BATCH_DROP
    )

    print(
        "Results directory:",
        run_dir
    )

    print("=" * 78)


    # ========================================================
    # LOAD FROZEN DATA SPLIT
    # ========================================================

    print(
        "\nLoading frozen "
        "source-disjoint split..."
    )

    train_ds, val_ds, _ = (
        load_source_disjoint_fold(
            args.split
        )
    )

    print(
        "Datasets loaded."
    )

    print(
        "Test set will NOT be "
        "evaluated during training."
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    print(
        "\nBuilding RSC model..."
    )

    model = MobileNetV2RSC(
        input_shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        num_classes=NUM_CLASSES,
        dropout_rate=0.30,
        feature_drop_fraction=(
            RSC_FEATURE_DROP
        ),
        batch_drop_fraction=(
            RSC_BATCH_DROP
        ),
        weights="imagenet",
        name=(
            "MobileNetV2_ImageNet_RSC_V1"
        ),
    )


    # Build subclass model explicitly.
    dummy = tf.zeros(
        shape=(
            1,
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        dtype=tf.float32,
    )

    _ = model(
        dummy,
        training=False,
    )


    # ========================================================
    # STAGE 1
    # ========================================================

    print()
    print("=" * 78)
    print(
        "STAGE 1 — FROZEN BACKBONE"
    )
    print("=" * 78)

    model.freeze_backbone()

    model.compile(
        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=(
                    STAGE1_LR
                )
            )
        )
    )

    print(
        "Stage 1 learning rate:",
        STAGE1_LR
    )

    print(
        "Trainable parameters:",
        count_trainable_parameters(
            model
        )
    )


    stage1_weights = (
        run_dir
        / "stage1_best.weights.h5"
    )

    stage1_csv = (
        run_dir
        / "stage1_history.csv"
    )

    stage1_callbacks = (
        make_callbacks(
            checkpoint_path=(
                stage1_weights
            ),
            csv_path=(
                stage1_csv
            ),
        )
    )


    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=(
            STAGE1_MAX_EPOCHS
        ),
        callbacks=(
            stage1_callbacks
        ),
        verbose=1,
    )


    (
        stage1_best_idx,
        stage1_best_loss,
        stage1_best_acc,
    ) = best_epoch_from_history(
        history_stage1
    )


    print()
    print(
        "Stage 1 best epoch:",
        stage1_best_idx + 1
    )

    print(
        "Stage 1 best val loss:",
        f"{stage1_best_loss:.6f}"
    )

    print(
        "Stage 1 best val accuracy:",
        f"{stage1_best_acc:.6f}"
    )


    # --------------------------------------------------------
    # Restore Stage 1 best checkpoint before fine-tuning
    # --------------------------------------------------------

    model.load_weights(
        stage1_weights
    )

    print(
        "Stage 1 best weights restored."
    )


    # ========================================================
    # STAGE 2
    # ========================================================

    print()
    print("=" * 78)

    print(
        "STAGE 2 — FINE-TUNE "
        "LAST 30 BACKBONE LAYERS"
    )

    print("=" * 78)


    model.unfreeze_last_layers(
        last_n=(
            FINE_TUNE_LAST_N
        )
    )


    # Recompile after changing trainability.
    model.compile(
        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=(
                    STAGE2_LR
                )
            )
        )
    )


    print(
        "Stage 2 learning rate:",
        STAGE2_LR
    )

    print(
        "Fine-tuned backbone layers:",
        FINE_TUNE_LAST_N
    )

    print(
        "BatchNormalization layers:"
        " frozen"
    )

    print(
        "Trainable parameters:",
        count_trainable_parameters(
            model
        )
    )


    stage2_weights = (
        run_dir
        / "stage2_best.weights.h5"
    )

    stage2_csv = (
        run_dir
        / "stage2_history.csv"
    )

    stage2_callbacks = (
        make_callbacks(
            checkpoint_path=(
                stage2_weights
            ),
            csv_path=(
                stage2_csv
            ),
        )
    )


    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=(
            STAGE2_MAX_EPOCHS
        ),
        callbacks=(
            stage2_callbacks
        ),
        verbose=1,
    )


    (
        stage2_best_idx,
        stage2_best_loss,
        stage2_best_acc,
    ) = best_epoch_from_history(
        history_stage2
    )


    print()
    print(
        "Stage 2 best epoch:",
        stage2_best_idx + 1
    )

    print(
        "Stage 2 best val loss:",
        f"{stage2_best_loss:.6f}"
    )

    print(
        "Stage 2 best val accuracy:",
        f"{stage2_best_acc:.6f}"
    )


    # ========================================================
    # MODEL SELECTION
    # ========================================================

    print()
    print("=" * 78)
    print(
        "OFFICIAL MODEL SELECTION"
    )
    print("=" * 78)


    if (
        stage2_best_loss
        < stage1_best_loss
    ):

        selected_stage = (
            "stage2"
        )

        selected_weights = (
            stage2_weights
        )

        selected_val_loss = (
            stage2_best_loss
        )

        selected_val_accuracy = (
            stage2_best_acc
        )

    else:

        selected_stage = (
            "stage1"
        )

        selected_weights = (
            stage1_weights
        )

        selected_val_loss = (
            stage1_best_loss
        )

        selected_val_accuracy = (
            stage1_best_acc
        )


    print(
        "Selected stage:",
        selected_stage
    )

    print(
        "Selected validation loss:",
        f"{selected_val_loss:.6f}"
    )

    print(
        "Selected validation accuracy:",
        f"{selected_val_accuracy:.6f}"
    )


    # --------------------------------------------------------
    # Restore selected checkpoint
    # --------------------------------------------------------

    model.load_weights(
        selected_weights
    )

    print(
        "Selected checkpoint restored."
    )


    # ========================================================
    # EXPORT CLEAN INFERENCE MODEL
    # ========================================================

    print(
        "\nExporting clean "
        "inference model..."
    )


    best_model = (
        model.build_inference_model()
    )


    best_model_path = (
        run_dir
        / "best_model.keras"
    )


    best_model.save(
        best_model_path
    )


    print(
        "Saved:",
        best_model_path
    )


    # --------------------------------------------------------
    # Verify exported model
    # --------------------------------------------------------

    verification_input = (
        tf.zeros(
            shape=(
                2,
                IMAGE_SIZE,
                IMAGE_SIZE,
                3,
            ),
            dtype=tf.float32,
        )
    )


    verification_output = (
        best_model(
            verification_input,
            training=False,
        )
    )


    probability_sums = (
        tf.reduce_sum(
            verification_output,
            axis=1,
        )
        .numpy()
    )


    print(
        "Exported model output:",
        verification_output.shape
    )

    print(
        "Probability sums:",
        probability_sums
    )


    # ========================================================
    # SAVE CONFIGURATION
    # ========================================================

    config = {

        "experiment":
            args.run_name,

        "model":
            "MobileNetV2_ImageNet_RSC_V1",

        "split":
            args.split,

        "seed":
            SEED,

        "image_size":
            IMAGE_SIZE,

        "batch_size":
            BATCH_SIZE,

        "num_classes":
            NUM_CLASSES,

        "pretraining":
            "ImageNet",

        "optimizer":
            "Adam",

        "stage1_learning_rate":
            STAGE1_LR,

        "stage2_learning_rate":
            STAGE2_LR,

        "stage1_max_epochs":
            STAGE1_MAX_EPOCHS,

        "stage2_max_epochs":
            STAGE2_MAX_EPOCHS,

        "fine_tune_last_n":
            FINE_TUNE_LAST_N,

        "batchnorm_frozen":
            True,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "selection_metric":
            "val_loss",

        "rsc_feature_drop_fraction":
            RSC_FEATURE_DROP,

        "rsc_batch_drop_fraction":
            RSC_BATCH_DROP,

        "augmentation": [
            "horizontal_and_vertical_flip",
            "rotation_0.08",
            "zoom_0.10",
            "contrast_0.10",
        ],

        "test_used_for_model_selection":
            False,
    }


    with open(
        run_dir
        / "config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )


    # ========================================================
    # SAVE HISTORIES
    # ========================================================

    history_json = {

        "stage1":
            pythonize_history(
                history_stage1.history
            ),

        "stage2":
            pythonize_history(
                history_stage2.history
            ),
    }


    with open(
        run_dir
        / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history_json,
            f,
            indent=4,
        )


    # ========================================================
    # SAVE VALIDATION SUMMARY
    # ========================================================

    summary = {

        "experiment":
            args.run_name,

        "split":
            args.split,

        "stage1_best_epoch":
            stage1_best_idx + 1,

        "stage1_best_val_loss":
            stage1_best_loss,

        "stage1_best_val_accuracy":
            stage1_best_acc,

        "stage2_best_epoch":
            stage2_best_idx + 1,

        "stage2_best_val_loss":
            stage2_best_loss,

        "stage2_best_val_accuracy":
            stage2_best_acc,

        "selected_stage":
            selected_stage,

        "selected_val_loss":
            selected_val_loss,

        "selected_val_accuracy":
            selected_val_accuracy,

        "official_model":
            str(
                best_model_path
            ),
    }


    with open(
        run_dir
        / "validation_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 78)
    print(
        "E015 TRAINING COMPLETE"
    )
    print("=" * 78)

    print(
        f"Stage 1 best val loss: "
        f"{stage1_best_loss:.6f}"
    )

    print(
        f"Stage 1 best val accuracy: "
        f"{stage1_best_acc:.6f}"
    )

    print(
        f"Stage 2 best val loss: "
        f"{stage2_best_loss:.6f}"
    )

    print(
        f"Stage 2 best val accuracy: "
        f"{stage2_best_acc:.6f}"
    )

    print(
        "Selected stage:",
        selected_stage
    )

    print(
        f"Selected val loss: "
        f"{selected_val_loss:.6f}"
    )

    print(
        f"Selected val accuracy: "
        f"{selected_val_accuracy:.6f}"
    )

    print(
        "Official model:",
        best_model_path
    )

    print()
    print(
        "TEST SET HAS NOT BEEN USED."
    )

    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()