"""
Multi-seed confirmation trainer for DG-AUG-V2.

Replicates E012:
    MobileNetV2 ImageNet
    + baseline augmentation
    + translation 0.05
    + brightness 0.08

Frozen protocol:
    SD1_FDL_AerialLandscape

Only training randomness changes between seeds.
The train/validation/test membership does NOT change.

Examples
--------
Seed 123:
python src/training/train_mobilenet_v2_dgv2_seeded.py \
    --seed 123 \
    --run-name E018_MobileNetV2_DGAugV2_SD1_seed123

Seed 2026:
python src/training/train_mobilenet_v2_dgv2_seeded.py \
    --seed 2026 \
    --run-name E019_MobileNetV2_DGAugV2_SD1_seed2026
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


# ============================================================
# FIXED PROTOCOL
# ============================================================

IMAGE_SIZE = 224
NUM_CLASSES = 3
BATCH_SIZE = 16

STAGE1_LR = 1e-3
STAGE2_LR = 1e-5

STAGE1_MAX_EPOCHS = 10
STAGE2_MAX_EPOCHS = 20

EARLY_STOPPING_PATIENCE = 5
FINE_TUNE_LAST_N = 30

DEFAULT_SPLIT = "SD1_FDL_AerialLandscape"


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Multi-seed confirmation of "
            "MobileNetV2 DG-AUG-V2."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Training randomness seed.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default=DEFAULT_SPLIT,
        help="Frozen source-disjoint split.",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Experiment output directory.",
    )

    return parser.parse_args()


# ============================================================
# DATASET RESEEDING
# ============================================================

def reseed_training_dataset(
    train_ds,
    seed,
):
    """
    Change training order only.

    Split membership remains frozen.
    """

    train_ds = train_ds.unbatch()

    train_ds = train_ds.shuffle(
        buffer_size=10000,
        seed=seed,
        reshuffle_each_iteration=True,
    )

    train_ds = train_ds.batch(
        BATCH_SIZE,
        drop_remainder=False,
    )

    train_ds = train_ds.prefetch(1)

    return train_ds


# ============================================================
# DG-AUG-V2
# ============================================================

def build_dg_v2_augmentation(
    seed,
):
    """
    Conservative DG-AUG-V2.

    Baseline E007 augmentation:
        flip
        rotation 0.08
        zoom 0.10
        contrast 0.10

    Additional E012 intervention:
        translation 0.05
        brightness 0.08

    No saturation.
    No Gaussian noise.
    """

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(
                "horizontal_and_vertical",
                seed=seed,
            ),

            tf.keras.layers.RandomRotation(
                0.08,
                fill_mode="reflect",
                seed=seed,
            ),

            tf.keras.layers.RandomZoom(
                0.10,
                fill_mode="reflect",
                seed=seed,
            ),

            tf.keras.layers.RandomContrast(
                0.10,
                seed=seed,
            ),

            tf.keras.layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="reflect",
                seed=seed + 1,
            ),

            tf.keras.layers.RandomBrightness(
                factor=0.08,
                value_range=(0.0, 1.0),
                seed=seed + 2,
            ),
        ],
        name="dg_aug_v2",
    )


# ============================================================
# MODEL
# ============================================================

def build_model(
    seed,
):

    augmentation = (
        build_dg_v2_augmentation(
            seed=seed
        )
    )

    backbone = (
        tf.keras.applications.MobileNetV2(
            input_shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
                3,
            ),
            include_top=False,
            weights="imagenet",
        )
    )

    backbone.trainable = False

    inputs = tf.keras.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        name="image",
    )

    x = augmentation(
        inputs
    )

    # Loader output: [0,1]
    # MobileNetV2 input: [-1,1]
    x = tf.keras.layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenet_preprocessing",
    )(x)

    x = backbone(
        x,
        training=False,
    )

    x = (
        tf.keras.layers
        .GlobalAveragePooling2D(
            name="global_average_pool"
        )(x)
    )

    x = tf.keras.layers.Dropout(
        0.30,
        seed=seed,
        name="classifier_dropout",
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=(
            "MobileNetV2_ImageNet_DG_AUG_V2"
        ),
    )

    return model, backbone


# ============================================================
# CALLBACKS
# ============================================================

def make_callbacks(
    checkpoint_path,
    csv_path,
):

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(
                checkpoint_path
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=(
                EARLY_STOPPING_PATIENCE
            ),
            restore_best_weights=False,
            verbose=1,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),

        tf.keras.callbacks.CSVLogger(
            filename=str(
                csv_path
            ),
            append=False,
        ),
    ]


# ============================================================
# HISTORY HELPERS
# ============================================================

def get_best_validation(
    history,
):

    val_losses = (
        history.history[
            "val_loss"
        ]
    )

    best_idx = int(
        np.argmin(
            val_losses
        )
    )

    best_loss = float(
        history.history[
            "val_loss"
        ][best_idx]
    )

    best_accuracy = float(
        history.history[
            "val_accuracy"
        ][best_idx]
    )

    return (
        best_idx,
        best_loss,
        best_accuracy,
    )


def json_safe_history(
    history,
):

    return {
        key: [
            float(value)
            for value in values
        ]
        for key, values
        in history.history.items()
    }


# ============================================================
# FINE-TUNING
# ============================================================

def unfreeze_last_layers(
    backbone,
    last_n,
):

    backbone.trainable = True

    # Freeze all first.
    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze last N except BatchNorm.
    for layer in backbone.layers[
        -last_n:
    ]:

        if isinstance(
            layer,
            tf.keras.layers.BatchNormalization,
        ):

            layer.trainable = False

        else:

            layer.trainable = True


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    seed = int(
        args.seed
    )

    if args.run_name:

        run_name = (
            args.run_name
        )

    else:

        run_name = (
            f"MobileNetV2_DGAugV2_"
            f"SD1_seed{seed}"
        )


    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    tf.keras.utils.set_random_seed(
        seed
    )

    try:

        tf.config.experimental.enable_op_determinism()

        print(
            "Deterministic TensorFlow "
            "operations enabled."
        )

    except Exception:

        print(
            "Deterministic TensorFlow "
            "operations unavailable."
        )


    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

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
        / run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 78)

    print(
        "DG-AUG-V2 MULTI-SEED "
        "CONFIRMATION"
    )

    print("=" * 78)

    print(
        "Experiment:",
        run_name
    )

    print(
        "Split:",
        args.split
    )

    print(
        "Seed:",
        seed
    )

    print(
        "Batch:",
        BATCH_SIZE
    )

    print(
        "DG translation:",
        0.05
    )

    print(
        "DG brightness:",
        0.08
    )

    print()
    print(
        "Frozen split membership."
    )

    print(
        "Only training randomness changes."
    )

    print("=" * 78)


    # ========================================================
    # DATA
    # ========================================================

    train_ds, val_ds, _ = (
        load_source_disjoint_fold(
            args.split
        )
    )

    train_ds = (
        reseed_training_dataset(
            train_ds,
            seed,
        )
    )

    print(
        "\nFrozen SD1 loaded."
    )

    print(
        "Training order seed:",
        seed
    )

    print(
        "Test set NOT used."
    )


    # ========================================================
    # MODEL
    # ========================================================

    model, backbone = (
        build_model(
            seed=seed
        )
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


    backbone.trainable = False


    model.compile(
        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=(
                    STAGE1_LR
                )
            )
        ),
        loss=(
            tf.keras.losses
            .SparseCategoricalCrossentropy()
        ),
        metrics=[
            "accuracy"
        ],
    )


    stage1_weights = (
        run_dir
        / "stage1_best.weights.h5"
    )

    stage1_history_path = (
        run_dir
        / "stage1_history.csv"
    )


    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=(
            STAGE1_MAX_EPOCHS
        ),
        callbacks=make_callbacks(
            stage1_weights,
            stage1_history_path,
        ),
        verbose=1,
    )


    (
        stage1_best_idx,
        stage1_best_loss,
        stage1_best_acc,
    ) = get_best_validation(
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


    # Restore best Stage 1 before Stage 2.
    model.load_weights(
        stage1_weights
    )


    # ========================================================
    # STAGE 2
    # ========================================================

    print()
    print("=" * 78)

    print(
        "STAGE 2 — FINE-TUNE "
        "LAST 30 LAYERS"
    )

    print("=" * 78)


    unfreeze_last_layers(
        backbone,
        FINE_TUNE_LAST_N,
    )


    model.compile(
        optimizer=(
            tf.keras.optimizers.Adam(
                learning_rate=(
                    STAGE2_LR
                )
            )
        ),
        loss=(
            tf.keras.losses
            .SparseCategoricalCrossentropy()
        ),
        metrics=[
            "accuracy"
        ],
    )


    stage2_weights = (
        run_dir
        / "stage2_best.weights.h5"
    )

    stage2_history_path = (
        run_dir
        / "stage2_history.csv"
    )


    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=(
            STAGE2_MAX_EPOCHS
        ),
        callbacks=make_callbacks(
            stage2_weights,
            stage2_history_path,
        ),
        verbose=1,
    )


    (
        stage2_best_idx,
        stage2_best_loss,
        stage2_best_acc,
    ) = get_best_validation(
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
    # OFFICIAL MODEL SELECTION
    # ========================================================

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


    model.load_weights(
        selected_weights
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    best_model_path = (
        run_dir
        / "best_model.keras"
    )

    model.save(
        best_model_path
    )


    # ========================================================
    # CONFIG
    # ========================================================

    config = {

        "experiment":
            run_name,

        "purpose":
            "multi_seed_confirmation",

        "reference_experiment":
            "E012",

        "model":
            "MobileNetV2_ImageNet_DG_AUG_V2",

        "split":
            args.split,

        "split_membership":
            "frozen",

        "training_seed":
            seed,

        "image_size":
            IMAGE_SIZE,

        "batch_size":
            BATCH_SIZE,

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

        "selection_metric":
            "val_loss",

        "augmentation": [
            "horizontal_and_vertical_flip",
            "rotation_0.08",
            "zoom_0.10",
            "contrast_0.10",
            "translation_0.05",
            "brightness_0.08",
        ],

        "test_used_for_selection":
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
    # HISTORIES
    # ========================================================

    history_data = {

        "stage1":
            json_safe_history(
                history_stage1
            ),

        "stage2":
            json_safe_history(
                history_stage2
            ),
    }


    with open(
        run_dir
        / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            history_data,
            f,
            indent=4,
        )


    # ========================================================
    # VALIDATION SUMMARY
    # ========================================================

    validation_summary = {

        "experiment":
            run_name,

        "seed":
            seed,

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
    }


    with open(
        run_dir
        / "validation_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            validation_summary,
            f,
            indent=4,
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 78)

    print(
        "DG-AUG-V2 MULTI-SEED "
        "TRAINING COMPLETE"
    )

    print("=" * 78)

    print(
        "Experiment:",
        run_name
    )

    print(
        "Seed:",
        seed
    )

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