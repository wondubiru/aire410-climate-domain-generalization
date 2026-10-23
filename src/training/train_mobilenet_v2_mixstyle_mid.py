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

from models.mobilenet_v2_mixstyle_mid import (
    build_mobilenet_v2_mixstyle_mid,
)

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 16

STAGE1_LR = 1e-3
STAGE2_LR = 1e-5

STAGE1_EPOCHS = 10
STAGE2_EPOCHS = 20

FINE_TUNE_LAST_N = 30


SUPPORTED_SPLITS = {
    "MIXED_RANDOM_V1",
    "SD1_FDL_AerialLandscape",
    "SD2_Louisiana_SatelliteClassification",
    "SD3_Disasters_AerialLandscape",
    "SD4_Cyclone_SatelliteClassification_Forest",
}


# ============================================================
# DATASET LOADER
# ============================================================

def load_experiment_split(split_name):

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
# CALLBACK FACTORY
# ============================================================

def make_callbacks(
    checkpoint_path,
    csv_path,
    patience=5,
):

    return [

        keras.callbacks.ModelCheckpoint(
            filepath=str(
                checkpoint_path
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),

        keras.callbacks.CSVLogger(
            str(csv_path)
        ),
    ]


# ============================================================
# COMPILE
# ============================================================

def compile_model(
    model,
    learning_rate,
):

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate
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
        ],
    )


# ============================================================
# ENABLE CONTROLLED FINE-TUNING
# ============================================================

def configure_fine_tuning(
    model,
    last_n_layers=30,
):

    backbone = model.get_layer(
        "mobilenetv2_1.00_224"
    )

    backbone.trainable = True

    # Freeze all but the last N backbone layers
    for layer in backbone.layers[
        :-last_n_layers
    ]:
        layer.trainable = False

    # Keep BatchNorm frozen.
    #
    # This is especially important with a relatively
    # small batch size such as 16.
    for layer in backbone.layers:
        if isinstance(
            layer,
            keras.layers.BatchNormalization
        ):
            layer.trainable = False

    trainable_backbone_layers = [
        layer.name
        for layer in backbone.layers
        if layer.trainable
    ]

    return trainable_backbone_layers


# ============================================================
# MAIN
# ============================================================

def main(
    run_name,
    split_name,
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
            "Could not explicitly enable deterministic operations."
        )


    # ========================================================
    # HARDWARE
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
            "No GPU detected."
        )


    # ========================================================
    # OUTPUT DIRECTORY
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
        "\nBuilding MobileNetV2..."
    )

    model = build_mobilenet_v2_mixstyle_mid(
    backbone_trainable=False,
    mixstyle_p=0.5,
    mixstyle_alpha=0.1,
   )
    model.summary()


    # ========================================================
    # SAVE EXPERIMENT CONFIGURATION
    # ========================================================

    configuration = {

        "experiment":
            run_name,

        "model":
           "MobileNetV2_ImageNet_MixStyle_Mid_V1",

        "dataset_split":
            split_name,

        "seed":
            SEED,

        "image_size":
            [224, 224],

        "batch_size":
            BATCH_SIZE,

        "optimizer":
            "Adam",

        "stage1": {

            "backbone":
                "frozen",

            "learning_rate":
                STAGE1_LR,

            "maximum_epochs":
                STAGE1_EPOCHS,
        },

        "stage2": {

            "fine_tune_last_n_layers":
                FINE_TUNE_LAST_N,

            "batchnorm_frozen":
                True,

            "learning_rate":
                STAGE2_LR,

            "maximum_epochs":
                STAGE2_EPOCHS,
        },

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

        "input_loader_range":
            "[0,1]",

        "mobilenet_internal_range":
            "[-1,1]",

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
    # STAGE 1
    # FROZEN BACKBONE
    # ========================================================

    print("\n" + "=" * 75)
    print("STAGE 1: FROZEN IMAGENET BACKBONE")
    print("=" * 75)

    compile_model(
        model,
        STAGE1_LR
    )

    stage1_checkpoint = (
        output_dir
        / "stage1_best.keras"
    )

    stage1_callbacks = make_callbacks(
        checkpoint_path=stage1_checkpoint,
        csv_path=(
            output_dir
            / "stage1_training_log.csv"
        ),
    )

    stage1_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        callbacks=stage1_callbacks,
    )


    # ========================================================
    # LOAD BEST STAGE 1 MODEL
    # ========================================================

    print(
        "\nLoading best Stage 1 checkpoint..."
    )

    model = keras.models.load_model(
        stage1_checkpoint
    )

    stage1_val_loss, stage1_val_accuracy = (
        model.evaluate(
            val_ds,
            verbose=1
        )
    )

    print(
        f"\nStage 1 best val loss: "
        f"{stage1_val_loss:.6f}"
    )

    print(
        f"Stage 1 best val accuracy: "
        f"{stage1_val_accuracy:.6f}"
    )


    # ========================================================
    # STAGE 2
    # CONTROLLED FINE-TUNING
    # ========================================================

    print("\n" + "=" * 75)
    print("STAGE 2: CONTROLLED FINE-TUNING")
    print("=" * 75)

    trainable_backbone_layers = (
        configure_fine_tuning(
            model,
            last_n_layers=FINE_TUNE_LAST_N,
        )
    )

    print(
        "\nTrainable MobileNetV2 "
        "backbone layers:"
    )

    print(
        len(trainable_backbone_layers)
    )

    for name in trainable_backbone_layers:
        print(
            "  ",
            name
        )


    # IMPORTANT:
    # Recompile after changing trainable layers.
    compile_model(
        model,
        STAGE2_LR
    )

    stage2_checkpoint = (
        output_dir
        / "stage2_best.keras"
    )

    stage2_callbacks = make_callbacks(
        checkpoint_path=stage2_checkpoint,
        csv_path=(
            output_dir
            / "stage2_training_log.csv"
        ),
    )

    stage2_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE2_EPOCHS,
        callbacks=stage2_callbacks,
    )


    # ========================================================
    # LOAD BEST STAGE 2 MODEL
    # ========================================================

    print(
        "\nLoading best Stage 2 checkpoint..."
    )

    stage2_model = (
        keras.models.load_model(
            stage2_checkpoint
        )
    )

    stage2_val_loss, stage2_val_accuracy = (
        stage2_model.evaluate(
            val_ds,
            verbose=1
        )
    )

    print(
        f"\nStage 2 best val loss: "
        f"{stage2_val_loss:.6f}"
    )

    print(
        f"Stage 2 best val accuracy: "
        f"{stage2_val_accuracy:.6f}"
    )


    # ========================================================
    # SELECT OFFICIAL MODEL USING VALIDATION ONLY
    # ========================================================

    print("\n" + "=" * 75)
    print("OFFICIAL MODEL SELECTION")
    print("=" * 75)

    if stage2_val_loss < stage1_val_loss:

        best_stage = "stage2"

        official_model = (
            stage2_model
        )

        official_val_loss = (
            stage2_val_loss
        )

        official_val_accuracy = (
            stage2_val_accuracy
        )

    else:

        best_stage = "stage1"

        official_model = (
            keras.models.load_model(
                stage1_checkpoint
            )
        )

        official_val_loss = (
            stage1_val_loss
        )

        official_val_accuracy = (
            stage1_val_accuracy
        )


    # ========================================================
    # SAVE OFFICIAL MODEL
    # ========================================================

    final_model_path = (
        output_dir
        / "final_model.keras"
    )

    official_model.save(
        final_model_path
    )

    # Also retain standard filename used by
    # the previous experiment family.
    best_model_path = (
        output_dir
        / "best_model.keras"
    )

    official_model.save(
        best_model_path
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    histories = {

        "stage1": {
            key: [
                float(v)
                for v in values
            ]
            for key, values
            in stage1_history.history.items()
        },

        "stage2": {
            key: [
                float(v)
                for v in values
            ]
            for key, values
            in stage2_history.history.items()
        },
    }

    with open(
        output_dir
        / "history.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            histories,
            f,
            indent=4
        )


    # ========================================================
    # SAVE VALIDATION RESULT
    # ========================================================

    validation_metrics = {

        "experiment":
            run_name,

        "dataset_split":
            split_name,

        "selected_stage":
            best_stage,

        "stage1_validation_loss":
            float(stage1_val_loss),

        "stage1_validation_accuracy":
            float(stage1_val_accuracy),

        "stage2_validation_loss":
            float(stage2_val_loss),

        "stage2_validation_accuracy":
            float(stage2_val_accuracy),

        "validation_loss":
            float(official_val_loss),

        "validation_accuracy":
            float(official_val_accuracy),

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

    print("\n" + "=" * 75)

    print(
        f"Experiment: {run_name}"
    )

    print(
        f"Split: {split_name}"
    )

    print(
        f"Selected stage: {best_stage}"
    )

    print(
        f"Validation loss: "
        f"{official_val_loss:.6f}"
    )

    print(
        f"Validation accuracy: "
        f"{official_val_accuracy:.6f}"
    )

    print(
        f"\nOfficial best model:\n"
        f"{best_model_path}"
    )

    print(
        "\nTEST SET HAS NOT BEEN USED."
    )

    print("=" * 75)


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Two-stage MobileNetV2 "
            "transfer-learning experiment."
        )
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default="E014_MobileNetV2_MixStyleMid_SD1",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="MIXED_RANDOM_V1",
        choices=sorted(
            SUPPORTED_SPLITS
        ),
    )

    args = parser.parse_args()

    main(
        run_name=args.run_name,
        split_name=args.split,
    )