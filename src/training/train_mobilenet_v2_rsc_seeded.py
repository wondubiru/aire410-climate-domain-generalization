"""
Multi-seed confirmation trainer for E015:
MobileNetV2 + Representation Self-Challenging (RSC).

This file is SELF-CONTAINED so that the experimental seed controls:

    - TensorFlow randomness
    - training shuffle
    - image augmentation
    - dropout

Frozen SD1 membership is NEVER changed.

Seed-42 reference:
    E015_MobileNetV2_RSC_SD1

Replication runs:
    E020 = seed 123
    E021 = seed 2026
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
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# PATH SETUP
# ============================================================

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


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

RSC_FEATURE_DROP = 1.0 / 3.0
RSC_BATCH_DROP = 1.0 / 3.0

DEFAULT_SPLIT = "SD1_FDL_AerialLandscape"


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Multi-seed MobileNetV2 + RSC "
            "confirmation."
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
    Preserve frozen train membership.

    Only change training sample order.
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
# RSC MODEL
# ============================================================

class MobileNetV2RSCSeeded(
    keras.Model
):

    def __init__(
        self,
        seed,
        input_shape=(224, 224, 3),
        num_classes=3,
        dropout_rate=0.30,
        feature_drop_fraction=1.0 / 3.0,
        batch_drop_fraction=1.0 / 3.0,
        weights="imagenet",
        **kwargs,
    ):

        super().__init__(
            **kwargs
        )

        self.seed = int(
            seed
        )

        self.num_classes = int(
            num_classes
        )

        self.feature_drop_fraction = float(
            feature_drop_fraction
        )

        self.batch_drop_fraction = float(
            batch_drop_fraction
        )


        # ====================================================
        # EXACT BASELINE AUGMENTATION
        # ====================================================

        self.augmentation = (
            keras.Sequential(
                [
                    layers.RandomFlip(
                        "horizontal_and_vertical",
                        seed=self.seed,
                    ),

                    layers.RandomRotation(
                        0.08,
                        seed=self.seed,
                    ),

                    layers.RandomZoom(
                        0.10,
                        seed=self.seed,
                    ),

                    layers.RandomContrast(
                        0.10,
                        seed=self.seed,
                    ),
                ],
                name="augmentation",
            )
        )


        # ====================================================
        # PREPROCESSING
        # ====================================================

        self.preprocessing = (
            layers.Rescaling(
                scale=2.0,
                offset=-1.0,
                name=(
                    "mobilenet_preprocessing"
                ),
            )
        )


        # ====================================================
        # IMAGENET BACKBONE
        # ====================================================

        self.backbone = (
            keras.applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights=weights,
            )
        )

        self.backbone.trainable = False


        # ====================================================
        # CLASSIFIER HEAD
        # ====================================================

        self.gap = (
            layers.GlobalAveragePooling2D(
                name="global_average_pool"
            )
        )

        self.dropout = (
            layers.Dropout(
                dropout_rate,
                seed=self.seed,
                name="classifier_dropout",
            )
        )

        self.classifier = (
            layers.Dense(
                num_classes,
                activation=None,
                name="predictions_logits",
            )
        )


        # ====================================================
        # LOSS / METRICS
        # ====================================================

        self.loss_fn = (
            keras.losses
            .SparseCategoricalCrossentropy(
                from_logits=True
            )
        )

        self.loss_tracker = (
            keras.metrics.Mean(
                name="loss"
            )
        )

        self.accuracy_tracker = (
            keras.metrics
            .SparseCategoricalAccuracy(
                name="accuracy"
            )
        )

        self.rsc_feature_keep_tracker = (
            keras.metrics.Mean(
                name="rsc_feature_keep"
            )
        )

        self.rsc_batch_challenge_tracker = (
            keras.metrics.Mean(
                name="rsc_batch_challenge"
            )
        )


    # ========================================================
    # METRICS
    # ========================================================

    @property
    def metrics(self):

        return [
            self.loss_tracker,
            self.accuracy_tracker,
            self.rsc_feature_keep_tracker,
            self.rsc_batch_challenge_tracker,
        ]


    # ========================================================
    # REPRESENTATION
    # ========================================================

    def extract_representation(
        self,
        inputs,
        training=False,
    ):

        x = self.augmentation(
            inputs,
            training=training,
        )

        x = self.preprocessing(
            x
        )

        # BN remains in inference mode.
        x = self.backbone(
            x,
            training=False,
        )

        x = self.gap(
            x
        )

        x = self.dropout(
            x,
            training=training,
        )

        return x


    # ========================================================
    # INFERENCE CALL
    # ========================================================

    def call(
        self,
        inputs,
        training=False,
    ):

        representation = (
            self.extract_representation(
                inputs,
                training=training,
            )
        )

        logits = self.classifier(
            representation
        )

        return tf.nn.softmax(
            logits,
            axis=-1,
        )


    # ========================================================
    # LINEAR PERCENTILE
    # ========================================================

    @staticmethod
    def _percentile_threshold(
        values,
        keep_fraction,
        axis,
    ):

        sorted_values = tf.sort(
            values,
            axis=axis,
        )

        n = tf.shape(
            values
        )[axis]

        position = (
            tf.cast(
                n - 1,
                tf.float32,
            )
            *
            tf.cast(
                keep_fraction,
                tf.float32,
            )
        )

        lower_index = tf.cast(
            tf.floor(
                position
            ),
            tf.int32,
        )

        upper_index = tf.cast(
            tf.math.ceil(
                position
            ),
            tf.int32,
        )

        weight = (
            position
            -
            tf.cast(
                lower_index,
                tf.float32,
            )
        )

        lower_value = tf.gather(
            sorted_values,
            lower_index,
            axis=axis,
        )

        upper_value = tf.gather(
            sorted_values,
            upper_index,
            axis=axis,
        )

        threshold = (
            lower_value
            +
            weight
            *
            (
                upper_value
                -
                lower_value
            )
        )

        return threshold


    # ========================================================
    # RSC TRAINING STEP
    # ========================================================

    def train_step(
        self,
        data,
    ):

        (
            images,
            labels,
            sample_weight,
        ) = (
            keras.utils
            .unpack_x_y_sample_weight(
                data
            )
        )

        if labels is None:

            raise ValueError(
                "RSC training requires labels."
            )


        labels = tf.cast(
            labels,
            tf.int32,
        )

        one_hot = tf.one_hot(
            labels,
            depth=self.num_classes,
            dtype=tf.float32,
        )


        # ====================================================
        # OUTER OPTIMIZATION TAPE
        # ====================================================

        with tf.GradientTape() as outer_tape:

            representation = (
                self.extract_representation(
                    images,
                    training=True,
                )
            )


            # ================================================
            # FEATURE IMPORTANCE
            # ================================================

            with tf.GradientTape() as inner_tape:

                inner_tape.watch(
                    representation
                )

                logits_before = (
                    self.classifier(
                        representation
                    )
                )

                true_class_score = (
                    tf.reduce_sum(
                        logits_before
                        *
                        one_hot
                    )
                )


            importance = (
                inner_tape.gradient(
                    true_class_score,
                    representation,
                )
            )

            importance = (
                tf.stop_gradient(
                    importance
                )
            )


            # ================================================
            # FEATURE CHALLENGE
            # ================================================

            feature_keep_fraction = (
                1.0
                -
                self.feature_drop_fraction
            )


            feature_threshold = (
                self._percentile_threshold(
                    importance,
                    keep_fraction=(
                        feature_keep_fraction
                    ),
                    axis=1,
                )
            )

            feature_threshold = (
                feature_threshold[
                    :,
                    None
                ]
            )


            feature_mask = tf.cast(
                importance
                <
                feature_threshold,
                tf.float32,
            )


            muted_representation = (
                representation
                *
                feature_mask
            )


            logits_muted = (
                self.classifier(
                    muted_representation
                )
            )


            # ================================================
            # BATCH CHALLENGE
            # ================================================

            probs_before = tf.nn.softmax(
                logits_before,
                axis=-1,
            )

            probs_muted = tf.nn.softmax(
                logits_muted,
                axis=-1,
            )


            true_prob_before = (
                tf.reduce_sum(
                    probs_before
                    *
                    one_hot,
                    axis=1,
                )
            )

            true_prob_after = (
                tf.reduce_sum(
                    probs_muted
                    *
                    one_hot,
                    axis=1,
                )
            )


            probability_change = (
                true_prob_before
                -
                true_prob_after
            )

            probability_change = (
                tf.stop_gradient(
                    probability_change
                )
            )


            batch_keep_fraction = (
                1.0
                -
                self.batch_drop_fraction
            )


            batch_threshold = (
                self._percentile_threshold(
                    probability_change,
                    keep_fraction=(
                        batch_keep_fraction
                    ),
                    axis=0,
                )
            )


            keep_entire_sample = (
                tf.cast(
                    probability_change
                    <
                    batch_threshold,
                    tf.float32,
                )
            )

            keep_entire_sample = (
                keep_entire_sample[
                    :,
                    None
                ]
            )


            # ================================================
            # FINAL MASK
            # ================================================

            final_mask = tf.maximum(
                feature_mask,
                keep_entire_sample,
            )

            final_mask = (
                tf.stop_gradient(
                    final_mask
                )
            )


            final_representation = (
                representation
                *
                final_mask
            )


            logits_final = (
                self.classifier(
                    final_representation
                )
            )


            loss = self.loss_fn(
                labels,
                logits_final,
                sample_weight=(
                    sample_weight
                ),
            )


            if self.losses:

                loss += tf.add_n(
                    self.losses
                )


        # ====================================================
        # OPTIMIZATION
        # ====================================================

        gradients = (
            outer_tape.gradient(
                loss,
                self.trainable_variables,
            )
        )


        grads_and_vars = [
            (gradient, variable)

            for gradient, variable
            in zip(
                gradients,
                self.trainable_variables,
            )

            if gradient is not None
        ]


        self.optimizer.apply_gradients(
            grads_and_vars
        )


        # ====================================================
        # METRICS
        # ====================================================

        self.loss_tracker.update_state(
            loss
        )

        self.accuracy_tracker.update_state(
            labels,
            logits_final,
            sample_weight=(
                sample_weight
            ),
        )


        feature_keep_ratio = (
            tf.reduce_mean(
                feature_mask
            )
        )


        challenged_ratio = (
            1.0
            -
            tf.reduce_mean(
                keep_entire_sample
            )
        )


        self.rsc_feature_keep_tracker.update_state(
            feature_keep_ratio
        )

        self.rsc_batch_challenge_tracker.update_state(
            challenged_ratio
        )


        return {
            metric.name:
                metric.result()

            for metric
            in self.metrics
        }


    # ========================================================
    # VALIDATION STEP
    # ========================================================

    def test_step(
        self,
        data,
    ):

        (
            images,
            labels,
            sample_weight,
        ) = (
            keras.utils
            .unpack_x_y_sample_weight(
                data
            )
        )


        labels = tf.cast(
            labels,
            tf.int32,
        )


        representation = (
            self.extract_representation(
                images,
                training=False,
            )
        )


        logits = self.classifier(
            representation
        )


        loss = self.loss_fn(
            labels,
            logits,
            sample_weight=(
                sample_weight
            ),
        )


        self.loss_tracker.update_state(
            loss
        )


        self.accuracy_tracker.update_state(
            labels,
            logits,
            sample_weight=(
                sample_weight
            ),
        )


        return {
            "loss":
                self.loss_tracker.result(),

            "accuracy":
                self.accuracy_tracker.result(),
        }


    # ========================================================
    # BACKBONE CONTROL
    # ========================================================

    def freeze_backbone(
        self,
    ):

        self.backbone.trainable = False


    def unfreeze_last_layers(
        self,
        last_n=30,
    ):

        self.backbone.trainable = True

        for layer in self.backbone.layers:

            layer.trainable = False


        for layer in self.backbone.layers[
            -last_n:
        ]:

            if isinstance(
                layer,
                layers.BatchNormalization,
            ):

                layer.trainable = False

            else:

                layer.trainable = True


    # ========================================================
    # CLEAN INFERENCE MODEL
    # ========================================================

    def build_inference_model(
        self,
    ):

        inputs = keras.Input(
            shape=(
                IMAGE_SIZE,
                IMAGE_SIZE,
                3,
            ),
            name="image",
        )


        x = self.augmentation(
            inputs,
            training=False,
        )


        x = self.preprocessing(
            x
        )


        x = self.backbone(
            x,
            training=False,
        )


        x = self.gap(
            x
        )


        x = self.dropout(
            x,
            training=False,
        )


        logits = self.classifier(
            x
        )


        outputs = (
            layers.Activation(
                "softmax",
                name="predictions",
            )(
                logits
            )
        )


        return keras.Model(
            inputs=inputs,
            outputs=outputs,
            name=(
                "MobileNetV2_ImageNet_RSC_V1"
            ),
        )


# ============================================================
# CALLBACKS
# ============================================================

def make_callbacks(
    checkpoint_path,
    csv_path,
):

    return [

        keras.callbacks.ModelCheckpoint(
            filepath=str(
                checkpoint_path
            ),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=(
                EARLY_STOPPING_PATIENCE
            ),
            restore_best_weights=False,
            verbose=1,
        ),

        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),

        keras.callbacks.CSVLogger(
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
            f"MobileNetV2_RSC_"
            f"SD1_seed{seed}"
        )


    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    keras.utils.set_random_seed(
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
    # OUTPUT
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
        /
        run_name
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
        "RSC MULTI-SEED CONFIRMATION"
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
        "RSC feature drop:",
        RSC_FEATURE_DROP
    )

    print(
        "RSC batch challenge:",
        RSC_BATCH_DROP
    )

    print()
    print(
        "Frozen SD1 membership."
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
        "\nFrozen split loaded."
    )

    print(
        "Training seed:",
        seed
    )

    print(
        "Test set NOT used."
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    model = (
        MobileNetV2RSCSeeded(
            seed=seed,
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
    )


    # Explicit build.
    dummy = tf.zeros(
        (
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
            keras.optimizers.Adam(
                learning_rate=(
                    STAGE1_LR
                )
            )
        )
    )


    stage1_weights = (
        run_dir
        /
        "stage1_best.weights.h5"
    )


    stage1_csv = (
        run_dir
        /
        "stage1_history.csv"
    )


    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=(
            STAGE1_MAX_EPOCHS
        ),
        callbacks=(
            make_callbacks(
                stage1_weights,
                stage1_csv,
            )
        ),
        verbose=1,
    )


    (
        stage1_best_idx,
        stage1_best_loss,
        stage1_best_acc,
    ) = (
        get_best_validation(
            history_stage1
        )
    )


    print()
    print(
        "Stage 1 best val loss:",
        f"{stage1_best_loss:.6f}"
    )

    print(
        "Stage 1 best val accuracy:",
        f"{stage1_best_acc:.6f}"
    )


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


    model.unfreeze_last_layers(
        last_n=(
            FINE_TUNE_LAST_N
        )
    )


    model.compile(
        optimizer=(
            keras.optimizers.Adam(
                learning_rate=(
                    STAGE2_LR
                )
            )
        )
    )


    stage2_weights = (
        run_dir
        /
        "stage2_best.weights.h5"
    )


    stage2_csv = (
        run_dir
        /
        "stage2_history.csv"
    )


    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=(
            STAGE2_MAX_EPOCHS
        ),
        callbacks=(
            make_callbacks(
                stage2_weights,
                stage2_csv,
            )
        ),
        verbose=1,
    )


    (
        stage2_best_idx,
        stage2_best_loss,
        stage2_best_acc,
    ) = (
        get_best_validation(
            history_stage2
        )
    )


    print()
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

    if (
        stage2_best_loss
        <
        stage1_best_loss
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
    # EXPORT CLEAN MODEL
    # ========================================================

    best_model = (
        model.build_inference_model()
    )


    best_model_path = (
        run_dir
        /
        "best_model.keras"
    )


    best_model.save(
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
            "E015",

        "model":
            "MobileNetV2_ImageNet_RSC_V1",

        "split":
            args.split,

        "split_membership":
            "frozen",

        "training_seed":
            seed,

        "batch_size":
            BATCH_SIZE,

        "stage1_learning_rate":
            STAGE1_LR,

        "stage2_learning_rate":
            STAGE2_LR,

        "fine_tune_last_n":
            FINE_TUNE_LAST_N,

        "batchnorm_frozen":
            True,

        "rsc_feature_drop_fraction":
            RSC_FEATURE_DROP,

        "rsc_batch_drop_fraction":
            RSC_BATCH_DROP,

        "selection_metric":
            "val_loss",

        "test_used_for_selection":
            False,
    }


    with open(
        run_dir
        /
        "config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )


    # ========================================================
    # HISTORY
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
        /
        "training_history.json",
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

    summary = {

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
        /
        "validation_metrics.json",
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
        "RSC MULTI-SEED TRAINING COMPLETE"
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
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()