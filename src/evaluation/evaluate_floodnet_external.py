"""
External evaluation on the frozen FloodNet UAV benchmark.

Benchmark
---------
EXT_UAV_FLOODNET_V1

True labels available:
    Flooded
    Neither

Model output remains the ORIGINAL three-class space:
    Desert
    Flooded
    Neither

IMPORTANT
---------
Desert is NOT removed.
Probabilities are NOT renormalized.
A Desert prediction is counted as an error.

Primary external metric:
    Supported-class Macro-F1
    = mean(F1_Flooded, F1_Neither)

The evaluator also reports:
    - full three-class cross-entropy loss
    - overall accuracy
    - supported-class macro precision
    - supported-class macro recall
    - per-class metrics
    - 2 x 3 confusion matrix
    - Desert prediction rate
    - AUROC / AUPRC for Flooded and Neither
    - complete per-image predictions

No training, tuning, or model selection occurs here.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# PROJECT PATHS
# ============================================================

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# CONSTANTS
# ============================================================

BENCHMARK_ID = "EXT_UAV_FLOODNET_V1"

CLASS_NAMES = [
    "Desert",
    "Flooded",
    "Neither",
]

CLASS_TO_INDEX = {
    "Desert": 0,
    "Flooded": 1,
    "Neither": 2,
}

SUPPORTED_CLASSES = [
    "Flooded",
    "Neither",
]

SUPPORTED_INDICES = [
    CLASS_TO_INDEX["Flooded"],
    CLASS_TO_INDEX["Neither"],
]

IMAGE_SIZE = 224


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen AIRE410 model "
            "on EXT_UAV_FLOODNET_V1."
        )
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help=(
            "Existing experiment directory under "
            "AIRE410_RESULTS_ROOT."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help=(
            "Skip SHA-256 verification. "
            "Not recommended for official evaluation."
        ),
    )

    return parser.parse_args()


# ============================================================
# HASH
# ============================================================

def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(
    path,
    label,
):

    image_bytes = tf.io.read_file(
        path
    )

    image = tf.io.decode_jpeg(
        image_bytes,
        channels=3,
    )

    image = tf.image.resize(
        image,
        size=(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        method="bilinear",
    )

    image = tf.cast(
        image,
        tf.float32,
    )

    # All our frozen models expect loader output
    # in [0,1] and perform MobileNet preprocessing
    # internally where applicable.
    image = image / 255.0

    return image, label


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    floodnet_root = Path(
        os.environ.get(
            "AIRE410_FLOODNET_ROOT",
            (
                "/content/drive/MyDrive/AIRE410/GoldData/"
                "FloodNet Challenge @ EARTHVISION 2021 - Track 1"
            ),
        )
    )

    results_root = Path(
        os.environ.get(
            "AIRE410_RESULTS_ROOT",
            "/content/drive/MyDrive/AIRE410/results",
        )
    )

    manifest_path = (
        REPO_ROOT
        / "data"
        / "external"
        / "EXT_UAV_FLOODNET_V1_manifest.csv"
    )

    run_dir = (
        results_root
        / args.run_name
    )

    model_path = (
        run_dir
        / "best_model.keras"
    )

    output_dir = (
        REPO_ROOT
        / "results"
        / "external_validation"
        / BENCHMARK_ID
        / args.run_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 78)

    print(
        "FROZEN EXTERNAL UAV EVALUATION"
    )

    print("=" * 78)

    print(
        "Benchmark:",
        BENCHMARK_ID
    )

    print(
        "Experiment:",
        args.run_name
    )

    print(
        "FloodNet root:",
        floodnet_root
    )

    print(
        "Manifest:",
        manifest_path
    )

    print(
        "Model:",
        model_path
    )

    print("=" * 78)


    # ========================================================
    # BASIC EXISTENCE CHECKS
    # ========================================================

    if not floodnet_root.exists():

        raise FileNotFoundError(
            f"FloodNet root not found:\n"
            f"{floodnet_root}"
        )

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"Frozen manifest not found:\n"
            f"{manifest_path}"
        )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{model_path}"
        )


    # ========================================================
    # LOAD FROZEN MANIFEST
    # ========================================================

    manifest = pd.read_csv(
        manifest_path
    )

    print(
        "\nManifest rows:",
        len(manifest)
    )


    # ========================================================
    # MANIFEST VALIDATION
    # ========================================================

    required_columns = {
        "benchmark_id",
        "source_dataset",
        "official_split",
        "acquisition_domain",
        "original_label",
        "aire410_label",
        "filename",
        "relative_path",
        "sha256",
    }

    missing_columns = (
        required_columns
        -
        set(manifest.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "Manifest missing columns: "
            f"{sorted(missing_columns)}"
        )


    if len(manifest) != 398:

        raise RuntimeError(
            "Expected exactly 398 frozen images, "
            f"found {len(manifest)}."
        )


    if not (
        manifest[
            "benchmark_id"
        ]
        ==
        BENCHMARK_ID
    ).all():

        raise RuntimeError(
            "Unexpected benchmark_id "
            "inside manifest."
        )


    observed_labels = set(
        manifest[
            "aire410_label"
        ].unique()
    )

    expected_labels = {
        "Flooded",
        "Neither",
    }

    if observed_labels != expected_labels:

        raise RuntimeError(
            "Unexpected mapped labels: "
            f"{observed_labels}"
        )


    if (
        manifest[
            "sha256"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate SHA-256 values "
            "found in frozen manifest."
        )


    print(
        "\nFrozen class counts:"
    )

    print(
        manifest[
            "aire410_label"
        ]
        .value_counts()
        .to_string()
    )


    # ========================================================
    # RESOLVE IMAGE PATHS
    # ========================================================

    image_paths = [
        floodnet_root
        / relative_path

        for relative_path
        in manifest[
            "relative_path"
        ]
    ]


    missing_files = [
        str(path)

        for path
        in image_paths

        if not path.exists()
    ]

    if missing_files:

        raise FileNotFoundError(
            "Frozen benchmark contains "
            f"{len(missing_files)} missing files.\n"
            f"First missing file:\n"
            f"{missing_files[0]}"
        )


    # ========================================================
    # VERIFY HASHES BEFORE MODEL ACCESS
    # ========================================================

    if not args.skip_hash_verification:

        print(
            "\nVerifying frozen SHA-256 hashes..."
        )

        mismatches = []

        for i, path in enumerate(
            image_paths
        ):

            actual_hash = sha256_file(
                path
            )

            expected_hash = str(
                manifest.iloc[i][
                    "sha256"
                ]
            )

            if actual_hash != expected_hash:

                mismatches.append({
                    "filename":
                        path.name,

                    "expected":
                        expected_hash,

                    "actual":
                        actual_hash,
                })


        if mismatches:

            mismatch_df = pd.DataFrame(
                mismatches
            )

            mismatch_path = (
                output_dir
                / "hash_mismatches.csv"
            )

            mismatch_df.to_csv(
                mismatch_path,
                index=False,
            )

            raise RuntimeError(
                "Frozen benchmark integrity "
                f"FAILED: {len(mismatches)} "
                "SHA-256 mismatches."
            )


        print(
            "SHA-256 verification: PASS"
        )

    else:

        print(
            "\nWARNING: hash verification skipped."
        )


    # ========================================================
    # TRUE LABELS
    # ========================================================

    y_true = np.array(
        [
            CLASS_TO_INDEX[
                label
            ]

            for label
            in manifest[
                "aire410_label"
            ]
        ],
        dtype=np.int32,
    )


    # ========================================================
    # BUILD DATASET
    # ========================================================

    path_strings = [
        str(path)
        for path
        in image_paths
    ]

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                path_strings,
                y_true,
            )
        )
    )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=(
            tf.data.AUTOTUNE
        ),
        deterministic=True,
    )

    dataset = dataset.batch(
        args.batch_size,
        drop_remainder=False,
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )


    # ========================================================
    # LOAD FROZEN MODEL
    # ========================================================

    print(
        "\nLoading frozen model..."
    )

    model = (
        tf.keras.models.load_model(
            model_path
        )
    )

    print(
        "Loaded model:",
        model.name
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    print(
        "\nRunning external predictions..."
    )

    probability_batches = []

    observed_labels_batches = []

    for images, labels in dataset:

        probabilities = model(
            images,
            training=False,
        ).numpy()

        probability_batches.append(
            probabilities
        )

        observed_labels_batches.append(
            labels.numpy()
        )


    y_prob = np.concatenate(
        probability_batches,
        axis=0,
    )

    y_true_check = np.concatenate(
        observed_labels_batches,
        axis=0,
    )


    if not np.array_equal(
        y_true,
        y_true_check,
    ):

        raise RuntimeError(
            "Dataset order changed unexpectedly."
        )


    if y_prob.shape != (
        len(manifest),
        3,
    ):

        raise RuntimeError(
            "Expected prediction shape "
            f"(398, 3), got {y_prob.shape}."
        )


    probability_sums = (
        y_prob.sum(
            axis=1
        )
    )


    if not np.allclose(
        probability_sums,
        1.0,
        atol=1e-4,
    ):

        raise RuntimeError(
            "Model outputs do not sum "
            "to approximately 1."
        )


    y_pred = np.argmax(
        y_prob,
        axis=1,
    )


    # ========================================================
    # FULL THREE-CLASS LOSS
    # ========================================================

    loss_fn = (
        tf.keras.losses
        .SparseCategoricalCrossentropy()
    )

    external_loss = float(
        loss_fn(
            y_true,
            y_prob,
        ).numpy()
    )


    # ========================================================
    # OVERALL ACCURACY
    #
    # Desert predictions remain errors.
    # ========================================================

    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )


    # ========================================================
    # SUPPORTED-CLASS MACRO METRICS
    #
    # Only true classes actually supported by FloodNet:
    # Flooded + Neither.
    #
    # Predictions still come from all 3 classes.
    # ========================================================

    macro_precision = float(
        precision_score(
            y_true,
            y_pred,
            labels=SUPPORTED_INDICES,
            average="macro",
            zero_division=0,
        )
    )

    macro_recall = float(
        recall_score(
            y_true,
            y_pred,
            labels=SUPPORTED_INDICES,
            average="macro",
            zero_division=0,
        )
    )

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=SUPPORTED_INDICES,
            average="macro",
            zero_division=0,
        )
    )


    # ========================================================
    # PER-CLASS METRICS
    # ========================================================

    class_precision = precision_score(
        y_true,
        y_pred,
        labels=SUPPORTED_INDICES,
        average=None,
        zero_division=0,
    )

    class_recall = recall_score(
        y_true,
        y_pred,
        labels=SUPPORTED_INDICES,
        average=None,
        zero_division=0,
    )

    class_f1 = f1_score(
        y_true,
        y_pred,
        labels=SUPPORTED_INDICES,
        average=None,
        zero_division=0,
    )


    class_rows = []

    for j, class_name in enumerate(
        SUPPORTED_CLASSES
    ):

        class_index = CLASS_TO_INDEX[
            class_name
        ]

        binary_truth = (
            y_true
            ==
            class_index
        ).astype(
            np.int32
        )


        auroc = float(
            roc_auc_score(
                binary_truth,
                y_prob[
                    :,
                    class_index
                ],
            )
        )


        auprc = float(
            average_precision_score(
                binary_truth,
                y_prob[
                    :,
                    class_index
                ],
            )
        )


        support = int(
            binary_truth.sum()
        )


        class_rows.append({
            "class":
                class_name,

            "precision":
                float(
                    class_precision[j]
                ),

            "recall":
                float(
                    class_recall[j]
                ),

            "f1":
                float(
                    class_f1[j]
                ),

            "support":
                support,

            "auroc":
                auroc,

            "auprc":
                auprc,
        })


    per_class_df = pd.DataFrame(
        class_rows
    )


    macro_auroc = float(
        per_class_df[
            "auroc"
        ].mean()
    )

    macro_auprc = float(
        per_class_df[
            "auprc"
        ].mean()
    )


    # ========================================================
    # 2 x 3 CONFUSION MATRIX
    # ========================================================

    cm = np.zeros(
        shape=(2, 3),
        dtype=np.int64,
    )


    true_row_map = {
        CLASS_TO_INDEX[
            "Flooded"
        ]: 0,

        CLASS_TO_INDEX[
            "Neither"
        ]: 1,
    }


    for true_label, pred_label in zip(
        y_true,
        y_pred,
    ):

        row = true_row_map[
            int(true_label)
        ]

        cm[
            row,
            int(pred_label)
        ] += 1


    cm_df = pd.DataFrame(
        cm,
        index=[
            "Flooded",
            "Neither",
        ],
        columns=CLASS_NAMES,
    )

    cm_df.index.name = "True"
    cm_df.columns.name = "Predicted"


    # ========================================================
    # DESERT PREDICTION RATE
    # ========================================================

    desert_index = CLASS_TO_INDEX[
        "Desert"
    ]

    desert_predictions = int(
        np.sum(
            y_pred
            ==
            desert_index
        )
    )

    desert_prediction_rate = float(
        desert_predictions
        /
        len(y_pred)
    )


    # ========================================================
    # PER-IMAGE PREDICTION TABLE
    # ========================================================

    predictions_df = manifest.copy()

    predictions_df[
        "true_class_index"
    ] = y_true

    predictions_df[
        "pred_class_index"
    ] = y_pred

    predictions_df[
        "predicted_label"
    ] = [
        CLASS_NAMES[i]
        for i in y_pred
    ]

    predictions_df[
        "prob_desert"
    ] = y_prob[:, 0]

    predictions_df[
        "prob_flooded"
    ] = y_prob[:, 1]

    predictions_df[
        "prob_neither"
    ] = y_prob[:, 2]

    predictions_df[
        "correct"
    ] = (
        y_true
        ==
        y_pred
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    metrics = {
        "benchmark_id":
            BENCHMARK_ID,

        "experiment":
            args.run_name,

        "model_name":
            model.name,

        "n_samples":
            int(
                len(y_true)
            ),

        "true_classes":
            [
                "Flooded",
                "Neither",
            ],

        "model_output_classes":
            CLASS_NAMES,

        "probability_renormalization":
            False,

        "desert_predictions_count":
            desert_predictions,

        "desert_prediction_rate":
            desert_prediction_rate,

        "cross_entropy_loss":
            external_loss,

        "accuracy":
            accuracy,

        "supported_macro_precision":
            macro_precision,

        "supported_macro_recall":
            macro_recall,

        "supported_macro_f1":
            macro_f1,

        "supported_macro_auroc":
            macro_auroc,

        "supported_macro_auprc":
            macro_auprc,

        "hash_verification":
            (
                "skipped"
                if args.skip_hash_verification
                else "passed"
            ),
    }


    metrics_path = (
        output_dir
        / "external_metrics.json"
    )

    per_class_path = (
        output_dir
        / "per_class_metrics.csv"
    )

    confusion_path = (
        output_dir
        / "confusion_matrix_2x3.csv"
    )

    predictions_path = (
        output_dir
        / "predictions.csv"
    )


    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )


    per_class_df.to_csv(
        per_class_path,
        index=False,
    )

    cm_df.to_csv(
        confusion_path
    )

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )


    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 78)

    print(
        f"{BENCHMARK_ID} — OFFICIAL EXTERNAL TEST"
    )

    print("=" * 78)

    print(
        "Experiment:",
        args.run_name
    )

    print(
        "Model:",
        model.name
    )

    print(
        "Samples:",
        len(y_true)
    )

    print()

    print(
        f"Loss:                       "
        f"{external_loss:.6f}"
    )

    print(
        f"Accuracy:                   "
        f"{accuracy:.6f}"
    )

    print(
        f"Supported Macro Precision:  "
        f"{macro_precision:.6f}"
    )

    print(
        f"Supported Macro Recall:     "
        f"{macro_recall:.6f}"
    )

    print(
        f"Supported Macro F1:         "
        f"{macro_f1:.6f}"
    )

    print(
        f"Supported Macro AUROC:      "
        f"{macro_auroc:.6f}"
    )

    print(
        f"Supported Macro AUPRC:      "
        f"{macro_auprc:.6f}"
    )

    print(
        f"Desert predictions:         "
        f"{desert_predictions}"
    )

    print(
        f"Desert prediction rate:     "
        f"{desert_prediction_rate:.6f}"
    )


    print(
        "\nPER-CLASS METRICS"
    )

    print(
        per_class_df.to_string(
            index=False
        )
    )


    print(
        "\n2 x 3 CONFUSION MATRIX"
    )

    print(
        cm_df
    )


    print(
        "\nSaved:"
    )

    print(
        metrics_path
    )

    print(
        per_class_path
    )

    print(
        confusion_path
    )

    print(
        predictions_path
    )

    print()
    print(
        "No external-data tuning was performed."
    )

    print("=" * 78)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()