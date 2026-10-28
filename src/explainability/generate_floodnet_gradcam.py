"""
Frozen FloodNet Grad-CAM analysis.

Uses the predeclared explainability cohort:

    3 Flooded consensus-correct
    3 Neither consensus-correct
    3 Neither -> Flooded consensus failures

Models compared at common seed 42:

    E007 = ERM
    E012 = DG-AUG-V2
    E015 = RSC

No image selection occurs in this script.
The cohort must already exist and remain unchanged.
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from PIL import Image

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

THIS_FILE = Path(__file__).resolve()

SRC_DIR = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[2]


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_SIZE = 224

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


# ============================================================
# FROZEN METHOD SUITE
# ============================================================

METHODS = [
    {
        "method": "ERM",
        "run_name": "E007_MobileNetV2_SD1",
    },
    {
        "method": "DG-AUG-V2",
        "run_name": "E012_MobileNetV2_DGAugV2_SD1",
    },
    {
        "method": "RSC",
        "run_name": "E015_MobileNetV2_RSC_SD1",
    },
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate frozen FloodNet Grad-CAM "
            "comparisons for ERM, DG-V2, and RSC."
        )
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap overlay opacity.",
    )

    return parser.parse_args()


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image_for_model(
    image_path,
):

    """
    Returns:
        display_image : uint8 RGB original-sized image
        model_input   : float32 [1,224,224,3] in [0,1]
    """

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

        display_image = np.asarray(
            image
        )

        resized = image.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            Image.Resampling.BILINEAR,
        )

        model_input = np.asarray(
            resized,
            dtype=np.float32,
        )

    model_input = (
        model_input
        / 255.0
    )

    model_input = np.expand_dims(
        model_input,
        axis=0,
    )

    return (
        display_image,
        model_input,
    )


# ============================================================
# FIND MOBILENET BACKBONE
# ============================================================

def find_backbone(
    model,
):

    """
    Locate the nested MobileNetV2 backbone.

    All frozen seed-42 models use a nested MobileNetV2
    whose output is the final 7x7 convolutional feature map.
    """

    candidates = []

    for layer in model.layers:

        name = layer.name.lower()

        if "mobilenetv2" in name:

            candidates.append(
                layer
            )

    if len(candidates) != 1:

        names = [
            layer.name
            for layer in model.layers
        ]

        raise RuntimeError(
            "Could not uniquely identify "
            "MobileNetV2 backbone.\n"
            f"Candidates: "
            f"{[x.name for x in candidates]}\n"
            f"Model layers: {names}"
        )

    backbone = candidates[0]

    return backbone


# ============================================================
# GRAD-CAM
# ============================================================

def make_gradcam(
    model,
    image_batch,
    target_class_index,
):

    """
    Grad-CAM using the final MobileNetV2 feature tensor.

    Target score:
        Frozen model probability for the specified class.

    Returns:
        heatmap
        probabilities
    """

    backbone = find_backbone(
        model
    )

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            backbone.output,
            model.output,
        ],
    )

    image_tensor = tf.convert_to_tensor(
        image_batch,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:

        (
            feature_maps,
            predictions,
        ) = grad_model(
            image_tensor,
            training=False,
        )

        target_score = (
            predictions[
                :,
                target_class_index
            ]
        )

    gradients = tape.gradient(
        target_score,
        feature_maps,
    )

    if gradients is None:

        raise RuntimeError(
            "Gradients are None."
        )

    # Average gradient over batch + spatial dimensions.
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(
            0,
            1,
            2,
        ),
    )

    feature_maps = (
        feature_maps[0]
    )

    heatmap = tf.reduce_sum(
        feature_maps
        *
        channel_weights,
        axis=-1,
    )

    # Positive evidence only.
    heatmap = tf.nn.relu(
        heatmap
    )

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value.numpy()) > 0:

        heatmap = (
            heatmap
            /
            max_value
        )

    heatmap = heatmap.numpy()

    return (
        heatmap,
        predictions[0].numpy(),
    )


# ============================================================
# HEATMAP OVERLAY
# ============================================================

def overlay_gradcam(
    display_image,
    heatmap,
    alpha,
):

    """
    Resize heatmap to original image dimensions and overlay it.
    """

    height = display_image.shape[0]
    width = display_image.shape[1]

    heatmap_uint8 = np.uint8(
        np.clip(
            heatmap,
            0.0,
            1.0,
        )
        * 255
    )

    heatmap_image = Image.fromarray(
        heatmap_uint8
    )

    heatmap_image = heatmap_image.resize(
        (
            width,
            height,
        ),
        Image.Resampling.BILINEAR,
    )

    heatmap_resized = (
        np.asarray(
            heatmap_image,
            dtype=np.float32,
        )
        / 255.0
    )

    cmap = plt.get_cmap(
        "jet"
    )

    heatmap_rgb = (
        cmap(
            heatmap_resized
        )[
            ...,
            :3
        ]
        * 255.0
    )

    original = display_image.astype(
        np.float32
    )

    overlay = (
        (1.0 - alpha)
        * original
        +
        alpha
        * heatmap_rgb
    )

    overlay = np.uint8(
        np.clip(
            overlay,
            0,
            255,
        )
    )

    return overlay


# ============================================================
# TARGET CLASS
# ============================================================

def get_target_class(
    cohort_group,
):

    """
    Predeclared explanation target.

    Correct Flooded:
        explain Flooded score.

    Correct Neither:
        explain Neither score.

    Neither -> Flooded failure:
        explain the incorrect Flooded score.
    """

    if (
        cohort_group
        ==
        "flooded_consensus_correct"
    ):

        return "Flooded"

    if (
        cohort_group
        ==
        "neither_consensus_correct"
    ):

        return "Neither"

    if (
        cohort_group
        ==
        "neither_to_flooded_consensus_failure"
    ):

        return "Flooded"

    raise ValueError(
        f"Unknown cohort group: "
        f"{cohort_group}"
    )


# ============================================================
# SAFE NAME
# ============================================================

def safe_name(
    text,
):

    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(">", "to")
    )


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


    cohort_path = (
        REPO_ROOT
        / "results"
        / "explainability"
        / "FloodNet_explainability_cohort.csv"
    )


    output_root = (
        REPO_ROOT
        / "results"
        / "explainability"
        / "FloodNet_gradcam_seed42"
    )

    individual_dir = (
        output_root
        / "individual"
    )

    comparison_dir = (
        output_root
        / "comparisons"
    )

    raw_dir = (
        output_root
        / "raw_heatmaps"
    )


    individual_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # VERIFY COHORT
    # ========================================================

    if not cohort_path.exists():

        raise FileNotFoundError(
            f"Frozen cohort missing:\n"
            f"{cohort_path}"
        )


    cohort = pd.read_csv(
        cohort_path
    )


    if len(cohort) != 9:

        raise RuntimeError(
            "Frozen explainability cohort "
            f"must contain 9 images. "
            f"Found {len(cohort)}."
        )


    print()
    print("=" * 80)

    print(
        "FLOODNET FROZEN GRAD-CAM ANALYSIS"
    )

    print("=" * 80)

    print(
        "Cohort:",
        cohort_path
    )

    print(
        "Images:",
        len(cohort)
    )

    print(
        "Methods:",
        [
            x["method"]
            for x in METHODS
        ]
    )

    print(
        "Total heatmaps:",
        len(cohort)
        * len(METHODS)
    )

    print("=" * 80)


    # ========================================================
    # LOAD ALL THREE MODELS ONCE
    # ========================================================

    models = {}

    print(
        "\nLoading frozen seed-42 models..."
    )


    for method_info in METHODS:

        method = (
            method_info[
                "method"
            ]
        )

        run_name = (
            method_info[
                "run_name"
            ]
        )

        model_path = (
            results_root
            / run_name
            / "best_model.keras"
        )


        if not model_path.exists():

            raise FileNotFoundError(
                model_path
            )


        model = (
            tf.keras.models.load_model(
                model_path
            )
        )


        backbone = find_backbone(
            model
        )


        print(
            f"{method:10s} | "
            f"{run_name} | "
            f"{model.name} | "
            f"Grad-CAM tensor: "
            f"{backbone.output.shape}"
        )


        models[
            method
        ] = model


    # ========================================================
    # GENERATE GRAD-CAMS
    # ========================================================

    summary_rows = []


    for cohort_index, row in (
        cohort.iterrows()
    ):

        group = row[
            "cohort_group"
        ]

        tier = row[
            "confidence_tier"
        ]

        filename = row[
            "filename"
        ]

        relative_path = row[
            "relative_path"
        ]

        true_label = row[
            "aire410_label"
        ]


        target_class = (
            get_target_class(
                group
            )
        )

        target_index = (
            CLASS_TO_INDEX[
                target_class
            ]
        )


        image_path = (
            floodnet_root
            / relative_path
        )


        if not image_path.exists():

            raise FileNotFoundError(
                image_path
            )


        (
            display_image,
            model_input,
        ) = load_image_for_model(
            image_path
        )


        # ----------------------------------------------------
        # Comparison figure:
        #
        # Original | ERM | DG-V2 | RSC
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(
                18,
                5,
            ),
        )


        axes[0].imshow(
            display_image
        )

        axes[0].set_title(
            (
                f"Original\n"
                f"True: {true_label}\n"
                f"{group} / {tier}"
            )
        )

        axes[0].axis(
            "off"
        )


        for method_index, method_info in enumerate(
            METHODS,
            start=1,
        ):

            method = (
                method_info[
                    "method"
                ]
            )

            run_name = (
                method_info[
                    "run_name"
                ]
            )

            model = models[
                method
            ]


            (
                heatmap,
                probabilities,
            ) = make_gradcam(
                model=model,
                image_batch=model_input,
                target_class_index=(
                    target_index
                ),
            )


            pred_index = int(
                np.argmax(
                    probabilities
                )
            )

            predicted_label = (
                CLASS_NAMES[
                    pred_index
                ]
            )

            target_probability = float(
                probabilities[
                    target_index
                ]
            )


            # ------------------------------------------------
            # Sanity check against frozen consensus cohort
            # ------------------------------------------------

            if (
                group
                ==
                "flooded_consensus_correct"
                and
                predicted_label
                != "Flooded"
            ):

                raise RuntimeError(
                    f"{method} no longer predicts "
                    f"Flooded for {filename}."
                )


            if (
                group
                ==
                "neither_consensus_correct"
                and
                predicted_label
                != "Neither"
            ):

                raise RuntimeError(
                    f"{method} no longer predicts "
                    f"Neither for {filename}."
                )


            if (
                group
                ==
                "neither_to_flooded_consensus_failure"
                and
                predicted_label
                != "Flooded"
            ):

                raise RuntimeError(
                    f"{method} no longer predicts "
                    f"Flooded for {filename}."
                )


            overlay = (
                overlay_gradcam(
                    display_image=(
                        display_image
                    ),
                    heatmap=heatmap,
                    alpha=args.alpha,
                )
            )


            axes[
                method_index
            ].imshow(
                overlay
            )


            axes[
                method_index
            ].set_title(
                (
                    f"{method}\n"
                    f"Pred: "
                    f"{predicted_label}\n"
                    f"P({target_class})="
                    f"{target_probability:.3f}"
                )
            )


            axes[
                method_index
            ].axis(
                "off"
            )


            # ------------------------------------------------
            # SAVE INDIVIDUAL OVERLAY
            # ------------------------------------------------

            base_name = (
                f"{cohort_index + 1:02d}_"
                f"{safe_name(group)}_"
                f"{tier}_"
                f"{Path(filename).stem}_"
                f"{safe_name(method)}"
            )


            individual_path = (
                individual_dir
                / f"{base_name}.png"
            )


            Image.fromarray(
                overlay
            ).save(
                individual_path
            )


            # ------------------------------------------------
            # SAVE RAW HEATMAP
            # ------------------------------------------------

            raw_path = (
                raw_dir
                / f"{base_name}.npy"
            )


            np.save(
                raw_path,
                heatmap,
            )


            # ------------------------------------------------
            # SUMMARY ROW
            # ------------------------------------------------

            summary_rows.append({
                "cohort_index":
                    cohort_index + 1,

                "cohort_group":
                    group,

                "confidence_tier":
                    tier,

                "filename":
                    filename,

                "true_label":
                    true_label,

                "target_class":
                    target_class,

                "method":
                    method,

                "run_name":
                    run_name,

                "predicted_label":
                    predicted_label,

                "prob_desert":
                    float(
                        probabilities[0]
                    ),

                "prob_flooded":
                    float(
                        probabilities[1]
                    ),

                "prob_neither":
                    float(
                        probabilities[2]
                    ),

                "target_probability":
                    target_probability,

                "heatmap_max":
                    float(
                        np.max(
                            heatmap
                        )
                    ),

                "heatmap_mean":
                    float(
                        np.mean(
                            heatmap
                        )
                    ),

                "individual_overlay":
                    str(
                        individual_path
                        .relative_to(
                            REPO_ROOT
                        )
                    ),

                "raw_heatmap":
                    str(
                        raw_path
                        .relative_to(
                            REPO_ROOT
                        )
                    ),
            })


        # ----------------------------------------------------
        # SAVE 4-COLUMN COMPARISON
        # ----------------------------------------------------

        plt.tight_layout()


        comparison_name = (
            f"{cohort_index + 1:02d}_"
            f"{safe_name(group)}_"
            f"{tier}_"
            f"{Path(filename).stem}"
            f"_comparison.png"
        )


        comparison_path = (
            comparison_dir
            / comparison_name
        )


        fig.savefig(
            comparison_path,
            dpi=180,
            bbox_inches="tight",
        )


        plt.close(
            fig
        )


        print(
            f"[{cohort_index + 1}/9] "
            f"{filename} | "
            f"{group} | "
            f"{tier}"
        )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_path = (
        output_root
        / "FloodNet_gradcam_summary.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False,
    )


    # ========================================================
    # RUN METADATA
    # ========================================================

    metadata = {
        "benchmark":
            "EXT_UAV_FLOODNET_V1",

        "cohort":
            str(
                cohort_path.relative_to(
                    REPO_ROOT
                )
            ),

        "n_images":
            9,

        "n_models":
            3,

        "n_gradcams":
            27,

        "models": METHODS,

        "gradcam_feature_tensor":
            "final MobileNetV2 backbone output",

        "gradcam_target":
            (
                "predeclared relevant class "
                "probability"
            ),

        "overlay_alpha":
            float(
                args.alpha
            ),

        "sample_selection":
            (
                "frozen before Grad-CAM generation"
            ),
    }


    metadata_path = (
        output_root
        / "gradcam_run_metadata.json"
    )


    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 80)

    print(
        "GRAD-CAM GENERATION COMPLETE"
    )

    print("=" * 80)

    print(
        "Images:",
        9
    )

    print(
        "Methods:",
        3
    )

    print(
        "Heatmaps:",
        27
    )

    print(
        "Comparison figures:",
        len(
            list(
                comparison_dir.glob(
                    "*.png"
                )
            )
        )
    )

    print(
        "\nSaved:"
    )

    print(
        comparison_dir
    )

    print(
        individual_dir
    )

    print(
        summary_path
    )

    print(
        metadata_path
    )

    print()
    print(
        "🔥 FROZEN FLOODNET GRAD-CAM RUN COMPLETE"
    )

    print("=" * 80)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()