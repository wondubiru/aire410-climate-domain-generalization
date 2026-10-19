import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42


# ============================================================
# DG-AUG-V1
# ============================================================

def build_dg_augmentation():
    """
    Source-agnostic augmentation for domain generalization.

    Baseline augmentation is preserved exactly:
        - horizontal + vertical flip
        - rotation = 0.08
        - zoom = 0.10
        - contrast = 0.10

    DG-AUG-V1 adds:
        - translation
        - brightness variation
        - saturation variation
        - Gaussian noise

    Goal:
        Reduce dependence on source-specific appearance,
        illumination, color, and acquisition artifacts.
    """

    return keras.Sequential(
        [
            # ------------------------------------------------
            # BASELINE AUGMENTATION — unchanged
            # ------------------------------------------------

            layers.RandomFlip(
                "horizontal_and_vertical",
                seed=SEED,
            ),

            layers.RandomRotation(
                0.08,
                fill_mode="reflect",
                seed=SEED,
            ),

            layers.RandomZoom(
                0.10,
                fill_mode="reflect",
                seed=SEED,
            ),

            layers.RandomContrast(
                0.10,
                seed=SEED,
            ),

            # ------------------------------------------------
            # DG-AUG-V1 additions
            # ------------------------------------------------

            layers.RandomTranslation(
                height_factor=0.10,
                width_factor=0.10,
                fill_mode="reflect",
                seed=SEED + 1,
            ),

            layers.RandomBrightness(
                factor=0.15,
                value_range=(0.0, 1.0),
                seed=SEED + 2,
            ),

            layers.RandomSaturation(
                factor=0.20,
                value_range=(0.0, 1.0),
                seed=SEED + 3,
            ),

            layers.GaussianNoise(
                stddev=0.03,
                seed=SEED + 4,
            ),

            # Gaussian noise can push values slightly
            # outside [0,1]. Clip them safely.
            layers.ReLU(
                max_value=1.0,
                name="clip_to_0_1",
            ),
        ],
        name="dg_aug_v1",
    )


# ============================================================
# MOBILENETV2 + DG-AUG-V1
# ============================================================

def build_mobilenet_v2_dg(
    input_shape=(224, 224, 3),
    num_classes=3,
    dropout_rate=0.30,
    backbone_trainable=False,
):
    """
    MobileNetV2 ImageNet transfer-learning model
    with source-agnostic DG-AUG-V1 augmentation.

    Loader input range:
        [0, 1]

    MobileNetV2 input range:
        [-1, 1]

    Only experimental change relative to the baseline
    MobileNetV2 model is the augmentation pipeline.
    """

    # --------------------------------------------------------
    # DG augmentation
    # --------------------------------------------------------

    augmentation = build_dg_augmentation()


    # --------------------------------------------------------
    # ImageNet pretrained backbone
    # --------------------------------------------------------

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = backbone_trainable


    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    inputs = keras.Input(
        shape=input_shape,
        name="image",
    )

    x = augmentation(
        inputs
    )


    # --------------------------------------------------------
    # Existing loader: [0,1]
    # MobileNetV2:      [-1,1]
    # --------------------------------------------------------

    x = layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenet_preprocessing",
    )(x)


    # --------------------------------------------------------
    # Pretrained feature extractor
    # --------------------------------------------------------

    x = backbone(
        x,
        training=False,
    )


    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D(
        name="global_average_pool"
    )(x)

    x = layers.Dropout(
        dropout_rate,
        seed=SEED,
        name="classifier_dropout",
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)


    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV2_ImageNet_DG_AUG_V1",
    )

    return model


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    model = build_mobilenet_v2_dg()

    model.summary()

    dummy = tf.random.uniform(
        shape=(2, 224, 224, 3),
        minval=0.0,
        maxval=1.0,
        seed=SEED,
    )

    # Inference path
    output_eval = model(
        dummy,
        training=False,
    )

    # Training path activates DG augmentation
    output_train = model(
        dummy,
        training=True,
    )

    print(
        "\nInference output shape:",
        output_eval.shape
    )

    print(
        "Training output shape:",
        output_train.shape
    )

    print(
        "Probability sums:",
        tf.reduce_sum(
            output_eval,
            axis=1
        ).numpy()
    )

    print(
        "Backbone trainable:",
        model.get_layer(
            "mobilenetv2_1.00_224"
        ).trainable
    )

    print(
        "DG augmentation:",
        model.get_layer(
            "dg_aug_v1"
        ).name
    )