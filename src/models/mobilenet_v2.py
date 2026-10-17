import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# MODEL CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 3
SEED = 42


# ============================================================
# MOBILENETV2 TRANSFER-LEARNING MODEL
# ============================================================

def build_mobilenet_v2(
    input_shape=(224, 224, 3),
    num_classes=3,
    dropout_rate=0.30,
    backbone_trainable=False,
):
    """
    Build MobileNetV2 transfer-learning classifier.

    Input:
        Images already normalized to [0, 1]
        by the existing AIRE410 tf.data loader.

    Internal preprocessing:
        [0, 1] -> [-1, 1]

    Initial E006 protocol:
        ImageNet pretrained MobileNetV2 backbone
        with backbone frozen.

    Output:
        3-class Softmax probabilities:
        Desert, Flooded, Neither.
    """

    # --------------------------------------------------------
    # Data augmentation
    #
    # Same augmentation family used by CustomCNN_V1.
    # --------------------------------------------------------

    augmentation = keras.Sequential(
        [
            layers.RandomFlip(
                "horizontal_and_vertical",
                seed=SEED,
            ),

            layers.RandomRotation(
                0.08,
                seed=SEED,
            ),

            layers.RandomZoom(
                0.10,
                seed=SEED,
            ),

            layers.RandomContrast(
                0.10,
                seed=SEED,
            ),
        ],
        name="augmentation",
    )


    # --------------------------------------------------------
    # ImageNet-pretrained backbone
    # --------------------------------------------------------

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = backbone_trainable


    # --------------------------------------------------------
    # Model input
    # --------------------------------------------------------

    inputs = keras.Input(
        shape=input_shape,
        name="image",
    )

    x = augmentation(
        inputs
    )


    # --------------------------------------------------------
    # Existing loader gives [0, 1].
    #
    # MobileNetV2 expects [-1, 1].
    #
    # Therefore:
    #
    #     x_new = 2*x - 1
    # --------------------------------------------------------

    x = layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenet_preprocessing",
    )(x)


    # --------------------------------------------------------
    # Pretrained feature extractor
    #
    # Keep BatchNorm layers in inference mode while
    # the backbone is frozen.
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
        name="MobileNetV2_ImageNet_V1",
    )

    return model


# ============================================================
# QUICK LOCAL TEST
# ============================================================

if __name__ == "__main__":

    model = build_mobilenet_v2()

    model.summary()

    dummy = tf.zeros(
        (2, 224, 224, 3)
    )

    output = model(
        dummy,
        training=False
    )

    print(
        "\nOutput shape:",
        output.shape
    )

    print(
        "Backbone trainable:",
        model.get_layer(
            "mobilenetv2_1.00_224"
        ).trainable
    )