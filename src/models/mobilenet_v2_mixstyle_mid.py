import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    from .mobilenet_v2_mixstyle import MixStyle
except ImportError:
    from mobilenet_v2_mixstyle import MixStyle


SEED = 42

# ------------------------------------------------------------
# E014 intervention point
# ------------------------------------------------------------

MIXSTYLE_TARGET_LAYER = "block_5_add"


def build_mobilenet_v2_mixstyle_mid(
    input_shape=(224, 224, 3),
    num_classes=3,
    dropout_rate=0.30,
    backbone_trainable=False,
    mixstyle_p=0.5,
    mixstyle_alpha=0.1,
):
    """
    E014:
        MobileNetV2 + mid-feature MixStyle.

    Controlled against E007 and E013.

    E007:
        no MixStyle

    E013:
        MixStyle after final backbone output
        ~7 x 7 x 1280

    E014:
        MixStyle after block_5_add
        ~28 x 28 x 32

    Everything else is unchanged.
    """

    # ========================================================
    # SAME E007 BASELINE AUGMENTATION
    # ========================================================

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


    # ========================================================
    # ORIGINAL IMAGENET BACKBONE
    # ========================================================

    original_backbone = (
        keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
        )
    )


    # Confirm target exists before modifying anything.
    target_layer = original_backbone.get_layer(
        MIXSTYLE_TARGET_LAYER
    )

    print(
        "MixStyle insertion target:",
        target_layer.name
    )

    print(
        "Target output shape:",
        target_layer.output.shape
    )


    # ========================================================
    # ONE MIXSTYLE INSTANCE
    # ========================================================

    mixstyle = MixStyle(
        p=mixstyle_p,
        alpha=mixstyle_alpha,
        name="mixstyle_mid",
    )


    # ========================================================
    # MODIFY FUNCTIONAL GRAPH
    # ========================================================

    def call_function(
        layer,
        *args,
        **kwargs,
    ):
        """
        Execute every original MobileNet layer normally.

        After block_5_add only:
            original output -> MixStyle -> next original layer
        """

        output = layer(
            *args,
            **kwargs,
        )

        if layer.name == MIXSTYLE_TARGET_LAYER:

            output = mixstyle(
                output
            )

        return output


    # Reuse the actual ImageNet-loaded layer objects.
    # Therefore pretrained weights are preserved.
    backbone = keras.models.clone_model(
        original_backbone,
        clone_function=lambda layer: layer,
        call_function=call_function,
    )

    backbone.trainable = (
        backbone_trainable
    )


    # ========================================================
    # FULL CLASSIFIER
    # ========================================================

    inputs = keras.Input(
        shape=input_shape,
        name="image",
    )

    x = augmentation(
        inputs
    )


    # Same MobileNetV2 preprocessing as E007.
    x = layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenet_preprocessing",
    )(x)


    # Full MobileNet still executes.
    #
    # MixStyle occurs internally after block_5_add.
    x = backbone()


    x = layers.GlobalAveragePooling2D(
        name="global_average_pool",
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


    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV2_ImageNet_MixStyle_Mid_V1",
    )

    return model


# ============================================================
# QUICK SANITY TEST
# ============================================================

if __name__ == "__main__":

    keras.utils.set_random_seed(
        SEED
    )

    model = (
        build_mobilenet_v2_mixstyle_mid()
    )

    print(
        "\nModel:",
        model.name
    )

    backbone = model.get_layer(
        "mobilenetv2_1.00_224"
    )

    print(
        "Backbone output:",
        backbone.output_shape
    )

    print(
        "Backbone trainable:",
        backbone.trainable
    )


    dummy = tf.random.uniform(
        shape=(4, 224, 224, 3),
        minval=0.0,
        maxval=1.0,
        seed=SEED,
    )


    # --------------------------------------------------------
    # Inference — MixStyle OFF
    # --------------------------------------------------------

    y_eval = model(
        dummy,
        training=False,
    )


    # --------------------------------------------------------
    # Training — MixStyle may activate
    # --------------------------------------------------------

    y_train = model(
        dummy,
        training=True,
    )


    print(
        "Inference shape:",
        y_eval.shape
    )

    print(
        "Training shape:",
        y_train.shape
    )

    print(
        "Inference sums:",
        tf.reduce_sum(
            y_eval,
            axis=1
        ).numpy()
    )

    print(
        "Training sums:",
        tf.reduce_sum(
            y_train,
            axis=1
        ).numpy()
    )


    # --------------------------------------------------------
    # Verify MixStyle was actually inserted into graph
    # --------------------------------------------------------

    mixstyle_layers = [
        layer.name
        for layer in backbone.layers
        if "mixstyle" in layer.name.lower()
    ]

    print(
        "Inserted MixStyle layers:",
        mixstyle_layers
    )