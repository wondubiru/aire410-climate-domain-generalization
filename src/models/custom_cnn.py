import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


NUM_CLASSES = 3
INPUT_SHAPE = (224, 224, 3)
SEED = 42


def build_custom_cnn(
    input_shape=INPUT_SHAPE,
    num_classes=NUM_CLASSES
):

    # ========================================================
    # TRAINING-TIME AUGMENTATION
    # ========================================================

    augmentation = keras.Sequential(
    [
        layers.RandomFlip(
            "horizontal_and_vertical",
            seed=SEED
        ),

        layers.RandomRotation(
            0.08,
            seed=SEED
        ),

        layers.RandomZoom(
            0.10,
            seed=SEED
        ),

        layers.RandomContrast(
            0.10,
            seed=SEED
        ),
    ],
    name="augmentation"
)


    # ========================================================
    # MODEL
    # ========================================================

    inputs = keras.Input(
        shape=input_shape,
        name="image"
    )

    # Augmentation runs only during training.
    x = augmentation(inputs)


    # --------------------------------------------------------
    # Block 1
    # --------------------------------------------------------

    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D()(x)


    # --------------------------------------------------------
    # Block 2
    # --------------------------------------------------------

    x = layers.Conv2D(
        64,
        kernel_size=3,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D()(x)


    # --------------------------------------------------------
    # Block 3
    # --------------------------------------------------------

    x = layers.Conv2D(
        128,
        kernel_size=3,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    x = layers.MaxPooling2D()(x)


    # ========================================================
    # CLASSIFICATION HEAD
    # ========================================================

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(
        0.40
    )(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions"
    )(x)


    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="CustomCNN_V1"
    )

    return model


if __name__ == "__main__":

    model = build_custom_cnn()

    model.summary()