import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42


# ============================================================
# MIXSTYLE
# ============================================================

@keras.utils.register_keras_serializable(
    package="AIRE410"
)
class MixStyle(layers.Layer):
    """
    Source-agnostic feature-statistics mixing.

    During training:
        1. Compute per-sample channel mean/std.
        2. Normalize each sample.
        3. Randomly pair samples within the mini-batch.
        4. Mix their feature statistics using lambda ~ Beta(alpha, alpha).
        5. Reconstruct features using the mixed statistics.

    During validation/test:
        Identity operation.

    No source labels are required.
    """

    def __init__(
        self,
        p=0.5,
        alpha=0.1,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)


    def _mix(self, x):

        # ----------------------------------------------------
        # Feature statistics
        #
        # x shape:
        #     N x H x W x C
        # ----------------------------------------------------

        mu = tf.reduce_mean(
            x,
            axis=[1, 2],
            keepdims=True,
        )

        var = tf.math.reduce_variance(
            x,
            axis=[1, 2],
            keepdims=True,
        )

        sigma = tf.sqrt(
            var + self.eps
        )


        # ----------------------------------------------------
        # Treat statistics as style descriptors.
        # Stop gradients through statistics themselves.
        # ----------------------------------------------------

        mu = tf.stop_gradient(mu)
        sigma = tf.stop_gradient(sigma)


        # ----------------------------------------------------
        # Normalize content
        # ----------------------------------------------------

        x_norm = (
            x - mu
        ) / sigma


        # ----------------------------------------------------
        # Random sample pairing within the batch
        # ----------------------------------------------------

        batch_size = tf.shape(x)[0]

        permutation = tf.random.shuffle(
            tf.range(batch_size)
        )

        mu_partner = tf.gather(
            mu,
            permutation
        )

        sigma_partner = tf.gather(
            sigma,
            permutation
        )


        # ----------------------------------------------------
        # lambda ~ Beta(alpha, alpha)
        #
        # Beta sample constructed using:
        #
        # lambda = G1 / (G1 + G2)
        #
        # where:
        # G1,G2 ~ Gamma(alpha, 1)
        # ----------------------------------------------------

        sample_shape = tf.stack([
            batch_size,
            1,
            1,
            1,
        ])

        gamma_1 = tf.random.gamma(
            shape=sample_shape,
            alpha=self.alpha,
            dtype=x.dtype,
        )

        gamma_2 = tf.random.gamma(
            shape=sample_shape,
            alpha=self.alpha,
            dtype=x.dtype,
        )

        lam = gamma_1 / (
            gamma_1
            + gamma_2
            + self.eps
        )


        # ----------------------------------------------------
        # Mix style statistics
        # ----------------------------------------------------

        mixed_mu = (
            lam * mu
            + (1.0 - lam) * mu_partner
        )

        mixed_sigma = (
            lam * sigma
            + (1.0 - lam) * sigma_partner
        )


        # ----------------------------------------------------
        # Restore normalized content using mixed style
        # ----------------------------------------------------

        return (
            x_norm * mixed_sigma
            + mixed_mu
        )


    def call(
        self,
        inputs,
        training=None,
    ):

        # Inference / validation:
        # MixStyle is disabled.
        if training is False or training is None:
            return inputs

        # If training is a normal Python bool
        if isinstance(training, bool):

            if not training:
                return inputs

            apply_mix = (
                tf.random.uniform(())
                < self.p
            )

            return tf.cond(
                apply_mix,
                lambda: self._mix(inputs),
                lambda: tf.identity(inputs),
            )

        # Graph-mode boolean
        return tf.cond(
            tf.cast(training, tf.bool),
            lambda: tf.cond(
                tf.random.uniform(()) < self.p,
                lambda: self._mix(inputs),
                lambda: tf.identity(inputs),
            ),
            lambda: tf.identity(inputs),
        )


    def get_config(self):

        config = super().get_config()

        config.update({
            "p": self.p,
            "alpha": self.alpha,
            "eps": self.eps,
        })

        return config


# ============================================================
# MOBILENETV2 + MIXSTYLE
# ============================================================

def build_mobilenet_v2_mixstyle(
    input_shape=(224, 224, 3),
    num_classes=3,
    dropout_rate=0.30,
    backbone_trainable=False,
    mixstyle_p=0.5,
    mixstyle_alpha=0.1,
):
    """
    MobileNetV2 ImageNet model with late-feature MixStyle.

    Controlled comparison against E007:
        Same baseline image augmentation.
        Same ImageNet backbone.
        Same classifier head.
        Same preprocessing.

    Experimental change:
        MixStyle is inserted between the convolutional
        backbone and global average pooling.
    """

    # --------------------------------------------------------
    # EXACT E007 BASELINE AUGMENTATION
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
    # IMAGENET BACKBONE
    # --------------------------------------------------------

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )

    backbone.trainable = (
        backbone_trainable
    )


    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    inputs = keras.Input(
        shape=input_shape,
        name="image",
    )

    x = augmentation(
        inputs
    )


    # --------------------------------------------------------
    # [0,1] -> [-1,1]
    # --------------------------------------------------------

    x = layers.Rescaling(
        scale=2.0,
        offset=-1.0,
        name="mobilenet_preprocessing",
    )(x)


    # --------------------------------------------------------
    # PRETRAINED FEATURES
    # --------------------------------------------------------

    x = backbone(
        x,
        training=False,
    )


    # --------------------------------------------------------
    # FEATURE-SPACE DOMAIN RANDOMIZATION
    # --------------------------------------------------------

    x = MixStyle(
        p=mixstyle_p,
        alpha=mixstyle_alpha,
        name="mixstyle",
    )(x)


    # --------------------------------------------------------
    # CLASSIFIER
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


    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="MobileNetV2_ImageNet_MixStyle_V1",
    )

    return model


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    keras.utils.set_random_seed(
        SEED
    )

    model = (
        build_mobilenet_v2_mixstyle()
    )

    model.summary()

    dummy = tf.random.uniform(
        shape=(4, 224, 224, 3),
        minval=0.0,
        maxval=1.0,
        seed=SEED,
    )

    # MixStyle OFF
    output_eval = model(
        dummy,
        training=False,
    )

    # MixStyle potentially ON
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
        "Inference probability sums:",
        tf.reduce_sum(
            output_eval,
            axis=1
        ).numpy()
    )

    print(
        "Training probability sums:",
        tf.reduce_sum(
            output_train,
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
        "MixStyle probability:",
        model.get_layer(
            "mixstyle"
        ).p
    )

    print(
        "MixStyle alpha:",
        model.get_layer(
            "mixstyle"
        ).alpha
    )