import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42


class MobileNetV2RSC(keras.Model):
    """
    MobileNetV2 trained with Representation Self-Challenging (RSC).

    RSC is applied to the pooled feature representation immediately
    before the classifier.

    Controlled against E007:
        - same MobileNetV2 ImageNet backbone
        - same image augmentation
        - same preprocessing
        - same dropout
        - same classifier
        - different TRAINING RULE only

    Default RSC parameters follow the common DomainBed formulation:
        feature_drop_fraction = 1/3
        batch_drop_fraction   = 1/3
    """

    def __init__(
        self,
        input_shape=(224, 224, 3),
        num_classes=3,
        dropout_rate=0.30,
        feature_drop_fraction=1.0 / 3.0,
        batch_drop_fraction=1.0 / 3.0,
        weights="imagenet",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes

        self.feature_drop_fraction = float(
            feature_drop_fraction
        )

        self.batch_drop_fraction = float(
            batch_drop_fraction
        )


        # ====================================================
        # EXACT E007 BASELINE AUGMENTATION
        # ====================================================

        self.augmentation = keras.Sequential(
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


        # ====================================================
        # MOBILENET PREPROCESSING
        # [0,1] -> [-1,1]
        # ====================================================

        self.preprocessing = layers.Rescaling(
            scale=2.0,
            offset=-1.0,
            name="mobilenet_preprocessing",
        )


        # ====================================================
        # IMAGENET BACKBONE
        # ====================================================

        self.backbone = keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights=weights,
        )

        self.backbone.trainable = False


        # ====================================================
        # SAME E007 CLASSIFIER HEAD
        # ====================================================

        self.gap = layers.GlobalAveragePooling2D(
            name="global_average_pool"
        )

        self.dropout = layers.Dropout(
            dropout_rate,
            seed=SEED,
            name="classifier_dropout",
        )

        # Keep logits here because RSC needs class scores.
        self.classifier = layers.Dense(
            num_classes,
            activation=None,
            name="predictions_logits",
        )


        # ====================================================
        # LOSS / METRICS
        # ====================================================

        self.loss_fn = (
            keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )
        )

        self.loss_tracker = keras.metrics.Mean(
            name="loss"
        )

        self.accuracy_tracker = (
            keras.metrics.SparseCategoricalAccuracy(
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


    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.accuracy_tracker,
            self.rsc_feature_keep_tracker,
            self.rsc_batch_challenge_tracker,
        ]


    # ========================================================
    # FEATURE EXTRACTION
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

        x = self.preprocessing(x)

        # Keep BatchNorm in inference mode exactly as in our
        # existing MobileNet transfer-learning protocol.
        #
        # Trainable convolutional weights still receive
        # gradients during Stage 2.
        x = self.backbone(
            x,
            training=False,
        )

        x = self.gap(x)

        x = self.dropout(
            x,
            training=training,
        )

        return x


    # ========================================================
    # STANDARD INFERENCE
    # ========================================================

    def call(
        self,
        inputs,
        training=False,
    ):

        features = self.extract_representation(
            inputs,
            training=training,
        )

        logits = self.classifier(
            features
        )

        return tf.nn.softmax(
            logits,
            axis=-1,
        )


    # ========================================================
    # PERCENTILE HELPER
    # ========================================================

    @staticmethod
    def _percentile_threshold(
        values,
        keep_fraction,
        axis,
        ):
        """
        Return threshold corresponding approximately to the
        requested fraction of values to KEEP below threshold.

        No TensorFlow Probability dependency required.
        """

        sorted_values = tf.sort(
            values,
            axis=axis,
        )

        n = tf.shape(values)[axis]

        position = (
             tf.cast(n - 1, tf.float32)
            * tf.cast(
                    keep_fraction,
                    tf.float32,
            )
       )

        lower_index = tf.cast(
          tf.floor(position),
          tf.int32,
           )

        upper_index = tf.cast(
           tf.math.ceil(position),
           tf.int32,
               )

        weight = (
           position
          - tf.cast(
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
         + weight
         * (
            upper_value
            - lower_value
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

        images, labels, sample_weights = (keras.utils.unpack_x_y_sample_weight(data))

        labels = tf.cast(
            labels,
            tf.int32,
        )

        one_hot = tf.one_hot(
            labels,
            depth=self.num_classes,
            dtype=tf.float32,
        )


        # ----------------------------------------------------
        # OUTER TAPE:
        # actual optimization step
        # ----------------------------------------------------

        with tf.GradientTape() as outer_tape:

            representation = (
                self.extract_representation(
                    images,
                    training=True,
                )
            )


            # ------------------------------------------------
            # Equation-style RSC step:
            # Find features most responsible for the
            # correct-class score.
            # ------------------------------------------------

            with tf.GradientTape() as inner_tape:

                inner_tape.watch(
                    representation
                )

                logits_before = self.classifier(
                    representation
                )

                true_class_score = tf.reduce_sum(
                    logits_before
                    * one_hot
                )


            importance = inner_tape.gradient(
                true_class_score,
                representation,
            )

            importance = tf.stop_gradient(
                importance
            )


            # ------------------------------------------------
            # FEATURE MASK
            #
            # Standard RSC default:
            # drop top 1/3 dominant representation features.
            # ------------------------------------------------

            feature_keep_fraction = (
                1.0
                - self.feature_drop_fraction
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
                feature_threshold[:, None]
            )

            feature_mask = tf.cast(
                importance
                < feature_threshold,
                tf.float32,
            )


            # ------------------------------------------------
            # First challenged prediction
            # ------------------------------------------------

            muted_representation = (
                representation
                * feature_mask
            )

            logits_muted = self.classifier(
                muted_representation
            )


            # ------------------------------------------------
            # BATCH PERCENTAGE
            #
            # Determine which examples were actually affected
            # most strongly by feature suppression.
            # ------------------------------------------------

            probs_before = tf.nn.softmax(
                logits_before,
                axis=-1,
            )

            probs_muted = tf.nn.softmax(
                logits_muted,
                axis=-1,
            )

            true_prob_before = tf.reduce_sum(
                probs_before
                * one_hot,
                axis=1,
            )

            true_prob_after = tf.reduce_sum(
                probs_muted
                * one_hot,
                axis=1,
            )

            probability_change = (
                true_prob_before
                - true_prob_after
            )

            probability_change = tf.stop_gradient(
                probability_change
            )


            batch_keep_fraction = (
                1.0
                - self.batch_drop_fraction
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


            # Samples with small probability change are NOT
            # challenged; they retain every feature.
            #
            # Samples with largest change retain feature_mask.
            keep_entire_sample = tf.cast(
                probability_change
                < batch_threshold,
                tf.float32,
            )

            keep_entire_sample = (
                keep_entire_sample[:, None]
            )


            # DomainBed-style OR:
            #
            # low-effect sample:
            #     all features restored
            #
            # high-effect sample:
            #     dominant features remain suppressed
            final_mask = tf.maximum(
                feature_mask,
                keep_entire_sample,
            )

            final_mask = tf.stop_gradient(
                final_mask
            )


            # ------------------------------------------------
            # FINAL CHALLENGED PREDICTION
            # ------------------------------------------------

            final_representation = (
                representation
                * final_mask
            )

            logits_final = self.classifier(
                final_representation
            )


            loss = self.loss_fn(
                labels,
                logits_final,
                sample_weight=sample_weights,
            )

            if self.losses:

                loss += tf.add_n(
                    self.losses
                )


        # ----------------------------------------------------
        # OPTIMIZATION
        # ----------------------------------------------------

        gradients = outer_tape.gradient(
            loss,
            self.trainable_variables,
        )

        grads_and_vars = [
            (g, v)
            for g, v in zip(
                gradients,
                self.trainable_variables,
            )
            if g is not None
        ]

        self.optimizer.apply_gradients(
            grads_and_vars
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        self.loss_tracker.update_state(
            loss
        )

        self.accuracy_tracker.update_state(
            labels,
            logits_final,
            sample_weight=sample_weights
        )


        feature_keep_ratio = tf.reduce_mean(
            feature_mask
        )

        challenged_ratio = (
            1.0
            - tf.reduce_mean(
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
            metric.name: metric.result()
            for metric in self.metrics
        }


    # ========================================================
    # NORMAL VALIDATION
    # ========================================================

    def test_step(
        self,
        data,
    ):

        images, labels, sample_weights = (keras.utils.unpack_x_y_sample_weight(data))   

        if labels is None:
            raise ValueError(
                "Labels are required for test_step."
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
            sample_weight=sample_weights
        )

        self.loss_tracker.update_state(
            loss
        )

        self.accuracy_tracker.update_state(
            labels,
            logits,
            sample_weight=sample_weights
        )

        return {
            "loss":
                self.loss_tracker.result(),

            "accuracy":
                self.accuracy_tracker.result(),
        }


    # ========================================================
    # TRANSFER-LEARNING CONTROL
    # ========================================================

    def freeze_backbone(self):

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

            # Same policy as E007:
            # BatchNorm stays frozen.
            if isinstance(
                layer,
                layers.BatchNormalization,
            ):
                layer.trainable = False

            else:
                layer.trainable = True


    # ========================================================
    # STANDARD INFERENCE MODEL
    # ========================================================

    def build_inference_model(self):
        """
        Produce a normal Functional Keras model containing no
        custom RSC training logic.

        This is what we save as best_model.keras.
        """

        inputs = keras.Input(
            shape=(224, 224, 3),
            name="image",
        )

        x = self.augmentation(
            inputs,
            training=False,
        )

        x = self.preprocessing(x)

        x = self.backbone(
            x,
            training=False,
        )

        x = self.gap(x)

        x = self.dropout(
            x,
            training=False,
        )

        logits = self.classifier(x)

        outputs = layers.Activation(
            "softmax",
            name="predictions",
        )(logits)

        return keras.Model(
            inputs=inputs,
            outputs=outputs,
            name="MobileNetV2_ImageNet_RSC_V1",
        )


# ============================================================
# QUICK SMOKE TEST
# ============================================================

if __name__ == "__main__":

    keras.utils.set_random_seed(
        SEED
    )

    model = MobileNetV2RSC(
        weights="imagenet",
    )

    model.freeze_backbone()

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3
        )
    )

    x = tf.random.uniform(
        (8, 224, 224, 3),
        minval=0.0,
        maxval=1.0,
        seed=SEED,
    )

    y = tf.constant(
        [0, 1, 2, 0, 1, 2, 0, 1],
        dtype=tf.int32,
    )


    print("\nNormal inference:")

    pred = model(
        x,
        training=False,
    )

    print(
        "Prediction shape:",
        pred.shape
    )

    print(
        "Probability sums:",
        tf.reduce_sum(
            pred,
            axis=1
        ).numpy()
    )


    print("\nRSC one-batch training test:")

    result = model.train_on_batch(
        x,
        y,
        return_dict=True,
    )

    print(result)


    inference_model = (
        model.build_inference_model()
    )

    print(
        "\nInference model:",
        inference_model.name
    )

    print(
        "Inference output:",
        inference_model(x).shape
    )