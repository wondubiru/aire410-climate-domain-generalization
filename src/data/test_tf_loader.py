import tensorflow as tf

from tf_loader import (
    load_mixed_datasets,
    CLASS_NAMES
)


print("TensorFlow version:")
print(tf.__version__)

print("\nClasses:")
print(CLASS_NAMES)


train_ds, val_ds, test_ds = load_mixed_datasets()


print("\nLoading one training batch...")


for images, labels in train_ds.take(1):

    print("\nImage batch shape:")
    print(images.shape)

    print("\nLabel batch shape:")
    print(labels.shape)

    print("\nPixel minimum:")
    print(float(tf.reduce_min(images)))

    print("\nPixel maximum:")
    print(float(tf.reduce_max(images)))

    print("\nLabels:")
    print(labels.numpy())


print("\nTensorFlow loader sanity check PASSED.")