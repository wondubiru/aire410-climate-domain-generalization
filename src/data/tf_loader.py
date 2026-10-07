from pathlib import Path
import pandas as pd
import tensorflow as tf
import os


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    os.environ.get(
        "AIRE410_DATASET_ROOT",
        r"C:\ADU_BSC_All_years\Fourth_year\Fall_2026_27\AIRE410\project\NewDataset"
    )
)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

CLASS_NAMES = [
    "Desert",
    "Flooded",
    "Neither",
]

CLASS_TO_INDEX = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(path, label):

    image_bytes = tf.io.read_file(path)

    # Handles JPG / JPEG / PNG
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    # decode_image has dynamic shape, so define rank explicitly
    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    # Normalize to [0, 1]
    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, label


# ============================================================
# CREATE DATASET FROM CSV
# ============================================================

def create_dataset(
    csv_path,
    training=False,
    batch_size=BATCH_SIZE,
):

    df = pd.read_csv(csv_path)

    paths = [
        str(DATASET_ROOT / relative_path)
        for relative_path in df["relative_path"]
    ]

    labels = [
        CLASS_TO_INDEX[label]
        for label in df["class_label"]
    ]

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )
    options = tf.data.Options()
    options.experimental_deterministic = True
    dataset = dataset.with_options(options)

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(df),
            seed=42,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=4
    )

    dataset = dataset.batch(
        batch_size
    )

    # Safer for the laptop than aggressive AUTOTUNE prefetching
    dataset = dataset.prefetch(1)

    return dataset


# ============================================================
# MIXED BASELINE
# ============================================================

def load_mixed_datasets():

    split_root = Path(
        "data/splits/MIXED_RANDOM_V1"
    )

    train_ds = create_dataset(
        split_root / "train.csv",
        training=True
    )

    val_ds = create_dataset(
        split_root / "val.csv",
        training=False
    )

    test_ds = create_dataset(
        split_root / "test.csv",
        training=False
    )

    return train_ds, val_ds, test_ds


# ============================================================
# SOURCE-DISJOINT FOLD
# ============================================================

def load_source_disjoint_fold(fold_name):

    split_root = (
        Path("data/splits")
        / fold_name
    )

    train_ds = create_dataset(
        split_root / "train.csv",
        training=True
    )

    val_ds = create_dataset(
        split_root / "val.csv",
        training=False
    )

    test_ds = create_dataset(
        split_root / "test.csv",
        training=False
    )

    return train_ds, val_ds, test_ds