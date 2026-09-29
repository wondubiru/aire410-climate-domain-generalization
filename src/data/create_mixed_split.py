from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MANIFEST_PATH = Path(
    "data/provenance_manifest_v2.csv"
)

OUTPUT_DIR = Path(
    "data/splits/MIXED_RANDOM_V1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    MANIFEST_PATH
)


# ============================================================
# ADD IDENTIFIERS
# ============================================================

folder_map = {
    "Flooded": "Flooded_NewDataset",
    "Desert": "Desert_NewDataset",
    "Neither": "Neither_NewDataset",
}

df["class_folder"] = (
    df["class_label"].map(folder_map)
)

df["image_id"] = (
    df["class_label"].astype(str)
    + "/"
    + df["file_name"].astype(str)
)

df["relative_path"] = (
    df["class_folder"].astype(str)
    + "/"
    + df["file_name"].astype(str)
)


# ============================================================
# STEP 1:
# CREATE TEST SPLIT
#
# 1/7 ≈ 14.3%
# ============================================================

test_splitter = StratifiedGroupKFold(
    n_splits=7,
    shuffle=True,
    random_state=SEED
)

train_val_idx, test_idx = next(
    test_splitter.split(
        X=df,
        y=df["class_label"],
        groups=df["sha256"]
    )
)

train_val_df = (
    df.iloc[train_val_idx]
    .copy()
)

test_df = (
    df.iloc[test_idx]
    .copy()
)


# ============================================================
# STEP 2:
# SPLIT REMAINING DATA INTO TRAIN + VALIDATION
#
# Validation = 1/6 of remaining 6/7
#
# (1/6)*(6/7) = 1/7 ≈ 14.3% total
#
# Final approximate ratio:
#
# Train = 5/7 ≈ 71.4%
# Val   = 1/7 ≈ 14.3%
# Test  = 1/7 ≈ 14.3%
# ============================================================

val_splitter = StratifiedGroupKFold(
    n_splits=6,
    shuffle=True,
    random_state=SEED
)

train_idx, val_idx = next(
    val_splitter.split(
        X=train_val_df,
        y=train_val_df["class_label"],
        groups=train_val_df["sha256"]
    )
)

train_df = (
    train_val_df.iloc[train_idx]
    .copy()
)

val_df = (
    train_val_df.iloc[val_idx]
    .copy()
)


# ============================================================
# VERIFICATION
# ============================================================

print("=" * 75)
print("MIXED RANDOM SPLIT")
print("=" * 75)


# ------------------------------------------------------------
# Image-ID leakage
# ------------------------------------------------------------

assert set(
    train_df["image_id"]
).isdisjoint(
    set(val_df["image_id"])
)

assert set(
    train_df["image_id"]
).isdisjoint(
    set(test_df["image_id"])
)

assert set(
    val_df["image_id"]
).isdisjoint(
    set(test_df["image_id"])
)


# ------------------------------------------------------------
# Exact-image leakage
# ------------------------------------------------------------

assert set(
    train_df["sha256"]
).isdisjoint(
    set(val_df["sha256"])
), "Exact duplicate leaked between train and validation."

assert set(
    train_df["sha256"]
).isdisjoint(
    set(test_df["sha256"])
), "Exact duplicate leaked between train and test."

assert set(
    val_df["sha256"]
).isdisjoint(
    set(test_df["sha256"])
), "Exact duplicate leaked between validation and test."


# ------------------------------------------------------------
# Dataset completeness
# ------------------------------------------------------------

assert (
    len(train_df)
    + len(val_df)
    + len(test_df)
    == len(df)
)


# ------------------------------------------------------------
# All classes present
# ------------------------------------------------------------

expected_classes = {
    "Flooded",
    "Desert",
    "Neither",
}

assert (
    set(train_df["class_label"])
    == expected_classes
)

assert (
    set(val_df["class_label"])
    == expected_classes
)

assert (
    set(test_df["class_label"])
    == expected_classes
)


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_split(split_df, filename):

    columns = [
        "image_id",
        "relative_path",
        "file_name",
        "class_label",
        "source_dataset",
        "broad_domain",
        "strict_domain",
        "width",
        "height",
        "sha256",
    ]

    split_df[
        columns
    ].sort_values(
        "image_id"
    ).to_csv(
        OUTPUT_DIR / filename,
        index=False
    )


save_split(
    train_df,
    "train.csv"
)

save_split(
    val_df,
    "val.csv"
)

save_split(
    test_df,
    "test.csv"
)


# ============================================================
# REPORT
# ============================================================

def print_split(name, split_df):

    print(
        "\n" + "=" * 75
    )

    print(name)

    print(
        "=" * 75
    )

    print(
        "\nClass counts:"
    )

    print(
        split_df[
            "class_label"
        ].value_counts()
    )

    print(
        f"\nTotal = {len(split_df)}"
    )

    print(
        "\nSource counts:"
    )

    print(
        split_df[
            "source_dataset"
        ].value_counts()
    )

    print(
        "\nSOURCE × CLASS:"
    )

    print(
        pd.crosstab(
            split_df["source_dataset"],
            split_df["class_label"]
        )
    )


print_split(
    "TRAIN",
    train_df
)

print_split(
    "VALIDATION",
    val_df
)

print_split(
    "TEST",
    test_df
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for split_name, split_df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:

    counts = (
        split_df[
            "class_label"
        ].value_counts()
    )

    summary_rows.append({
        "split": split_name,
        "total": len(split_df),
        "Flooded": counts.get(
            "Flooded", 0
        ),
        "Desert": counts.get(
            "Desert", 0
        ),
        "Neither": counts.get(
            "Neither", 0
        ),
    })


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUTPUT_DIR
    / "split_summary.csv",
    index=False
)


# ============================================================
# SOURCE PRESENCE CHECK
# ============================================================

all_sources = set(
    df["source_dataset"]
)

print(
    "\n" + "=" * 75
)

print(
    "SOURCE PRESENCE CHECK"
)

print(
    "=" * 75
)

for split_name, split_df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:

    present = set(
        split_df[
            "source_dataset"
        ]
    )

    missing = (
        all_sources - present
    )

    if len(missing) == 0:

        print(
            f"{split_name}: "
            "all 8 sources present"
        )

    else:

        print(
            f"{split_name}: "
            f"missing sources = {missing}"
        )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "MIXED RANDOM SPLIT PASSED."
)

print(
    "=" * 75
)

print(
    "\nSummary:"
)

print(
    summary
)

print(
    f"\nSaved under:\n"
    f"{OUTPUT_DIR.resolve()}"
)