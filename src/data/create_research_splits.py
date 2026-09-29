from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MANIFEST_PATH = Path("data/provenance_manifest_v2.csv")
OUTPUT_ROOT = Path("data/splits")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# SOURCE-DISJOINT TEST FOLDS
# ============================================================

FOLDS = {

    # Clean UAV-oriented external-source test:
    # FDL provides Flooded + Neither
    # Aerial Landscape provides Desert
    "SD1_FDL_AerialLandscape": [
        "FDL_UAV_Flooded_Areas",
        "Aerial_Landscape_Images",
    ],

    # Independent airborne + remote-sensing combination
    "SD2_Louisiana_SatelliteClassification": [
        "Louisiana_Flood_2016",
        "Satellite_Image_Classification",
    ],

    # Harder test dominated by Disasters Dataset
    "SD3_Disasters_AerialLandscape": [
        "Disasters_Dataset",
        "Aerial_Landscape_Images",
    ],

    # Large stress-test fold
    "SD4_Cyclone_SatelliteClassification_Forest": [
        "Cyclone_Wildfire_Flood_Earthquake",
        "Satellite_Image_Classification",
        "Forest_Aerial_Images",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def add_identifiers(df):
    """
    Create reproducible unique identifiers and relative paths.
    """

    folder_map = {
        "Flooded": "Flooded_NewDataset",
        "Desert": "Desert_NewDataset",
        "Neither": "Neither_NewDataset",
    }

    df = df.copy()

    df["class_folder"] = df["class_label"].map(folder_map)

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

    return df


def save_split(df, path):
    """
    Save only useful research/training metadata.
    """

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

    df[columns].sort_values("image_id").to_csv(
        path,
        index=False
    )


def split_train_validation(train_pool):
    """
    Create approximately 85/15 train/validation split.

    Exact duplicate files are grouped by SHA-256 so identical
    images cannot appear in both training and validation.

    Stratification attempts to preserve class proportions.
    """

    splitter = StratifiedGroupKFold(
        n_splits=7,
        shuffle=True,
        random_state=SEED
    )

    X = train_pool
    y = train_pool["class_label"]
    groups = train_pool["sha256"]

    train_idx, val_idx = next(
        splitter.split(X, y, groups)
    )

    train_df = train_pool.iloc[train_idx].copy()
    val_df = train_pool.iloc[val_idx].copy()

    return train_df, val_df


def verify_split(train_df, val_df, test_df, fold_name):

    print(f"\nVerifying {fold_name}...")

    # --------------------------------
    # Image identity leakage
    # --------------------------------

    assert set(train_df["image_id"]).isdisjoint(
        set(val_df["image_id"])
    )

    assert set(train_df["image_id"]).isdisjoint(
        set(test_df["image_id"])
    )

    assert set(val_df["image_id"]).isdisjoint(
        set(test_df["image_id"])
    )

    # --------------------------------
    # Exact-image leakage
    # --------------------------------

    assert set(train_df["sha256"]).isdisjoint(
        set(val_df["sha256"])
    ), "Exact duplicate appears in train and validation."

    assert set(train_df["sha256"]).isdisjoint(
        set(test_df["sha256"])
    ), "Exact duplicate appears in train and test."

    assert set(val_df["sha256"]).isdisjoint(
        set(test_df["sha256"])
    ), "Exact duplicate appears in validation and test."

    # --------------------------------
    # Source-disjoint test
    # --------------------------------

    train_sources = set(train_df["source_dataset"])
    val_sources = set(val_df["source_dataset"])
    test_sources = set(test_df["source_dataset"])

    assert train_sources.isdisjoint(
        test_sources
    ), "Training source leaked into test."

    assert val_sources.isdisjoint(
        test_sources
    ), "Validation source leaked into test."

    # --------------------------------
    # All three classes must exist
    # --------------------------------

    expected_classes = {
        "Flooded",
        "Desert",
        "Neither"
    }

    assert set(train_df["class_label"]) == expected_classes
    assert set(val_df["class_label"]) == expected_classes
    assert set(test_df["class_label"]) == expected_classes

    print("PASS")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(MANIFEST_PATH)

df = add_identifiers(df)

print("=" * 75)
print("DATASET")
print("=" * 75)

print(f"Total images: {len(df)}")

print("\nClass counts:")
print(df["class_label"].value_counts())

print("\nSource counts:")
print(df["source_dataset"].value_counts())


# ============================================================
# CREATE FOLDS
# ============================================================

summary_rows = []

for fold_name, test_sources in FOLDS.items():

    print("\n" + "=" * 75)
    print(fold_name)
    print("=" * 75)

    fold_dir = OUTPUT_ROOT / fold_name
    fold_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Fixed held-out sources
    # --------------------------------------------------------

    test_df = df[
        df["source_dataset"].isin(test_sources)
    ].copy()

    train_pool = df[
        ~df["source_dataset"].isin(test_sources)
    ].copy()

    print("\nHeld-out test sources:")
    for source in test_sources:
        print(" -", source)

    # --------------------------------------------------------
    # Exact duplicate protection across source boundary
    # --------------------------------------------------------

    test_hashes = set(test_df["sha256"])

    leakage_mask = train_pool["sha256"].isin(
        test_hashes
    )

    quarantined = train_pool[
        leakage_mask
    ].copy()

    if len(quarantined) > 0:

        print(
            f"\nWARNING: {len(quarantined)} exact duplicate "
            "images found across train/test source boundary."
        )

        quarantined.to_csv(
            fold_dir / "quarantined_cross_source_duplicates.csv",
            index=False
        )

        train_pool = train_pool[
            ~leakage_mask
        ].copy()

    else:

        print("\nCross-source exact duplicates: 0")

    # --------------------------------------------------------
    # Train / validation
    # --------------------------------------------------------

    train_df, val_df = split_train_validation(
        train_pool
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    verify_split(
        train_df,
        val_df,
        test_df,
        fold_name
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_split(
        train_df,
        fold_dir / "train.csv"
    )

    save_split(
        val_df,
        fold_dir / "val.csv"
    )

    save_split(
        test_df,
        fold_dir / "test.csv"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\nTRAIN:")
    print(train_df["class_label"].value_counts())
    print(f"Total = {len(train_df)}")

    print("\nVALIDATION:")
    print(val_df["class_label"].value_counts())
    print(f"Total = {len(val_df)}")

    print("\nTEST:")
    print(test_df["class_label"].value_counts())
    print(f"Total = {len(test_df)}")

    print("\nTEST SOURCE × CLASS:")
    print(
        pd.crosstab(
            test_df["source_dataset"],
            test_df["class_label"]
        )
    )

    for split_name, split_df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:

        counts = split_df[
            "class_label"
        ].value_counts()

        summary_rows.append({
            "fold": fold_name,
            "split": split_name,
            "total": len(split_df),
            "Flooded": counts.get("Flooded", 0),
            "Desert": counts.get("Desert", 0),
            "Neither": counts.get("Neither", 0),
        })


# ============================================================
# GLOBAL SUMMARY
# ============================================================

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_ROOT / "source_disjoint_split_summary.csv",
    index=False
)

print("\n" + "=" * 75)
print("ALL SOURCE-DISJOINT SPLITS CREATED")
print("=" * 75)

print(summary)

print(
    "\nSaved under:\n"
    f"{OUTPUT_ROOT.resolve()}"
)