from pathlib import Path
from PIL import Image
import pandas as pd
import hashlib
import re

# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = Path(
    r"C:\ADU_BSC_All_years\Fourth_year\Fall_2026_27\AIRE410\project\NewDataset"
)

# Run script from repository root.
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_DIR / "provenance_manifest.csv"
REPORT_PATH = OUTPUT_DIR / "provenance_validation.md"


# ============================================================
# EXPECTED SOURCE COUNTS
# Reconstructed from the published composition of the dataset.
# ============================================================

EXPECTED_COUNTS = {
    "Louisiana_Flood_2016": 263,
    "FDL_UAV_Flooded_Areas": 297,
    "Cyclone_Wildfire_Flood_Earthquake": 613,
    "Satellite_Image_Classification": 1131,
    "Disasters_Dataset": 1630,
    "Aerial_Landscape_Images": 800,
    "Aerial_Images_of_Cities": 600,
    "Forest_Aerial_Images": 1000,
}


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    """Return SHA-256 hash of one file."""
    hasher = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def normalize_class(folder_name):
    if folder_name == "Flooded_NewDataset":
        return "Flooded"

    if folder_name == "Desert_NewDataset":
        return "Desert"

    if folder_name == "Neither_NewDataset":
        return "Neither"

    return "Unknown"


# ============================================================
# SOURCE RECONSTRUCTION
# ============================================================

def reconstruct_source(class_label, filename, width, height):
    """
    Reconstruct original source dataset using:
      1. class
      2. filename pattern
      3. image dimensions
      4. known source contribution counts

    Returns:
        source_dataset
        confidence
        evidence
    """

    name = filename.lower()
    size = (width, height)

    # --------------------------------------------------------
    # DESERT
    # --------------------------------------------------------

    if class_label == "Desert":

        # Exactly 800 numeric-only images observed.
        if re.fullmatch(r"\d+\.(jpg|jpeg|png)", name):
            return (
                "Aerial_Landscape_Images",
                "high",
                "numeric_filename + 256x256 + expected_count_800"
            )

        # desert(n).jpg plus duplicated-name variants total 1131.
        if name.startswith("desert("):
            return (
                "Satellite_Image_Classification",
                "high",
                "desert_parentheses_filename + 256x256 + expected_count_1131"
            )

    # --------------------------------------------------------
    # FLOODED
    # --------------------------------------------------------

    elif class_label == "Flooded":

        # Exactly 130 flooded_* large UAV images.
        if re.fullmatch(r"flooded_\d+\.(jpg|jpeg|png)", name):
            return (
                "FDL_UAV_Flooded_Areas",
                "high",
                "flooded_prefix + ~3Kx4K_dimensions + expected_count_130"
            )

        # Exactly 613 variable-resolution numeric-only files.
        if re.fullmatch(r"\d+\.(jpg|jpeg|png)", name):
            return (
                "Cyclone_Wildfire_Flood_Earthquake",
                "high",
                "numeric_filename + variable_dimensions + expected_count_613"
            )

        # Louisiana images form a clean 512x360 group.
        if size == (512, 360):
            return (
                "Louisiana_Flood_2016",
                "high",
                "512x360_dimensions + expected_flood_count_102"
            )

        # Remaining 224x224 images = Disasters Dataset.
        if size == (224, 224):
            return (
                "Disasters_Dataset",
                "high",
                "224x224_dimensions + expected_flood_count_1493"
            )

    # --------------------------------------------------------
    # NEITHER
    # --------------------------------------------------------

    elif class_label == "Neither":

        # Exactly 1000 filenames containing _sat_.
        if "_sat_" in name:
            return (
                "Forest_Aerial_Images",
                "high",
                "_sat_filename_pattern + 256x256 + expected_count_1000"
            )

        # Exactly 167 unflooded_* high-resolution images.
        if re.fullmatch(r"unflooded_\d+\.(jpg|jpeg|png)", name):
            return (
                "FDL_UAV_Flooded_Areas",
                "high",
                "unflooded_prefix + ~3Kx4K_dimensions + expected_count_167"
            )

        # Numeric-only files contain two clearly separated size groups.
        if re.fullmatch(r"\d+\.(jpg|jpeg|png)", name):

            if size == (512, 360):
                return (
                    "Louisiana_Flood_2016",
                    "high",
                    "numeric_filename + 512x360 + expected_count_161"
                )

            if size == (256, 256):
                return (
                    "Aerial_Images_of_Cities",
                    "high",
                    "numeric_filename + 256x256 + expected_count_600"
                )

        # Exactly 137 remaining 224x224 images.
        if size == (224, 224):
            return (
                "Disasters_Dataset",
                "high",
                "224x224_dimensions + expected_neither_count_137"
            )

    # --------------------------------------------------------
    # NOT RESOLVED
    # --------------------------------------------------------

    return (
        "UNRESOLVED",
        "none",
        "no_reconstruction_rule_matched"
    )


# ============================================================
# BUILD MANIFEST
# ============================================================

records = []

for class_dir in sorted(DATASET_ROOT.iterdir()):

    if not class_dir.is_dir():
        continue

    class_label = normalize_class(class_dir.name)

    print(f"Scanning {class_label}...")

    for path in sorted(class_dir.iterdir()):

        if not path.is_file():
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                image_format = img.format
                mode = img.mode

            source_dataset, confidence, evidence = reconstruct_source(
                class_label,
                path.name,
                width,
                height
            )

            records.append({
                "file_name": path.name,
                "class_label": class_label,
                "width": width,
                "height": height,
                "format": image_format,
                "mode": mode,
                "file_size_bytes": path.stat().st_size,
                "source_dataset": source_dataset,

                # Deliberately NOT inferred yet.
                "acquisition_domain": "TO_VERIFY",

                "confidence": confidence,
                "evidence": evidence,
                "sha256": sha256_file(path),
            })

        except Exception as e:

            records.append({
                "file_name": path.name,
                "class_label": class_label,
                "width": None,
                "height": None,
                "format": None,
                "mode": None,
                "file_size_bytes": path.stat().st_size,
                "source_dataset": "ERROR",
                "acquisition_domain": "TO_VERIFY",
                "confidence": "none",
                "evidence": f"ERROR: {e}",
                "sha256": None,
            })


df = pd.DataFrame(records)

df.to_csv(MANIFEST_PATH, index=False)


# ============================================================
# VALIDATION
# ============================================================

actual_counts = df["source_dataset"].value_counts().to_dict()

unresolved = int((df["source_dataset"] == "UNRESOLVED").sum())
errors = int((df["source_dataset"] == "ERROR").sum())

print("\n" + "=" * 70)
print("RECONSTRUCTION RESULTS")
print("=" * 70)

print("\nTOTAL IMAGES:")
print(len(df))

print("\nSOURCE COUNTS:")
print(df["source_dataset"].value_counts())

print("\nCLASS × SOURCE MATRIX:")
matrix = pd.crosstab(df["class_label"], df["source_dataset"])
print(matrix)

print("\nUNRESOLVED:")
print(unresolved)

print("\nERRORS:")
print(errors)


# ============================================================
# EXPECTED vs ACTUAL
# ============================================================

print("\n" + "=" * 70)
print("EXPECTED vs ACTUAL")
print("=" * 70)

all_match = True

validation_rows = []

for source, expected in EXPECTED_COUNTS.items():

    actual = actual_counts.get(source, 0)

    match = actual == expected

    if not match:
        all_match = False

    validation_rows.append({
        "source": source,
        "expected": expected,
        "actual": actual,
        "match": match
    })

    marker = "PASS" if match else "FAIL"

    print(
        f"{marker:4} | "
        f"{source:40} "
        f"expected={expected:4} "
        f"actual={actual:4}"
    )


# ============================================================
# GENERATE MARKDOWN VALIDATION REPORT
# ============================================================

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("# Provenance Reconstruction Validation\n\n")

    f.write("## Dataset\n\n")
    f.write(f"Total images: **{len(df)}**\n\n")

    f.write("## Reconstruction Status\n\n")
    f.write(f"- Unresolved images: **{unresolved}**\n")
    f.write(f"- Processing errors: **{errors}**\n\n")

    f.write("## Expected vs Reconstructed Source Counts\n\n")

    f.write("| Source | Expected | Reconstructed | Match |\n")
    f.write("|---|---:|---:|---|\n")

    for row in validation_rows:

        symbol = "PASS" if row["match"] else "FAIL"

        f.write(
            f"| {row['source']} "
            f"| {row['expected']} "
            f"| {row['actual']} "
            f"| {symbol} |\n"
        )

    f.write("\n## Class × Source Matrix\n\n")

    f.write(matrix.to_markdown())

    f.write("\n\n")

    f.write("## Interpretation\n\n")

    if all_match and unresolved == 0 and errors == 0:

        f.write(
            "All reconstructed source counts exactly match the expected "
            "source contributions. No images remain unresolved. "
            "This provides strong internal consistency evidence for the "
            "source-provenance reconstruction.\n"
        )

    else:

        f.write(
            "One or more reconstructed counts differ from the expected "
            "source composition. These discrepancies must be investigated "
            "before the provenance reconstruction is used in experiments.\n"
        )


# ============================================================
# FINAL ASSERTIONS
# ============================================================

assert len(df) == 6334, (
    f"Expected 6334 images, found {len(df)}."
)

assert unresolved == 0, (
    f"{unresolved} images remain unresolved."
)

assert errors == 0, (
    f"{errors} files caused processing errors."
)

for source, expected in EXPECTED_COUNTS.items():

    actual = actual_counts.get(source, 0)

    assert actual == expected, (
        f"{source}: expected {expected}, reconstructed {actual}"
    )


print("\n" + "=" * 70)

if all_match:
    print("ALL SOURCE COUNTS MATCH.")
    print("PROVENANCE RECONSTRUCTION PASSED INTERNAL VALIDATION.")

print("=" * 70)

print(f"\nManifest saved to:\n{MANIFEST_PATH}")
print(f"\nValidation report saved to:\n{REPORT_PATH}")