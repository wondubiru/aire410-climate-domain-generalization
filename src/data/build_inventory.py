
from pathlib import Path
from PIL import Image
import hashlib
import pandas as pd

# CHANGE THIS ONLY
DATASET_ROOT = Path(
    r"C:\ADU_BSC_All_years\Fourth_year\Fall_2026_27\AIRE410\project\NewDataset"
)

records = []

for class_dir in DATASET_ROOT.iterdir():
    if not class_dir.is_dir():
        continue

    print(f"Scanning: {class_dir.name}")

    for path in sorted(class_dir.iterdir()):
        if not path.is_file():
            continue

        try:
            with Image.open(path) as img:
                width, height = img.size
                mode = img.mode
                image_format = img.format

            # Exact-content fingerprint
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

            records.append({
                "class_folder": class_dir.name,
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "width": width,
                "height": height,
                "mode": mode,
                "format": image_format,
                "file_size_bytes": path.stat().st_size,
                "sha256": sha256,
            })

        except Exception as e:
            records.append({
                "class_folder": class_dir.name,
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "width": None,
                "height": None,
                "mode": None,
                "format": None,
                "file_size_bytes": path.stat().st_size,
                "sha256": None,
                "error": str(e),
            })

df = pd.DataFrame(records)

output = DATASET_ROOT.parent / "compiled_dataset_inventory.csv"
df.to_csv(output, index=False)

print("\n=== TOTAL ===")
print(len(df))

print("\n=== CLASS COUNTS ===")
print(df["class_folder"].value_counts())

print("\n=== IMAGE DIMENSIONS ===")
print(
    df.groupby(["class_folder", "width", "height"])
      .size()
      .sort_values(ascending=False)
      .head(30)
)

print("\n=== FILE TYPES ===")
print(df["extension"].value_counts())

print(f"\nSaved inventory to:\n{output}")
