from pathlib import Path
from PIL import Image

DATASET_ROOT = Path(
    r"C:\ADU_BSC_All_years\Fourth_year\Fall_2026_27\AIRE410\project\NewDataset"
)

for class_dir in sorted(DATASET_ROOT.iterdir()):

    if not class_dir.is_dir():
        continue

    print("\n" + "=" * 70)
    print(class_dir.name)
    print("=" * 70)

    files = sorted(class_dir.iterdir(), key=lambda p: p.name)

    # First 20
    print("\nFIRST 20")
    for path in files[:20]:
        with Image.open(path) as img:
            print(path.name, img.size, path.suffix)

    # Middle region
    mid = len(files) // 2

    print("\nMIDDLE 20")
    for path in files[mid-10:mid+10]:
        with Image.open(path) as img:
            print(path.name, img.size, path.suffix)

    # Last 20
    print("\nLAST 20")
    for path in files[-20:]:
        with Image.open(path) as img:
            print(path.name, img.size, path.suffix)