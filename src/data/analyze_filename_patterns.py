from pathlib import Path
from PIL import Image
from collections import Counter, defaultdict
import re

DATASET_ROOT = Path(
    r"C:\ADU_BSC_All_years\Fourth_year\Fall_2026_27\AIRE410\project\NewDataset"
)


def classify_filename(name):

    lower = name.lower()

    if re.fullmatch(r"\d+\.(jpg|jpeg|png)", lower):
        return "numeric_only"

    if re.fullmatch(r"desert\(\d+\)\.(jpg|jpeg|png)", lower):
        return "desert_parentheses"

    if re.fullmatch(r"flooded_\d+\.(jpg|jpeg|png)", lower):
        return "flooded_prefix"

    if re.fullmatch(r"unflooded_\d+\.(jpg|jpeg|png)", lower):
        return "unflooded_prefix"

    if re.fullmatch(r"03_\d+\.(jpg|jpeg|png)", lower):
        return "03_prefix"

    if "_sat_" in lower:
        return "sat_pattern"

    return "other"


for class_dir in sorted(DATASET_ROOT.iterdir()):

    if not class_dir.is_dir():
        continue

    pattern_counts = Counter()
    dimensions = defaultdict(Counter)
    examples = defaultdict(list)

    for path in class_dir.iterdir():

        if not path.is_file():
            continue

        pattern = classify_filename(path.name)

        with Image.open(path) as img:
            size = img.size

        pattern_counts[pattern] += 1
        dimensions[pattern][size] += 1

        if len(examples[pattern]) < 5:
            examples[pattern].append(path.name)

    print("\n" + "=" * 80)
    print(class_dir.name)
    print("=" * 80)

    for pattern, count in pattern_counts.most_common():

        print(f"\n{pattern}: {count}")

        print("Examples:")
        for x in examples[pattern]:
            print("   ", x)

        print("Top dimensions:")
        for size, n in dimensions[pattern].most_common(10):
            print(f"    {size}: {n}")