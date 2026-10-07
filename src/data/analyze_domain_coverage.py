import pandas as pd

MANIFEST = "data/provenance_manifest.csv"
TAXONOMY = "data/source_domain_taxonomy.csv"

df = pd.read_csv(MANIFEST)
taxonomy = pd.read_csv(TAXONOMY)

# Attach domain metadata to every image
df = df.merge(
    taxonomy,
    on="source_dataset",
    how="left"
)

# Save enriched manifest
df.to_csv(
    "data/provenance_manifest_v2.csv",
    index=False
)

print("\n=== SOURCE × CLASS ===")
source_class = pd.crosstab(
    df["source_dataset"],
    df["class_label"]
)
print(source_class)

print("\n=== BROAD DOMAIN × CLASS ===")
broad = pd.crosstab(
    df["broad_domain"],
    df["class_label"]
)
print(broad)

print("\n=== STRICT DOMAIN × CLASS ===")
strict = pd.crosstab(
    df["strict_domain"],
    df["class_label"]
)
print(strict)

source_class.to_csv(
    "data/source_class_coverage.csv"
)

broad.to_csv(
    "data/broad_domain_class_coverage.csv"
)

strict.to_csv(
    "data/strict_domain_class_coverage.csv"
)

print("\nSaved:")
print("data/provenance_manifest_v2.csv")
print("data/source_class_coverage.csv")
print("data/broad_domain_class_coverage.csv")
print("data/strict_domain_class_coverage.csv")