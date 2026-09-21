# Provenance Reconstruction Validation

## Dataset

Total images: **6334**

## Reconstruction Status

- Unresolved images: **0**
- Processing errors: **0**

## Expected vs Reconstructed Source Counts

| Source | Expected | Reconstructed | Match |
|---|---:|---:|---|
| Louisiana_Flood_2016 | 263 | 263 | PASS |
| FDL_UAV_Flooded_Areas | 297 | 297 | PASS |
| Cyclone_Wildfire_Flood_Earthquake | 613 | 613 | PASS |
| Satellite_Image_Classification | 1131 | 1131 | PASS |
| Disasters_Dataset | 1630 | 1630 | PASS |
| Aerial_Landscape_Images | 800 | 800 | PASS |
| Aerial_Images_of_Cities | 600 | 600 | PASS |
| Forest_Aerial_Images | 1000 | 1000 | PASS |

## Class × Source Matrix

| class_label   |   Aerial_Images_of_Cities |   Aerial_Landscape_Images |   Cyclone_Wildfire_Flood_Earthquake |   Disasters_Dataset |   FDL_UAV_Flooded_Areas |   Forest_Aerial_Images |   Louisiana_Flood_2016 |   Satellite_Image_Classification |
|:--------------|--------------------------:|--------------------------:|------------------------------------:|--------------------:|------------------------:|-----------------------:|-----------------------:|---------------------------------:|
| Desert        |                         0 |                       800 |                                   0 |                   0 |                       0 |                      0 |                      0 |                             1131 |
| Flooded       |                         0 |                         0 |                                 613 |                1493 |                     130 |                      0 |                    102 |                                0 |
| Neither       |                       600 |                         0 |                                   0 |                 137 |                     167 |                   1000 |                    161 |                                0 |

## Interpretation

All reconstructed source counts exactly match the expected source contributions. No images remain unresolved. This provides strong internal consistency evidence for the source-provenance reconstruction.
