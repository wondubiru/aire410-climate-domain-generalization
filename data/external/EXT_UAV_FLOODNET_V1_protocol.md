# EXT_UAV_FLOODNET_V1

## Purpose
External UAV validation benchmark for frozen AIRE410
Flooded / Desert / Neither classifiers.

## Included samples
398 labeled RGB images:
- 51 Flooded
- 347 Non-Flooded

## Mapping
- FloodNet Flooded -> AIRE410 Flooded
- FloodNet Non-Flooded -> AIRE410 Neither

The original three-class model output remains unchanged:
- Desert
- Flooded
- Neither

Desert predictions are errors.

## Leakage audit
Compared against all 6,334 original AIRE410 images.

- Exact SHA-256 overlaps: 0
- Minimum nearest pHash distance: 12
- pHash distance <= 4: 0
- pHash distance <= 8: 0

## Prohibited use
This benchmark must not be used for:
- training
- fine-tuning
- threshold tuning
- hyperparameter optimization
- augmentation selection
- seed selection
- model selection

Status: FROZEN BEFORE FIRST MODEL EVALUATION.
