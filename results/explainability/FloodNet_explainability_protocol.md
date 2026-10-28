# FloodNet Explainability Cohort

## Purpose

Select representative external UAV examples for Grad-CAM and
failure analysis without manual image cherry-picking.

## Frozen model suite

Selection uses predictions from all nine frozen models:

ERM:
- seed 42
- seed 123
- seed 2026

DG-AUG-V2:
- seed 42
- seed 123
- seed 2026

RSC:
- seed 42
- seed 123
- seed 2026

## Cohort

Nine images total:

1. Three Flooded consensus-correct images
   - every one of the 9 models predicts Flooded

2. Three Neither consensus-correct images
   - every one of the 9 models predicts Neither

3. Three Neither -> Flooded consensus failures
   - every one of the 9 models predicts Flooded

## Confidence tiers

Within each group, images are selected deterministically:

- low confidence
- median confidence
- high confidence

Confidence is computed as the mean relevant class probability
across all nine frozen models.

No visual inspection is used during sample selection.

## Grad-CAM comparison

For each frozen cohort image, primary method comparison will use
the seed-42 models:

- E007 ERM
- E012 DG-AUG-V2
- E015 RSC

Using a common seed avoids confounding method comparison with
training-seed variation.

## Prohibited

The explainability cohort must not be changed after viewing
Grad-CAM heatmaps.

Status: FROZEN BEFORE GRAD-CAM GENERATION.
