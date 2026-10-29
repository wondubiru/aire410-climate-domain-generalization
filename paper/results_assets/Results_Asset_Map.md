# Results Section Asset Map

## IV-A — Mixed and Source-Disjoint Performance

Use Table I.

It reports Mixed, SD1--SD4, and the unweighted SD1--SD4 mean.

The mean row directly supports the observed difference:

- CustomCNN: 0.657910
- MobileNetV2 pipeline: 0.747631
- Difference: +8.972 percentage points

Do not attribute this difference solely to ImageNet pretraining,
because architecture and training strategy also differ.

## IV-B — Multi-Seed Evaluation

Use Table II immediately after introducing seeds 42, 123, and 2026.

Primary message:

Neither DG-AUG-V2 nor RSC consistently outperforms ERM.

## IV-C — External FloodNet UAV Validation

Use Table III.

All displayed metrics are three-seed means except the explicitly
reported Macro-F1 standard deviation.

Supported Macro-F1 is computed over Flooded and Neither ground-truth
classes. The three-class model output remains unchanged and Desert
predictions remain errors.

## IV-D — Grad-CAM

Main-paper Grad-CAM figure should contain representative frozen
examples only:

1. consensus-correct Flooded,
2. consensus-correct Neither,
3. high-confidence consensus Neither-to-Flooded failure (10840.jpg).

Each example:

Original | ERM | DG-AUG-V2 | RSC

The remaining frozen Grad-CAM figures belong in supplementary material
or an appendix.

## IV-E — Segmentation Shortcut Audit

Use Table IV followed by:

1. Fig_segmentation_class_presence_rates.png
2. Fig_segmentation_mean_pixel_coverage.png
3. Fig_water_coverage_threshold_curve.png

For a space-limited conference paper, combine the first two figures
into one two-panel figure and move the threshold curve to supplementary
material.

Table IV reports Holm-adjusted p-values across all ten segmentation
classes.

The analysis is post-hoc exploratory and establishes association, not
causality.

Road-flooded and Building-flooded were absent from both consensus
groups and are therefore reported descriptively rather than treated
as informative hypothesis tests.
