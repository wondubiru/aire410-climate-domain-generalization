# AIRE410 Master Evidence Summary

## Core research question

How robustly do climate-scene classifiers generalize across image
sources and acquisition conditions, and what visual/contextual cues
are associated with their failures?

## Evidence chain

1. Mixed/random evaluation produces very high classification
   performance.

2. Source-disjoint evaluation reveals substantial cross-source
   generalization loss.

3. The MobileNetV2 ImageNet transfer-learning pipeline achieves higher
   mean source-disjoint Macro-F1 than the CustomCNN baseline across
   SD1-SD4.

   The observed difference is approximately +8.97 percentage points.
   Because architecture and training strategy also differ, this gain
   must not be attributed solely to ImageNet pretraining.

4. DG-AUG-V2 and RSC do not consistently outperform ERM across the
   three evaluated seeds (42, 123, and 2026).

5. External FloodNet UAV evaluation independently reproduces the
   asymmetric Neither-to-Flooded failure pattern.

6. FloodNet external Macro-F1 is reported over the two supported
   ground-truth classes, Flooded and Neither. The model retains all
   three outputs, and Desert predictions remain errors.

7. Among 347 external true-Neither scenes:

   - 178 (51.3%) are predicted Flooded by all nine frozen models.
   - 56 (16.1%) are predicted Neither by all nine frozen models.
   - 113 (32.6%) show disagreement across models.

8. Frozen Grad-CAM examples show recurring activation around
   water-, road-, and boundary-adjacent regions in several
   false-flood cases.

9. Official FloodNet segmentation masks quantitatively show that
   consensus failures contain substantially more ordinary Water and
   non-flooded Road context than consensus-correct Neither images.

10. Road-flooded and Building-flooded pixels are absent from both
    consensus groups used in the segmentation shortcut comparison.

## Key quantitative findings

### Mixed versus source-disjoint performance

CustomCNN:

- Mixed Macro-F1: 0.945335
- SD1 Macro-F1: 0.339019

MobileNetV2 ImageNet pipeline:

- Mixed Macro-F1: 0.977511
- SD1 Macro-F1: 0.527433

### Mean source-disjoint performance across SD1-SD4

- CustomCNN: 0.657910
- MobileNetV2 ImageNet pipeline: 0.747631
- Difference: +0.089721 (+8.972 percentage points)

### SD1 three-seed replication

- ERM: 0.498805 ± 0.025926
- DG-AUG-V2: 0.483064 ± 0.044320
- RSC: 0.488298 ± 0.032900

### FloodNet external UAV evaluation

Supported-class Macro-F1:

- ERM: 0.375282 ± 0.046831
- DG-AUG-V2: 0.372935 ± 0.082410
- RSC: 0.361866 ± 0.076359

ERM mean class recalls:

- Flooded recall: 0.954248
- Neither recall: 0.305476

### Segmentation shortcut audit

Ordinary Water presence:

- Consensus failure: 51.7%
- Consensus correct: 5.4%
- Fisher exact p = 3.028e-11

Non-flooded Road presence:

- Consensus failure: 73.0%
- Consensus correct: 14.3%
- Fisher exact p = 3.949e-15

Mean Grass coverage:

- Consensus failure: 51.88%
- Consensus correct: 85.25%
- Mann-Whitney p = 2.022e-17

## Main interpretation

The complete evidence is consistent with the classifiers learning
flood-associated contextual cues that do not fully capture the
semantic distinction between ordinary water/infrastructure and true
inundation.

The external segmentation analysis supports this interpretation but
does not establish causality.

The segmentation comparisons and associated statistical tests are
post-hoc exploratory analyses.

Grad-CAM is used as qualitative explanatory evidence and should not be
interpreted as causal proof of the model's decision mechanism.

## Experimental stopping rule

The major model-development phase is complete.

Additional experiments should only be introduced if they:

1. satisfy an explicit remaining course requirement,
2. answer a clearly defined unresolved research question, or
3. provide a justified independent external validation that materially
   strengthens the final study.
