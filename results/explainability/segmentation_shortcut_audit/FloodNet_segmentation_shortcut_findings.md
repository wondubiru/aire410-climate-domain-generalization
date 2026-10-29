# FloodNet Segmentation Shortcut Audit

## Status

This analysis is a post-hoc exploratory explanatory analysis performed
after the external FloodNet evaluation and the frozen Grad-CAM cohort.

The consensus groups were defined from nine already-frozen models:

- Consensus-correct Neither: 56 images
- Consensus Neither→Flooded failure: 178 images
- Mixed/disagreement: 113 images

No image was added, removed, or reselected based on its segmentation
composition.

## Water

Ordinary Water was present in:

- 92/178
  (51.7%)
  consensus failures
- 3/56
  (5.4%)
  consensus-correct Neither images

Presence-rate difference:

+46.3 percentage points

Fisher exact p-value:

3.02822e-11

Mean Water coverage:

- Failure: 13.44%
- Correct: 0.53%

Mann-Whitney p-value:

1.1592e-09

## Non-flooded roads

Road-non-flooded was present in:

- 73.0% of consensus failures
- 14.3% of consensus-correct images

Mean Road-non-flooded coverage:

- Failure: 8.84%
- Correct: 0.58%

Fisher exact p-value:

3.94897e-15

Mann-Whitney p-value:

4.2779e-14

## Flooded infrastructure

Road-flooded presence:

- Failure: 0.0%
- Correct: 0.0%

Building-flooded presence:

- Failure: 0.0%
- Correct: 0.0%

Thus, the Water and Road-non-flooded associations are not explained
by officially annotated flooded roads or flooded buildings within
these consensus groups.

## Vegetation composition

Mean Grass coverage:

- Failure: 51.88%
- Correct: 85.25%

Mann-Whitney p-value:

2.02191e-17

Mean Tree coverage:

- Failure: 18.78%
- Correct: 13.22%

Mann-Whitney p-value:

0.000103636

## Interpretation

Consensus false-flood predictions are strongly associated with
water-rich and infrastructure-rich non-flooded UAV scenes.

Together with the frozen Grad-CAM examples, the results are consistent
with classifiers relying on flood-associated contextual cues rather
than robustly distinguishing ordinary water and infrastructure from
true inundation.

This analysis does not establish causality. The statistical tests are
exploratory and post-hoc and should be reported as such.

## Frozen exemplar observations

8176.jpg:
official segmentation contained Grass, Road-non-flooded, and Tree,
with no Water, Road-flooded, or Building-flooded.

10840.jpg:
official segmentation contained Water, Road-non-flooded, Vehicle, and
Grass, with no Road-flooded or Building-flooded.

These examples illustrate two distinct routes to consensus
false-flood predictions: flood-like contextual appearance without
annotated water, and ordinary water adjacent to explicitly
non-flooded infrastructure.
