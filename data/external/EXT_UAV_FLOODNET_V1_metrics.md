# EXT_UAV_FLOODNET_V1 — Evaluation Metrics

## Ground-truth classes
- Flooded
- Neither

## Model output classes
- Desert
- Flooded
- Neither

No class is removed.
No probabilities are renormalized.
Desert predictions count as errors.

## Primary metric
Supported-class Macro-F1:

Macro-F1 = (F1_Flooded + F1_Neither) / 2

## Secondary metrics
- Accuracy
- Supported-class macro precision
- Supported-class macro recall
- Per-class precision / recall / F1
- AUROC / AUPRC
- Desert prediction count and rate
- Full 2 x 3 confusion matrix
- Per-image probabilities

## Frozen model suite

ERM:
- E007
- E016
- E017

DG-AUG-V2:
- E012
- E018
- E019

RSC:
- E015
- E020
- E021

All nine models must be reported.

Status: FROZEN BEFORE FIRST MODEL EVALUATION.
