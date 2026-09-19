
# AIRE410 Research Protocol v1

## Working Title

**Robust Cross-Domain Climate Scene Classification Across UAV and Satellite Imagery**

---

## 1. Core Problem

The official task is to classify aerial images into:

- Flooded
- Desert
- Neither

Our research extension focuses on whether deep-learning models remain reliable when the image domain changes.

---

## 2. Main Research Goal

> Measure, explain, and reduce cross-domain generalization failure between UAV and satellite imagery.

---

## 3. Research Questions

### RQ1
How accurately can CNN-based models classify climate-related aerial scenes as Flooded, Desert, or Neither?

### RQ2
How much does model performance degrade when training and testing image domains differ?

### RQ3
Do different CNN architectures exhibit different levels of cross-domain robustness?

### RQ4
Can a domain-generalization strategy improve performance on unseen aerial-imaging domains?

---

## 4. Hypotheses

### H1 — Domain Shift
Cross-domain performance will be lower than matched-domain performance.

### H2 — Architecture Sensitivity
Different CNN architectures will exhibit different generalization gaps.

### H3 — Domain Generalization
A domain-generalization method can improve out-of-domain performance compared with ordinary training.

### H4 — External Validation
Observed generalization behavior should also be tested on independent external datasets.

---

## 5. Primary Dataset

Official AIRE410 Climate Change Dataset:

- approximately 6,334 aerial images
- 3 classes:
  - Flooded
  - Desert
  - Neither

The dataset must be audited before model training.

---

## 6. Dataset Audit Questions

We must determine:

- class counts
- UAV image counts
- satellite image counts
- image dimensions
- original data sources
- geographic provenance where available
- duplicate images
- near-duplicate images
- corrupted files
- possible data leakage
- whether UAV/Satellite source labels can be reliably reconstructed

---

## 7. Initial Models

- Custom CNN
- MobileNetV2
- ResNet50
- EfficientNetB0

Framework:

**TensorFlow**

---

## 8. Internal Experiments

### E1
Mixed → Mixed

Purpose:
Standard classification baseline.

### E2
UAV → UAV

Purpose:
Measure UAV in-domain performance.

### E3
Satellite → Satellite

Purpose:
Measure satellite in-domain performance.

### E4
UAV → Satellite

Purpose:
Measure UAV-to-satellite generalization.

### E5
Satellite → UAV

Purpose:
Measure satellite-to-UAV generalization.

---

## 9. External / Gold Data Experiments

Potential external experiments:

### E6
Independent UAV evaluation

### E7
Global flood stress test

### E8
Independent satellite flood evaluation

### E9
Independent desertification evaluation

### E10
Repeat important evaluations after domain-generalization improvement

External datasets must be audited before inclusion.

---

## 10. Primary Metric

**Macro-F1**

Secondary metrics:

- Accuracy
- Precision
- Recall
- AUROC
- AUPRC
- Confusion Matrix

---

## 11. Generalization Gap

For compatible experiments:

G = F1_ID - F1_OOD

where:

- ID = in-domain performance
- OOD = out-of-domain performance
- G = domain-generalization gap

---

## 12. Failure Analysis

When performance decreases, investigate:

- class-specific failures
- source-specific failures
- resolution sensitivity
- geographic variation
- viewing-angle differences
- model confidence
- calibration
- Grad-CAM
- feature-space visualization
- possible shortcut learning

---

## 13. Domain-Generalization Method

Not fixed yet.

The proposed method must be selected only after:

1. literature review
2. baseline experiments
3. failure analysis
4. research-gap identification

Possible directions may include:

- robust augmentation
- feature alignment
- consistency learning
- domain-invariant representation learning

No novelty claim will be made before literature verification.

---

## 14. Reproducibility Rules

Every important experiment must record:

- experiment ID
- model
- dataset
- train domain
- validation domain
- test domain
- preprocessing
- augmentation
- image size
- batch size
- optimizer
- learning rate
- random seed
- epochs
- checkpoint
- final metrics
- notes

---

## 15. Compute Strategy

Laptop:

- development
- debugging
- small data tests
- analysis
- visualization

Cloud GPU:

- full model training
- transfer learning
- fine-tuning
- expensive cross-domain experiments
- multi-seed experiments

Persistent storage:

- datasets
- checkpoints
- large experiment outputs

---

## 16. Research Success

The project is successful if it produces a rigorous answer to the research questions.

Success does NOT require the proposed method to win.

Valid outcomes include:

- meaningful cross-domain degradation discovered
- architecture-dependent robustness discovered
- external generalization failure discovered
- interpretable failure mode discovered
- domain-generalization improvement demonstrated
- unexpected cross-domain robustness demonstrated

---

## 17. Final Target

- complete AIRE410 requirements
- reproducible TensorFlow implementation
- rigorous dataset audit
- CNN benchmark
- UAV ↔ Satellite domain evaluation
- Gold / external data validation
- failure analysis
- proposed improvement
- ablation study
- explainability
- journal-style paper
- GitHub repository
- project demo
