
# Dataset Sources and Provenance

## Purpose

This document records the origin, role, status, and research-use notes for every dataset considered in this project.

No external dataset should be used in experiments before its provenance, licensing, class compatibility, and independence from the primary dataset are checked.

---

## 1. Primary Course Dataset

### Name
Climate Change Dataset

### Role
Primary training and internal evaluation dataset.

### Official Project Classes
- Flooded
- Desert
- Neither

### Approximate Size
6,334 aerial images

### Required Internal Evaluations
- Mixed → Mixed
- UAV → UAV
- Satellite → Satellite
- UAV → Satellite
- Satellite → UAV

### Current Status
UNDER AUDIT

### Questions to Resolve
- What are the exact original source datasets?
- Which images are UAV?
- Which images are satellite?
- Can source-domain labels be reconstructed reliably?
- Are there exact duplicates?
- Are there near-duplicates?
- Are there train/test leakage risks?
- Are classes balanced?
- What are image dimensions and formats?
- What geographic regions are represented?

### Decision
PENDING

---

# 2. External / Gold Data Candidates

Each candidate must be classified as:

- KEEP
- MAYBE
- REJECT

before being included in the study.

---

## Candidate A — Independent UAV Flood Dataset

### Dataset Name
TBD

### Intended Role
External UAV-domain evaluation.

### Desired Labels
- Flood
- Normal / Non-flood

### Research Question
Does the model generalize to previously unseen UAV imagery?

### Independence Check
PENDING

### License Check
PENDING

### Duplicate Check
PENDING

### Decision
PENDING

---

## Candidate B — Global Flood Stress-Test Dataset

### Dataset Name
TBD

### Intended Role
Global flood-recognition stress test.

### Desired Characteristics
- multiple countries
- multiple flood events
- varied camera/viewing conditions
- geographically diverse imagery

### Research Question
Does flood recognition remain reliable across unseen geographic and acquisition conditions?

### Independence Check
PENDING

### License Check
PENDING

### Duplicate Check
PENDING

### Decision
PENDING

---

## Candidate C — Independent Satellite Flood Dataset

### Dataset Name
TBD

### Intended Role
External satellite-domain evaluation.

### Desired Characteristics
- satellite imagery
- clearly identified flood regions
- independent of primary dataset sources

### Research Question
Does a model trained on the course dataset generalize to independent satellite imagery?

### Independence Check
PENDING

### License Check
PENDING

### Duplicate Check
PENDING

### Decision
PENDING

---

## Candidate D — Desertification Dataset

### Dataset Name
TBD

### Intended Role
External desert / desertification stress test.

### Desired Labels
Preferably severity-based labels such as:
- None
- Light
- Moderate
- Severe
- Extreme

### Research Question
Does model confidence in the Desert class behave meaningfully across increasing desertification severity?

### Independence Check
PENDING

### License Check
PENDING

### Duplicate Check
PENDING

### Decision
PENDING

---

# 3. Dataset Acceptance Criteria

A dataset is accepted only if it satisfies the relevant requirements below.

## Scientific Relevance
- supports one of the research questions
- represents a meaningful domain shift
- contributes more than just extra training samples

## Independence
- not derived from the same original images as the primary dataset
- no known overlap with training data
- duplicate and near-duplicate audit completed

## Provenance
- original paper or official repository identified
- source is traceable
- acquisition conditions are reasonably documented

## Labels
- labels are reliable
- class mapping is scientifically defensible
- ambiguous labels are not forced into artificial categories

## Licensing
- research use is allowed
- redistribution restrictions are understood
- attribution requirements are recorded

## Technical Suitability
- downloadable
- readable image format
- sufficient sample size
- usable metadata where available

---

# 4. Important Rule

External datasets will not automatically be merged into the training set.

Their default role is:

> Independent external evaluation.

This preserves their value for measuring real out-of-domain generalization.

---

# 5. Locked External Test Principle

Once a final external evaluation set is defined:

- do not repeatedly tune on it
- do not select hyperparameters using it
- do not inspect it after every training run
- use validation data for model selection
- reserve external evaluation for final or milestone testing

---

# 6. Dataset Decision Log

| Dataset | Role | Status | Independence | License | Duplicate Audit | Final Decision |
|---|---|---|---|---|---|---|
| Primary Climate Change Dataset | Training/Internal | Under audit | N/A | Pending | Pending | Pending |
| UAV Candidate | External UAV | Pending | Pending | Pending | Pending | Pending |
| Global Flood Candidate | Stress test | Pending | Pending | Pending | Pending | Pending |
| Satellite Candidate | External satellite | Pending | Pending | Pending | Pending | Pending |
| Desertification Candidate | Desert stress test | Pending | Pending | Pending | Pending | Pending |
