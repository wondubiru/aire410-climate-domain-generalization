
# AIRE410 — Today Sprint

## Date
19 September 2026

## Core Team
- Wondu
- Mohammed

---

## Today's Objective

Establish the project foundation so the research can begin without confusion.

The goal is NOT to train full models today.

The goal is to make sure:

- repository structure is ready
- research protocol is documented
- experiment tracking exists
- dataset auditing has started
- TensorFlow data loading is tested
- both core members understand the workflow

---

# Wondu — GitHub / Research Infrastructure

## Tasks

- [x] Create private GitHub repository
- [x] Add Mohammed as collaborator
- [x] Create project folder structure
- [x] Create `src/` subfolders
- [x] Configure `.gitignore`
- [x] Write `README.md`
- [x] Create `requirements.txt`
- [x] Create working branch
- [x] Create `research_protocol_v1.md`
- [x] Create `experiment_registry.csv`
- [x] Create `data/dataset_audit_template.csv`
- [x] Create `data/dataset_sources.md`

## Remaining Today

- [ ] Review repository structure
- [ ] Check that no large files are tracked
- [ ] Explain repository workflow to Mohammed
- [ ] Review Mohammed's TensorFlow/data work
- [ ] Merge approved setup into `main`

---

# Mohammed — Data / TensorFlow Foundation

## Tasks

- [ ] Access official project dataset
- [ ] Inspect folder structure
- [ ] Confirm class labels
- [ ] Count sample images per class
- [ ] Inspect image dimensions
- [ ] Build minimal TensorFlow loader
- [ ] Resize images to initial target size
- [ ] Normalize images
- [ ] Create a small batch
- [ ] Visualize images + labels
- [ ] Record any dataset issues

## Success Condition

The pipeline should prove:

```text
Image files
   ↓
TensorFlow dataset
   ↓
Resize
   ↓
Normalize
   ↓
Batch
   ↓
Correct images + labels
