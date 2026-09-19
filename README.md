# AIRE410 — Climate Domain Generalization

Research project for **AIRE410 / AIN410 Deep Learning**.

## Project Title

**Robust Cross-Domain Climate Scene Classification Across UAV and Satellite Imagery**

## Team

- Wondu
- Mohammed
- Member 3

## Official Course Problem

The project focuses on classifying aerial imagery into:

- Flooded
- Desert
- Neither

The official AIRE410 project also requires investigation of:

- random evaluation
- source-aware evaluation
- UAV → Satellite transfer
- Satellite → UAV transfer
- cross-source generalization
- class-wise performance
- explainability

## Research Goal

The project goes beyond ordinary image classification.

Our main research objective is:

> Evaluate whether deep-learning models trained on one aerial-imaging domain remain reliable when the image source, sensor, geography, or acquisition conditions change.

The research pipeline is:

```text
Classification
      ↓
Domain-shift evaluation
      ↓
Generalization-gap measurement
      ↓
Failure analysis
      ↓
Domain-generalization method
      ↓
External / Gold Data validation
