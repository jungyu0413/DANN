# Process vs. Equipment Fault Classification using Domain Adaptation  
**Project: SK hynix Semiconductor Process Fast Track Program 2**

---

## Overview

This repository presents a deep learning-based classification system designed to analyze causes of failure in semiconductor wafer processes.  
The goal is to develop a model that can **efficiently distinguish between Process-related and Equipment-related anomalies**, improving diagnostic accuracy and operational efficiency in semiconductor fabs.

---

## Research Highlights

- In real-world semiconductor facilities, it is challenging to build a separate model for each individual piece of equipment due to environment diversity and cost.
- In collaboration with domain experts, we define fault types as:
  - **Process-related** (공정 기인)
  - **Equipment-related** (장비 기인)
- The model is trained using domain-specific pretraining and enhanced via **Domain Adaptation**, which helps generalize across devices despite pattern differences.

---

## Approach & Model

- Pretraining on domain-specific datasets followed by adversarial domain adaptation
- Pattern difference across devices is mitigated using gradient reversal layers
- Final model is able to classify shared-fault causes across heterogeneous equipment setups

---

## Key Results

- Achieved **82.7% accuracy** on real field test data
- Successfully deployed to production environments
- Reduced the need to maintain multiple per-equipment models
- Improved generalization and **reduced operational & maintenance cost**

---
