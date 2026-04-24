# DDLC-RSLAD CIFAR-10 Experiment Code

This repository provides the implementation of RSLAD and its enhanced version on the CIFAR-10 dataset. 
The primary goal is to demonstrate the effectiveness of our proposed enhancement strategy (DDLC) 
and to serve as a reproducible baseline for comparative experiments.

---

## Project Structure

- `rslad_cifar10.py`  
  Implementation of the original RSLAD training pipeline.  
  This serves as the baseline method for comparison.

- `rslad_ddlc_cifar10.py`  
  Implementation of the enhanced training pipeline with DDLC.  
  This is the main method proposed in our work.

- `rslad_loss.py`  
  Contains adversarial example generation (e.g., PGD) and loss computation used in RSLAD.

- `extra_data.py`  
  Implements the loading and mixing strategy for additional generated data used in DDLC.

- `L_pearson.py`  
  Implementation of the Linear Correlation Matching (LCM) loss.  
  This module computes the correlation-based alignment between teacher and student outputs, 
  encouraging the student model to preserve the relative structural relationships learned by the teacher.

---

## Method Overview

### Baseline: RSLAD

The script `rslad_cifar10.py` follows the original RSLAD training procedure and is used as a baseline 
to evaluate the effectiveness of our enhancement strategy.

### Proposed Method: RSLAD + DDLC

The script `rslad_ddlc_cifar10.py` implements the enhanced method with DDLC, which introduces:

- Data-driven augmentation using generated samples
- Structural alignment constraints between teacher and student models

This version is the core contribution of our work and is used in all main experimental results.

---

## Customizing Teacher and Student Models

Users can easily modify the teacher-student configuration by editing the training scripts.

Typical modifications include:

- Changing the **student model**  
  ```python
  student = resnet18()
  # → replace with
  student = mobilenet_v2()
