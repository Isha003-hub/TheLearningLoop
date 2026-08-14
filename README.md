# KB-Predict™ — Graph Neural Network Screening for Drug-Resistant *Klebsiella pneumoniae*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/PyTorch%20Geometric-2.7.0-orange" />
  <img src="https://img.shields.io/badge/RDKit-2026.3.1-green" />
  <img src="https://img.shields.io/badge/License-Academic-lightgrey" />
</p>

**The Learning Loop** · AIGC-5005 Capstone, Humber Polytechnic
**Industry Sponsor:** [Living In Silico Inc.](https://livinginsilico.ca) · **Supervisor:** Prof. Timothy Wong

An edge-aware Graph Isomorphism Network (GINE) that screens candidate small molecules for
inhibitory activity against the **SSB–PriA protein–protein interaction** — a DNA-replication-repair
target found in multidrug-resistant *Klebsiella pneumoniae*, a WHO critical-priority pathogen with
no prior computational screening pipeline built against it.

The model is deployed live as part of Living In Silico's **KB-Predict™** platform.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Team](#team)
- [The Problem](#the-problem)
- [Dataset](#dataset)
- [Methods](#methods)
  - [1. Data Integrity — Canonical De-duplication](#1-data-integrity--canonical-de-duplication)
  - [2. Featurization](#2-featurization)
  - [3. Model Architecture Evolution](#3-model-architecture-evolution)
  - [4. Class Imbalance Handling](#4-class-imbalance-handling)
  - [5. Training Configuration](#5-training-configuration)
  - [6. Decision Threshold Selection (GHOST)](#6-decision-threshold-selection-ghost)
  - [7. Validation Strategy](#7-validation-strategy)
  - [8. Ensembling](#8-ensembling)
- [Results](#results)
- [Deployment](#deployment)
- [Real-World Validation](#real-world-validation)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Publication](#publication)
- [References](#references)
- [Acknowledgments](#acknowledgments)

---

## Overview

Antibiotic resistance is accelerating faster than new drug development, and carbapenem-resistant
*K. pneumoniae* is ranked the WHO's single highest-priority resistant pathogen, with reported
mortality up to 50% in resistant infections. This project builds a computational triage tool that
ranks candidate compounds for a specific, druggable, low-toxicity-risk target — the SSB–PriA
interaction required for bacterial DNA replication restart — so a wet lab only needs to test the
most promising candidates instead of screening blind.

**Headline result:** 86.10% test accuracy, ROC-AUC 0.8928, independently reproduced from saved
model weights, deployed as a live inference endpoint.

## Team

| Member | Role |
|---|---|
| Ruchi Shah (Lead) | Model architecture, imbalance strategy, evaluation metrics, sponsor requirements |
| Isha Shah | Data analysis, web page design, documentation |
| Krutik Babariya | Azure infrastructure, encoding & training pipeline support |
| Dilkhush Yadav | Model-to-web-app deployment, frontend, bug fixing |

---

## The Problem

| | |
|---|---|
| Target | SSB–PriA protein–protein interaction (DNA replication restart under bacterial stress) |
| Organism | *Klebsiella pneumoniae*, WHO critical-priority multidrug-resistant pathogen |
| Prior computational work on this target | None found |
| Screening library | 441,232 compounds, PubChem BioAssay AID 1272365 |
| True active rate | 0.59% (2,568 confirmed actives) |
| Real-world imbalance | ≈ 1:171 active:inactive |
| Sponsor requirement | Model accuracy > 85% |

## Dataset

- **Source:** [PubChem BioAssay AID 1272365](https://pubchem.ncbi.nlm.nih.gov/bioassay/1272365) —
  "SSB-PriA antibiotic resistant target AlphaScreen," Small Molecule Screening Facility, University
  of Wisconsin–Madison.
- **Raw:** 441,232 compound records, 2,595 raw actives.
- **After canonical SMILES de-duplication:** 2,568 unique confirmed actives (27 duplicates removed),
  13 molecules excluded as ambiguous (conflicting Active/Inactive labels across assay records).
- **Split:** 70% train / 15% validation / 15% test, partitioned on active compounds first, with
  inactives matched at a fixed 2:1 ratio for validation/test and dynamically resampled during
  training. A disjointness assertion runs before every training pass, verifying zero shared
  canonical SMILES across all three partitions.

---

## Methods

### 1. Data Integrity — Canonical De-duplication

Every SMILES string is converted to its RDKit canonical form before any split is made. A molecule
can be written as multiple valid SMILES strings; without canonicalization, duplicate molecules can
leak across train/test partitions and inflate reported performance. An earlier, non-deduplicated
run of this pipeline reported ROC-AUC 0.9011 — the honest, deduplicated result is 0.88. This audit
directly informed every subsequent methodological choice in the project.

### 2. Featurization

**Atom features (35-dim):** element identity (10), heavy-atom degree (6), hydrogen count (5),
formal charge (3), hybridization state (5), **chirality** (4), aromaticity + ring membership (2).

**Bond features (7-dim):** bond order (4), conjugation, ring membership, stereochemistry.

**Global fingerprint:** 2048-bit Morgan fingerprint (radius 2), projected down to 128 dimensions
through a learned bottleneck, preventing the model from using the sparse, high-dimensional
fingerprint as a memorization channel.

### 3. Model Architecture Evolution

**Phase 1 — GAT baseline (`FinalUltraModel`).** `GATConv(10→64, 4 heads)` → `GATConv(256→128, 1 head)`,
fused with a 2048→512 fingerprint branch. Accuracy 86.06%, AUC-ROC 0.8800 on a fixed 1:2
balanced subsample (7,785 molecules). Retained in this repository as the documented Phase 1
baseline.

**Phase 2/3 — Edge-aware GINE (final, deployed model).** Superseded GAT for both theoretical and
empirical reasons: GIN's sum-based neighbor aggregation is provably as expressive as the
Weisfeiler–Leman graph isomorphism test — the theoretical ceiling for this model family — whereas
GAT and GCN's averaging-based aggregation can lose neighbor-count information. GINEConv extends
this to consume bond features directly in message passing, which plain GIN ignores.

**Final architecture:**
- Node/edge encoders projecting 35-dim atom and 7-dim bond features into a shared 128-dim space
- 3× `GINEConv` layers, BatchNorm, ReLU, dropout (0.3), residual connection after layer 1
- Jumping-Knowledge readout: sum- and mean-pooling after **every** layer, concatenated → 768-dim
- Fingerprint bottleneck: 2048-bit Morgan FP → 128-dim
- Joint classifier: 896 → 128 → 64 → 2, BatchNorm + ReLU + dropout (0.5) per layer
- **493,893 trainable parameters** (verified directly from the saved `state_dict`)

### 4. Class Imbalance Handling

- **Dynamic resampling:** rather than a single fixed 1:2 subsample, a fresh random 2:1
  inactive:active sample is drawn every training epoch from a pool of ~58,000 unique inactive
  compounds — measured ~9.5% molecule overlap between consecutive epochs, ~6.7× more unique
  inactive chemistry seen over 10 epochs versus a fixed sample.
- **Focal loss** (γ = 2), replacing plain cross-entropy, to concentrate gradient signal on hard,
  borderline examples rather than letting the easy majority class dominate training.
- Four imbalance strategies were compared empirically on identical data (fixed sample + class
  weights; dynamic resampling + class weights; dynamic resampling, no weights; dynamic resampling +
  focal loss). The focal-loss variant minimized the train/validation overfitting gap (7.57 points)
  and maximized balanced accuracy (81.67%).

### 5. Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3×10⁻⁴ |
| Weight decay | 1×10⁻² |
| Batch size | 32 |
| Max epochs | 40 (early stopping, patience = 8, min Δ = 1×10⁻³) |
| LR scheduler | ReduceLROnPlateau (factor 0.5) on validation PR-AUC |
| Gradient clipping | max norm 2.0 |
| Label smoothing | 0.05 |
| Model selection | best validation PR-AUC (never test) |

### 6. Decision Threshold Selection (GHOST)

The model outputs a probability, not a label. The deployed decision threshold is selected on the
**validation set only**, informed by GHOST (Esposito et al., 2021) — sweeping candidate thresholds
and evaluating Cohen's κ — subject to a validation-accuracy safety margin, so the deployed test
accuracy reliably clears the sponsor's 85% requirement. **Deployed threshold: 0.55.** The threshold
is never tuned against the test set.

### 7. Validation Strategy

- **Train/validation/test gap monitoring** every epoch — final model shows a 0.22-point
  validation-vs-test accuracy gap.
- **5-fold cross-validation** over the full active-compound set: every one of 2,568 actives
  evaluated exactly once, out-of-fold (7,704 pooled predictions vs. 1,158 in a single split).
  Result: **ROC-AUC 0.8842 ± 0.0069** — a tight spread confirming the result is stable, not a
  favorable random split. Cross-validation also revealed the sponsor's original combined target
  (accuracy > 85% *and* recall > 80%) is not simultaneously reachable at any single threshold; the
  model is data-limited (2,568 confirmed actives) rather than architecture-limited.
- **Independent reproduction:** the deployed model's saved weights were reloaded from scratch in a
  separate session and re-evaluated on a freshly reconstructed held-out test set, reproducing the
  reported metrics within 0.1 points.

### 8. Ensembling

Final predictions average the output of **3 independently trained models** (seeds 42, 7, 2024),
reducing variance from individual training runs without changing the architecture.

---

## Results

**Final deployed model, held-out test set (n = 1,158), threshold = 0.55:**

| Metric | Value |
|---|---|
| Accuracy | **86.10%** |
| ROC-AUC | 0.8928 |
| PR-AUC | 0.8549 |
| Recall | 77.20% |
| Precision | 80.32% |
| F1-score | 78.73% |
| Balanced accuracy | 83.87% |
| Cohen's κ | 0.684 |
| Matthews Correlation Coefficient | 0.684 |
| Cross-validated ROC-AUC | 0.8842 ± 0.0069 |

**Confusion matrix (n = 1,158):**

| | Predicted Inactive | Predicted Active |
|---|---|---|
| **Actual Inactive** (772) | 699 (TN) | 73 (FP) |
| **Actual Active** (386) | 88 (FN) | 298 (TP) |

Balanced metrics (κ, MCC) are reported alongside accuracy deliberately: at the true population
imbalance of ≈1:171, a trivial classifier predicting "Inactive" for everything would score ~99.4%
raw accuracy while finding zero real inhibitors.

---

## Deployment

- **Live platform:** [KB-Predict™](https://livinginsilico.ca) on the Living In Silico website, and
  directly at the deployed Azure endpoint.
- **Serving target:** Azure Machine Learning managed online endpoint (Docker container), CPU
  inference — the model is small enough (~494K parameters) that GPU adds cold-start latency with no
  real benefit.
- **Features:** single-compound SMILES lookup, and bulk CSV upload/download for batch screening.
- **Model artifact:** a single checkpoint containing the 3-seed ensemble `state_dict`s (not a
  pickled model object — more portable across library versions), architecture hyperparameters,
  the RDKit/featurizer version, and the locked decision threshold, so the scoring endpoint
  reproduces training-time featurization exactly. `model.eval()` is called once at container
  startup for deterministic inference.

## Real-World Validation

Beyond held-out test metrics, the deployed model was used to screen a real, independently sourced
commercial compound library (15,538 compounds, Life Chemicals antibacterial-focused catalog) —
producing candidate rankings on molecules the model never saw during training or validation.

---

## Repository Structure

```
.
├── notebooks/
│   ├── Final_GAT_GNN_Model_The_learning_Loop.ipynb     # Phase 1 baseline (GAT)
│   └── GIN_Final_Azure_TheLearningLoop.ipynb           # Final GINE pipeline, training -> deployment
├── models/
│   ├── gin_kpne_deploy.pth       # 3-seed ensemble checkpoint (deployed)
│   ├── gin_kpne_single.pth       # single best-seed checkpoint
│   └── gin_model_info.json       # metadata: hyperparameters, threshold, metrics
├── data/
│   └── AID_1272365_datatable.csv # PubChem source data
├── docs/
│   └── IEEE_LearningLoop_FinalPaper.docx
└── README.md
```

*(Adjust paths above to match your actual repository layout.)*

## Usage

```bash
pip install torch torch_geometric rdkit pandas numpy scikit-learn --break-system-packages

# Predict on a single molecule
python predict.py --smiles "CC1=CC(=NNC2=CC=CC=C2[N+](=O)[O-])CC(C1)(C)C"

# Batch predict from a CSV of SMILES
python predict.py --csv compounds.csv --output results.csv
```

The prediction script loads `gin_kpne_deploy.pth`, rebuilds the exact training-time featurization
(35-dim atom features, 7-dim bond features, 2048-bit Morgan fingerprint), runs the 3-seed ensemble,
and applies the locked 0.55 decision threshold.

## Publication

This work is being prepared for academic publication in partnership with Living In Silico Inc.

## References

1. K. Xu, W. Hu, J. Leskovec, and S. Jegelka, "How Powerful are Graph Neural Networks?," *ICLR*, 2019.
2. W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. Pande, and J. Leskovec, "Strategies for
   Pre-training Graph Neural Networks," *ICLR*, 2020.
3. C. Esposito, G. A. Landrum, N. Schneider, N. Stiefl, and S. Riniker, "GHOST: Adjusting the
   Decision Threshold to Handle Imbalanced Data in Machine Learning," *J. Chem. Inf. Model.*, 61(6),
   2021.
4. D. Chicco and G. Jurman, "The Advantages of the Matthews Correlation Coefficient (MCC) over F1
   Score and Accuracy in Binary Classification Evaluation," *BMC Genomics*, 21, 2020.
5. A. Bender et al., "Artificial Intelligence in Drug Discovery — What It Is, Where We Stand and
   the Path Forward," *Nature Reviews Drug Discovery*, 2026.
6. PubChem BioAssay AID 1272365, National Center for Biotechnology Information, U.S. National
   Library of Medicine.

*(See the full manuscript for the complete reference list.)*

## Acknowledgments

- **Living In Silico Inc.** — industry sponsorship, project direction, and deployment platform.
- **Prof. Timothy Wong** — academic supervision, Humber Polytechnic.
- Built with PyTorch, PyTorch Geometric, and RDKit.

---

*AIGC-5005 Capstone Project · Faculty of Applied Sciences & Technology, Humber Polytechnic*
