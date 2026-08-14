# Deep Learning-Based Identification of SSB-PriA Interaction Inhibitors Against Multidrug-Resistant *Klebsiella pneumoniae* Using Graph Neural Networks

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/PyTorch%20Geometric-2.7.0-orange" />
  <img src="https://img.shields.io/badge/RDKit-2026.3.1-green" />
  <img src="https://img.shields.io/badge/License-Academic-lightgrey" />
</p>

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Team](#team)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## Introduction

Antibiotic resistance is one of the most urgent public health crises of the 21st century. Bacteria are evolving faster than new drugs can be developed, and *Klebsiella pneumoniae* sits at the centre of this crisis — a gram-negative pathogen commonly found in healthcare settings that causes severe infections including pneumonia, bloodstream infections, and urinary tract infections. Carbapenem-resistant strains carry a mortality rate of up to **50%**, and the World Health Organization (WHO) has classified them as critical-priority pathogens.

Traditional drug discovery — testing thousands of compounds in the lab — is slow, expensive, and increasingly unable to keep pace with bacterial evolution. This project takes a computational approach: instead of physical lab testing, we use **Graph Neural Networks (GNNs)** to predict which chemical compounds are most likely to block a key bacterial survival mechanism, the **SSB-PriA protein-protein interaction (PPI)**.

SSB (single-stranded DNA-binding protein) protects exposed DNA during replication, while PriA helicase restarts stalled replication forks under bacterial stress. Together, they are essential for bacterial DNA repair and replication. Crucially, this interaction is conserved across gram-negative pathogens and has no functional equivalent in human cells — making it an ideal, selective antibiotic target. Yet no deep learning-based computational pipeline had previously been applied to screen SSB-PriA inhibitors at scale.

This project fills that gap.

---

## Project Overview

**Course:** AIGC-5005 — Capstone Project, Humber Polytechnic  
**Industry Sponsor:** [Living In Silico Inc.](https://www.livinginsilico.com)  
**Group Name:** The Learning Loop   

We developed **FinalUltraModel**, a hybrid deep learning architecture that combines:
- **Graph Attention Networks (GAT)** — to learn local structural patterns from molecular graphs
- **Morgan Fingerprint vectors** — to capture global chemical topology

The model was trained on the **PubChem BioAssay AID 1272365** dataset (~440,000 experimentally tested compounds) and evaluated on a balanced held-out test set of 1,557 molecules, achieving an **AUC-ROC of 0.8800** and **86.06% accuracy**.

---

## Team
 
| Name | Role |
|---|---|
| **Ruchi Shah** | Lead |
| Isha Shah | Member |
| Krutik Babariya | Member |
| Dilkhush Yadav | Member |

---

## Model Architecture

`FinalUltraModel` is a dual-branch hybrid classifier implemented in PyTorch and PyTorch Geometric.

```
Input SMILES
     │
     ├──────────────────────────────┐
     │                              │
[Molecular Graph]           [Morgan Fingerprint]
     │                         (2048-bit, radius=2)
     │                              │
GATConv(10→64, 4 heads)     Linear(2048→512)
+ ELU activation             + ReLU + Dropout(0.5)
     │                              │
GATConv(256→128, 1 head)           │
+ ELU activation                   │
     │                              │
global_mean_pool                   │
(128-dim graph embedding)   (512-dim fingerprint embedding)
     │                              │
     └──────────┬───────────────────┘
                │
     Concatenate (640-dim)
                │
     Linear(640→256) + ReLU + Dropout(0.6)
                │
     Linear(256→2)
                │
     Active / Inactive
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 3×10⁻⁵ |
| Weight Decay | 1×10⁻⁴ |
| Batch Size | 16 |
| Max Epochs | 50 (early stop at 86% accuracy) |
| Loss Function | Cross-Entropy Loss |
| Random Seed | 42 |
| Dropout (fingerprint branch) | 0.5 |
| Dropout (classifier) | 0.6 |

---

## Dataset

**Primary Dataset:** [PubChem BioAssay AID 1272365](https://pubchem.ncbi.nlm.nih.gov/bioassay/1272365)

| Parameter | Value |
|---|---|
| Original entries | ~441,244 compounds |
| Active compounds (original) | ~0.78% of valid entries |
| Balanced subset used | 7,785 molecules |
| Active (balanced) | 2,595 |
| Inactive (balanced) | 5,190 (1:2 ratio) |
| Train / Test split | 6,228 / 1,557 (80/20) |
| Molecular features | 10-dim node features per atom |
| Fingerprint size | 2048-bit Morgan (radius=2) |

**Preprocessing steps:**
1. Remove metadata header rows and records missing SMILES or activity labels
2. Parse SMILES strings with RDKit; invalid molecules discarded
3. Encode atom-level node features: atom type (one-hot: C, N, O, S, F, Cl), degree, aromaticity → 10-dimensional vectors
4. Build bidirectional edge index from bond connectivity
5. Compute 2048-bit Morgan fingerprints using `AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)`
6. Undersample inactive class to 2:1 ratio to address extreme class imbalance

> **Note:** The dataset CSV (`AID_1272365_datatable.csv`) is not included in this repository due to file size. Download it directly from [PubChem](https://pubchem.ncbi.nlm.nih.gov/bioassay/1272365) and place it in the `/Dataset/` folder before running.

---

## Results

Evaluated on a held-out test set of **1,557 molecules**:

| Metric | Value |
|---|---|
| **Accuracy** | 86.06% |
| **Precision** | 0.8102 |
| **Recall** | 0.7216 |
| **F1-Score** | 0.7634 |
| **AUC-ROC** | **0.8800** |

**Confusion Matrix:**

|  | Predicted Inactive | Predicted Active |
|---|---|---|
| **Actual Inactive** | 990 (TN) | 82 (FP) |
| **Actual Active** | 135 (FN) | 350 (TP) |

The model trained for **33 epochs** before reaching the 86% accuracy target. Training loss dropped from 0.6399 (Epoch 1) to 0.0192 (Epoch 33), demonstrating stable and consistent learning throughout.

The AUC-ROC of **0.8800** — well above both random chance (0.5) and the project target of 0.80 — confirms strong discriminative power for ranking active inhibitors over inactive compounds across all classification thresholds.

---

## Installation

**Prerequisites:** Python 3.x, pip or conda

```bash
# Clone the repository
git clone https://github.com/Isha003-hub/TheLearningLoop.git
cd TheLearningLoop

# Install dependencies
pip install pandas numpy rdkit torch torch-geometric scikit-learn matplotlib seaborn tqdm
```

**Dependencies:**

| Library | Version |
|---|---|
| PyTorch | 2.x |
| PyTorch Geometric | 2.7.0 |
| RDKit | 2026.3.1 |
| NumPy | 2.3.5 |
| Pandas | 2.3.3 |
| Scikit-learn | 1.7.2 |
| Matplotlib / Seaborn | Latest |

> GPU is optional but recommended for faster training. The model was originally developed and tested on CPU hardware.

---

## Usage

1. **Download the dataset** from [PubChem AID 1272365](https://pubchem.ncbi.nlm.nih.gov/bioassay/1272365) and place the CSV in `/Dataset/`.

2. **Open and run the notebook:**
```bash
jupyter notebook "Final GAT_GNN Model_The learning Loop.ipynb"
```

3. **Run cells in order:**
   - Cell 1: Install/verify dependencies
   - Cell 2: Import libraries and set seed
   - Cell 4: Load and inspect dataset
   - Cell 5–6: Preprocess data and handle class imbalance
   - Cell 7: Visualize class and molecular weight distributions
   - Cell 8: Convert SMILES to molecular graphs
   - Cell 9: Define `FinalUltraModel` architecture
   - Cell 10: Train model (saves `best_model_8510.pth` when target is hit)
   - Cell 11: Plot training accuracy and loss curves
   - Cell 12–13: Evaluate model and visualize confusion matrix

---

## Project Structure

```
TheLearningLoop/
│
├── Dataset/
│   └── AID_1272365_datatable.csv          # Download from PubChem (not included)
│
├── Final GAT_GNN Model_The learning Loop.ipynb   # Main model notebook
│
├── Reports/
│   ├── AIGC5005_ThelearningLoop_CapstoneProposal.pdf
│   └── IEEE_LearningLoop_FinalPaper.pdf
│
├── best_model_8510.pth                    # Saved model checkpoint (generated on run)
│
└── README.md
```

---

## Limitations

- **No wet-lab validation:** All results are computational predictions; experimentally confirmed activity may differ.
- **Simplified atom features:** Node features encode only atom type, degree, and aromaticity (10 dimensions). Richer features (formal charge, ring membership, implicit valence, hydrogen count) could improve performance.
- **No 3D structural information:** Only 2D molecular graphs and Morgan fingerprints are used. 3D conformations are particularly relevant for PPI binding prediction.
- **Binary classification only:** Continuous inhibition percentages are thresholded to Active/Inactive, which may misclassify borderline compounds.
- **CPU-only training:** Hardware constraints limited architecture search and hyperparameter tuning.
- **No external validation set:** Model was evaluated on an 80/20 split of the same balanced subset; performance on entirely unseen chemical libraries remains to be tested.

---

## Future Work

- Wet-lab validation of top-ranked predicted inhibitors
- Incorporation of 3D molecular descriptors and conformational features
- Exploration of richer atom feature encodings
- Extension to other gram-negative bacterial targets beyond *K. pneumoniae*
- Integration with molecular docking pipelines for downstream validation
- Explainability analysis (attention weight visualization) to identify key pharmacophoric substructures

---

## Acknowledgements

- **Industry Sponsor:** Living In Silico Inc. — for project direction and real-world drug discovery context
- **Data Source:** National Center for Biotechnology Information (NCBI) — PubChem BioAssay AID 1272365
- **Tools:** RDKit, PyTorch, PyTorch Geometric, NCBI PubChem

---

> *This project was completed as part of AIGC-5005: Capstone Project at Humber Polytechnic, Faculty of Applied Sciences & Technology, April 2026.*









---------------------------------------------------

# KB-Predict™ — Graph Neural Network Screening for Drug-Resistant *Klebsiella pneumoniae*

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" />
  <img src="https://img.shields.io/badge/PyTorch%20Geometric-2.7.0-orange" />
  <img src="https://img.shields.io/badge/RDKit-2026.3.1-green" />
  <img src="https://img.shields.io/badge/License-Academic-lightgrey" />
</p>

**The Learning Loop** · AIGC-5005 Capstone, Humber Polytechnic
**Industry Sponsor:** [Living In Silico Inc.](https://livinginsilico.ca) · **Supervisors:** Prof. Timothy Wong, Dr. Asama Nseaf

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
- **Prof. Timothy Wong** and **Dr. Asama Nseaf** — academic supervision, Humber Polytechnic.
- Built with PyTorch, PyTorch Geometric, and RDKit.

---

*AIGC-5005 Capstone Project · Faculty of Applied Sciences & Technology, Humber Polytechnic*
