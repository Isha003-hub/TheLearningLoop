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
|---|---|---|
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
- **Supervisor:** Dr. Asama Nseaf, Humber Polytechnic — for mentorship and guidance
- **Data Source:** National Center for Biotechnology Information (NCBI) — PubChem BioAssay AID 1272365
- **Tools:** RDKit, PyTorch, PyTorch Geometric, NCBI PubChem

---

> *This project was completed as part of AIGC-5005: Capstone Project at Humber Polytechnic, Faculty of Applied Sciences & Technology, April 2026.*
