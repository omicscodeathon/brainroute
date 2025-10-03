# mtoralzml

![Project](https://img.shields.io/badge/Project-mtoralzml-lightblue)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub Repo stars](https://img.shields.io/github/stars/omicscodeathon/mtoralzml)
[![GitHub contributors](https://img.shields.io/github/contributors/omicscodeathon/mtoralzml.svg)](https://GitHub.com/omicscodeathon/mtoralzml/graphs/contributors/)
[![Github tag](https://badgen.net/github/tag/omicscodeathon/mtoralzml)](https://github.com/omicscodeathon/mtoralzml/tags/)

**mtoralzml** is an open-source computational pipeline for predicting blood-brain barrier (BBB) permeability of mTOR inhibitors, with a focus on Alzheimer’s disease drug discovery. The project integrates cheminformatics and machine learning to streamline candidate selection and accelerate CNS drug development.

---

## Overview

NeuroGate

NeuroGate is an interactive platform designed to predict the blood–brain barrier (BBB) permeability of small molecules, addressing a key challenge in central nervous system (CNS) drug discovery. The BBB acts as a highly selective barrier, limiting the entry of therapeutic agents into the brain, which complicates the development of treatments for neurodegenerative diseases.

NeuroGate combines machine learning models and neural network approaches with molecular datasets to predict BBB permeability accurately. Traditional models like K-Nearest Neighbors (KNN) have demonstrated high predictive performance (F1 score: 92%), highlighting the value of classical methods, particularly for limited or complex datasets.

The platform integrates a large language model (LLM) to provide contextual, molecule-specific insights. For each compound, users can ask questions and receive answers informed by predicted BBB permeability, confidence scores, and molecular properties such as molecular weight and lipophilicity. All interactions, predictions, and chat histories can be downloaded as CSV files for offline analysis and research documentation.

NeuroGate offers a user-friendly interface built with Python Streamlit, featuring a responsive design, custom CSS styling, and session-based caching to ensure fast loading of models and prediction results. This makes the platform intuitive, interactive, and suitable for researchers exploring CNS drug candidates.

To showcase its practical utility, NeuroGate was applied to mTOR pathway inhibitors, relevant to Alzheimer’s disease pathology, allowing researchers to rapidly assess BBB permeability and prioritize therapeutic candidates for further investigation.

Overall, NeuroGate provides a comprehensive, interdisciplinary pipeline for accelerating preclinical CNS research, combining predictive modeling with contextual AI-driven insights to support drug discovery.

---

## Table of Contents

- [Objectives](#objectives)
- [Workflow](#workflow)
- [Getting Started](#getting-started)
- [Data Sources](#data-sources)
- [Descriptor Calculation](#descriptor-calculation)
- [Modeling & Analysis](#modeling--analysis)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

---

## Objectives

- [x] Data Integration – Aggregate and harmonize high-quality BBB permeability datasets from multiple public sources.
- [x] Calculate molecular descriptors (RDKit, Mordred)
- [x] Integrate BBB permeability data
- [x] Model Development – Build multiple machine learning architectures for BBB classification, from traditional QSAR approaches to deep learning methods.
- [x] Evaluate machine learning models for BBB prediction
- [x] Uncertainty Quantification – Incorporate prediction confidence and applicability domain detection for safer decision-making.
- [x] Interpretability – Provide mechanistic insights into which molecular features drive BBB penetration predictions.
- [x] Build platform (Web application) to predict BBB permeability
- [x] Platform Deployment – Create a publicly accessible web interface and API for molecule evaluation.
- [x] Domain Demonstration – Apply the platform to mTOR inhibitors and other Alzheimer’s-related compounds as a real-world neurodegenerative disease use case.

---

## Workflow

![workflow flowchart](workflow/neurogate.png).

### 1. Data collection
Molecular data for blood-brain barrier (BBB) penetration were obtained from two publicly available repositories: <a href="https://github.com/theochem/B3DB/blob/main/B3DB/B3DB_regression.tsv">B3DB</a>   and MoleculeNet (moleculenet reference) . Both datasets provide curated annotations of compounds with experimentally determined BBB permeability status (BBB+ vs BBB-). Simplified Molecular-Input Line-Entry System (SMILES)  strings were used as the primary chemical representation. An initial amount of 9857 molecules were pulled from  both data sources, 6523 BBB permeable molecules(BBB+) and 3334 BBB  impermeable molecules(BBB-). 

### 2. Molecular Descriptors Calculations

   - Molecular descriptors of the [molecules](data/B3BD/smiles.smi) were calculated using [RDKIT](https://www.rdkit.org/docs/GettingStartedInPython.html) version 25.03.6.
   - The descriptors were calculated using the ['RDKIT_descriptors'](scripts/RDKIT_descriptors.py) python script.
   - A total of 217 physicochemical and topological descriptors were gotten for each molecule, capturing lipophilicity, polar surface area, molecular weight, hydrogen bonding potential, and other                properties relevant to BBB permeability.

### 3. Data Preprocessing

   - Duplicate smiles were removed
   - entries with completely missing descriptors were dropped
   - Other issing descriptors were filled with zero(0)
   - Data was standardized to have zero mean and unit standard deviation using [StandardScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to enable          models converge quickly.
   - Balanced out BBB+ and BBB- classes in data by generating synthetic data for BBB minority class using [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) 
   - After preprocessing, the number of molecules in data was brought to 12716; 6358 BBB permeable and 6358 BBB impermeable.

### 4. 
     
---
 
## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/mtoralzml.git
   cd mtoralzml
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run descriptor calculation scripts**

   ```bash
   python scripts/extract_smiles.py
   python scripts/RDKIT_descriptors.py
   python scripts/mordred_descr_cal.py
   python scripts/PADEL_descriptor_cal.py
   ```

4. **Explore and analyze data**
   - Use the Jupyter notebooks in `notebooks/` for data curation and model building.

---

## Data Sources

### Core BBB Permeability data

- MoleculeNet BBBp (2,059 compounds, BBB+/– labels)
- PubChem BioAssays (AID 628 and related BBB assays)
- DrugBank CNS/non-CNS drug lists
- Literature datasets of brain/plasma ratios for known drugs

### Domain-Specific Case Study Data

- ChEMBL bioactivity datasets for mTOR inhibitors (Target CHEMBL2842)
- BindingDB mTOR ligand binding affinities
- Alzheimer’s-related small molecule datasets from AD Knowledge Portal and PubChem

---
