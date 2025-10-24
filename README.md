# BrainRoute

![Project](https://img.shields.io/badge/Project-mtoralzml-lightblue)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![GitHub Repo stars](https://img.shields.io/github/stars/omicscodeathon/mtoralzml)
[![GitHub contributors](https://img.shields.io/github/contributors/omicscodeathon/mtoralzml.svg)](https://GitHub.com/omicscodeathon/mtoralzml/graphs/contributors/)
[![Github tag](https://badgen.net/github/tag/omicscodeathon/mtoralzml)](https://github.com/omicscodeathon/mtoralzml/tags/)

**BrainRoute** is an open-source computational platform designed to predict blood–brain barrier (BBB) permeability of drug candidates, leveraging artificial intelligence to accelerate drug discovery. The project serves as a resource for advancing research in central nervous system (CNS) therapeutics and supporting the development of novel treatment strategies. 

---

## Overview

BrainRoute is an interactive platform designed to predict the blood–brain barrier (BBB) permeability of small molecules, addressing a key challenge in central nervous system (CNS) drug discovery. The BBB acts as a highly selective barrier, limiting the entry of therapeutic agents into the brain, which complicates the development of treatments for neurodegenerative diseases.

BrainRoute combines machine learning models and neural network approaches with molecular datasets to predict BBB permeability accurately. Traditional models like K-Nearest Neighbors (KNN) have demonstrated high predictive performance (F1 score: 92%), highlighting the value of classical methods, particularly for limited or complex datasets.

The platform integrates a large language model (LLM) to provide contextual, molecule-specific insights. For each compound, users can ask questions and receive answers informed by predicted BBB permeability, confidence scores, and molecular properties such as molecular weight and lipophilicity. All interactions, predictions, and chat histories can be downloaded as CSV files for offline analysis and research documentation.

BrainRoute offers a user-friendly interface built with Python Streamlit, featuring a responsive design, custom CSS styling, and session-based caching to ensure fast loading of models and prediction results. This makes the platform intuitive, interactive, and suitable for researchers exploring CNS drug candidates.

To showcase its practical utility, BrainRoute was applied to mTOR pathway inhibitors, relevant to Alzheimer’s disease pathology, allowing researchers to rapidly assess BBB permeability and prioritize therapeutic candidates for further investigation.

Overall, BrainRoute provides a comprehensive, interdisciplinary pipeline for accelerating preclinical CNS research, combining predictive modeling with contextual AI-driven insights to support drug discovery.

---

## Table of Contents

- [Objectives](#objectives)
- [Workflow](#workflow)
  - [Data collection](#1-data-collection)
  - [Molecular Descriptors Calculations](#2-molecular-descriptors-calculations)
  - [Data Preprocessing](#3-data-preprocessing)
  - [Model Development](#m4-odel-development)
  - [platform Development](#5-platform-development)
- [Getting Started](#getting-started)
- [Data Sources](#data-sources)
- [Modeling & Analysis](#modeling--analysis)
- [Results](#results)
- [Reproducibility](#reproducibility)
- [Domain-Specific Case Study Data](#case-study)
- [Contributions](#contributions)
- [License](#license)
- [Citation](#citation)

---

## Objectives

- [x] Develop **BrainRoute**, a predictive pipeline for assessing blood–brain barrier (BBB) permeability of small molecules in CNS drug discovery.  
- [x] Aggregate and harmonize high-quality BBB datasets and calculate molecular descriptors (RDKit, Mordred).  
- [x] Build and evaluate predictive models using **machine learning and neural network approaches**.  
- [x] Provide **uncertainty estimates and interpretability features**, highlighting which molecular properties drive predictions.  
- [x] Create an **interactive web platform** with a chat feature, enabling researchers to explore predictions, confidence scores, and contextual insights about molecules of interest.  
- [x] Demonstrate utility through application to **mTOR inhibitors** as a case study in neurodegenerative disease research.  


---

## Workflow

![workflow flowchart](workflow/BrainRoute.png)


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

### 4. Model Development

- The dataset was split to training set (which was further split during training of XGBoost and DL models 80:20 fo rtrain and test) and test set (which served as an external data) in the ration 80:20 respectively'
- An initial benchmark was performed with [LazyPredict](https://pypi.org/project/lazypredict/), which quickly screened a wide range of algorithms to identify high-performing candidates from Accuracy, balanced Accuracy, ROC-AUC and F1 score. Based on this evaluation, targeted [models](output/models) were developed and tuned, including:
  
    * K-Nearest Neighbors (KNN)
    * Support Vector Machine (SVM)
    * Random Forests (RF)
    * Logistic Regression (LR)
    * XGBoost (XGB)
    * Multilayer Perceptron (MLP) (implemented with TensorFlow)
     
- The data was split using [Stratified k-fold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) splitting  to preserve the percentage of samples for each class. Models were cross validated and performance was evaluated on a held-out test set, with metrics including Accuracy, F1-score, Precision, Recall, Specificity, ROC-AUC, PR-AUC and matthews correlation coef(MCC) [metrics](figures/model_plots/eval_metrics.png)
- The model's ability to generalize on unseen data was assessed on the test data(external data) using the F1-score, Precision, Recall, ROC-AUC, PR-AUC and matthews correlation coef(MCC) [metrics](figures/model_plots/gen_metrics)
- Among classical models, KNN and XGB achieved particularly strong performance (F1-score: 0.91), highlighting the effectiveness of traditional approaches for BBB classification when working with limited and heterogeneous datasets.

This workflow integrates classical QSAR methods with modern neural architectures, enabling both predictive robustness and biological interpretability in BBB permeability prediction task.


### 5. Platform Development

- The main user interface platform was built using the [Streamlit library]([reference](https://streamlit.io/)) available for the Python language. To retrieve information about the user input molecule, we used ChEMBL web services [API](https://www.ebi.ac.uk/chembl/) and PubChem [PUG REST API](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest).
- The platform user interface was developed using the Python Streamlit library, enabling easy deployment and interactive web applications. We used custom CSS styling to create a visually appealing and organized layout.
- The Platform works as such;
        - takes as input SMILES or Compound name
        - calculate descriptors and grabs information about compound from CHEMBL and PubChem at the background
        - Predict BBB permeability of compound
        - performs Uncertainty quantification 
        - displays results and relevant information about compound
        - provides option for batch processing of compounds/smiles, download of prediction results etc.
- To help users learn more about the molecule of interest, we integrated a Large Language Model (LLM) namedly [Llama3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) with our platform. Users can prompt the model to explain results, for information about molecules and other details of interest to user.
- Llama model params were set to a max of 250 tokes, to comply with our free API usage tier while ensuring necessary responses, temperature to 0.7 to introduce some variability in responses, while maintaining factual consistency and  top-p to 0.9, enabling us to balance diversity and coherence in model responses.


---
 
## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/mtoralzml.git
   cd mtoralzml
   ```

2. **Install dependencies**

Intel - 

   ```bash
   pip install -r requirements.txt
   ```
Apple silicon - 

 ```bash
   pip install -r requirements_macos.txt
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

--- 

## Results 

Our evaluation demonstrated strong predictive performance across multiple models for BBB permeability classification. The K-Nearest Neighbors (KNN) model achieved the highest performance with an F1-score of 0.92, while other approaches such as Random Forests and Neural Networks also showed reliable accuracy and robustness.

Beyond predictive modeling, we successfully developed and deployed BrainRoute as an interactive web platform. The system integrates with external resources including the Hugging Face inference API (for large language model–driven insights), ChEMBL (for bioactivity data), and the PubMed API (for literature retrieval). The interface, built in Streamlit and deployed via Hugging Face Spaces, enables researchers to evaluate molecules, explore predictions, and interact with contextual LLM-powered explanations in a user-friendly environment.

## Reproducibility

The workflow implemented in BrainRoute is fully reproducible. Users can follow the steps outlined below to clone the repository, install dependencies, and run the provided scripts and notebooks to recreate the modeling pipeline or extend it with new datasets and features. The platform has been designed for transparency and modularity, making it easy for researchers to build upon and adapt for their own projects. We also plan to continue improving the platform by refining models, expanding datasets, and enhancing the interactive interface.

Please install all requirements based on your machine with the instructions mentioned above. 


## Case Study

The Mtor cellular pathway is a crucial pathway that acts as a central regulator for cell metabolism, growth, proliferation, and survival. Because it integrates signals from various upstream pathways (like growth factors and nutrients), inhibiting the main mTOR kinase can simultaneously block numerous downstream cellular processes. This makes it a powerful target for therapies, particularly in oncology. However, this broad inhibition can also lead to significant side effects, trigger complex feedback loops within cells, or result in only partial pathway inhibition.

The study concludes that none of the 26 approved mTOR inhibitors analyzed are predicted to cross the Blood-Brain Barrier, highlighting a major challenge in developing drugs for brain-related conditions that involve this pathway.

Technical Specifications & Workflow
The analysis in the notebook is carried out using Python with several key libraries:

- pandas: Used for data manipulation and to load the dataset of mTOR inhibitors.
- rdkit: A cheminformatics toolkit used to process the chemical structures from their SMILES strings and calculate molecular descriptors.
- joblib: Used to load pre-trained machine learning models.
- tensorflow.keras: Employed for loading a pre-trained Multi-Layer Perceptron (MLP) neural network model.

The workflow involves loading the molecules, calculating their chemical properties (descriptors), and then feeding these properties into various predictive models (KNN, SVM, Random Forest, etc.) to assess BBB permeability. 
**The batch processing functionality of the platform allows users to complete such studies with the UI itself. Users can upload CSV files and generate cumulative results.**

## Contributions 

## License

## Citation


---
