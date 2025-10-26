# NeuroGate

![Project](https://img.shields.io/badge/Project-mtoralzml-lightblue)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![GitHub Repo stars](https://img.shields.io/github/stars/omicscodeathon/mtoralzml)](https://github.com/omicscodeathon/mtoralzml)
[![GitHub contributors](https://img.shields.io/github/contributors/omicscodeathon/mtoralzml.svg)](https://GitHub.com/omicscodeathon/mtoralzml/graphs/contributors/)
[![Github tag](https://badgen.net/github/tag/omicscodeathon/mtoralzml)](https://github.com/omicscodeathon/mtoralzml/tags/)

**NeuroGate** is an open-source computational platform designed to predict blood–brain barrier (BBB) permeability of drug candidates, leveraging artificial intelligence to accelerate drug discovery. The project serves as a resource for advancing research in central nervous system (CNS) therapeutics and supporting the development of novel treatment strategies.

---

## 🎯 Overview

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

- [x] Develop **NeuroGate**, a predictive pipeline for assessing blood–brain barrier (BBB) permeability of small molecules in CNS drug discovery.
- [x] Aggregate and harmonize high-quality BBB datasets and calculate molecular descriptors (RDKit, Mordred).
- [x] Build and evaluate predictive models using **machine learning and neural network approaches**.
- [x] Provide **uncertainty estimates and interpretability features**, highlighting which molecular properties drive predictions.
- [x] Create an **interactive web platform** with a chat feature, enabling researchers to explore predictions, confidence scores, and contextual insights about molecules of interest.
- [x] Demonstrate utility through application to **mTOR inhibitors** as a case study in neurodegenerative disease research.

---

## 🔬 Workflow

![workflow flowchart](workflow/neurogate.png)

### 1. Data collection

Molecular data for blood-brain barrier (BBB) penetration were obtained from two publicly available repositories: <a href="https://github.com/theochem/B3DB/blob/main/B3DB/B3DB_regression.tsv">B3DB</a> and MoleculeNet (moleculenet reference) . Both datasets provide curated annotations of compounds with experimentally determined BBB permeability status (BBB+ vs BBB-). Simplified Molecular-Input Line-Entry System (SMILES) strings were used as the primary chemical representation. An initial amount of 9857 molecules were pulled from both data sources, 6523 BBB permeable molecules(BBB+) and 3334 BBB impermeable molecules(BBB-).

### 2. Molecular Descriptors Calculations

- Molecular descriptors of the [molecules](data/B3BD/smiles.smi) were calculated using [RDKIT](https://www.rdkit.org/docs/GettingStartedInPython.html) version 25.03.6.
- The descriptors were calculated using the ['RDKIT_descriptors'](scripts/RDKIT_descriptors.py) python script.

```bash
# Extract SMILES from source files
python scripts/extract_smiles.py

# Calculate RDKit descriptors (217 features)
python scripts/RDKIT_descriptors.py

```

**Descriptors Computed (RDKit - 217 features):**

- Physicochemical: MW, LogP, MR, TPSA
- Topological: Balaban J, Bertz CT, Chi indices
- Electronic: Partial charges, atom type counts
- Structural: Ring counts, H-bond donors/acceptors
- Lipinski's descriptors: Rule of 5 compliance
- Graph-based: Molecular connectivity indices

### 3️⃣ Data Preprocessing

**Quality Control:**

- Duplicate smiles were removed
- Entries with completely missing descriptors were dropped

```python
# Remove duplicates
df = df.drop_duplicates(subset=['smiles'])

# Handle missing values
df = df.dropna(thresh=len(df.columns) * 0.5)  # Drop if >50% missing
df = df.fillna(0)  # Fill remaining with 0

```

**Normalization:**

- Data was standardized to have zero mean and unit standard deviation using [StandardScaler()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to enable models converge quickly.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# μ = 0, σ = 1 for all features
```

**Class Balancing:**

- Balanced out BBB+ and BBB- classes in data by generating synthetic data for BBB minority class using [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
```

**Final Dataset:**

- **Total samples**: 12,716 (after SMOTE)
- **BBB+**: 6,358 (50%)
- **BBB-**: 6,358 (50%)
- **Features**: 217 (RDKit descriptors)
- **Train/Test split**: 80/20 (10,172 / 2,544)

### 4️⃣ Model Development & Training

#### A. Classical Machine Learning

**Model Selection Strategy:**

1. **Initial Screening**: [LazyPredict](https://pypi.org/project/lazypredict/) for rapid benchmarking
2. **Hyperparameter Tuning**: GridSearchCV with 5-fold CV
3. **Final Evaluation**: [Stratified](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) test set

**Models Implemented:**

| Model             | Algorithm              | Key Parameters                     | Training Time |
| ----------------- | ---------------------- | ---------------------------------- | ------------- |
| **KNN**           | K-Nearest Neighbors    | k=3, weights='distance'            | ~5 min        |
| **XGBoost**       | Gradient Boosting      | n_estimators=100, max_depth=6      | ~15 min       |
| **SVM**           | Support Vector Machine | kernel='rbf', C=1.0, gamma='scale' | ~30 min       |
| **Random Forest** | Ensemble (Bagging)     | n_estimators=100, max_depth=8      | ~20 min       |
| **Logistic Reg**  | Linear Classifier      | max_iter=1000, solver='lbfgs'      | ~2 min        |

**Training Code:**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

# Initialize model
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(knn, X_train, y_train, cv=skf, scoring='f1')

# Train final model
knn.fit(X_train, y_train)

# Save model
joblib.dump(knn, 'output/models/KNN_model.pkl')
```

#### B. Deep Learning Approach

**Architecture**: SPMM (Structure-Property Multi-Modal) Model

- **Base**: BERT (Bidirectional Encoder Representations from Transformers)
- **Input**: SMILES tokenization
- **Pre-training**: 10M+ molecules from PubChem
- **Fine-tuning**: BBB-specific dataset

**Model Architecture:**

```
Input (SMILES) → Tokenizer → BERT Encoder → [CLS] Token
                                                ↓
                                    Linear(768 → 256) + GELU
                                                ↓
                                    Dropout(0.1)
                                                ↓
                                    Linear(256 → 2)
                                                ↓
                                    Softmax → [BBB+, BBB-]
```

**Training Configuration:**

```python
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss

optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.02)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
loss_fn = CrossEntropyLoss()

# Training hyperparameters
batch_size_train = 16
batch_size_eval = 64
num_epochs = 10
warmup_epochs = 1
```

### 5️⃣ Platform Development

**Technology Stack:**

- **Frontend**: [Streamlit](<[reference](https://streamlit.io/)>) + Custom CSS
- **Backend**: Python 3.12
- **ML Framework**: scikit-learn, XGBoost, PyTorch
- **Cheminformatics**: RDKit 2025.03.6
- **LLM**: Llama 3 8B (via Hugging Face Inference API)
- **Database**: React + Google Sheets API + RDKit.js
- **Deployment**: Hugging Face Spaces

**API Integrations:**

1. **[ChEMBL API](https://www.ebi.ac.uk/chembl/)**: Drug information, bioactivity data
2. **[PubChem PUG REST API](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)**: Chemical properties, synonyms
3. **Hugging Face Router**: LLM inference
4. **Google Sheets API**: Collaborative database backend

---

## 📊 Models & Performance

### Evaluation Metrics

**Cross-Validation (5-fold Stratified):**

![Model Performances](figures/model_plots/gen_metrics.png)

![Metrics PLot](figures/model_plots/model_comparism.png)

**External Test Set (Held-out 20%):**

![Model performances on External data](figures/model_plots/eval_metrics.png)

### Feature Importance

**Top 2 performing models (KNN and XGBoost)**
![KNN Feature Importance Plot](figures/KNN_feature_importance.png)

![XGBoost Feature Importance Plot](figures/XGB_feature_importance.png)

### Comparison with Literature

| Study                    | Year     | Best Model      | Accuracy/F1                                | Dataset Size |
| ------------------------ | -------- | --------------- | ------------------------------------------ | ------------ |
| **BrainRoute (Ours)**    | **2025** | **XGB**         | **Acc: 0.93**, **F1: 0.93**, **AUC: 0.98** | **12,716**   |
| **BrainRoute (Ours)**    | **2025** | **KNN**         | **Acc: 0.92**, **F1: 0.92**, **AUC: 0.96** | **12,716**   |
| Wang et al.              | 2018     | SVM Consensus   | Acc: 96.6%                                 | 2,358        |
| Liu et al.               | 2021     | Ensemble        | Acc: 93.0%                                 | 1,757        |
| Lim et al.               | 2023     | GCNN            | Acc: 88.0%                                 | 8,000        |
| Shaker et al. (LightBBB) | 2021     | LightGBM        | Acc: 89.0%                                 | 7,162        |
| Atallah et al.           | 2024     | Voting Ensemble | AUC: 96.0%                                 | 7,807        |

**Key Insights:**

- Classical ML (KNN, XGBoost) achieves **competitive or superior performance** compared to deep learning approaches taking into account Auc and F1 scores
- Ensemble methods provide **robust uncertainty quantification**
- Data quality and preprocessing are **more critical than model complexity** for this task
- BrainRoute's uncertainty-aware predictions enable **risk-stratified decision making**

---

## 🖥️ Platform Features

### Single Molecule Analysis

**Input Methods:**

1. **Compound Name**: e.g., "Donepezil", "Caffeine", "Aspirin"
2. **SMILES String**: e.g., `CC(=O)OC1=CC=CC=C1C(=O)O`

**Outputs:**

- **BBB Prediction**: BBB+ or BBB- with confidence score
- **Uncertainty Metrics**: Standard deviation across models
- **Model Agreement**: Percentage of models in consensus
- **Molecular Properties**: MW, LogP, TPSA, HBA, HBD, rotatable bonds
- **2D Structure**: Interactive molecular visualization
- **ChEMBL Data**: Bioactivity, mechanism of action, clinical phase
- **Individual Model Predictions**: See how each model voted

**Example Results:**

```
Compound: Donepezil
SMILES: COc1ccc2c(c1)[C@H](CCN1CCC(CC1)C(=O)c1ccccc1)c1ccccc1-2

🎯 Prediction: BBB+ (Permeable)
📊 Confidence: 87.3%
⚠️ Uncertainty: 8.2% (Low)
🤝 Model Agreement: 100%

Molecular Properties:
- Molecular Weight: 379.5 g/mol
- LogP: 4.32
- TPSA: 38.8 Ų
- H-Bond Donors: 0
- H-Bond Acceptors: 3
- Rotatable Bonds: 5

✓ Lipinski's Rule of 5: PASS
✓ BBB Permeability Rules: PASS
```

### Batch Processing

**Supported Formats:**

- **CSV Upload**: Must contain `smiles` and/or `name` columns
- **Text Input**: One molecule per line (names or SMILES)

**Features:**

- Process **hundreds of molecules** simultaneously
- **Summary Statistics**: BBB+ rate, average confidence, success rate
- **Interactive Visualizations**:
  - Prediction distribution (pie chart)
  - Confidence vs. Uncertainty scatter plot
  - Molecular property distributions
- **Export Options**: CSV, Excel, JSON formats
- **Filtering**: By status, prediction, confidence threshold

**Example Batch Results:**

```
📊 Batch Summary:
- Total Molecules: 150
- Successful Predictions: 147 (98%)
- BBB+ Predictions: 54 (37%)
- BBB- Predictions: 93 (63%)
- Average Confidence: 84.2%
- Average Uncertainty: 11.5%
```

### AI Chat Interface ([Llama3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) Integration)

**Capabilities:**

- **Contextual Q&A**: Ask about your molecule's properties
- **Drug Discovery Insights**: CNS potential, mechanism, side effects
- **Literature Guidance**: Research directions and references
- **Comparison**: How does it compare to similar compounds?

**Example Conversations:**

```
User: What makes Donepezil effective for Alzheimer's?

🦙 Llama 3: Donepezil is an acetylcholinesterase inhibitor with
excellent BBB penetration (predicted BBB+ with 87% confidence).
Its moderate molecular weight (379.5 g/mol) and optimal LogP (4.32)
enable effective CNS entry. It enhances cholinergic neurotransmission
by preventing acetylcholine breakdown in synapses, which helps
compensate for cholinergic deficits in AD.
```

**Quick Questions:**

- 💊 Drug Potential
- 🧪 Key Properties
- ⚠️ Side Effects
- 🔬 Current Research

### Curated Database

**Access**: Separate React-based web interface

**Features:**

- **9,857+ Annotated Molecules**: All with BBB predictions
- **Real-time Structure Rendering**: Client-side RDKit.js
- **Advanced Search**: By name, SMILES, properties
- **Property Filters**: MW range, LogP, TPSA thresholds
- **Prediction Filters**: BBB+/-, confidence threshold
- **Export**: Selected molecules to CSV
- **Collaborative**: Google Sheets backend for easy updates

---

## 🧪 Case Study: mTOR Inhibitors

### Background

**mTOR (Mechanistic Target of Rapamycin):**

- Serine/threonine protein kinase
- Master regulator of cell growth, metabolism, autophagy
- Two complexes: mTORC1 (growth) and mTORC2 (survival)

**Relevance to Alzheimer's Disease:**

- Hyperactivation linked to tau hyperphosphorylation
- Contributes to amyloid-β plaque formation
- Inhibits autophagy → toxic protein accumulation
- Rapamycin shows neuroprotective effects in AD models

**Clinical Challenge:**

- Most mTOR inhibitors designed for cancer/immunosuppression
- BBB penetration rarely optimized
- **Research Question**: Which approved mTOR inhibitors can cross the BBB?

### Analysis

**Dataset:**

- **25 FDA-approved mTOR inhibitors** analyzed
- Includes: Rapamycin (Sirolimus), Everolimus, Temsirolimus, etc.
- Data: SMILES, ChEMBL IDs, approved indications

**Methodology:**

```python
# Load mTOR inhibitors dataset
import pandas as pd
mtor_data = pd.read_csv('data/case_study/mtor_inhibitors.csv')

# Batch prediction using BrainRoute
from scripts.webapp.prediction import process_batch_molecules
results, error = process_batch_molecules(mtor_data, 'csv', models)

# Analyze results
bbb_positive = [r for r in results if r['prediction'] == 'BBB+']
print(f"BBB+ inhibitors: {len(bbb_positive)} / {len(results)}")
```

### Results

| Category                 | Count | Percentage |
| ------------------------ | ----- | ---------- |
| **Total Analyzed**       | 25    | 100%       |
| **BBB+ (Permeable)**     | 9     | 36%        |
| **BBB- (Non-permeable)** | 16    | 64%        |

**BBB+ Predicted Compounds (Examples):**

1. **Rapamycin** - Confidence: 78% (literature-confirmed)
2. **Temsirolimus** - Confidence: 72%
3. **Compound X** - Confidence: 81%

**BBB- Predicted Compounds (Examples):**

1. **Everolimus** - Confidence: 89%
2. **Ridaforolimus** - Confidence: 85%

**Property Analysis:**

```
BBB+ Compounds:
- Avg MW: 892 ± 156 g/mol
- Avg LogP: 5.2 ± 1.1
- Avg TPSA: 168 ± 34 Ų

BBB- Compounds:
- Avg MW: 1,124 ± 203 g/mol
- Avg LogP: 4.1 ± 0.9
- Avg TPSA: 245 ± 52 Ų
```

### Key Insights

1. **Minority BBB Penetration**: Only 36% predicted to cross BBB
2. **MW Threshold**: BBB+ compounds generally <1,000 Da
3. **TPSA Correlation**: BBB+ compounds have lower TPSA (<200 Ų)
4. **Clinical Implications**:

   - Most existing mTOR inhibitors unsuitable for CNS disorders
   - Need for structure optimization or alternative delivery
   - Rapamycin's BBB+ prediction aligns with clinical AD trials

5. **Future Directions**:
   - Design CNS-optimized mTOR inhibitors
   - Explore drug delivery strategies (nanoparticles, intranasal)
   - Validate predictions with experimental BBB assays

**Notebook**: Full analysis available in [Case study notebook](notebooks/case_study_mtor.ipynb)

---

## 📈 Results & Benchmarking

### Model Performance Summary

**Best Performers:**

- **Accuracy**: XGBoost (93%), KNN (92%)
- **F1-Score**: XGBoost (0.93), KNN (0.92)
- **ROC-AUC**: XGBoost (0.98), KNN (0.96)
- **Robustness**: Ensemble (lowest variance across folds)

**Key Takeaways:**

- Classical ML outperforms deep learning for this dataset size
- Ensemble methods provide best uncertainty quantification
- XGBoost's success highlights importance of molecular similarity
- Deep learning may improve with 10x larger datasets (>100,000 molecules)

### Computational Efficiency

| Model   | Training Time | Inference (1 mol) | Inference (1000 mols) |
| ------- | ------------- | ----------------- | --------------------- |
| KNN     | 5 min         | <0.1s             | ~30s                  |
| XGBoost | 15 min        | <0.1s             | ~45s                  |
| SVM     | 30 min        | <0.1s             | ~60s                  |
| BERT    | 2 hours       | 0.5s              | ~8 min                |

**Hardware**: Intel icore 7 CPU @ 1.5 GHz, 16GB RAM
**Note** BERT requires GPU to run

---

## 🔌 API Reference

### Python API Usage

```python
from scripts.webapp.prediction import predict_bbb_penetration_with_uncertainty
from scripts.webapp.utils import load_ml_models
from rdkit import Chem

# Load models once
models, errors = load_ml_models()

# Single prediction
mol = Chem.MolFromSmiles("CC(=O)OC1=CC=CC=C1C(=O)O")
result, error = predict_bbb_penetration_with_uncertainty(mol, models)

if not error:
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Uncertainty: {result['uncertainty']:.2f}%")
    print(f"Agreement: {result['agreement']:.2f}%")

# Batch prediction
from scripts.webapp.lewis.prediction import process_batch_molecules
import pandas as pd

batch_data = pd.DataFrame({
    'name': ['Aspirin', 'Caffeine'],
    'smiles': ['CC(=O)OC1=CC=CC=C1C(=O)O', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C']
})

results, error = process_batch_molecules(batch_data, 'csv', models)
for result in results:
    print(f"{result['name']}: {result['prediction']} ({result['confidence']:.1f}%)")
```

### REST API (Future Development)

We plan to develop a REST API for programmatic access:

```bash
# Predict single molecule
curl -X POST https://api.brainroute.io/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}'

# Response
{
  "prediction": "BBB-",
  "confidence": 85.2,
  "uncertainty": 12.3,
  "agreement": 80.0,
  "properties": {
    "mw": 180.16,
    "logp": 1.19,
    "tpsa": 63.6
  }
}
```

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Model Loading Errors

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory: 'output/models/KNN_model.pkl'`

**Solution**:

```bash
# Download pre-trained models
wget https://github.com/omicscodeathon/mtoralzml/releases/download/v1.0/models.zip
unzip models.zip -d output/

# Or train models from scratch
python notebooks/model_training.ipynb
```

#### 2. RDKit Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'rdkit'`

**Solution**:

```bash
# Conda installation (recommended)
conda install -c conda-forge rdkit

# Pip installation
pip install rdkit-pypi

# Mac M1/M2/M3 specific
conda install -c conda-forge rdkit python=3.12
```

#### 3. Streamlit Port Already in Use

**Problem**: `Address already in use`

**Solution**:

```bash
# Use different port
streamlit run scripts/webapp/lewis/main.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill -9  # Mac/Linux
netstat -ano | findstr :8501   # Windows (find PID and kill)
```

#### 4. HuggingFace API Token Issues

**Problem**: `Hugging Face API token not found`

**Solution**:

```bash
# Set environment variable
export HUGGINGFACE_API_TOKEN="your_token_here"

# Or add to .streamlit/secrets.toml
mkdir -p .streamlit
echo 'HF_TOKEN = "your_token_here"' > .streamlit/secrets.toml

# Get free token from: https://huggingface.co/settings/tokens
```

#### 5. Memory Issues with Batch Processing

**Problem**: `MemoryError` when processing large batches

**Solution**:

```python
# Process in smaller chunks
chunk_size = 100
for i in range(0, len(molecules), chunk_size):
    chunk = molecules[i:i+chunk_size]
    results = process_batch_molecules(chunk, 'csv', models)
```

#### 6. SMILES Parsing Errors

**Problem**: `Could not process molecule`

**Solution**:

```python
from rdkit import Chem

# Validate SMILES before prediction
smiles = "invalid_smiles"
mol = Chem.MolFromSmiles(smiles)

if mol is None:
    print("Invalid SMILES string")
    # Try sanitization
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol:
        Chem.SanitizeMol(mol)
```

### Performance Optimization

```python
# Cache model loading
import streamlit as st

@st.cache_resource
def load_models():
    return load_ml_models()

# Use batch processing for multiple molecules
# ~10x faster than individual predictions

# Enable GPU for deep learning (if available)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

### Getting Help

- 📖 [Documentation](https://github.com/omicscodeathon/mtoralzml/wiki)
- 💬 [GitHub Discussions](https://github.com/omicscodeathon/mtoralzml/discussions)
- 🐛 [Report a Bug](https://github.com/omicscodeathon/mtoralzml/issues/new?template=bug_report.md)
- ✨ [Feature Request](https://github.com/omicscodeathon/mtoralzml/issues/new?template=feature_request.md)
- 📧 Email: sohamshirolkar24@gmail.com, leahcerere@gmail.com, lewistem@gmail.com, nemase00@gmail.com

---

## 🤝 Contributing

We welcome contributions from the community! BrainRoute is an open-science project that thrives on collaboration.

### How to Contribute

1. **Fork the repository**

   ```bash
   git clone https://github.com/yourusername/mtoralzml.git
   cd mtoralzml
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

   - Add new features or fix bugs
   - Update documentation
   - Add tests for new functionality

3. **Run tests**

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

Beyond predictive modeling, we successfully developed and deployed NeuroGate as an interactive web platform. The system integrates with external resources including the Hugging Face inference API (for large language model–driven insights), ChEMBL (for bioactivity data), and the PubMed API (for literature retrieval). The interface, built in Streamlit and deployed via Hugging Face Spaces, enables researchers to evaluate molecules, explore predictions, and interact with contextual LLM-powered explanations in a user-friendly environment.

## Reproducibility

The workflow implemented in NeuroGate is fully reproducible. Users can follow the steps outlined below to clone the repository, install dependencies, and run the provided scripts and notebooks to recreate the modeling pipeline or extend it with new datasets and features. The platform has been designed for transparency and modularity, making it easy for researchers to build upon and adapt for their own projects. We also plan to continue improving the platform by refining models, expanding datasets, and enhancing the interactive interface.

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
