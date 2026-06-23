# BrainRoute

BrainRoute is a platform for blood-brain barrier (BBB) permeability prediction. This repository is for the BrainRoute Streamlit prediction tool and the model training and validation pipeline used by that tool.

The main BrainRoute database platform is maintained separately in [omicscodeathon/brainroutedb](https://github.com/omicscodeathon/brainroutedb) and is available at:

https://omicscodeathon.github.io/brainroutedb/

The deployed Streamlit prediction tool is available at:

https://brainroute.streamlit.app/

## Repository Structure

- `brainroute_ml_validation/` contains the reproducible model training and validation workflow.
- `brainroute_ml_validation/configs/validation_config.yaml` defines input paths, seeds, feature settings, split settings, models, and output locations.
- `brainroute_ml_validation/scripts/` contains the ordered pipeline scripts.
- `brainroute_ml_validation/src/` contains shared chemistry, feature, split, preprocessing, and modeling utilities.
- `brainroute_ml_validation/models/` contains the three model artifacts used by the Streamlit prediction tool.
- `brainroute_ml_validation/data/external/` contains external validation datasets used by the validation workflow.
- `brainroute_ml_validation/reports/` contains generated metrics, split summaries, predictions, figures, leakage checks, and model comparison outputs.
- `scripts/webapp/` contains the Streamlit prediction tool.
- `legacy/` contains archived notebooks, earlier scripts, historical data, and non-deployed model artifacts.

## Prediction Tool

The Streamlit app uses three BBB permeability models:

- PaDEL + Morgan LightGBM
- PaDEL + Morgan Extra Trees
- PaDEL + Morgan + ChemBERTa XGBoost

For each molecule, the app computes the feature views required by the selected model, including PaDEL descriptors, RDKit Morgan fingerprints, and ChemBERTa SMILES embeddings when needed. Metadata such as molecule names, labels, source tags, InChIKeys, and scaffolds are not used as model features.

Run the app locally:

```bash
python -m streamlit run scripts/webapp/main.py
```

The app expects the deployed `.joblib` model files under `brainroute_ml_validation/models/`. Optional platform account-linking and AI assistant settings are configured through local or Streamlit secrets and are not required to run basic BBB predictions.

## Training and Validation Pipeline

The pipeline starts from the configured BBB dataset, standardizes molecules, creates model features, builds validation splits, evaluates leakage controls, trains models, and writes summary reports.

The ordered scripts are:

1. `01_standardize_and_audit.py`
2. `02_calculate_morgan_fingerprints.py`
3. `03_calculate_pretrained_embeddings.py`
4. `04_build_feature_matrices.py`
5. `05_create_validation_splits.py`
6. `06_near_duplicate_analysis.py`
7. `07_leakage_controls.py`
8. `08_train_models.py`
9. `09_external_validation.py`
10. `10_statistical_comparison.py`
11. `11_make_summary_tables.py`

The workflow includes molecule standardization, duplicate and label-conflict checks, PaDEL descriptors, Morgan fingerprints, optional ChemBERTa embeddings, random and duplicate-aware splits, scaffold-based validation, near-duplicate Tanimoto analysis, leakage controls, model training, external validation, statistical comparisons, and final summary tables.

## Reproducibility

Create an environment with the required Python packages:

```bash
python -m pip install -r requirements.txt
```

For Apple Silicon, a conda or miniforge environment is recommended for RDKit, PyTorch, LightGBM, and XGBoost compatibility.

Run the full workflow:

```bash
python brainroute_ml_validation/run_full_validation.py \
  --config brainroute_ml_validation/configs/validation_config.yaml
```

Run part of the workflow:

```bash
python brainroute_ml_validation/run_full_validation.py \
  --config brainroute_ml_validation/configs/validation_config.yaml \
  --start-at 06_near_duplicate_analysis.py \
  --stop-after 08_train_models.py
```

Run a single step:

```bash
python brainroute_ml_validation/scripts/06_near_duplicate_analysis.py \
  --config brainroute_ml_validation/configs/validation_config.yaml
```

Primary generated outputs are written under:

- `brainroute_ml_validation/data/processed/`
- `brainroute_ml_validation/data/splits/`
- `brainroute_ml_validation/models/`
- `brainroute_ml_validation/reports/`
- `brainroute_ml_validation/reports/figures/`

The current configuration points to the archived starting dataset at `legacy/data/padel_loop_results_BBB.csv`.

External validation inputs are stored under `brainroute_ml_validation/data/external/`, including `external_dataset_qsar.xlsx`.

## Configuration

Do not commit private credentials or local secrets. Local Streamlit secrets should use `scripts/webapp/.streamlit/secrets.toml`, which is ignored by git. A template is available at `scripts/webapp/.streamlit/secrets.toml.example`.

Common optional settings include Supabase connection values for account-linked prediction logging and a Hugging Face token for the AI assistant. Basic BBB prediction only requires the model artifacts and the chemistry/modeling dependencies.
