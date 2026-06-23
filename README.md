# BrainRoute

**BrainRoute** is an open machine-learning platform for blood-brain barrier (BBB) permeability prediction and community-led data curation and verification. This repository holds the code for the BrainRoute prediction tool, a Streamlit app connected to the main BrainRoute platform for BBB permeability prediction. 



Live demo: https://brainroute.streamlit.app/
Main BrainRoute platform: https://omicscodeathon.github.io/brainroutedb/ 

## Repository Layout

- `brainroute_ml_validation/` - reproducible BBB model-training and validation pipeline.
- `brainroute_ml_validation/configs/validation_config.yaml` - single configuration file for seeds, paths, feature settings, split settings, model settings, and runtime options.
- `brainroute_ml_validation/scripts/` - ordered validation scripts from standardization through summary tables.
- `brainroute_ml_validation/src/` - reusable chemistry, feature-building, splitting, preprocessing, and modeling code.
- `brainroute_ml_validation/models/` - only the three deployed Streamlit model artifacts.
- `brainroute_ml_validation/reports/` - saved split summaries, metrics, predictions, figures, leakage checks. 
- `scripts/webapp/` - Streamlit application and deployment requirements.
- `legacy/` - historical notebooks, old models, previous figures, raw legacy data, archived scripts, and non-deployed model artifacts.

## Deployed Prediction Models

The Streamlit app uses three strict-validation model artifacts:

- PaDEL + Morgan LightGBM, trained on `duplicate_aware_seed5`
- PaDEL + Morgan Extra Trees, trained on `duplicate_aware_seed5`
- PaDEL + Morgan + ChemBERTa embeddings XGBoost, trained on `scaffold_cv_fold1`

At inference time, the app calculates the same feature representations required by each model: PaDEL descriptors, RDKit Morgan fingerprints, and frozen ChemBERTa SMILES embeddings when needed. Metadata columns such as SMILES, InChIKey, scaffold, source tags, and labels are not used as model features.

## Validation Pipeline

The validation workflow addresses leakage, chemical redundancy, near-duplicate similarity, scaffold bias, and reproducibility. It uses:

- RDKit molecule standardization, canonical SMILES, InChIKey, and Bemis-Murcko scaffolds.
- Duplicate and conflicting-label audits before modeling.
- PaDEL descriptors, Morgan fingerprints, and optional frozen pretrained ChemBERTa embeddings.
- Fixed random, duplicate-aware, repeated duplicate-aware, scaffold holdout, and 5-fold scaffold-CV splits.
- Near-duplicate Tanimoto analysis using Morgan fingerprints.
- Fold-local preprocessing only: missingness filtering, low-variance filtering, median imputation, correlation filtering, and scaling where appropriate.
- Small GridSearchCV/RandomizedSearchCV model searches using balanced accuracy as the primary selection metric.
- Saved predictions, selected features, split files, metrics, figures, leakage controls, and statistical comparisons.

The primary reviewer-facing validation evidence is scaffold 5-fold cross-validation. Duplicate-aware repeated splits are reported as a secondary validation setting, and random 80/20 splitting is retained only as a conventional baseline.

## Reproducibility

The pipeline is designed to be rerunnable from fixed inputs and fixed configuration:

- All random seeds are stored in `brainroute_ml_validation/configs/validation_config.yaml`.
- All train/test and CV split files are saved under `brainroute_ml_validation/data/splits/`.
- Processed feature matrices are saved under `brainroute_ml_validation/data/processed/`.
- Model-level predictions and metrics are saved under `brainroute_ml_validation/reports/`.
- Reviewer-facing method text is saved in `brainroute_ml_validation/reports/reviewer_methods_text.md`.
- The historical PaDEL starting dataset is archived at `legacy/data/padel_loop_results_BBB.csv`, and the config points to that file.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```
`
For Apple Silicon, a conda or miniforge environment is recommended so RDKit, NumPy, PyTorch, LightGBM, and XGBoost resolve to compatible ARM64 builds.

Run the full validation workflow:

```bash
python brainroute_ml_validation/run_full_validation.py \
  --config brainroute_ml_validation/configs/validation_config.yaml
```

Run individual steps:

```bash
python brainroute_ml_validation/scripts/01_standardize_and_audit.py --config brainroute_ml_validation/configs/validation_config.yaml
python brainroute_ml_validation/scripts/02_calculate_morgan_fingerprints.py --config brainroute_ml_validation/configs/validation_config.yaml
python brainroute_ml_validation/scripts/03_calculate_pretrained_embeddings.py --config brainroute_ml_validation/configs/validation_config.yaml
python brainroute_ml_validation/scripts/04_build_feature_matrices.py --config brainroute_ml_validation/configs/validation_config.yaml
python brainroute_ml_validation/scripts/05_create_validation_splits.py --config brainroute_ml_validation/configs/validation_config.yaml
python brainroute_ml_validation/scripts/06_near_duplicate_analysis.py --config brainroute_ml_validation/configs/validation_config.yaml
python brainroute_ml_validation/scripts/07_leakage_controls.py --config brainroute_ml_validation/configs/validation_config.yaml
python brainroute_ml_validation/scripts/08_train_models.py --config brainroute_ml_validation/configs/validation_config.yaml
python brainroute_ml_validation/scripts/09_external_validation.py --config brainroute_ml_validation/configs/validation_config.yaml
python brainroute_ml_validation/scripts/10_statistical_comparison.py --config brainroute_ml_validation/configs/validation_config.yaml
python brainroute_ml_validation/scripts/11_make_summary_tables.py --config brainroute_ml_validation/configs/validation_config.yaml
```

## Running the Streamlit App

```bash
python -m streamlit run scripts/webapp/main.py
```

The app loads model paths from `scripts/webapp/config.py` and expects the three deployed `.joblib` files in `brainroute_ml_validation/models/`.

## Project Team

Lead authors:

- Soham Shirolkar - University of South Florida - ORCID: 0009-0004-4798-899X - sohamshirolkar24@gmail.com
- Lewis Tem - University of Buea - lewistem8@gmail.com
- Olaitan I. Awe - Institute for Genomic Medicine Research & ASBCB - ORCID: 0000-0002-4257-3611 - laitanawe@gmail.com

See `CONTRIBUTORS.md` for the full BrainRoute team.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
