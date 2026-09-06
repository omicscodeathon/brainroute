# Matched 3D and quantum-chemistry benchmark inputs

The `artifacts` directory contains the final inputs used for the conflict-excluded, molecule-matched benchmark reported in the revised manuscript.

- `artifacts/benchmark_manifest.csv`: locked internal and external cohort with standardized identities, labels, scaffolds, and calculation structures.
- `artifacts/matched_benchmark_analysis_manifest.csv`: final analysis cohort after representation and source-conflict eligibility checks.
- `artifacts/excluded_source_label_conflicts.csv`: the 24 retained modeling molecules excluded because reconstructed source records disagreed.
- `artifacts/conflict_exclusion_audit.json`: row counts and SHA-256 checksums for the conflict-exclusion step.
- `artifacts/xtb_features.csv`: 19 GFN2-xTB variables for the matched molecules.
- `artifacts/unimol_v1_index.csv` and `artifacts/unimol_v1_representations.npy`: row index and frozen 512-dimensional Uni-Mol v1 representation matrix.

The corresponding prediction-level results, fold metrics, nested-tuning choices, paired tests, chemical-space analyses, and verification record are in `brainroute_ml_validation/reports`. The 15 fitted outer-fold models are in `brainroute_ml_validation/models/matched_3d_qm_benchmark`.
