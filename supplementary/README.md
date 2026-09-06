# BrainRoute manuscript supplementary material

This directory collects the submission-ready supplementary tables, their principal machine-readable source tables, and the manuscript figure files. The broader prediction-level outputs, provenance audits, calibration data, split audits, and feature-importance tables remain in [`brainroute_ml_validation/reports`](../brainroute_ml_validation/reports/).

## Supplementary tables

- `BrainRoute_supplementary_tables.docx` contains Supplementary Tables S1-S6 in landscape format.
- Table S1: `tables/source_endpoint_harmonization.csv`
- Table S2: `tables/matched_benchmark_performance_summary.csv` and `tables/matched_benchmark_paired_comparisons.csv`
- Table S3: `tables/deployed_external_bootstrap_ci.csv`
- Table S4: `tables/revision_external_attrition_audit.csv`
- Table S5: `tables/source_validation_metrics.csv`
- Table S6: `tables/revision_model_weight_policy.csv` and `tables/revision_validation_strategy_summary.csv`

The table copies in this directory are identical to the corresponding files in `brainroute_ml_validation/reports`.

## Benchmark artifacts

The final conflict-excluded inputs for the matched 3D and quantum-chemistry analysis are in [`brainroute_ml_validation/data/benchmarks/matched_3d_qm_conflict_excluded`](../brainroute_ml_validation/data/benchmarks/matched_3d_qm_conflict_excluded/). They include the locked cohort manifest, the 19 GFN2-xTB variables, frozen 512-dimensional Uni-Mol v1 representations and index, and the 24-molecule source-label-conflict exclusion audit.

All 15 fitted outer-fold models are in [`brainroute_ml_validation/models/matched_3d_qm_benchmark`](../brainroute_ml_validation/models/matched_3d_qm_benchmark/): five PaDEL plus Morgan controls, five PaDEL plus Morgan plus GFN2-xTB models, and five frozen Uni-Mol v1 models.

The external drive was used for resumable calculation caches and downloaded Uni-Mol/Hugging Face weights. Those caches are not study outputs and are not required to inspect the reported results. The compact final inputs, fitted benchmark models, prediction-level outputs, and summary tables are included in this repository.

## Figures

The `figures` directory contains Figures 1-9 used in the manuscript. Figures 4-7 and 9 are the higher-resolution final composites.
