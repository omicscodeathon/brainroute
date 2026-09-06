# Matched 3D and quantum-chemistry benchmark

This additive workflow evaluates a modern 3D baseline and quantum-chemistry-informed extension without replacing the models used by the BrainRoute platform.

## Locked comparison

Three XGBoost configurations are evaluated on the same successful molecules:

1. PaDEL descriptors and radius-2 Morgan fingerprints (control).
2. The same PaDEL and Morgan features augmented with GFN2-xTB electronic descriptors.
3. Frozen 512-dimensional Uni-Mol v1 molecular representations.

The original five outer Bemis-Murcko scaffold folds are retained. Hyperparameters are selected only within each outer training partition using three-fold `StratifiedGroupKFold`, grouped by Bemis-Murcko scaffold. The independent external cohort has no exact InChIKey overlap with the internal cohort. Source-exclusive sensitivity tests train on B3DB-only molecules and test on MoleculeNet BBBP-only molecules, and vice versa; molecules represented in both sources are excluded from both sides of those tests.

## Molecular preparation

The modeling identity remains the standardized InChIKey. For 3D and quantum calculations, the largest organic fragment is selected deterministically where a disconnected standardized record is present. Formal charge and stereochemical SMILES are retained. RDKit ETKDGv3 generates one conformer, followed by MMFF94s minimization when parameters are available and UFF otherwise.

GFN2-xTB single-point calculations provide total and electronic energy, HOMO and LUMO energies, HOMO-LUMO gap, molecular dipole, molecular polarizability, partial-charge summaries, atomic-dipole summaries, and Wiberg bond-order summaries. Calculations are cached per molecule and failures are recorded. Uni-Mol v1 is used as a frozen pretrained 3D representation model. Molecules for which either xTB or Uni-Mol does not obtain a valid 3D result are excluded from all three configurations so the comparison remains molecule-matched.

## Reproduction

The published configuration uses repository-relative workspaces under `brainroute_ml_validation/data/benchmarks/`. For large runs, the locations can be moved to another disk with the `--workspace`, `--source-workspace`, or `--target-workspace` options and by updating `external_workspace` in the selected configuration file. The original run used `/Volumes/SS1TB/brainroute-3d-qm-benchmark` and `/Volumes/SS1TB/brainroute-3d-qm-benchmark-conflict-excluded`. Condensed result tables are copied to `brainroute_ml_validation/reports/` after evaluation.

The repository includes the final conflict-excluded xTB and Uni-Mol inputs in `brainroute_ml_validation/data/benchmarks/matched_3d_qm_conflict_excluded/`, all 15 fitted outer-fold models in `brainroute_ml_validation/models/matched_3d_qm_benchmark/`, and the complete compact result set in `brainroute_ml_validation/reports/`. The multi-gigabyte calculation and package-download caches are optional and are not required to inspect or verify the reported results.

```bash
python brainroute_ml_validation/scripts/15_prepare_matched_benchmark.py

python brainroute_ml_validation/scripts/16_calculate_xtb_features.py --workers 4

UNIMOL_WEIGHT_DIR=brainroute_ml_validation/data/benchmarks/matched_3d_qm/cache/unimol_weights \
HF_HOME=brainroute_ml_validation/data/benchmarks/matched_3d_qm/cache/huggingface \
python \
  brainroute_ml_validation/scripts/17_calculate_unimol_representations.py --batch-size 32

python \
  brainroute_ml_validation/scripts/20_materialize_conflict_excluded_benchmark.py

python \
  brainroute_ml_validation/scripts/18_run_matched_benchmark.py \
  --config brainroute_ml_validation/configs/3d_qm_benchmark_conflict_excluded.yaml

python brainroute_ml_validation/scripts/21_verify_matched_benchmark.py
```

## Interpretation limits

The Uni-Mol experiment is a frozen-representation baseline with a common shallow classifier, not end-to-end Uni-Mol fine-tuning. The xTB features come from one force-field-minimized conformer and gas-phase single-point calculations; they do not represent a conformer ensemble, explicit solvent, pH-dependent microspecies enumeration, active transport, or transporter binding. These choices make the requested experiment reproducible on the available computer while preserving a direct comparison with the platform's existing descriptor framework.

The legacy PaDEL matrix contains non-finite values for undefined descriptors. PaDEL's maximum IEEE-754 floating-point sentinel in `gmin` is treated as missing before conversion to XGBoost's float32 input format. XGBoost handles those missing values natively. This conversion is applied identically to the PaDEL/Morgan control and xTB-augmented view.
