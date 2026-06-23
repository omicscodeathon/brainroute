# Methods Text

Random 80/20 splitting was retained only as a baseline because it can overestimate performance when exact duplicates, close analogs, or related scaffolds appear in both training and test sets.

Molecules were standardized with RDKit, converted to canonical SMILES and InChIKey identifiers, and assigned Bemis-Murcko scaffolds. Exact duplicate audits were performed by InChIKey. Molecules with conflicting labels for the same InChIKey were excluded by default and preserved in audit files.

Duplicate-aware splits used InChIKey groups so an exact molecule could not appear in both train and test sets. Scaffold holdout and five-fold scaffold cross-validation used Bemis-Murcko scaffold groups so the same scaffold was not shared across train/test folds. Scaffold cross-validation was treated as the primary model-selection evidence.

Morgan fingerprints were computed with radius 2, 2048 bits, and chirality enabled. For each split, every test molecule was compared with all training molecules by Tanimoto similarity, and nearest-neighbor similarity summaries were saved at 0.80, 0.85, and 0.90 thresholds.

PaDEL descriptors, Morgan fingerprints, and optional frozen pretrained SMILES-transformer embeddings were treated as separate feature representations. Descriptor missingness filters, variance filters, correlation filters, imputers, scalers, and any model-specific preprocessing were fit only inside the training fold through scikit-learn pipelines.

Class weighting was used as the default imbalance strategy to preserve chemical diversity. External validation, when configured, was performed after standardization, exact-overlap removal, and near-duplicate similarity annotation; the external set was not used for model tuning.

All split files, seeds, configuration files, scripts, predictions, metrics, selected-feature lists, and audit tables are written to disk to support independent reproducibility.
