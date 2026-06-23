#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.chemistry import (
    calculate_morgan_matrix,
    max_tanimoto_to_train,
    morgan_bitvect,
    standardize_smiles,
)
from brainroute_ml_validation.src.modeling import metric_dict, predict_scores
from brainroute_ml_validation.src.utils import (
    LOGGER,
    ensure_dirs,
    load_config,
    normalize_binary_label,
    project_path,
    read_table,
    resolve_path,
    script_arg_parser,
    set_global_seed,
)


RESULTS_SUBDIR = "results/external_validation"
DEFAULT_EXTERNAL_PATH = "data/external/B3DB_classification_external.tsv"
DEFAULT_EXTERNAL_TWO_PATH = "/Users/soham/Downloads/external_dataset_qsar.xlsx"
THRESHOLD = 0.5
PADEL_CHUNK_SIZE = 24
PADEL_CHUNK_TIMEOUT = 180
PADEL_CHUNK_MAX_RUNTIME = 120
PADEL_SINGLE_TIMEOUT = 18
PADEL_SINGLE_MAX_RUNTIME = 12


@dataclass(frozen=True)
class ModelConfig:
    display_name: str
    feature_view: str
    model_name: str
    artifact_regex: str


SELECTED_MODELS = [
    ModelConfig(
        display_name="PaDEL + Morgan LightGBM duplicate-aware",
        feature_view="padel_morgan",
        model_name="lightgbm",
        artifact_regex=r"^padel_morgan__lightgbm__(duplicate_aware_seed\d+)\.joblib$",
    ),
    ModelConfig(
        display_name="PaDEL + Morgan Extra Trees duplicate-aware",
        feature_view="padel_morgan",
        model_name="extra_trees",
        artifact_regex=r"^padel_morgan__extra_trees__(duplicate_aware_seed\d+)\.joblib$",
    ),
    ModelConfig(
        display_name="PaDEL + Morgan + ChemBERTa XGBoost scaffold-CV",
        feature_view="padel_morgan_embeddings",
        model_name="xgboost",
        artifact_regex=r"^padel_morgan_embeddings__xgboost__(scaffold_cv_fold\d+)\.joblib$",
    ),
]


def out_dir(cfg: dict) -> Path:
    path = project_path(cfg, cfg.get("_external_results_subdir", RESULTS_SUBDIR))
    path.mkdir(parents=True, exist_ok=True)
    (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOGGER.info("Wrote %s", path)


def resolve_external_path(cfg: dict) -> Path | None:
    configured = cfg.get("paths", {}).get("external_validation_path")
    path = resolve_path(configured, cfg)
    if path and path.exists():
        return path
    fallback = project_path(cfg, DEFAULT_EXTERNAL_PATH)
    if fallback.exists():
        return fallback
    return None


def read_external_two_short(path: str | Path = DEFAULT_EXTERNAL_TWO_PATH) -> pd.DataFrame:
    ext = pd.read_excel(path, header=2)
    short = ext[["SMILES", "CAS", "Name", "Species", "Activity score"]].copy()
    return short.rename(
        columns={
            "SMILES": "smiles",
            "CAS": "cas",
            "Name": "name",
            "Species": "species",
            "Activity score": "BBB label",
        }
    )


def choose_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower().replace("\ufeff", ""): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace("\ufeff", "")
        if key in lower:
            return lower[key]
    if required:
        raise ValueError(f"Could not find any of these columns: {candidates}")
    return None


def standardize_external(ext: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = cfg.get("columns", {})
    smiles_col = choose_column(ext, [cols.get("smiles", ""), "SMILES", "smiles", "Original_SMILES"])
    label_col = choose_column(ext, [cols.get("label", ""), "BBB label", "BBB+/BBB-", "BBB", "label"])
    name_col = choose_column(ext, [cols.get("molecule_name", ""), "compound_name", "name", "Original_Name"], required=False)
    cas_col = choose_column(ext, ["cas", "CAS"], required=False)
    species_col = choose_column(ext, ["species", "Species"], required=False)
    inchi_col = choose_column(ext, ["Inchi", "InChI", "inchi"], required=False)

    valid_columns = [
        "external_row",
        "compound_name",
        "cas",
        "species",
        "input_smiles",
        "input_inchi",
        "canonical_smiles",
        "inchikey",
        "murcko_scaffold",
        "label",
        "label_raw",
    ]
    failed_columns = [
        "external_row",
        "compound_name",
        "cas",
        "species",
        "input_smiles",
        "input_inchi",
        "label_raw",
        "standardization_error",
        "label_error",
    ]
    valid_rows: list[dict] = []
    failed_rows: list[dict] = []
    seen_keys: set[str] = set()
    for external_row, row in ext.iterrows():
        label = normalize_binary_label(row.get(label_col))
        std = standardize_smiles(row.get(smiles_col))
        if not std.valid or pd.isna(label):
            failed_rows.append(
                {
                    "external_row": external_row,
                    "compound_name": row.get(name_col) if name_col else None,
                    "cas": row.get(cas_col) if cas_col else None,
                    "species": row.get(species_col) if species_col else None,
                    "input_smiles": row.get(smiles_col),
                    "input_inchi": row.get(inchi_col) if inchi_col else None,
                    "label_raw": row.get(label_col),
                    "standardization_error": std.error if not std.valid else None,
                    "label_error": "label_missing_or_unrecognized" if pd.isna(label) else None,
                }
            )
            continue
        duplicate_key = std.inchikey or std.canonical_smiles
        if duplicate_key in seen_keys:
            failed_rows.append(
                {
                    "external_row": external_row,
                    "compound_name": row.get(name_col) if name_col else None,
                    "cas": row.get(cas_col) if cas_col else None,
                    "species": row.get(species_col) if species_col else None,
                    "input_smiles": row.get(smiles_col),
                    "input_inchi": row.get(inchi_col) if inchi_col else None,
                    "label_raw": row.get(label_col),
                    "standardization_error": "duplicate_external_inchikey",
                    "label_error": None,
                }
            )
            continue
        seen_keys.add(duplicate_key)
        valid_rows.append(
            {
                "external_row": external_row,
                "compound_name": row.get(name_col) if name_col else None,
                "cas": row.get(cas_col) if cas_col else None,
                "species": row.get(species_col) if species_col else None,
                "input_smiles": row.get(smiles_col),
                "input_inchi": row.get(inchi_col) if inchi_col else None,
                "canonical_smiles": std.canonical_smiles,
                "inchikey": std.inchikey,
                "murcko_scaffold": std.murcko_scaffold,
                "label": int(label),
                "label_raw": row.get(label_col),
            }
        )
    return pd.DataFrame(valid_rows, columns=valid_columns), pd.DataFrame(failed_rows, columns=failed_columns)


def remove_training_overlaps(ext_std: pd.DataFrame, internal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    internal_smiles = set(internal["canonical_smiles"].dropna().astype(str))
    internal_keys = set(internal["inchikey"].dropna().astype(str))
    overlap_mask = ext_std["canonical_smiles"].isin(internal_smiles) | ext_std["inchikey"].isin(internal_keys)
    overlaps = ext_std.loc[overlap_mask].copy()
    if not overlaps.empty:
        overlaps["overlap_by_inchikey"] = overlaps["inchikey"].isin(internal_keys)
        overlaps["overlap_by_canonical_smiles"] = overlaps["canonical_smiles"].isin(internal_smiles)
    cleaned = ext_std.loc[~overlap_mask].reset_index(drop=True).copy()
    return cleaned, overlaps.reset_index(drop=True)


def calculate_padel_2d_only(smiles: list[str], timeout: int, maxruntime_seconds: int) -> list[dict]:
    from padelpy.wrapper import padeldescriptor

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        smi_path = tmp / "molecules.smi"
        out_path = tmp / "padel_descriptors.csv"
        smi_path.write_text("\n".join(smiles), encoding="utf-8")
        padeldescriptor(
            mol_dir=str(smi_path),
            d_file=str(out_path),
            d_2d=True,
            d_3d=False,
            fingerprints=False,
            retainorder=True,
            threads=1,
            sp_timeout=timeout,
            maxruntime=maxruntime_seconds * 1000,
        )
        return pd.read_csv(out_path).to_dict("records")


def calculate_single_full_padel_worker(args: tuple[int, str, int, int]) -> tuple[int, dict | None, str | None]:
    idx, smiles, timeout, maxruntime = args
    from padelpy import from_smiles

    try:
        row = from_smiles(
            smiles,
            fingerprints=False,
            descriptors=True,
            timeout=timeout,
            maxruntime=maxruntime,
            threads=1,
        )
        return idx, row, None
    except Exception as exc:
        return idx, None, str(exc)


def calculate_padel_features(
    smiles: list[str],
    two_d_only: bool = False,
    chunk_size: int = PADEL_CHUNK_SIZE,
    chunk_timeout: int = PADEL_CHUNK_TIMEOUT,
    chunk_max_runtime: int = PADEL_CHUNK_MAX_RUNTIME,
    single_timeout: int = PADEL_SINGLE_TIMEOUT,
    single_max_runtime: int = PADEL_SINGLE_MAX_RUNTIME,
    workers: int = 1,
) -> tuple[pd.DataFrame, list[int]]:
    from padelpy import from_smiles

    rows: list[dict | None] = [None] * len(smiles)
    failed: list[int] = []
    if workers > 1 and not two_d_only:
        LOGGER.info("Calculating full PaDEL descriptors molecule-by-molecule with %s workers", workers)
        tasks = [(idx, value, single_timeout, single_max_runtime) for idx, value in enumerate(smiles)]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(calculate_single_full_padel_worker, task): task[0] for task in tasks}
            for future in as_completed(futures):
                idx, row, error = future.result()
                if row is None:
                    LOGGER.warning("PaDEL failed for external row %s: %s", idx, error)
                    failed.append(idx)
                else:
                    rows[idx] = row
        X = pd.DataFrame([row if row is not None else {} for row in rows])
        X = X.drop(columns=["Name"], errors="ignore")
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.add_prefix("padel__"), sorted(failed)

    for start in range(0, len(smiles), chunk_size):
        stop = min(start + chunk_size, len(smiles))
        chunk = smiles[start:stop]
        try:
            if two_d_only:
                chunk_rows = calculate_padel_2d_only(chunk, chunk_timeout, chunk_max_runtime)
            else:
                chunk_rows = from_smiles(
                    chunk,
                    fingerprints=False,
                    descriptors=True,
                    timeout=chunk_timeout,
                    maxruntime=chunk_max_runtime,
                    threads=1,
                )
            if isinstance(chunk_rows, dict):
                chunk_rows = [chunk_rows]
            for offset, descriptor_row in enumerate(chunk_rows):
                rows[start + offset] = descriptor_row
            continue
        except Exception as exc:
            LOGGER.warning(
                "PaDEL chunk %s:%s failed; retrying one molecule at a time: %s",
                start,
                stop,
                exc,
            )

        for offset, value in enumerate(chunk):
            idx = start + offset
            try:
                if two_d_only:
                    single_rows = calculate_padel_2d_only([value], single_timeout, single_max_runtime)
                    rows[idx] = single_rows[0] if single_rows else {}
                else:
                    rows[idx] = from_smiles(
                        value,
                        fingerprints=False,
                        descriptors=True,
                        timeout=single_timeout,
                        maxruntime=single_max_runtime,
                        threads=1,
                    )
            except Exception as inner_exc:
                LOGGER.warning("PaDEL failed for external row %s: %s", idx, inner_exc)
                failed.append(idx)
    X = pd.DataFrame([row if row is not None else {} for row in rows])
    X = X.drop(columns=["Name"], errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.add_prefix("padel__"), failed


def calculate_morgan_features(smiles: list[str], cfg: dict) -> pd.DataFrame:
    morgan = cfg.get("morgan", {})
    n_bits = int(morgan.get("nBits", 2048))
    matrix, valid_positions, _ = calculate_morgan_matrix(
        smiles,
        radius=int(morgan.get("radius", 2)),
        n_bits=n_bits,
        use_chirality=bool(morgan.get("useChirality", True)),
    )
    X = pd.DataFrame(0, index=range(len(smiles)), columns=[f"morgan__morgan_{i}" for i in range(n_bits)])
    if valid_positions:
        X.iloc[valid_positions, :] = matrix
    return X


def calculate_embedding_features(smiles: list[str], cfg: dict) -> pd.DataFrame:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

    import torch
    from transformers import AutoModel, AutoTokenizer

    emb_cfg = cfg.get("pretrained_embeddings", {})
    model_name = emb_cfg.get("model_name", "DeepChem/ChemBERTa-77M-MLM")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_size = int(emb_cfg.get("batch_size", 32))
    max_length = int(emb_cfg.get("max_length", 256))
    pooling = emb_cfg.get("pooling", "cls")
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(smiles), batch_size):
            batch = smiles[start : start + batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            outputs = model(**tokens)
            hidden = outputs.last_hidden_state
            if pooling == "mean":
                mask = tokens["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden[:, 0, :]
            embeddings.append(pooled.cpu().numpy())
    arr = np.vstack(embeddings)
    return pd.DataFrame(arr, columns=[f"emb__emb_{i}" for i in range(arr.shape[1])])


def model_expected_features(model) -> list[str]:
    finite = getattr(model, "named_steps", {}).get("finite") if hasattr(model, "named_steps") else None
    if finite is not None and hasattr(finite, "feature_names_in_"):
        return list(finite.feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise ValueError("Saved model does not expose fitted feature_names_in_; cannot align external features safely.")


def build_feature_views(ext: pd.DataFrame, cfg: dict) -> tuple[dict[str, pd.DataFrame], list[int]]:
    smiles = ext["canonical_smiles"].tolist()
    cache = out_dir(cfg)
    padel_cache = cache / "external_padel_features_raw.csv"
    padel_failed_cache = cache / "external_padel_failed_positions.json"
    if padel_cache.exists() and padel_failed_cache.exists():
        LOGGER.info("Loading cached external PaDEL descriptors from %s", padel_cache)
        padel_X = pd.read_csv(padel_cache)
        with padel_failed_cache.open("r", encoding="utf-8") as handle:
            padel_failed = json.load(handle)
    else:
        LOGGER.info("Calculating external PaDEL descriptors for %s molecules", len(smiles))
        padel_X, padel_failed = calculate_padel_features(
            smiles,
            two_d_only=bool(cfg.get("_padel_2d_only", False)),
            chunk_size=int(cfg.get("_padel_chunk_size", PADEL_CHUNK_SIZE)),
            chunk_timeout=int(cfg.get("_padel_chunk_timeout", PADEL_CHUNK_TIMEOUT)),
            chunk_max_runtime=int(cfg.get("_padel_chunk_max_runtime", PADEL_CHUNK_MAX_RUNTIME)),
            single_timeout=int(cfg.get("_padel_single_timeout", PADEL_SINGLE_TIMEOUT)),
            single_max_runtime=int(cfg.get("_padel_single_max_runtime", PADEL_SINGLE_MAX_RUNTIME)),
            workers=int(cfg.get("_padel_workers", 1)),
        )
        padel_X.to_csv(padel_cache, index=False)
        with padel_failed_cache.open("w", encoding="utf-8") as handle:
            json.dump(padel_failed, handle)
        LOGGER.info("Cached external PaDEL descriptors at %s", padel_cache)
    LOGGER.info("Calculating external Morgan fingerprints")
    morgan_X = calculate_morgan_features(smiles, cfg)
    feature_views = {
        "padel_morgan": pd.concat([padel_X.reset_index(drop=True), morgan_X.reset_index(drop=True)], axis=1)
    }
    needs_embeddings = any(model.feature_view == "padel_morgan_embeddings" for model in SELECTED_MODELS)
    if needs_embeddings:
        LOGGER.info("Calculating external ChemBERTa embeddings")
        emb_X = calculate_embedding_features(smiles, cfg)
        feature_views["padel_morgan_embeddings"] = pd.concat(
            [padel_X.reset_index(drop=True), morgan_X.reset_index(drop=True), emb_X.reset_index(drop=True)],
            axis=1,
        )
    return feature_views, padel_failed


def selected_artifacts(cfg: dict, model_cfg: ModelConfig) -> list[tuple[str, Path]]:
    model_dir = project_path(cfg, "models")
    pattern = re.compile(model_cfg.artifact_regex)
    matches: list[tuple[str, Path]] = []
    for path in sorted(model_dir.glob("*.joblib")):
        match = pattern.match(path.name)
        if match:
            matches.append((match.group(1), path))
    if not matches:
        raise FileNotFoundError(f"No artifacts found for {model_cfg.display_name} in {model_dir}")
    return matches


def predict_with_artifact(
    model_path: Path,
    X_view: pd.DataFrame,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict, object]:
    model = joblib.load(model_path)
    expected = model_expected_features(model)
    X_aligned = X_view.reindex(columns=expected, fill_value=np.nan)
    y_score = predict_scores(model, X_aligned)
    if y_score is None:
        raise ValueError(f"Model does not expose probabilities or scores: {model_path}")
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= THRESHOLD).astype(int)
    metrics = metric_dict(y_true, y_pred, y_score)
    return y_pred, y_score, metrics, model


def mean_std_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc", "roc_auc", "auprc"]
    rows = []
    for name, group in fold_metrics.groupby("model_configuration", sort=False):
        row = {
            "model_configuration": name,
            "n_folds_or_seeds": len(group),
            "folds_or_seeds": ";".join(group["fold_or_seed"].astype(str)),
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
            row[f"{metric}_mean_std"] = f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def probability_averaged_results(
    per_artifact_predictions: pd.DataFrame,
    ext: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    prediction_cols = []
    base_cols = [
        col
        for col in ["external_row", "compound_name", "cas", "species", "canonical_smiles", "inchikey", "label", "label_raw"]
        if col in ext.columns
    ]
    base = ext[base_cols].copy()
    for model_name, group in per_artifact_predictions.groupby("model_configuration", sort=False):
        pivot = group.pivot(index="external_position", columns="fold_or_seed", values="y_score").sort_index()
        avg_score = pivot.mean(axis=1).to_numpy()
        y_true = ext.loc[pivot.index, "label"].astype(int).to_numpy()
        y_pred = (avg_score >= THRESHOLD).astype(int)
        metrics = metric_dict(y_true, y_pred, avg_score)
        metrics.update(
            {
                "model_configuration": model_name,
                "n_folds_or_seeds": pivot.shape[1],
                "folds_or_seeds": ";".join(map(str, pivot.columns.tolist())),
            }
        )
        rows.append(metrics)
        safe = safe_name(model_name)
        base[f"{safe}_probability"] = avg_score
        base[f"{safe}_prediction"] = np.where(y_pred == 1, "BBB+", "BBB-")
        prediction_cols.append(f"{safe}_probability")
    return pd.DataFrame(rows), base


def brainroute_ensemble_results(probability_predictions: pd.DataFrame, y_true: np.ndarray) -> pd.DataFrame:
    prob_cols = [c for c in probability_predictions.columns if c.endswith("_probability")]
    if not prob_cols:
        raise ValueError("No probability columns available for BrainRoute ensemble.")
    ensemble_score = probability_predictions[prob_cols].mean(axis=1).to_numpy()
    ensemble_pred = (ensemble_score >= THRESHOLD).astype(int)
    metrics = metric_dict(y_true, ensemble_pred, ensemble_score)
    metrics.update({"model_configuration": "BrainRoute selected-model ensemble", "n_model_configurations": len(prob_cols)})
    out = pd.DataFrame([metrics])
    probability_predictions["brainroute_ensemble_probability"] = ensemble_score
    probability_predictions["brainroute_ensemble_prediction"] = np.where(ensemble_pred == 1, "BBB+", "BBB-")
    return out


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def add_similarity_to_training(ext: pd.DataFrame, internal: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    m = cfg.get("morgan", {})
    radius = int(m.get("radius", 2))
    n_bits = int(m.get("nBits", 2048))
    chirality = bool(m.get("useChirality", True))
    train_fps = [morgan_bitvect(s, radius, n_bits, chirality) for s in internal["canonical_smiles"]]
    train_fps = [fp for fp in train_fps if fp is not None]
    test_fps = [morgan_bitvect(s, radius, n_bits, chirality) for s in ext["canonical_smiles"]]
    sim = max_tanimoto_to_train(test_fps, train_fps)
    ext = ext.copy()
    ext["max_tanimoto_to_training"] = [row["max_tanimoto"] for row in sim]
    return ext


def save_figures(
    fold_metrics: pd.DataFrame,
    probability_metrics: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
    probability_predictions: pd.DataFrame,
    ext: pd.DataFrame,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

    metric_subset = ["balanced_accuracy", "auprc", "mcc", "f1"]
    summary = mean_std_summary(fold_metrics)
    long_rows = []
    for _, row in summary.iterrows():
        for metric in metric_subset:
            long_rows.append(
                {
                    "model_configuration": row["model_configuration"],
                    "metric": metric,
                    "mean": row[f"{metric}_mean"],
                    "std": row[f"{metric}_std"],
                }
            )
    long = pd.DataFrame(long_rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=long, x="model_configuration", y="mean", hue="metric", ax=ax, errorbar=None)
    for container, metric in zip(ax.containers, metric_subset):
        subset = long[long["metric"] == metric].reset_index(drop=True)
        ax.errorbar(
            x=[bar.get_x() + bar.get_width() / 2 for bar in container],
            y=subset["mean"],
            yerr=subset["std"],
            fmt="none",
            color="black",
            linewidth=1,
            capsize=3,
        )
    ax.set_ylabel("External validation metric")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "figures/external_selected_model_metrics_barplot.png", dpi=300)
    plt.close(fig)

    fold_long = fold_metrics.melt(
        id_vars=["model_configuration", "fold_or_seed"],
        value_vars=metric_subset,
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=fold_long, x="model_configuration", y="value", hue="metric", ax=ax)
    sns.stripplot(data=fold_long, x="model_configuration", y="value", hue="metric", dodge=True, ax=ax, color="black", alpha=0.45)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: len(metric_subset)], labels[: len(metric_subset)], title="metric")
    ax.set_ylabel("Fold/seed metric")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output / "figures/external_fold_seed_metric_boxplots.png", dpi=300)
    plt.close(fig)

    y_true = ext["label"].astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    for col in [c for c in probability_predictions.columns if c.endswith("_probability") and not c.startswith("brainroute_")]:
        RocCurveDisplay.from_predictions(y_true, probability_predictions[col], name=col.replace("_probability", ""), ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    fig.tight_layout()
    fig.savefig(output / "figures/external_probability_averaged_roc_curves.png", dpi=300)
    plt.close(fig)

    prevalence = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(7, 6))
    for col in [c for c in probability_predictions.columns if c.endswith("_probability") and not c.startswith("brainroute_")]:
        PrecisionRecallDisplay.from_predictions(y_true, probability_predictions[col], name=col.replace("_probability", ""), ax=ax)
    ax.axhline(prevalence, linestyle="--", color="gray", linewidth=1, label=f"prevalence={prevalence:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures/external_probability_averaged_pr_curves.png", dpi=300)
    plt.close(fig)

    for col in [c for c in probability_predictions.columns if c.endswith("_prediction")]:
        y_pred = (probability_predictions[col] == "BBB+").astype(int).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["BBB-", "BBB+"], yticklabels=["BBB-", "BBB+"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(col.replace("_prediction", ""))
        fig.tight_layout()
        fig.savefig(output / f"figures/confusion_matrix__{col.replace('_prediction', '')}.png", dpi=300)
        plt.close(fig)

    if "max_tanimoto_to_training" in ext.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(ext["max_tanimoto_to_training"].dropna(), bins=20, ax=ax)
        ax.set_xlabel("Max Morgan Tanimoto similarity to internal training set")
        ax.set_ylabel("External molecule count")
        fig.tight_layout()
        fig.savefig(output / "figures/external_nearest_training_tanimoto_distribution.png", dpi=300)
        plt.close(fig)


def print_text_summary(
    curation: dict,
    fold_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    probability_metrics: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
) -> None:
    print("\nExternal validation summary")
    print("===========================")
    print(f"Training overlaps removed: {curation['overlaps_removed']}")
    print(f"Final external validation molecules: {curation['final_external_validation_size']}")
    print(
        "External class balance: "
        f"BBB+={curation['final_bbb_positive_count']}, BBB-={curation['final_bbb_negative_count']}"
    )
    print("\nEvaluated folds/seeds:")
    for name, group in fold_metrics.groupby("model_configuration", sort=False):
        print(f"- {name}: {', '.join(group['fold_or_seed'].astype(str))}")
    print("\nMean +/- std performance:")
    for _, row in summary.iterrows():
        print(
            f"- {row['model_configuration']}: "
            f"balanced_accuracy={row['balanced_accuracy_mean_std']}, "
            f"auprc={row['auprc_mean_std']}, "
            f"mcc={row['mcc_mean_std']}, "
            f"f1={row['f1_mean_std']}"
        )
    print("\nProbability-averaged performance:")
    for _, row in probability_metrics.iterrows():
        print(
            f"- {row['model_configuration']}: "
            f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
            f"auprc={row['auprc']:.4f}, mcc={row['mcc']:.4f}, f1={row['f1']:.4f}"
        )
    ens = ensemble_metrics.iloc[0]
    print("\nBrainRoute ensemble performance:")
    print(
        f"- balanced_accuracy={ens['balanced_accuracy']:.4f}, "
        f"auprc={ens['auprc']:.4f}, mcc={ens['mcc']:.4f}, f1={ens['f1']:.4f}"
    )


def main() -> None:
    parser = script_arg_parser("External validation for selected BrainRoute platform models.")
    parser.add_argument("--external-two", action="store_true", help="Use the second Excel external set as df_external_two_short.")
    parser.add_argument("--external-two-path", default=DEFAULT_EXTERNAL_TWO_PATH)
    parser.add_argument("--padel-2d-only", action="store_true", help="Calculate only 2D PaDEL descriptors for external molecules.")
    parser.add_argument("--padel-chunk-size", type=int, default=PADEL_CHUNK_SIZE)
    parser.add_argument("--padel-chunk-timeout", type=int, default=PADEL_CHUNK_TIMEOUT)
    parser.add_argument("--padel-chunk-max-runtime", type=int, default=PADEL_CHUNK_MAX_RUNTIME)
    parser.add_argument("--padel-single-timeout", type=int, default=PADEL_SINGLE_TIMEOUT)
    parser.add_argument("--padel-single-max-runtime", type=int, default=PADEL_SINGLE_MAX_RUNTIME)
    parser.add_argument("--padel-workers", type=int, default=1)
    parser.add_argument("--results-subdir", default=RESULTS_SUBDIR)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["_external_results_subdir"] = args.results_subdir
    cfg["_padel_2d_only"] = args.padel_2d_only
    cfg["_padel_chunk_size"] = args.padel_chunk_size
    cfg["_padel_chunk_timeout"] = args.padel_chunk_timeout
    cfg["_padel_chunk_max_runtime"] = args.padel_chunk_max_runtime
    cfg["_padel_single_timeout"] = args.padel_single_timeout
    cfg["_padel_single_max_runtime"] = args.padel_single_max_runtime
    cfg["_padel_workers"] = args.padel_workers
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)
    output = out_dir(cfg)

    if args.external_two:
        LOGGER.info("Loading second external validation data from %s", args.external_two_path)
        ext_raw = read_external_two_short(args.external_two_path)
    else:
        ext_path = resolve_external_path(cfg)
        if ext_path is None:
            raise FileNotFoundError(
                "No external validation file configured and fallback data/external/B3DB_classification_external.tsv was not found."
            )
        LOGGER.info("Loading external validation data from %s", ext_path)
        ext_raw = read_table(ext_path)
    internal = read_table(project_path(cfg, "data/processed/standardized_molecules.csv"))

    ext_std, invalid = standardize_external(ext_raw, cfg)
    cleaned, overlaps = remove_training_overlaps(ext_std, internal)
    cleaned = add_similarity_to_training(cleaned, internal, cfg)
    write_output(invalid, output / "external_invalid_or_failed_standardization.csv")
    write_output(overlaps, output / "external_training_overlaps_removed.csv")

    feature_views, padel_failed = build_feature_views(cleaned, cfg)
    if padel_failed:
        padel_failed_set = set(padel_failed)
        padel_failed_df = cleaned.iloc[sorted(padel_failed_set)].copy()
        write_output(padel_failed_df, output / "external_padel_failed_molecules.csv")
        cleaned = cleaned.loc[[i for i in cleaned.index if i not in padel_failed_set]].reset_index(drop=True)
        for view, X in feature_views.items():
            feature_views[view] = X.loc[[i for i in X.index if i not in padel_failed_set]].reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No external molecules remained after curation and feature generation.")

    curation = {
        "original_external_set_size": int(len(ext_raw)),
        "invalid_or_failed_molecules": int(len(invalid) + len(padel_failed)),
        "overlaps_removed": int(len(overlaps)),
        "final_external_validation_size": int(len(cleaned)),
        "final_bbb_positive_count": int((cleaned["label"] == 1).sum()),
        "final_bbb_negative_count": int((cleaned["label"] == 0).sum()),
    }
    write_output(pd.DataFrame([curation]), output / "external_dataset_curation_summary.csv")

    y_true = cleaned["label"].astype(int).to_numpy()
    fold_metric_rows = []
    prediction_rows = []
    for model_cfg in SELECTED_MODELS:
        X_view = feature_views.get(model_cfg.feature_view)
        if X_view is None:
            raise ValueError(f"Feature view unavailable for {model_cfg.display_name}: {model_cfg.feature_view}")
        for fold_or_seed, model_path in selected_artifacts(cfg, model_cfg):
            LOGGER.info("Evaluating %s / %s", model_cfg.display_name, fold_or_seed)
            y_pred, y_score, metrics, _ = predict_with_artifact(model_path, X_view, y_true)
            metrics.update(
                {
                    "model_configuration": model_cfg.display_name,
                    "feature_view": model_cfg.feature_view,
                    "model": model_cfg.model_name,
                    "fold_or_seed": fold_or_seed,
                    "model_artifact": model_path.name,
                    "threshold": THRESHOLD,
                }
            )
            fold_metric_rows.append(metrics)
            for pos, (pred, score) in enumerate(zip(y_pred, y_score)):
                prediction_rows.append(
                    {
                        "external_position": pos,
                        "external_row": cleaned.iloc[pos]["external_row"],
                        "model_configuration": model_cfg.display_name,
                        "fold_or_seed": fold_or_seed,
                        "model_artifact": model_path.name,
                        "y_true": int(y_true[pos]),
                        "y_pred": int(pred),
                        "y_score": float(score),
                    }
                )

    fold_metrics = pd.DataFrame(fold_metric_rows)
    per_artifact_predictions = pd.DataFrame(prediction_rows)
    summary = mean_std_summary(fold_metrics)
    probability_metrics, probability_predictions = probability_averaged_results(per_artifact_predictions, cleaned)
    ensemble_metrics = brainroute_ensemble_results(probability_predictions, y_true)

    write_output(fold_metrics, output / "external_validation_fold_level_metrics.csv")
    write_output(summary, output / "external_validation_summary_mean_std.csv")
    write_output(probability_metrics, output / "external_validation_probability_averaged_results.csv")
    write_output(ensemble_metrics, output / "external_validation_brainroute_ensemble_results.csv")
    write_output(per_artifact_predictions, output / "external_validation_per_artifact_predictions.csv")
    write_output(probability_predictions, output / "external_validation_cleaned_predictions.csv")

    duplicate_prediction_cols = [col for col in probability_predictions.columns if col in cleaned.columns and col != "external_row"]
    cleaned_with_outputs = cleaned.merge(
        probability_predictions.drop(columns=duplicate_prediction_cols),
        on="external_row",
        how="left",
    )
    write_output(cleaned_with_outputs, output / "cleaned_non_overlapping_external_validation_dataframe.csv")

    save_figures(fold_metrics, probability_metrics, ensemble_metrics, probability_predictions, cleaned, output)
    print_text_summary(curation, fold_metrics, summary, probability_metrics, ensemble_metrics)


if __name__ == "__main__":
    main()
