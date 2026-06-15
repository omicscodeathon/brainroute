#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from brainroute_ml_validation.src.utils import LOGGER, ensure_dirs, load_config, project_path, script_arg_parser, set_global_seed, write_csv


SCAFFOLD_FOLDS = [f"scaffold_cv_fold{i}" for i in range(1, 6)]


def read_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def table_md(df: pd.DataFrame, title: str, max_rows: int = 20) -> str:
    if df.empty:
        return f"\n## {title}\n\nNot available.\n"
    small = df.head(max_rows).fillna("")
    header = "| " + " | ".join(map(str, small.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(small.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in small.to_numpy()]
    return f"\n## {title}\n\n" + "\n".join([header, separator, *rows]) + "\n"


def scaffold_cv_ranking(perf: pd.DataFrame) -> pd.DataFrame:
    if perf.empty:
        return pd.DataFrame()
    primary = perf[perf["split"].str.startswith("scaffold_cv_fold", na=False)].copy()
    if primary.empty:
        return pd.DataFrame()
    rank = (
        primary.groupby(["feature_view", "model"])
        .agg(
            n=("balanced_accuracy", "size"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            auprc_mean=("auprc", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            mcc_mean=("mcc", "mean"),
            f1_mean=("f1", "mean"),
        )
        .reset_index()
        .sort_values(["balanced_accuracy_mean", "auprc_mean", "mcc_mean"], ascending=False)
    )
    return rank


def safe_name(*parts: str) -> str:
    return "__".join(str(p).replace("/", "_").replace(" ", "_") for p in parts)


def savefig(path: Path, dpi: int = 220) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def model_feature_importance(model) -> pd.DataFrame:
    from brainroute_ml_validation.src.modeling import selected_features_from_pipeline

    if not hasattr(model, "named_steps") or "clf" not in model.named_steps:
        return pd.DataFrame()
    features = selected_features_from_pipeline(model)
    clf = model.named_steps["clf"]
    values = None
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        coef = clf.coef_
        values = abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else abs(coef)
    if values is None or len(features) != len(values):
        return pd.DataFrame()
    out = pd.DataFrame({"feature": features, "importance": values})
    return out.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)


def aggregate_scaffold_feature_importance(cfg: dict, feature_view: str, model_name: str) -> pd.DataFrame:
    import joblib

    model_dir = project_path(cfg, "models")
    frames = []
    for fold in SCAFFOLD_FOLDS:
        path = model_dir / f"{feature_view}__{model_name}__{fold}.joblib"
        if not path.exists():
            continue
        importance = model_feature_importance(joblib.load(path))
        if importance.empty:
            continue
        importance["fold"] = fold
        frames.append(importance)
    if not frames:
        return pd.DataFrame()
    all_imp = pd.concat(frames, ignore_index=True)
    return (
        all_imp.groupby("feature")
        .agg(mean_importance=("importance", "mean"), std_importance=("importance", "std"), n_folds=("fold", "nunique"))
        .reset_index()
        .sort_values("mean_importance", ascending=False)
    )


def make_feature_importance_figures(cfg: dict, top_rank: pd.DataFrame) -> pd.DataFrame:
    if top_rank.empty:
        return pd.DataFrame()
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = project_path(cfg, "reports/figures")
    report_dir = project_path(cfg, "reports")
    summary = []
    for _, row in top_rank.head(10).iterrows():
        view = row["feature_view"]
        model = row["model"]
        imp = aggregate_scaffold_feature_importance(cfg, view, model)
        if imp.empty:
            summary.append({"feature_view": view, "model": model, "status": "feature_importance_unavailable"})
            continue
        out_csv = report_dir / f"feature_importance_top10__{safe_name(view, model)}.csv"
        imp.to_csv(out_csv, index=False)
        plot_df = imp.head(20).iloc[::-1]
        plt.figure(figsize=(8, 6))
        sns.barplot(data=plot_df, x="mean_importance", y="feature", color="#4C78A8")
        plt.xlabel("Mean feature importance across scaffold CV folds")
        plt.ylabel("")
        plt.title(f"{view} / {model}")
        savefig(fig_dir / f"feature_importance_top20__{safe_name(view, model)}.png")
        summary.append({"feature_view": view, "model": model, "status": "written", "csv": str(out_csv)})
    return pd.DataFrame(summary)


def make_best_model_feature_correlation_heatmap(cfg: dict, top_rank: pd.DataFrame) -> None:
    if top_rank.empty:
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    from brainroute_ml_validation.src.preprocessing import finite_dataframe

    best = top_rank.iloc[0]
    view = best["feature_view"]
    model = best["model"]
    imp = aggregate_scaffold_feature_importance(cfg, view, model)
    if imp.empty:
        return
    top_features = imp.head(15)["feature"].tolist()
    feature_path = project_path(cfg, f"data/processed/features_{view}.csv")
    if not feature_path.exists():
        return
    available = pd.read_csv(feature_path, nrows=0).columns.tolist()
    usecols = [f for f in top_features if f in available]
    if len(usecols) < 2:
        return
    X = pd.read_csv(feature_path, usecols=usecols)
    corr = finite_dataframe(X).corr(method="pearson")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, square=True, linewidths=0.2)
    plt.title(f"Top-feature correlation: {view} / {model}")
    savefig(project_path(cfg, "reports/figures/best_model_top15_feature_correlation_heatmap.png"), dpi=240)
    pd.DataFrame({"feature": usecols}).to_csv(project_path(cfg, "reports/best_model_top15_correlation_features.csv"), index=False)


def make_curve_figures(cfg: dict, top_rank: pd.DataFrame) -> None:
    if top_rank.empty:
        return
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, precision_recall_curve, roc_curve

    pred_path = project_path(cfg, "reports/model_predictions_all_splits.csv")
    if not pred_path.exists():
        return
    preds = pd.read_csv(pred_path, usecols=["y_true", "y_score", "feature_view", "model", "split"])
    preds = preds[preds["split"].str.startswith("scaffold_cv_fold", na=False)].dropna(subset=["y_true", "y_score"])
    if preds.empty:
        return

    fig_dir = project_path(cfg, "reports/figures")
    top10 = top_rank.head(10)[["feature_view", "model"]].itertuples(index=False, name=None)

    plt.figure(figsize=(8, 7))
    for view, model in top10:
        df = preds[(preds["feature_view"] == view) & (preds["model"] == model)]
        if df.empty or df["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(df["y_true"], df["y_score"])
        plt.plot(fpr, tpr, linewidth=1.7, label=f"{view}/{model} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Scaffold-CV ROC curves for top 10 models")
    plt.legend(fontsize=7, loc="lower right")
    savefig(fig_dir / "roc_curves_top10_scaffold_cv.png")

    plt.figure(figsize=(8, 7))
    for view, model in top_rank.head(10)[["feature_view", "model"]].itertuples(index=False, name=None):
        df = preds[(preds["feature_view"] == view) & (preds["model"] == model)]
        if df.empty or df["y_true"].nunique() < 2:
            continue
        precision, recall, _ = precision_recall_curve(df["y_true"], df["y_score"])
        plt.plot(recall, precision, linewidth=1.7, label=f"{view}/{model} (AUPRC={auc(recall, precision):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Scaffold-CV PR curves for top 10 models")
    plt.legend(fontsize=7, loc="lower left")
    savefig(fig_dir / "pr_curves_top10_scaffold_cv.png")


def make_figures(cfg: dict, perf: pd.DataFrame) -> None:
    if perf.empty:
        return
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig_dir = project_path(cfg, "reports/figures")
    primary = perf[perf["split"].str.startswith("scaffold_cv_fold", na=False)]
    top_rank = scaffold_cv_ranking(perf)
    if not top_rank.empty:
        top_rank.to_csv(project_path(cfg, "reports/scaffold_cv_model_ranking.csv"), index=False)

    def blue_palette_for(data: pd.DataFrame) -> list:
        n = max(data["feature_view"].nunique(), 1)
        return sns.color_palette("Blues", n_colors=n + 3)[2 : 2 + n]

    for metric in ["balanced_accuracy", "auprc", "roc_auc", "mcc", "f1"]:
        if primary.empty or metric not in primary:
            continue
        plt.figure(figsize=(11, 5.5))
        sns.barplot(
            data=primary,
            x="model",
            y=metric,
            hue="feature_view",
            errorbar="sd",
            capsize=0.12,
            err_kws={"linewidth": 1.2, "color": "#1f2937"},
            palette=blue_palette_for(primary),
        )
        plt.xticks(rotation=30, ha="right")
        plt.title(f"Scaffold-CV {metric.replace('_', ' ')}")
        savefig(fig_dir / f"scaffold_cv_{metric}_by_model_feature_view.png")

    duplicate = perf[perf["split"].str.startswith("duplicate_aware_seed", na=False)]
    for metric in ["balanced_accuracy", "auprc"]:
        if duplicate.empty or metric not in duplicate:
            continue
        plt.figure(figsize=(11, 5.5))
        sns.barplot(
            data=duplicate,
            x="model",
            y=metric,
            hue="feature_view",
            errorbar="sd",
            capsize=0.12,
            err_kws={"linewidth": 1.2, "color": "#1f2937"},
            palette=blue_palette_for(duplicate),
        )
        plt.xticks(rotation=30, ha="right")
        plt.title(f"Duplicate-aware repeated split {metric.replace('_', ' ')}")
        savefig(fig_dir / f"duplicate_aware_{metric}_by_model_feature_view.png")

    near = read_if_exists(project_path(cfg, "reports/near_duplicate_analysis_all_splits.csv"))
    if not near.empty:
        plot_df = near[near["split"].str.contains("random80|scaffold_split", regex=True, na=False)]
        if not plot_df.empty:
            plt.figure(figsize=(8, 5))
            sns.histplot(data=plot_df, x="max_tanimoto", hue="split", bins=30, element="step", stat="density")
            savefig(fig_dir / "nearest_neighbor_tanimoto_random_vs_scaffold.png")

    make_curve_figures(cfg, top_rank)
    fi_summary = make_feature_importance_figures(cfg, top_rank)
    if not fi_summary.empty:
        fi_summary.to_csv(project_path(cfg, "reports/feature_importance_figure_summary.csv"), index=False)
    make_best_model_feature_correlation_heatmap(cfg, top_rank)


def main() -> None:
    args = script_arg_parser("Create final summary tables, figures, and reviewer-facing text.").parse_args()
    cfg = load_config(args.config)
    set_global_seed(int(cfg.get("random_seed", 42)))
    ensure_dirs(cfg)
    reports = project_path(cfg, "reports")
    accounting = read_if_exists(reports / "data_accounting.csv")
    duplicates = read_if_exists(reports / "duplicate_audit.csv")
    splits = read_if_exists(reports / "split_summary.csv")
    near = read_if_exists(reports / "near_duplicate_similarity_summary.csv")
    perf = read_if_exists(reports / "model_performance_all_splits.csv")
    stats = read_if_exists(reports / "model_statistical_comparison.csv")
    external = read_if_exists(reports / "external_validation_metrics.csv")

    primary = perf[perf["split"].str.startswith("scaffold_cv_fold", na=False)] if not perf.empty else pd.DataFrame()
    duplicate_perf = perf[perf["split"].str.startswith("duplicate_aware_seed", na=False)] if not perf.empty else pd.DataFrame()
    summary_rows = []
    for name, df in [("scaffold_cv_primary", primary), ("duplicate_aware_repeated", duplicate_perf)]:
        if not df.empty:
            s = df.groupby(["feature_view", "model"])[["balanced_accuracy", "auprc", "mcc", "f1"]].agg(["mean", "std"]).reset_index()
            s.columns = ["_".join([str(x) for x in col if x]) for col in s.columns]
            s["validation"] = name
            summary_rows.append(s)
    final_summary = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    write_csv(final_summary, reports / "final_model_results_summary.csv", cfg)

    md = "# BrainRoute strict-validation results summary\n"
    md += table_md(accounting, "Data Accounting")
    md += table_md(duplicates[duplicates.get("is_duplicate", False) == True] if not duplicates.empty else duplicates, "Duplicate Audit")
    md += table_md(splits, "Split Summary")
    md += table_md(near, "Near-Duplicate Similarity Summary")
    md += table_md(final_summary, "Model Performance Summary")
    md += table_md(external, "External Validation")
    md += table_md(stats, "Statistical Comparison")
    (reports / "final_model_results_summary.md").write_text(md, encoding="utf-8")
    (reports / "reviewer_validation_summary.md").write_text(md, encoding="utf-8")

    methods = """# Reviewer Methods Text

Random 80/20 splitting was retained only as a baseline because it can overestimate performance when exact duplicates, close analogs, or related scaffolds appear in both training and test sets.

Molecules were standardized with RDKit, converted to canonical SMILES and InChIKey identifiers, and assigned Bemis-Murcko scaffolds. Exact duplicate audits were performed by InChIKey. Molecules with conflicting labels for the same InChIKey were excluded by default and preserved in audit files.

Duplicate-aware splits used InChIKey groups so an exact molecule could not appear in both train and test sets. Scaffold holdout and five-fold scaffold cross-validation used Bemis-Murcko scaffold groups so the same scaffold was not shared across train/test folds. Scaffold cross-validation was treated as the primary model-selection evidence.

Morgan fingerprints were computed with radius 2, 2048 bits, and chirality enabled. For each split, every test molecule was compared with all training molecules by Tanimoto similarity, and nearest-neighbor similarity summaries were saved at 0.80, 0.85, and 0.90 thresholds.

PaDEL descriptors, Morgan fingerprints, and optional frozen pretrained SMILES-transformer embeddings were treated as separate feature representations. Descriptor missingness filters, variance filters, correlation filters, imputers, scalers, and any model-specific preprocessing were fit only inside the training fold through scikit-learn pipelines.

Class weighting was used as the default imbalance strategy to preserve chemical diversity. External validation, when configured, was performed after standardization, exact-overlap removal, and near-duplicate similarity annotation; the external set was not used for model tuning.

All split files, seeds, configuration files, scripts, predictions, metrics, selected-feature lists, and audit tables are written to disk to support independent reproducibility.
"""
    (reports / "reviewer_methods_text.md").write_text(methods, encoding="utf-8")
    make_figures(cfg, perf)
    LOGGER.info("Wrote final summaries and reviewer methods text.")


if __name__ == "__main__":
    main()
