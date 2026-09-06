from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


NON_MODEL_PATTERNS = [
    r"^label$",
    r"^bbb$",
    r"bbb\+",
    r"bbb-",
    r"smiles",
    r"inchi",
    r"inchikey",
    r"scaffold",
    r"murcko",
    r"source",
    r"dataset",
    r"name",
    r"compound",
    r"iupac",
    r"^cid$",
    r"reference",
    r"comment",
    r"tag",
    r"verification",
    r"prediction",
    r"molecule_id",
]


def finite_dataframe(X, columns=None, max_abs_value: float = 1e12) -> pd.DataFrame:
    """Return numeric dataframe with infinities/extreme descriptor values as NaN."""
    X_df = pd.DataFrame(X, columns=columns)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    return X_df.mask(X_df.abs() > max_abs_value)


def excluded_column_reason(column: str) -> str | None:
    c = str(column).lower()
    for pattern in NON_MODEL_PATTERNS:
        if re.search(pattern, c):
            return f"matched pattern '{pattern}'"
    return None


def numeric_model_columns(df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    excluded = []
    keep = []
    for col in df.columns:
        reason = excluded_column_reason(col)
        if reason:
            excluded.append({"column": col, "reason": reason})
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            excluded.append({"column": col, "reason": "non_numeric_or_all_missing"})
            continue
        keep.append(col)
    return keep, pd.DataFrame(excluded)


class MissingnessFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        missing = X_df.isna().mean()
        self.keep_mask_ = (missing <= self.threshold).to_numpy()
        self.selected_features_ = list(np.array(self.feature_names_in_)[self.keep_mask_])
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names_in_).loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


class NonFiniteCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, max_abs_value: float = 1e12):
        self.max_abs_value = max_abs_value

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        return self

    def transform(self, X):
        return finite_dataframe(X, columns=self.feature_names_in_, max_abs_value=self.max_abs_value)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class LowVarianceFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        variances = X_df.var(axis=0, skipna=True)
        self.keep_mask_ = (variances > self.threshold).fillna(False).to_numpy()
        self.selected_features_ = list(np.array(self.feature_names_in_)[self.keep_mask_])
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names_in_).loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)


class MedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        self.medians_ = X_df.median(axis=0, skipna=True).fillna(0)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names_in_)
        return X_df.fillna(self.medians_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold: float = 0.95, max_features: int | None = None):
        self.threshold = threshold
        self.max_features = max_features

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        self.feature_names_in_ = list(X_df.columns)
        if X_df.shape[1] <= 1:
            self.selected_features_ = self.feature_names_in_
            return self

        corr = X_df.corr(method="pearson").abs()
        missing = X_df.isna().mean()
        variance = X_df.var(axis=0, skipna=True).fillna(0)
        to_drop: set[str] = set()
        cols = list(corr.columns)
        for i, col_i in enumerate(cols):
            if col_i in to_drop:
                continue
            high = corr.index[(corr[col_i] > self.threshold) & (corr.index != col_i)].tolist()
            for col_j in high:
                if col_j in to_drop:
                    continue
                if missing[col_i] > missing[col_j]:
                    drop = col_i
                elif missing[col_j] > missing[col_i]:
                    drop = col_j
                else:
                    drop = col_i if variance[col_i] < variance[col_j] else col_j
                to_drop.add(drop)

        selected = [c for c in self.feature_names_in_ if c not in to_drop]
        if self.max_features is not None and len(selected) > self.max_features:
            ranked = variance[selected].sort_values(ascending=False)
            selected = ranked.index[: self.max_features].tolist()
        self.selected_features_ = selected
        self.dropped_features_ = sorted(to_drop)
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.feature_names_in_).loc[:, self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.selected_features_, dtype=object)
