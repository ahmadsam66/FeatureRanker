#!/usr/bin/env python3
"""
FeatureRank: Multi-method feature correlation and ranking tool for CSV/TSV files.

Supports 1 or 2 target columns and automatically adapts scoring between
regression and classification tasks for each target.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, r2_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def detect_separator(filepath: Path) -> str:
    """Detect delimiter from file extension or header line."""
    suffix = filepath.suffix.lower()
    if suffix in {".tsv", ".tab"}:
        return "\t"
    if suffix == ".csv":
        return ","
    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    return "\t" if first_line.count("\t") > first_line.count(",") else ","


def is_classification_target(y: pd.Series) -> bool:
    """Infer whether a target column should be treated as classification."""
    if y.dtype.kind in {"b", "O", "U", "S"} or isinstance(y.dtype, pd.CategoricalDtype):
        return True
    non_null = y.dropna()
    n_unique = non_null.nunique()
    if n_unique <= 2:
        return True
    if y.dtype.kind in {"i", "u"} and n_unique <= 15:
        return True
    return False


def prepare_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prepare a clean numeric feature matrix for correlation and statistical tests,
    while returning column lists for full preprocessing in ML pipelines.
    """
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    parts: List[pd.DataFrame] = []

    if num_cols:
        num_imp = SimpleImputer(strategy="median")
        num_arr = num_imp.fit_transform(X[num_cols])
        parts.append(pd.DataFrame(num_arr, columns=num_cols, index=X.index))

    if cat_cols:
        # Frequency encoding gives a compact numeric representation for correlation/ANOVA
        cat_encoded = pd.DataFrame(index=X.index)
        for col in cat_cols:
            s = X[col].astype("string").fillna("__MISSING__")
            freq = s.value_counts(normalize=True)
            cat_encoded[col] = s.map(freq).astype(float)
        parts.append(cat_encoded)

    X_numeric = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=X.index)
    return X_numeric, num_cols, cat_cols


def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """Build a scikit-learn ColumnTransformer for ML models."""
    transformers = []
    if num_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def compute_correlations(
    X_num: pd.DataFrame,
    targets_df: pd.DataFrame,
    out_dir: Path,
    top_n_heatmap: int = 30,
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations between features and target(s),
    save correlations.tsv, and render a correlation heatmap PDF per target.
    """
    records = []

    for target_col in targets_df.columns:
        y_raw = targets_df[target_col]
        if is_classification_target(y_raw):
            y_num, _ = pd.factorize(y_raw.astype("string").fillna("__MISSING__"))
            y_series = pd.Series(y_num, index=targets_df.index, name=target_col, dtype=float)
        else:
            y_series = pd.to_numeric(y_raw, errors="coerce")

        valid_mask = y_series.notna()
        x_valid = X_num.loc[valid_mask]
        y_valid = y_series.loc[valid_mask]

        pearson_vals = {}
        spearman_vals = {}

        for col in x_valid.columns:
            xv = x_valid[col].to_numpy(dtype=float)
            yv = y_valid.to_numpy(dtype=float)

            if np.std(xv) == 0 or np.std(yv) == 0:
                pearson_vals[col] = 0.0
                spearman_vals[col] = 0.0
                continue

            p_corr = float(np.corrcoef(xv, yv)[0, 1])
            s_corr, _ = spearmanr(xv, yv)
            pearson_vals[col] = 0.0 if np.isnan(p_corr) else p_corr
            spearman_vals[col] = 0.0 if np.isnan(s_corr) else float(s_corr)

        for col in X_num.columns:
            records.append(
                {
                    "target": target_col,
                    "feature": col,
                    "pearson_r": pearson_vals[col],
                    "abs_pearson_r": abs(pearson_vals[col]),
                    "spearman_rho": spearman_vals[col],
                    "abs_spearman_rho": abs(spearman_vals[col]),
                }
            )

        # Build correlation matrix for top correlated features + target for heatmap PDF
        rank_series = pd.Series({c: abs(pearson_vals[c]) for c in x_valid.columns}).sort_values(ascending=False)
        selected_cols = rank_series.head(top_n_heatmap).index.tolist()
        plot_df = x_valid[selected_cols].copy()
        plot_df[target_col] = y_valid
        corr_matrix = plot_df.corr(method="spearman")

        fig_size = max(7.0, min(18.0, 0.38 * len(corr_matrix.columns) + 2.0))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
        sns.heatmap(
            corr_matrix,
            cmap="coolwarm",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            square=True,
            linewidths=0.4,
            cbar_kws={"shrink": 0.8, "label": "Spearman Correlation"},
            ax=ax,
        )
        ax.set_title(f"Spearman Correlation Heatmap — Target: {target_col}", fontsize=13, pad=12)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        safe_target = "".join(c if c.isalnum() or c in {"_", "-"} else "_" for c in target_col)
        pdf_path = out_dir / f"{safe_target}_correlation_heatmap.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    corr_df = pd.DataFrame(records)
    corr_df.to_csv(out_dir / "correlations.tsv", sep="\t", index=False)
    return corr_df


def _map_transformed_importances_to_features(
    preprocessor: ColumnTransformer,
    num_cols: List[str],
    cat_cols: List[str],
    transformed_importances: np.ndarray,
) -> Dict[str, float]:
    """Aggregate importances from one-hot / scaled columns back to original input features."""
    scores: Dict[str, float] = {col: 0.0 for col in (num_cols + cat_cols)}
    idx = 0

    if num_cols:
        n_num = len(num_cols)
        for col, imp in zip(num_cols, transformed_importances[idx : idx + n_num]):
            scores[col] += float(max(0.0, imp))
        idx += n_num

    if cat_cols:
        ohe: OneHotEncoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = ohe.get_feature_names_out(cat_cols)
        for out_name, imp in zip(cat_feature_names, transformed_importances[idx : idx + len(cat_feature_names)]):
            # Match back to original categorical column prefix
            matched_col = None
            for col in cat_cols:
                if out_name == col or out_name.startswith(f"{col}_"):
                    matched_col = col
                    break
            if matched_col is not None:
                scores[matched_col] += float(max(0.0, imp))

    return scores


def rank_features_for_target(
    X_raw: pd.DataFrame,
    X_num: pd.DataFrame,
    y_raw: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute feature importance scores and ranks using:
      1. Spearman Rank Correlation (|rho|) [Statistical / Non-parametric]
      2. Mutual Information [Information-theoretic]
      3. ANOVA F-score / F-regression [Linear / Parametric statistical]
      4. Extra Trees Ensemble Importance (ML Method 1)
      5. Permutation Importance with Random Forest (ML Method 2)
    """
    is_clf = is_classification_target(y_raw)
    if is_clf:
        y_target, _ = pd.factorize(y_raw.astype("string").fillna("__MISSING__"))
        y_arr = np.asarray(y_target, dtype=int)
    else:
        y_num = pd.to_numeric(y_raw, errors="coerce")
        y_arr = y_num.to_numpy(dtype=float)

    valid_mask = ~np.isnan(y_arr) if not is_clf else np.ones(len(y_arr), dtype=bool)
    X_num_v = X_num.loc[valid_mask]
    X_raw_v = X_raw.loc[valid_mask]
    y_v = y_arr[valid_mask]

    features = list(X_num.columns)

    # Method 1 (Classic 1): Absolute Spearman correlation
    s1_scores: Dict[str, float] = {}
    for col in features:
        xv = X_num_v[col].to_numpy(dtype=float)
        if np.std(xv) == 0 or np.std(y_v) == 0:
            s1_scores[col] = 0.0
        else:
            rho, _ = spearmanr(xv, y_v)
            s1_scores[col] = 0.0 if np.isnan(rho) else float(abs(rho))

    # Method 2 (Classic 2): Mutual Information
    if len(features) > 0:
        if is_clf:
            mi_vals = mutual_info_classif(X_num_v, y_v, random_state=random_state)
        else:
            mi_vals = mutual_info_regression(X_num_v, y_v, random_state=random_state)
        s2_scores = {col: float(max(0.0, val)) for col, val in zip(features, mi_vals)}
    else:
        s2_scores = {}

    # Method 3 (Classic 3): ANOVA F-test / F-regression statistic
    if len(features) > 0:
        if is_clf:
            f_vals, _ = f_classif(X_num_v, y_v)
        else:
            f_vals, _ = f_regression(X_num_v, y_v)
        s3_scores = {
            col: float(max(0.0, 0.0 if np.isnan(val) or np.isinf(val) else val))
            for col, val in zip(features, f_vals)
        }
    else:
        s3_scores = {}

    # ML Method 1 (Method 4): Extremely Randomized Trees (ExtraTrees) impurity importance
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_proc = preprocessor.fit_transform(X_raw_v)

    if is_clf:
        et_model = ExtraTreesClassifier(n_estimators=200, random_state=random_state, n_jobs=-1)
    else:
        et_model = ExtraTreesRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)

    et_model.fit(X_proc, y_v)
    s4_scores = _map_transformed_importances_to_features(
        preprocessor, num_cols, cat_cols, et_model.feature_importances_
    )

    # ML Method 2 (Method 5): Random Forest Permutation Importance
    if is_clf:
        rf_model = RandomForestClassifier(n_estimators=150, random_state=random_state, n_jobs=-1)
        scoring = "accuracy" if len(np.unique(y_v)) > 2 else "roc_auc"
    else:
        rf_model = RandomForestRegressor(n_estimators=150, random_state=random_state, n_jobs=-1)
        scoring = "r2"

    rf_model.fit(X_proc, y_v)
    # Use a held-out split if dataset is large enough, else fit set for fast deterministic ranking
    n_samples = len(y_v)
    if n_samples >= 30:
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n_samples)
        split = int(0.8 * n_samples)
        eval_idx = perm[split:]
        perm_res = permutation_importance(
            rf_model,
            X_proc[eval_idx],
            y_v[eval_idx],
            n_repeats=8,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1,
        )
    else:
        perm_res = permutation_importance(
            rf_model,
            X_proc,
            y_v,
            n_repeats=8,
            random_state=random_state,
            scoring=scoring,
            n_jobs=-1,
        )

    s5_scores = _map_transformed_importances_to_features(
        preprocessor, num_cols, cat_cols, perm_res.importances_mean
    )

    method_score_map = {
        "spearman_abs": s1_scores,
        "mutual_info": s2_scores,
        "anova_f": s3_scores,
        "ml_extratrees": s4_scores,
        "ml_permutation_rf": s5_scores,
    }

    df = pd.DataFrame({"feature": features})
    norm_cols = []

    for m_name, score_dict in method_score_map.items():
        raw_s = df["feature"].map(score_dict).fillna(0.0).astype(float)
        df[f"score_{m_name}"] = raw_s
        df[f"rank_{m_name}"] = raw_s.rank(ascending=False, method="min").astype(int)

        max_val = raw_s.max()
        norm_s = (raw_s / max_val) if max_val > 0 else pd.Series(0.0, index=df.index)
        norm_col = f"norm_{m_name}"
        df[norm_col] = norm_s
        norm_cols.append(norm_col)

    df["consensus_score"] = df[norm_cols].mean(axis=1)
    df["consensus_rank"] = df["consensus_score"].rank(ascending=False, method="min").astype(int)
    df["task_type"] = "classification" if is_clf else "regression"

    return df.sort_values("consensus_rank", ascending=True).reset_index(drop=True)


def run_featurerank(
    input_path: Path,
    targets: List[str],
    out_dir: Path,
    top_n_heatmap: int = 30,
    random_state: int = 42,
) -> None:
    if not (1 <= len(targets) <= 2):
        raise ValueError("Please provide 1 or 2 target columns via --targets.")

    sep = detect_separator(input_path)
    df = pd.read_csv(input_path, sep=sep)

    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        raise ValueError(f"Target column(s) not found in input file: {missing_targets}")

    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in targets]
    if not feature_cols:
        raise ValueError("No feature columns remain after excluding target column(s).")

    X_raw = df[feature_cols]
    targets_df = df[targets]

    X_num, num_cols, cat_cols = prepare_features(X_raw)

    # 1. Correlation TSV & Heatmap PDF(s)
    compute_correlations(X_num=X_num, targets_df=targets_df, out_dir=out_dir, top_n_heatmap=top_n_heatmap)

    # 2. Feature Rankings across 3 classic + 2 ML methods
    all_rankings: List[pd.DataFrame] = []
    long_rows: List[dict] = []

    methods = [
        ("spearman_abs", "statistical"),
        ("mutual_info", "statistical"),
        ("anova_f", "statistical"),
        ("ml_extratrees", "ml"),
        ("ml_permutation_rf", "ml"),
    ]

    for target_col in targets:
        t_rank_df = rank_features_for_target(
            X_raw=X_raw,
            X_num=X_num,
            y_raw=targets_df[target_col],
            num_cols=num_cols,
            cat_cols=cat_cols,
            random_state=random_state,
        )
        t_rank_df.insert(0, "target", target_col)
        all_rankings.append(t_rank_df)

        for _, row in t_rank_df.iterrows():
            for m_name, m_group in methods:
                long_rows.append(
                    {
                        "target": target_col,
                        "task_type": row["task_type"],
                        "feature": row["feature"],
                        "method": m_name,
                        "method_category": m_group,
                        "score": row[f"score_{m_name}"],
                        "normalized_score": row[f"norm_{m_name}"],
                        "rank": int(row[f"rank_{m_name}"]),
                        "consensus_rank": int(row["consensus_rank"]),
                    }
                )

    combined_rankings = pd.concat(all_rankings, ignore_index=True)
    combined_rankings.to_csv(out_dir / "feature_rankings.tsv", sep="\t", index=False)

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(out_dir / "feature_rankings_long.tsv", sep="\t", index=False)

    print(f"Done. Outputs written to: {out_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="featurerank",
        description="Rank features from a CSV or TSV file using 3 statistical methods + 2 ML methods.",
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Path to input .csv or .tsv file.")
    parser.add_argument(
        "-t",
        "--targets",
        required=True,
        nargs="+",
        help="One or two target column names (e.g., -t label or -t target1 target2).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=Path("featurerank_output"),
        type=Path,
        help="Directory where output files will be saved (default: featurerank_output).",
    )
    parser.add_argument(
        "--top-n-heatmap",
        default=30,
        type=int,
        help="Maximum top features to include in each correlation heatmap PDF (default: 30).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_featurerank(
        input_path=args.input,
        targets=args.targets,
        out_dir=args.output_dir,
        top_n_heatmap=args.top_n_heatmap,
        random_state=args.seed,
    )