# FeatureRanker (`FeatureRanker.py`)

**FeatureRanker** is a lightweight command-line tool in Python for exploratory feature correlation and multi-method feature ranking on **CSV** and **TSV** datasets with **1 or 2 target columns**. It automatically handles missing values, mixed numerical and categorical predictors, and task inference (classification vs. regression) so you can quickly identify top predictive features and relationship patterns across multiple statistical and machine learning perspectives.

Created and maintained by **Ahmad Mousavi** ahmad.moousavi@gmail.com

---

## Introduction

Feature selection and exploratory correlation analysis are foundational steps in data science and machine learning workflows. Relying on a single metric can miss non-linear relationships, interaction effects, or model-specific behavior. **FeatureRanker** combines **3 statistical/information-theoretic methods** with **2 tree-based machine learning methods** and summarizes them into a unified consensus ranking per target:

1. **Spearman Rank Correlation (`spearman_abs`)** — Measures monotonic non-parametric association (`|\rho|`) between each feature and the target.
2. **Mutual Information (`mutual_info`)** — Captures both linear and non-linear statistical dependency (`mutual_info_classif` or `mutual_info_regression`).
3. **ANOVA F-score / F-regression (`anova_f`)** — Evaluates parametric variance ratios (`f_classif` for classification, `f_regression` for regression).
4. **Extra Trees Importance (`ml_extratrees`)** — Uses an ensemble of Extremely Randomized Trees (`ExtraTreesClassifier` / `ExtraTreesRegressor`) and aggregates impurity importances back to original input features.
5. **Random Forest Permutation Importance (`ml_permutation_rf`)** — Measures the drop in validation/evaluation performance (`roc_auc` or `accuracy` for classification, `r^2` for regression) when a feature's values are permuted.

Each method's score is normalized to $[0, 1]$ by dividing by the maximum score for that method, and a mean `consensus_score` (`consensus_rank`) is calculated across all five methods.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ahmadsam66/FeatureRanker.git
cd FeatureRanker
```

*(If you placed `FeatureRanker.py` directly in your working directory, you can skip cloning and run it directly.)*

### 2. Install Dependencies

Make sure you have **Python 3.9+** installed, then install the required packages:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

---

## Usage

Run `FeatureRanker.py` from the command line with an input dataset and one or two target column names.

### Single Target (CSV)

```bash
python FeatureRanker.py -i data/dataset.csv -t outcome -o results/single_target
```

### Two Targets (TSV)

```bash
python FeatureRanker.py -i data/measurements.tsv -t target_a target_b -o results/dual_target
```

### Customizing Heatmap Size and Random Seed

```bash
python FeatureRanker.py \
  -i data/features.csv \
  -t label \
  -o results/custom_run \
  --top-n-heatmap 20 \
  --seed 123
```

### Command-Line Arguments

| Flag | Long Flag | Required | Default | Description |
| :--- | :--- | :---: | :--- | :--- |
| `-i` | `--input` | Yes | — | Path to the input `.csv` or `.tsv` file. |
| `-t` | `--targets` | Yes | — | One or two target column names separated by spaces (e.g., `-t y` or `-t y1 y2`). |
| `-o` | `--output-dir` | No | `FeatureRanker_output` | Output directory where generated files are written. |
| — | `--top-n-heatmap` | No | `30` | Maximum number of top features included alongside the target in each correlation heatmap PDF. |
| — | `--seed` | No | `42` | Random seed used for reproducibility in ML models, mutual information, and permutation importance. |

---

## Inputs & Preprocessing

### Input Requirements
- **File Formats**: Comma-separated (`.csv`) or tab-separated (`.tsv` / `.tab`) files. Delimiters are detected automatically from the file extension or header line.
- **Target Columns**: **1 or 2** existing column names passed via `--targets`. All remaining columns are treated as candidate input features.
- **Target Task Inference**: For each target column, FeatureRanker automatically determines whether to treat it as **classification** or **regression** based on data type (`bool`, `object`, `string`, `category`) and unique value count.

### Feature Handling
- **Numerical & Boolean Columns**: Missing values are imputed using the median. For ML pipelines (`ExtraTrees` and `RandomForest`), numerical features are standardized with `StandardScaler`.
- **Categorical Columns**: Missing values are handled (`__MISSING__` for numeric/correlation encoding and most-frequent imputation for ML pipelines). Frequency encoding is used for correlation and statistical scoring, while `OneHotEncoder(handle_unknown="ignore")` is used in the ML pipelines and mapped back to original feature names.

---

## Outputs

All outputs are saved inside the specified `--output-dir`:

1. **`<target>_correlation_heatmap.pdf`** *(one PDF per target)*
   - A publication-ready Spearman correlation heatmap showing the top correlated features (up to `--top-n-heatmap`) together with the target column on a diverging `[-1, 1]` scale.
2. **`correlations.tsv`**
   - Tab-separated table containing pairwise feature-to-target correlation metrics for every feature and target:
     - `target`, `feature`
     - `pearson_r`, `abs_pearson_r`
     - `spearman_rho`, `abs_spearman_rho`
3. **`feature_rankings.tsv`**
   - Wide-format tab-separated ranking table per target with raw scores (`score_<method>`), ranks (`rank_<method>`), normalized `[0, 1]` scores (`norm_<method>`), `consensus_score`, `consensus_rank`, and inferred `task_type` (`classification` or `regression`).
4. **`feature_rankings_long.tsv`**
   - Tidy/long-format tab-separated table with columns `target`, `task_type`, `feature`, `method`, `method_category` (`statistical` or `ml`), `score`, `normalized_score`, `rank`, and `consensus_rank`—ideal for custom visualization scripts (e.g., `ggplot2`, `seaborn`, `pandas`) or filtering.

---

## Applications

- **Tabular Machine Learning & Feature Selection**: Quickly narrow down high-dimensional feature tables before training gradient boosting, linear models, or neural networks.
- **Multi-Outcome / Dual-Target Studies**: Compare how the same set of features ranks across two related targets or outcomes (for example, two clinical endpoints, two performance metrics, or two experimental readouts) in a single run.
- **Bioinformatics & Omics / Lab Data**: Screen assay measurements, clinical variables, or engineered descriptors for linear and non-linear associations with phenotypes or continuous responses.
- **Business, Operations & Finance Analytics**: Identify key drivers of churn, conversion, revenue, or risk while checking for non-linear tree-based importance that simple linear correlation might overlook.
- **Rapid EDA & Reporting**: Generate clean PDF correlation heatmaps and structured TSV tables for reproducible lab notebooks, pipelines, and team reviews.
