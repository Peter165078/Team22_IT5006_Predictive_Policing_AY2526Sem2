# Handoff Note for Phase 2 Part 3 and Part 4

This note is for teammates responsible for:

1. `Model Evaluation & Comparison`
2. `Model Interpretation & Business Insights`

It summarizes what has already been completed in Part 1 and Part 2, what artifacts are available, and how to continue from the current repository state.

## 1. What Part 1 and Part 2 have completed

### 1.1 Problem definition

The current modeling task is framed as a **binary classification problem**:

- input: a location-time instance
- output: whether a crime occurrence is predicted (`target = 1`) or not (`target = 0`)

This is an occurrence-forecasting setup rather than crime-type prediction. The intention is to estimate whether a district/grid and time context should be considered higher risk.

### 1.2 Data preparation decisions already made

The Part 1 and Part 2 pipeline uses the Chicago crime archives that are already split by year under:

- `apps/dashboard/split_data_by_year/`

For the current reproducible benchmark, the training run uses:

- years: `2022-2024`
- rows sampled per year: `20,000`
- total positive rows before negative sampling: `60,000`
- total rows after 1:1 negative generation: `120,000`

The processed dataset summary is stored in:

- `artifacts/metrics/phase2_data_summary.json`

Key numbers from the current run:

| Item | Value |
| --- | ---: |
| Positive rows | 60,000 |
| Negative rows | 60,000 |
| Total rows | 120,000 |
| Feature count | 37 |
| Train rows | 83,820 |
| Validation rows | 17,999 |
| Test rows | 17,999 |
| Historical windows | 7d, 14d, 30d, 90d |

### 1.3 Leakage and feature policy

Please keep this assumption consistent when writing Parts 3 and 4:

- We are predicting **crime occurrence**, not crime category.
- Therefore, post-incident fields such as `Primary Type`, `Description`, `IUCR`, `Arrest`, and `Domestic` were deliberately removed from model inputs.

This was an intentional correction to avoid target leakage. Any interpretation section should state that the evaluation is based on **pre-event spatial-temporal and historical features only**.

### 1.4 Feature groups used by the models

The current models are trained on three main groups of features:

- **Temporal features**
  - hour
  - day of week
  - month
  - day of year
  - week of year
  - cyclic encodings
  - weekend / night / rush-hour indicators
- **Spatial features**
  - district
  - ward
  - beat
  - community area
  - latitude / longitude
  - coordinate-based grid cell
- **Historical features**
  - prior crime counts in the same district and grid cell over `7d`, `14d`, `30d`, and `90d`

The exact saved feature list is in:

- `artifacts/metrics/phase2_feature_columns.txt`

## 2. Models already implemented

Three tabular models have been fully trained and saved:

1. `Logistic Regression`
2. `Random Forest`
3. `HistGradientBoosting`

Saved model files:

- `artifacts/models/logistic_regression.pkl`
- `artifacts/models/random_forest.pkl`
- `artifacts/models/hist_gradient_boosting.pkl`

The training script is:

- `src/scripts/train.py`

The preprocessing pipeline is:

- `src/data/processor.py`

The written explanation for Part 1 and Part 2 is:

- `docs/phase2_part1_part2_draft.md`

## 3. Current model results you can directly use

The main result table is:

- `artifacts/metrics/phase2_model_metrics.csv`

The parameter search / trial table is:

- `artifacts/metrics/phase2_model_trials.csv`

Current validation and test summary:

| Model | Validation AUROC | Validation AUPRC | Test AUROC | Test AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.579 | 0.549 | 0.577 | 0.550 |
| Random Forest | 0.586 | 0.560 | 0.584 | 0.568 |
| HistGradientBoosting | 0.589 | 0.563 | 0.591 | 0.573 |

Current best model:

- `HistGradientBoosting`

Reason:

- it achieved the strongest validation and test AUROC among the currently saved models
- it also had the strongest AUPRC among the currently saved models

## 4. Files Part 3 can use immediately

For `Model Evaluation & Comparison`, the following files are ready to use:

- `artifacts/metrics/phase2_model_metrics.csv`
- `artifacts/metrics/phase2_model_trials.csv`
- `artifacts/metrics/predictions/logistic_regression_val.csv`
- `artifacts/metrics/predictions/logistic_regression_test.csv`
- `artifacts/metrics/predictions/random_forest_val.csv`
- `artifacts/metrics/predictions/random_forest_test.csv`
- `artifacts/metrics/predictions/hist_gradient_boosting_val.csv`
- `artifacts/metrics/predictions/hist_gradient_boosting_test.csv`

These prediction files already contain:

- `y_true`
- `y_prob`
- `y_pred`

So teammates working on Part 3 can directly use them to:

- redraw metric summary tables
- plot ROC / PR curves
- compare precision, recall, and F1 across models
- analyze threshold behavior if needed

## 5. What Part 3 still needs to add

Part 3 can start from the saved outputs, but these items still need to be written or extended:

- clearer comparison commentary across the three models
- spatial accuracy discussion
- temporal accuracy discussion
- robustness analysis
- cross-validation or a justified alternative if full CV is too expensive

Important note:

- The current artifacts already support standard classification comparison.
- Spatial and temporal evaluation will likely require teammates to derive additional grouped analyses from prediction outputs or from rerunning the pipeline with extra bookkeeping.

## 6. What Part 4 should say consistently

For `Model Interpretation & Business Insights`, the write-up should stay aligned with the modeling assumptions:

- the model predicts **occurrence risk**, not exact crime category
- the strongest signals are expected to come from **recent historical crime intensity**, **time-of-day patterns**, and **location context**
- results should be framed as **decision-support evidence**, not deterministic crime certainty

The Part 4 section can build on the following logic:

- if recent district or grid-cell crime counts are predictive, this supports targeted patrol planning or monitoring in repeatedly active areas
- if time-based features are predictive, this supports scheduling and timing decisions
- if location context remains important across models, this supports hotspot-oriented intervention planning

## 7. Caveats teammates should not ignore

Please keep these caveats in the evaluation and business sections:

- the current benchmark uses a sampled `2022-2024` dataset for reproducibility and iteration speed
- this is a local experimental setup, not yet the final full-scale benchmark described in the course brief
- synthetic negatives are necessary for the binary task, so interpretation should acknowledge this design choice
- model performance is moderate, not production-grade, so business claims should stay realistic

## 8. Suggested handoff wording

If you want to explain the handoff briefly to teammates, you can use this:

> Part 1 and Part 2 are done as a reproducible binary occurrence-prediction pipeline. The preprocessing, feature engineering, and multi-model training are already implemented and documented. Three models have been trained and saved, and the metrics and prediction outputs are in `artifacts/metrics/`. HistGradientBoosting currently performs best. For Part 3, you can directly use the saved prediction files and metric tables for comparison and additional evaluation plots. For Part 4, please interpret the results as occurrence-risk prediction based on temporal, spatial, and historical context, and note the sampling and synthetic-negative limitations.
