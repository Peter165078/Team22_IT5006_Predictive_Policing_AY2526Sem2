# Final Pipeline Handoff

This note summarizes the current end-to-end project state after the April 2026
refactor. It should be treated as the most accurate project handoff reference
for slides, final report writing, and demo preparation.

## Final modeling definition

- Modeling unit: district-hour space-time cells
- Task: binary classification
- Objective: estimate `P(Y = 1 | X)` where:
  - `Y = 1` means at least one recorded crime occurs in a district-hour cell
  - `Y = 0` means no recorded crime occurs in that district-hour cell

This replaces the earlier event-level setup that depended on synthetic
negative-label construction.

## Temporal scope

- EDA / dashboard exploration: yearly Chicago crime archives available in
  `apps/dashboard/split_data_by_year/`
- Model development and training: `2015-2024`
- Holdout evaluation year: `2025`
- Validation/test policy: chronological split within the `2025` holdout

## Current data assets

- Yearly ZIP archives available locally: `2014-2025`
- Refactored modeling dataset:
  - `data/raw/chicago_crime_district_hour_2015_2025_phase2.csv`

## Modeling dataset summary

Generated from the refactored district-hour builder:

- Total rows: `2,217,936`
- Positive rows: `1,409,469`
- Negative rows: `808,467`
- Overall positive rate: `0.6355`
- Distinct districts: `23`

After preprocessing and historical-feature cold-start removal:

- Train rows: `2,015,280`
- Validation rows: `100,740`
- Test rows: `100,740`
- Feature count: `37`

## Core pipeline files

- Data builder:
  - `src/data/dataset_builder.py`
- Explicit year-based split logic:
  - `src/data/split_strategy.py`
- Shared preprocessing:
  - `src/data/processor.py`
- Dataset preparation entrypoint:
  - `src/scripts/prepare_phase2_data.py`
- Training entrypoint:
  - `src/scripts/train.py`
- Spatial-temporal evaluation:
  - `src/scripts/evaluate_phase2.py`
- Feature importance:
  - `src/scripts/feature_importance.py`
- Optional yearly archive downloader:
  - `src/scripts/download_chicago_year_archive.py`
- Streamlit demo:
  - `apps/dashboard/app.py`

## What the refactor changed

1. The default modeling scope is no longer a sampled `2022-2024` subset.
2. The project no longer depends on copied event rows to synthesize negatives.
3. The main task now uses explicit district-hour units, which gives natural
   positive and negative labels.
4. The default split now follows:
   - train: `2015-2024`
   - holdout: `2025`
5. Historical features are now computed with a vectorized lookup path that can
   support the full refactored dataset.

## Final benchmark results

Validation / test metrics from `artifacts/metrics/phase2_model_metrics.csv`:

| Model | Validation AUROC | Test AUROC | Validation AUPRC | Test AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7186 | 0.7177 | 0.7792 | 0.7867 |
| Decision Tree | 0.7362 | 0.7334 | 0.7950 | 0.8003 |
| Random Forest | 0.7402 | 0.7370 | 0.8001 | 0.8056 |
| HistGradientBoosting | 0.7412 | 0.7381 | 0.8011 | 0.8063 |

Best benchmark:

- `HistGradientBoosting`
- Test AUROC: `0.7381`
- Test AUPRC: `0.8063`

## Spatial-temporal evaluation highlights

From `artifacts/metrics/phase2_spatiotemporal_metrics.csv`:

- HistGradientBoosting test district correlation: `0.9996`
- HistGradientBoosting test top-5 overlap: `1.0`
- HistGradientBoosting test hourly correlation: `0.9900`
- HistGradientBoosting test day-of-week correlation: `0.7311`

Interpretation:

- Temporal alignment is very strong.
- District-level aggregate ranking is also much stronger than in the earlier
  event-level benchmark.
- The refactored district-hour task is materially more learnable and more
  operationally interpretable than the old sampled event-row setup.

## Feature-importance highlights

From `artifacts/metrics/feature_importance/feature_importance_summary.json`:

Top HistGradientBoosting features include:

- `hour`
- `hour_sin`
- `District`
- `crimes_last_30d`
- `crimes_last_14d`
- `day_of_week`

Grouped importance shares:

- temporal: `0.659`
- historical: `0.201`
- spatial: `0.140`

Interpretation:

- Temporal signals are still the strongest drivers.
- Historical crime windows add clear value.
- Spatial information remains useful, especially district identity.

## Demo state

The main Streamlit app now includes:

- Welcome page
- historical dashboard
- prediction demo page

The prediction demo uses the saved `HistGradientBoosting` model and the same
refactored preprocessing pipeline to return:

- predicted probability
- risk band
- recent positive-hour counts for the selected district

## Safe presentation wording

Use this wording in slides / report:

- “We model district-hour risk, not individual incident records.”
- “Negative labels arise naturally from district-hour cells with zero recorded incidents.”
- “The final benchmark trains on 2015-2024 and evaluates on 2025.”
- “HistGradientBoosting is the strongest overall benchmark, but the gap over Random Forest is small.”
- “The system is a decision-support proof-of-concept, not an automated enforcement tool.”

Avoid this wording:

- “We used 20,000 rows per year as the final setup.”
- “The final model depends on synthetic negatives.”
- “The system predicts exact crimes.”
- “The app should directly guide enforcement decisions.”
