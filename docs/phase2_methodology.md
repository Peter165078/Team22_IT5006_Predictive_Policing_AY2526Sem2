# Phase 2 Methodology Notes

This note is meant to support the `Problem Definition & Data Preparation` and `Model Implementation & Training` sections for Milestone 2.

## Problem definition

- Task type: binary classification.
- Prediction target: given an observed feature vector `X` for a location-time instance, estimate `P(Y = 1 | X)`, where `Y = 1` denotes that at least one crime incident occurs in that space-time unit and `Y = 0` denotes no recorded incident.
- Motivation: this framing lets the team compare classification models on the same temporal and spatial feature space before moving to the evaluation and interpretation sections owned by other teammates.

## Data preparation

- Source data: yearly Chicago crime archives stored in `apps/dashboard/split_data_by_year/`.
- Recommended local workflow:
  1. Merge selected yearly ZIP archives into one raw Phase 2 CSV with `src/scripts/prepare_phase2_data.py`.
  2. Run the shared `DataProcessor` in `src/data/processor.py`.
- Current preprocessing pipeline:
  - chronological split to reduce leakage
  - synthetic negative-label construction because raw crime data contains only observed crimes and therefore lacks explicit `Y = 0` rows
  - removal of post-incident fields such as crime type, IUCR, arrest, and domestic flags because they are not available at prediction time for occurrence forecasting
  - temporal feature extraction: hour, weekday, month, cyclic encodings, weekend/night/rush-hour flags
  - spatial feature treatment: district, ward, beat, community area, coordinates, grid cell
  - historical features based on prior crime counts in the same district and grid cell
  - train-only fitting for scaling and category handling
  - model-side weighting (`class_weight` or weighted sampling) for any residual imbalance during fitting

- Rolling-history windows:
  - `7d` for weekly rhythm
  - `14d` for short-term carryover across two weeks
  - `30d` for monthly seasonality
  - `90d` for quarter-scale persistence
- These windows were selected to represent operationally interpretable short-, medium-, and longer-horizon recency effects rather than arbitrary cutoffs.

## Modeling strategy

- The training pipeline in `src/scripts/train.py` now supports multiple models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - HistGradientBoosting
  - PyTorch CrimeMLP
- The default report-ready benchmark focuses on four classical baselines:
  - Logistic Regression as a transparent linear reference
  - Decision Tree as a single-tree nonlinear baseline
  - Random Forest as the bagged-tree ensemble extension of the decision-tree family
  - HistGradientBoosting as a stronger boosted-tree benchmark for tabular data
- Each model is trained with a small validation-oriented parameter grid so the report can describe a concrete tuning methodology instead of a single untuned run.
- Reproducible outputs are saved under:
  - `artifacts/models/`
  - `artifacts/metrics/`
  - `artifacts/metrics/predictions/`

## Practical note

- The default script configuration uses a capped number of rows per year so local runs finish in a reasonable time.
- For final report-quality experiments, increase the year span and row cap if your machine budget allows it, then regenerate the saved metrics and model artifacts.
