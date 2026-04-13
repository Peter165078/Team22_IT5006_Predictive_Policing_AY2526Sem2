# Refactor Scope Notes

This note records the refactored default temporal setup for the project so the
codebase, README, report drafting, and presentation materials stay aligned.

## Target temporal design

- EDA and dashboard exploration: use the yearly dashboard archives that are
  available in `apps/dashboard/split_data_by_year/`
- Modeling unit: district-hour space-time cells
- Model development and training: `2015-2024`
- Holdout evaluation year: `2025`
- Validation/test policy: split the `2025` holdout chronologically into two
  consecutive parts

## Why this refactor was made

The earlier sampled `2022-2024` benchmark relied on a reduced event-level setup
and synthetic negative construction. The refactor removes that reduced-scope
assumption and switches the main modeling task to explicit district-hour units,
where negative labels arise naturally.

## Code changes already applied

- `src/scripts/prepare_phase2_data.py`
  now defaults to building `data/raw/chicago_crime_district_hour_2015_2025_phase2.csv`
- `src/scripts/train.py`
  now defaults to:
  - dataset years `2015-2025`
  - train years `2015-2024`
  - holdout year `2025`
- `src/scripts/download_chicago_year_archive.py`
  can fetch a missing yearly archive and normalize the timestamp format into the
  same ZIP structure already used by the dashboard
- `src/scripts/evaluate_phase2.py`
  and `src/scripts/feature_importance.py`
  now follow the same temporal split assumptions
- `src/data/dataset_builder.py`
  now builds a district-hour modeling table and raises a clear error if any
  requested yearly archive is missing
- `src/data/processor.py`
  now exposes explicit split assignment, supports natural labels when a `target`
  column is already present, and computes historical windows with a vectorized
  lookup path

## Final-report guidance

- Do not describe `2022-2024` with `20,000` rows per year as the final project
  setup.
- Do not describe the final modeling pipeline as depending on synthetic
  negatives copied from event rows.
- Use this wording instead:
  - modeling unit: district-hour space-time cells
  - training scope: `2015-2024`
  - holdout scope: `2025`
  - evaluation policy: chronological future-year holdout
