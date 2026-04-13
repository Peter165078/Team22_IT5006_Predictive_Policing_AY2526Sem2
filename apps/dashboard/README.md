# Dashboard Data Notes

The Streamlit dashboard reads yearly Chicago crime archives from:

- `apps/dashboard/split_data_by_year/`

These ZIP files are kept in the project because they support the demo workflow
without requiring a separate preprocessing step.

By contrast, the following are considered local training artifacts and are not
intended to be committed by default:

- `data/raw/chicago_crime_district_hour_2015_2025_phase2.csv`
- `artifacts/models/*.pkl`
- `artifacts/models/*.pt`
- `artifacts/metrics/predictions/*.csv`

If you regenerate the dashboard source files from a different raw export, keep
the filenames in the form `chicago_crime_<year>.csv.zip` so `app.py` can find
them automatically.

The dashboard itself can browse whichever yearly ZIPs are present. The
refactored modeling pipeline, however, expects a complete `2015-2025` range so
that model training can use `2015-2024`, keep `2025` as a holdout year, and
build an explicit district-hour modeling table with natural negative labels.
