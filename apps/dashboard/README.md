# Dashboard Data Notes

The Streamlit dashboard reads yearly Chicago crime archives from:

- `apps/dashboard/split_data_by_year/`

These ZIP files are kept in the project because they support the demo workflow
without requiring a separate preprocessing step.

By contrast, the following are considered local training artifacts and are not
intended to be committed by default:

- `data/raw/chicago_crime_2022_2024_phase2.csv`
- `artifacts/models/*.pkl`
- `artifacts/models/*.pt`
- `artifacts/metrics/predictions/*.csv`

If you regenerate the dashboard source files from a different raw export, keep
the filenames in the form `chicago_crime_<year>.csv.zip` so `app.py` can find
them automatically.
