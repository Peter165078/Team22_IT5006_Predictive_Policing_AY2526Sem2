# Dashboard Data Notes

The Streamlit application reads yearly Chicago crime archives from:

- `apps/dashboard/split_data_by_year/`

These ZIP files are kept in the project because they support the demo workflow
without requiring a separate preprocessing step.

The Phase 3 `Prediction Demo` page also expects:

- `artifacts/models/hist_gradient_boosting.pkl`

At app startup, the prediction view builds a temporary Phase 2-style dataset
from the 2022-2024 yearly ZIP files, applies the shared preprocessing pipeline,
and then loads the saved HistGradientBoosting model for inference.

By contrast, the following are considered regenerated artifacts and do not need
to be refreshed for a normal demo run:

- `data/raw/chicago_crime_2022_2024_phase2.csv`
- `artifacts/models/*.pt`
- `artifacts/metrics/predictions/*.csv`

If you regenerate the dashboard source files from a different raw export, keep
the filenames in the form `chicago_crime_<year>.csv.zip` so `app.py` can find
them automatically.
