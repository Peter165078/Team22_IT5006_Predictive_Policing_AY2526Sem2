# Phase 3 Delivery Checklist

This checklist is for the non-report portion of the final submission.

## Core deliverables

- GitHub repository link is available and public to the teaching team
- Live Streamlit URL is accessible
- `apps/dashboard/app.py` opens without import errors in the target environment
- `Prediction Demo` page returns a probability and risk label
- `README.md` explains setup, demo flow, and deployment assumptions

## Demo readiness

- Welcome page loads correctly
- Dashboard page can show a map and at least one temporal chart
- Prediction Demo accepts district, date, and hour inputs
- Validation prevents impossible inputs
- Demo result shows:
  - probability
  - risk band
  - recent district counts
- Presenter can explain that the system is for planning support, not automated enforcement

## Files to double-check before submission

- `README.md`
- `apps/dashboard/app.py`
- `apps/dashboard/README.md`
- `artifacts/models/hist_gradient_boosting.pkl`
- `apps/dashboard/split_data_by_year/chicago_crime_2022.csv.zip`
- `apps/dashboard/split_data_by_year/chicago_crime_2023.csv.zip`
- `apps/dashboard/split_data_by_year/chicago_crime_2024.csv.zip`

## Local run commands

```bash
pip install -r requirements.txt
python -m streamlit run apps/dashboard/app.py
```

## Optional model-training commands

```bash
python src/scripts/prepare_phase2_data.py --start-year 2022 --end-year 2024 --overwrite
python src/scripts/train.py --start-year 2022 --end-year 2024
python src/scripts/evaluate_phase2.py
```

## Risk notes for presentation

- Performance differences between models are small; avoid claiming a dramatic winner
- Spatial hotspot ranking is weaker than overall discrimination
- Synthetic negatives are a modeling approximation, not observed no-crime ground truth
- The app demonstrates a proof-of-concept workflow rather than a production policing system
