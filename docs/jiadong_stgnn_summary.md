# Jiadong STGNN Contribution

## Why this was added

This repository already integrated the agency-level demo experience inspired by the separate `JIaDLu` project, but that alone did not fully reflect Lu Jiadong's technical contribution.

To make his work visible in the main submission repository, we also integrated the core spatiotemporal modeling module that appears in his project report:

- graph-based spatial modeling over Chicago community areas
- temporal modeling with interchangeable `LSTM`, `GRU`, and `MHA` encoders
- a unified preprocessing, adjacency-building, training, and evaluation pipeline

## What is now in the repository

The integrated research module lives under:

- `src/experimental/jiadong_stgnn/`

Key files:

- `src/experimental/jiadong_stgnn/config.py`
- `src/experimental/jiadong_stgnn/models/stgnn.py`
- `src/experimental/jiadong_stgnn/models/temporal_modules.py`
- `src/experimental/jiadong_stgnn/utils/data_pipeline.py`
- `src/experimental/jiadong_stgnn/utils/adjacency.py`
- `src/experimental/jiadong_stgnn/utils/experiment.py`
- `src/experimental/jiadong_stgnn/train.py`
- `src/experimental/jiadong_stgnn/evaluate.py`
- `src/scripts/run_jiadong_stgnn.py`

## Model framing

This module reformulates crime prediction as a spatiotemporal forecasting problem.

- Input shape: `(B, T, N, F)`
- `B`: batch size
- `T`: 7-day sliding window
- `N`: 77 Chicago community areas
- `F`: 6 features

Feature set:

- `crime_count`
- `theft_count`
- `battery_count`
- `day_of_week`
- `is_weekend`
- `month`

Output shape:

- `(B, N)`

Each output row predicts next-step crime intensity for all 77 regions.

## What this does and does not imply

This integration is meant to preserve and expose Jiadong's method contribution inside the main GitHub repository.

It does **not** mean the main Streamlit demo currently runs full STGNN inference online. The current integrated dashboard still uses the existing Team22 demo flow plus the local preview agency map page.

## Local reproduction

The STGNN module expects a combined Chicago CSV at:

- `data/exp_data/Chicago_Crimes_2015_2025.csv`

Optional extra dependencies for this module are listed in:

- `requirements-stgnn.txt`

Example run:

```bash
python src/scripts/run_jiadong_stgnn.py --temporal lstm
```

Outputs are written under:

- `artifacts/models/jiadong_stgnn/`
- `artifacts/metrics/jiadong_stgnn/`

## Why this is the right integration scope

This keeps the repository clean while still making the contribution inspectable:

- the main Team22 app remains the single primary demo entry point
- the agency-map demo contribution stays visible in `apps/dashboard/app.py`
- the underlying STGNN research contribution is now preserved in code, not just described in prose

