# IT5006 Predictive Policing Project

An end-to-end data analytics project on **predictive policing**, using historical crime data to explore temporal and spatial patterns, build predictive models, and deploy a demo system for decision support.

This project is developed as part of the IT5006 course and is organized into three phases:
- **Phase 1** — Literature Review + EDA + Dashboard
- **Phase 2** — Model Training + Evaluation
- **Phase 3** — Deployment + Demo

---

# Phase 1 🔍

Phase 1 focuses on understanding crime data, identifying temporal and spatial patterns, and building an interactive dashboard to communicate insights.

## EDA Results
![overview](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg@main/img/overview_grid.png)

## Dashboard Demo
Interactive dashboard built for exploring crime data.

Features include:
- Time filtering (hour/day/month)
- Crime-type filtering
- Map-based visualization of incidents or hotspots

**Live Demo:**  
https://team22it5006predictivepolicingay2526sem2-5xjmr8cwbappeurrspssw.streamlit.app/

## Demo Video
A short walkthrough of the dashboard:  
![dashboard_preview](https://cdn.jsdelivr.net/gh/JIaDLu/BlogImg@main/img/dashboard_preview_out.gif)

---

# Phase 2 🛠️

Phase 2 now includes a reproducible pipeline for:
- building a single training CSV from yearly Chicago archives
- applying the shared preprocessing and feature engineering pipeline
- training multiple models for Milestone 2
- saving model files, prediction outputs, and metric summaries under `artifacts/`

---

# Phase 3 🚀

Phase 3 now uses a single Streamlit application to support two presentation
workflows:
- **Dashboard mode** for historical exploration of Chicago crime patterns
- **Prediction Demo mode** for entering a district, date, hour, and optional
  spatial details to obtain a model-based crime-risk estimate

The deployed demo is intentionally lightweight: it focuses on prediction
functionality, input validation, and explainable outputs rather than a polished
production UI.

---

# Reproduce Our Results

## Installation (Project Setup)

This project supports both **uv** and **conda** for dependency and environment management, and uses **Python 3.10**.

### Option 1: Using `uv`

If you are using `uv`, follow these steps:

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. Create and activate the virtual environment with Python 3.10:

   ```bash
   uv venv -p 3.10
   source .venv/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   uv pip install -r requirements.txt
   ```

### Option 2: Using `conda`

If you prefer using `conda`, you can set up the environment using the following steps:

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. Create a new `conda` environment with Python 3.10:

   ```bash
   conda create --name <env-name> python=3.10
   ```

3. Activate the `conda` environment:

   ```bash
   conda activate <env-name>
   ```

4. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Optional Notebook / EDA Dependencies

If you want to rerun the exploratory notebooks and mapping-heavy EDA workflow,
install:

```bash
pip install -r requirements-analysis.txt
```

### Optional MLP Dependency

The default Phase 2 and Phase 3 workflows do **not** require PyTorch. If you
want to rerun the optional `crime_mlp` benchmark, install:

```bash
pip install -r requirements-mlp.txt
```
---

# Data Source 📊
## Chicago Crime Dataset (2001–Present)
Source: Official dataset from the Chicago Police Department’s CLEAR (Citizen Law Enforcement Analysis and Reporting) system.  

Access:  
[https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)  

About the data:  
[https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data)

## How to Download
1. Open the dataset link  
2. Click **Export**  
3. Choose **CSV**  
4. Download the file  

## Where to Place the File
After downloading:  
- Rename if needed (e.g., `Crimes_-_2001_to_Present_20260216.csv`)  
- Place it under:  
  ```
  data/raw/
  ```
  Example:  
  ```
  data/raw/Crimes_-_2001_to_Present_20260216.csv
  ```

> Note: The repository does not include raw data due to file size.

## Repository Data Policy

For GitHub submission, we distinguish between:

- **demo assets kept in the repo**: yearly dashboard ZIP files under `apps/dashboard/split_data_by_year/`
- **deployment artifact kept in the repo**: `artifacts/models/hist_gradient_boosting.pkl`, which powers the Prediction Demo page
- **local training outputs that can be regenerated**: `data/raw/chicago_crime_2022_2024_phase2.csv` and prediction dumps under `artifacts/metrics/predictions/`

This keeps the dashboard and prediction demo runnable while avoiding raw-data
sprawl in version control.

---

# Phase 1 Quick Start

## Run EDA Notebooks

Open and run in order:  
1. `notebooks/01_eda_overview.ipynb`  
2. `notebooks/02_build_standard_data.ipynb`  
3. `notebooks/03_eda_pipeline.ipynb`  
4. `notebooks/04_results_overview.ipynb`  

Generated figures will be saved to:  
```
artifacts/
└── figures/
└── overview_grid.png
```

---

# Run Demo App
The Streamlit app provides both interactive historical exploration and a
prediction-oriented Phase 3 demo.  

Example (Streamlit):  
```bash
python -m streamlit run apps/dashboard/app.py
```

Then open the local URL shown in the terminal (typically):  
```
http://localhost:8501
```

The app expects:
- yearly ZIP files in `apps/dashboard/split_data_by_year/`
- `artifacts/models/hist_gradient_boosting.pkl` for the Prediction Demo page

## Prediction Demo Workflow

The Phase 3 demo is designed to satisfy the course requirement for a basic,
accessible prediction interface. In `Prediction Demo`, the user:

1. selects a police district
2. chooses a date and hour
3. optionally refines ward, community area, beat, latitude, and longitude
4. clicks `Predict Risk`
5. receives a probability score, a low/medium/high risk label, and recent
   district activity counts

Built-in validation includes:
- date restrictions to the modeled horizon
- bounded numeric inputs for district, ward, beat, community area, latitude,
  and longitude
- rejection of cold-start cases where historical context is unavailable

---

# Phase 2 Quick Start

## 1. Build the local Phase 2 dataset

```bash
python src/scripts/prepare_phase2_data.py --start-year 2022 --end-year 2024 --overwrite
```

This reads the yearly ZIP archives from `apps/dashboard/split_data_by_year/` and writes a consolidated CSV to `data/raw/`. By default, the Phase 2 helper script caps each year at `20,000` rows to keep local report-generation runs reproducible and practical.

## 2. Train multiple models

```bash
python src/scripts/train.py --start-year 2022 --end-year 2024
```

The default report-oriented benchmark trains four classical baselines: Logistic Regression, Decision Tree, Random Forest, and HistGradientBoosting. The optional PyTorch MLP remains available through `--models crime_mlp`.

## 3. Export spatial-temporal evaluation summaries

```bash
python src/scripts/evaluate_phase2.py
```

This writes compact district-level and temporal-alignment metrics to `artifacts/metrics/phase2_spatiotemporal_metrics.csv`.

Generated outputs will be saved to:

```text
artifacts/
├── metrics/
│   ├── phase2_data_summary.json
│   ├── phase2_model_metrics.csv
│   ├── phase2_model_trials.csv
│   └── predictions/
└── models/
```

Additional write-up notes for the Milestone 2 report are in:

```text
docs/phase2_methodology.md
```

---

# Phase 3 Quick Start

## 1. Install demo dependencies

```bash
pip install -r requirements.txt
```

## 2. Launch the Streamlit app

```bash
python -m streamlit run apps/dashboard/app.py
```

## 3. Demo flow to show in presentation

- Start on `Welcome`
- Open `Dashboard` and briefly show the historical map and temporal heatmap
- Switch to `Prediction Demo`
- Enter a district, date, and hour
- Show the predicted probability, risk band, and recent district counts
- Emphasize that the tool is for planning support, not automated enforcement

## 4. Deployment note

The current demo is packaged so that the Streamlit app can run directly from
the repository without rebuilding the training pipeline first. The prediction
page uses the bundled `HistGradientBoosting` model artifact under
`artifacts/models/`.

---

# Repository Structure

### `apps/`
User-facing applications (demo/UI layer).  
- `apps/dashboard/`  
  Streamlit application used in Phase 1 for EDA communication and in Phase 3 for the final prediction demo. It now includes both a historical dashboard view and a model-backed prediction page.  
- `apps/web_frontend/`  
  Optional frontend experimentation area. It is not required for the submitted demo workflow.  

> The submitted Phase 3 demo runs from `apps/dashboard/app.py`.

### `artifacts/`
Generated outputs produced by training and evaluation runs (not raw code).  
- `artifacts/figures/`  
  Exported plots for reports (EDA charts, model performance plots, maps, etc.).  
- `artifacts/models/`  
  Saved model artifacts used for reproducibility and deployment. The Streamlit prediction demo currently loads `hist_gradient_boosting.pkl`.  

> Keep demo-critical artifacts small and versioned when they are required for live presentation.

### `data/`
Datasets and intermediate data files.  
- `data/raw/` — Original raw datasets (usually ignored by Git)  
- `data/processed/` — Optional cleaned/feature-engineered datasets used for training  

For the final demo, the app rebuilds a temporary 2022-2024 modeling dataset
from the yearly ZIP files rather than depending on a checked-in processed CSV.

### `docs/`
Supporting documentation for the course submission and internal handoff.  
Examples include:
- Phase 2 methodology notes
- revision notes responding to grader feedback
- final report handoff guidance
- Phase 3 delivery checklist

### `notebooks/`
Jupyter notebooks used primarily for exploration, experimentation, and reporting.  
These notebooks support EDA and report figure generation. Reusable logic should
stay in `src/`, not be duplicated inside notebooks.

**Rule of thumb:** Notebooks are for exploration and visualization; reusable logic should live in `src/`.

### `reports/`
PDF deliverables for each phase submission.  
- `reports/Phase1_Report.pdf` — Phase 1 report (literature + EDA + dashboard summary)  
- `reports/Phase2_Report.pdf` / `reports/Phase3_Report.pdf` — final compiled deliverables when ready

### `src/`
Core reusable Python package code (shared by notebooks, scripts, dashboard, and API).  
- `src/data/`  
  Dataset construction and the shared preprocessing pipeline used by both training and inference  
- `src/scripts/`  
  Reproducible CLI entrypoints for preparing the Phase 2 dataset, training models, exporting metrics, and feature importance summaries  
- `src/__init__.py`  
  Package initialization when present

> Goal: All “single source of truth” logic should live in `src/` to avoid duplicating code across notebooks and the Streamlit demo.

## Root Files
- `.gitignore` — Git ignore rules (should exclude large raw data and local caches)  
- `requirements.txt` — default dependencies for the final demo and standard project workflows
- `requirements-analysis.txt` — optional extras for notebooks and EDA utilities
- `requirements-mlp.txt` — optional extra dependency for rerunning the PyTorch MLP benchmark
- `README.md` — High-level overview and repo navigation (this file)  

## Phase Mapping (How this structure supports the course)
- **Phase 1 (Literature + EDA + Dashboard)**  
  `notebooks/` + `apps/dashboard/` + outputs in `artifacts/figures/` + submission in `reports/`  
- **Phase 2 (Model Training + Evaluation)**  
  `src/` (train/eval code) + saved outputs in `artifacts/models/` and `artifacts/metrics/`  
- **Phase 3 (Deployment + Demo)**  
  `apps/dashboard/` (demo UI) + model loaded from `artifacts/models/` + supporting notes in `docs/`  

---
