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
In development.

---

# Phase 3 🚀
In development.

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

# Run Dashboard
The dashboard provides interactive visualization of crime data.  

Example (Streamlit):  
```bash
streamlit run apps/dashboard/app.py
```

Then open the local URL shown in the terminal (typically):  
```
http://localhost:8501
```

---

# Repository Structure (Still Designing & Developing)

### `apps/`
User-facing applications (demo/UI layer).  
- `apps/dashboard/`  
  Interactive dashboard used in Phase 1 (EDA presentation) and later reused for Phase 3 demo (visualizing inputs/outputs, maps, filters, etc.).  
- `apps/web_frontend/`  
  Optional separate web frontend (e.g., React/Vue). Use this if you plan to build a standalone UI that calls the API from `deployment/`.  

> Recommendation: If you use Streamlit for the dashboard, keep it under `apps/dashboard/`.

### `artifacts/`
Generated outputs produced by training and evaluation runs (not raw code).  
- `artifacts/figures/`  
  Exported plots for reports (EDA charts, model performance plots, maps, etc.).  
- `artifacts/models/`  
  Saved model artifacts (e.g., `.pkl`/`.joblib`) used for reproducibility and deployment.  

> Note: Large artifacts may be excluded from Git and stored via releases or external storage depending on team preference.

### `data/`
Datasets and intermediate data files.  
- `data/raw/` — Original raw datasets (usually ignored by Git)  
- `data/processed/` — Cleaned/feature-engineered datasets used for training  

### `deployment/`
Deployment layer for serving predictions (Phase 3).  
This directory contains:  
- A Flask (or FastAPI) API server  
- Configuration for serving the trained model from `artifacts/models/`  
- Optional Docker files and deployment scripts  

### `docs/`
Project documentation source intended for Read the Docs (or similar).  
Use this for detailed technical documentation beyond the `README.md`, such as:  
- Data dictionary  
- Modeling approach  
- API contract  
- Reproducibility guide  

### `notebooks/`
Jupyter notebooks used primarily for exploration, experimentation, and reporting.  
Current notebooks:  
- `01_eda_overview.ipynb` — Overall dataset understanding and cleaning notes  
- `02_eda_time_patterns.ipynb` — Temporal pattern analysis  
- `03_eda_spatial_patterns.ipynb` — Spatial analysis (hotspots, maps)  

**Rule of thumb:** Notebooks are for exploration and visualization; reusable logic should live in `src/`.

### `reports/`
PDF deliverables for each phase submission.  
- `reports/Phase1_Report.pdf` — Phase 1 report (literature + EDA + dashboard summary)  
- (Later) `reports/Phase2_Report.pdf`, `reports/Phase3_Report.pdf`  

### `scripts/`
Command-line convenience scripts to standardize common workflows.  
- `scripts/train.sh` — Runs training pipeline end-to-end (Phase 2)  
- `scripts/deploy.sh` — Starts deployment stack or API server (Phase 3)  

These scripts should call into `src/` so that training/deployment stay consistent and reproducible.

### `src/`
Core reusable Python package code (shared by notebooks, scripts, dashboard, and API).  
- `src/config/`  
  Configuration files (paths, parameters, feature lists, model configs, etc.).  
- `src/data/`  
  Data ingestion and feature engineering:  
  - `load.py` — Data loading utilities  
  - `features.py` — Feature engineering shared by training and inference  
- `src/models/`  
  Modeling pipelines:  
  - `train.py` — Training entrypoints and model selection logic  
- `src/evaluation/`  
  Evaluation utilities:  
  - `metrics.py` — Metrics and evaluation routines  
- `src/utils/`  
  Common helper utilities (logging, IO helpers, shared constants, etc.).  
- `src/__init__.py`  
  Package initialization.  

> Goal: All “single source of truth” logic should live in `src/` to avoid duplicating code across notebooks and deployment.

## Root Files
- `.gitignore` — Git ignore rules (should exclude large raw data and local caches)  
- `.python-version` — Python version pin (pyenv compatible)  
- `pyproject.toml` / `uv.lock` — Dependency management and reproducible environments  
- `README.md` — High-level overview and repo navigation (this file)  

## Phase Mapping (How this structure supports the course)
- **Phase 1 (Literature + EDA + Dashboard)**  
  `notebooks/` + `apps/dashboard/` + outputs in `artifacts/figures/` + submission in `reports/`  
- **Phase 2 (Model Training + Evaluation)**  
  `src/` (train/eval code) + `scripts/train.sh` + saved outputs in `artifacts/models/` and `artifacts/metrics/`  
- **Phase 3 (Deployment + Demo)**  
  `deployment/` (API server) + `apps/dashboard/` or `apps/web_frontend/` (demo UI) + model loaded from `artifacts/models/`  

---