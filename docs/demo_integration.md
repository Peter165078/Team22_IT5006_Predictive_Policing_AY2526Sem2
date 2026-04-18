# Demo Integration Notes

## What is already integrated

The main deliverable lives in this repository and runs from:

- `apps/dashboard/app.py`

The integrated Streamlit app now contains:

- the original Team22 Chicago dashboard and analysis pages
- the Team22 prediction demo, hotspot page, and group-pattern view
- the NIBRS generalization summary page
- a new `Agency Map Demo` page that adapts the JIaDLu agency-level map interaction into the Team22 app
- an integrated `src/experimental/jiadong_stgnn/` research module that preserves Lu Jiadong's STGNN code inside the main repository

## What was not copied wholesale

The separate `JIaDLu` project was used as a reference and local integration source, but this repository does **not** contain that project as a second standalone application, nested Git repository, or duplicate folder tree.

In other words, the useful demo behavior was merged into the Team22 dashboard, rather than embedding the entire external repository.

The main repository now also preserves the most important method-level contribution from that work: the STGNN training and evaluation module described in Jiadong's report.

## Local-only assets

The integrated agency demo expects a local top-level folder named:

- `NIBRS data/`

That directory is treated as a local dataset store for preview/demo use. It is intentionally ignored in Git because the files are large and environment-specific.

## Current preview limitation

The integrated `Agency Map Demo` currently uses a 7-day average preview fallback when the original Baseline-GRU checkpoint assets are not available locally. The page still preserves the agency-level interaction pattern and bar-marker map style for local demonstrations.

## Recommended repository mental model

If you need to explain the project quickly, the cleanest description is:

- `team22_phase3_clean` is the main integrated submission repository
- the Team22 Streamlit app is the single primary entry point
- the JIaDLu demo ideas have been folded into that app instead of being kept as a separate product inside the repo
