# Team 22 Phase 3 Presentation Notes

This note is designed for a 10-minute presentation plus 5-minute Q&A.

## Slide 1 — Title

- Introduce the project as a predictive policing proof-of-concept.
- Emphasize that the project has three parts: pipeline, model comparison, and deployed demo.
- Keep this short; do not start with details yet.

## Slide 2 — Problem Definition

- State the task precisely: estimate `P(Y=1 | X)`.
- Explain that `Y=1` means at least one crime occurs in a time-location unit.
- Clarify that this is occurrence prediction, not crime-type classification.

## Slide 3 — Data and Main Challenge

- Data comes from yearly Chicago crime archives.
- The refactored pipeline uses 2015-2024 for model development and reserves 2025 as the holdout year.
- The refactored model uses district-hour units, so positive and negative labels both arise naturally.
- This removes the need for the earlier synthetic-negative workaround.

## Slide 4 — Pipeline

- Walk through the architecture from yearly ZIP files to dataset construction, preprocessing, training, evaluation, and demo deployment.
- Mention the main code modules:
  - `src/data/dataset_builder.py`
  - `src/data/processor.py`
  - `src/scripts/train.py`
  - `src/scripts/evaluate_phase2.py`
  - `apps/dashboard/app.py`

## Slide 5 — Feature Engineering

- Explain the three feature families:
  - temporal
  - spatial
  - historical
- Mention leakage control:
  - post-incident variables are dropped
  - chronological split is used
  - historical counts use only events before time `T`

## Slide 6 — Model Results

- The classical benchmarks are Logistic Regression, Decision Tree, Random Forest, and HistGradientBoosting.
- HistGradientBoosting is best overall, but only slightly.
- Do not say there is a dramatic winner.

## Slide 7 — Interpretation

- Temporal alignment is stronger than spatial hotspot ranking.
- Feature importance shows that temporal signals dominate.
- The current pipeline is more reliable for timing-related support than for precise district ranking.

## Slide 8 — Demo

- Show that the app has two modes:
  - dashboard
  - prediction demo
- In the demo, show one input example and explain the output:
  - probability
  - risk level
  - recent district counts

## Slide 9 — Conclusion

- Reiterate the three outcomes:
  - reproducible pipeline
  - benchmark comparison
  - deployed proof-of-concept
- End by framing the system as a decision-support prototype, not a production enforcement tool.

## High-probability Q&A

### Why use district-hour units instead of event rows?

Because event rows contain only observed crimes. By switching to explicit district-hour units, we obtain natural `Y=0` cases whenever no incident occurs in that district and hour.

### Why use chronological split instead of random split?

Because this is a time-dependent prediction task. Random splitting would leak future information into training.

### Why use 7, 14, 30, and 90-day windows?

They correspond to weekly, biweekly, monthly, and quarterly recency effects and are operationally interpretable.

### Why is HistGradientBoosting only slightly better?

Because the signal in the available features is modest. The task is difficult, and the models perform in a similar range.

### Why is spatial performance weaker than temporal performance?

The current feature set captures temporal crime rhythm more strongly than district-level hotspot ranking. Better spatial performance would likely require richer contextual features.

### Is this suitable for real-world policing?

Not as a production system. It should be presented as a proof-of-concept decision-support tool with clear limitations.
