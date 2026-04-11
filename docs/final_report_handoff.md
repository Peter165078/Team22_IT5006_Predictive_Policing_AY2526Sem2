# Final Report Handoff

This document is a handoff guide for the teammate responsible for assembling the final report. It summarizes what the project currently contains, which files should be used, and how to frame the final write-up consistently.

## 1. Project one-paragraph summary

This project studies predictive policing as a decision-support problem using Chicago crime data. The final system combines exploratory data analysis, an interactive Streamlit dashboard, and a reproducible machine-learning benchmark that estimates the probability of crime occurrence for a location-time instance. The final narrative should emphasize responsible use: the model is better at learning short-term temporal risk patterns than at producing reliable district-wide hotspot rankings, so it should be presented as a human-in-the-loop planning aid rather than as an automated decision engine.

## 2. Recommended report structure

1. Introduction and problem motivation
2. Dataset description and preprocessing
3. Exploratory data analysis
4. Problem formulation and feature engineering
5. Model implementation and training
6. Model evaluation and comparison
7. Interpretation, business insights, and limitations
8. Deployment/demo system
9. Conclusion and future work

## 3. Core files to use

### Project overview and demo

- [README.md](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/README.md)
- [app.py](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/apps/dashboard/app.py)

### Phase 2 write-up drafts

- [phase2_part1_part2_draft.md](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/docs/phase2_part1_part2_draft.md)
- [phase2_part3_part4_draft.md](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/docs/phase2_part3_part4_draft.md)
- [phase2_part4_feature_importance_addendum.md](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/docs/phase2_part4_feature_importance_addendum.md)
- [phase2_feedback_revision_notes.md](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/docs/phase2_feedback_revision_notes.md)

### Quantitative outputs

- [phase2_model_metrics.csv](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/artifacts/metrics/phase2_model_metrics.csv)
- [phase2_spatiotemporal_metrics.csv](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/artifacts/metrics/phase2_spatiotemporal_metrics.csv)
- [feature_importance_summary.json](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/artifacts/metrics/feature_importance/feature_importance_summary.json)

### Figures

- [overview_grid.png](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/artifacts/overview_grid.png)
- [crime_trend_hourly.png](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/artifacts/figures/crime_trend_hourly.png)
- [crime_trend_weekday.png](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/artifacts/figures/crime_trend_weekday.png)
- [crime_spatial_distribution.png](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/artifacts/figures/crime_spatial_distribution.png)
- [hist_gradient_boosting_feature_importance_top15.png](/Users/jiahao/Documents/Team22_IT5006_Predictive_Policing_AY2526Sem2-main/Team22_IT5006_Predictive_Policing_AY2526Sem2/artifacts/metrics/feature_importance/hist_gradient_boosting_feature_importance_top15.png)

## 4. Numbers that should stay consistent

Use these values consistently across the final report unless the team reruns the benchmark:

- Data scope for modeling: 2022 to 2024
- Sampled rows per year: 20,000
- Positive rows: 60,000
- Constructed negative rows: 60,000
- Total modeled rows: 120,000
- Feature count after preprocessing: 37
- Train / validation / test rows after cold-start filtering: 83,820 / 17,999 / 17,999

### Record-level model summary

- Logistic Regression: test AUROC 0.577, test AUPRC 0.550
- Decision Tree: test AUROC 0.581, test AUPRC 0.552
- Random Forest: test AUROC 0.583, test AUPRC 0.567
- HistGradientBoosting: test AUROC 0.591, test AUPRC 0.573

### Spatial-temporal summary

- HistGradientBoosting and Random Forest are strongest on hourly temporal correlation.
- District-level hotspot ranking is weak across models.
- Decision Tree and Logistic Regression can match or exceed the ensembles on some aggregated district metrics, so the final report must avoid claiming one model is uniformly best in every sense.

### Feature-importance summary for HistGradientBoosting

- Temporal features: 94.27%
- Spatial features: 4.08%
- Historical features: 1.36%
- Missing indicators: 0.29%

## 5. Writing guidance by section

### Introduction

- Frame the project as predictive policing for decision support.
- Avoid language that implies automated police action.
- Use probability language instead of deterministic prediction language.

### Data and preprocessing

- Explicitly define the task as estimating `P(Y = 1 | X)`.
- Explain why negative labels had to be constructed.
- State clearly that negative construction is a pragmatic approximation.
- Explain the 7d, 14d, 30d, and 90d windows as interpretable recency horizons.

### Modeling

- Explain why each of the four models was included.
- Mention that Decision Tree was added to address milestone feedback and to compare single-tree versus ensemble behavior.
- Explain hyperparameters in plain English, not just as raw Python parameter names.

### Evaluation

- Distinguish record-level metrics from spatial and temporal aggregation metrics.
- State that HistGradientBoosting is the strongest record-level benchmark.
- Do not claim that HistGradientBoosting is best on every operational dimension.

### Interpretation and business insights

- Emphasize that temporal structure is the strongest predictive pattern in the benchmark.
- Keep interpretation descriptive, not causal.
- Present business recommendations as modest and human-in-the-loop.

### Limitations

- Sampled data only
- Constructed negative labels
- Weak district-level spatial ranking
- Possible reporting bias in crime data
- No fairness or policy validation for real-world deployment

## 6. What the report should not say

- Do not say the model can accurately rank districts city-wide.
- Do not say the model should automate policing decisions.
- Do not say feature importance proves causal drivers of crime.
- Do not treat synthetic negatives as naturally observed no-crime labels.
- Do not overstate the margin between HistGradientBoosting and Random Forest.

## 7. Suggested final-report storyline

The best storyline for the final report is:

1. The team first explored the data and built an interactive dashboard.
2. The team then translated the problem into a reproducible binary classification benchmark.
3. Multiple baseline models were compared fairly.
4. The best model learned meaningful short-term temporal patterns, but spatial hotspot ranking remained weak.
5. Therefore, the project should be framed as a responsible decision-support system, not as a high-confidence predictive-policing engine.

## 8. What still needs final polishing

- Full PDF writing style and grammar smoothing
- Consistent figure numbering and captions
- Unified section numbering across merged report sections
- Final author contribution statement if required by the course
