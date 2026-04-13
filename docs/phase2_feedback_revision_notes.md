# Phase 2 Feedback Revision Notes

This note tracks the concrete project updates made in response to the Milestone 2 feedback so the revised material can be shared with the instructor and then folded into the final report.

## Revisions completed in the project

1. **Problem formulation tightened**
   - The predictive task is now stated explicitly as estimating `P(Y = 1 | X)` for a location-time feature vector `X`.
   - The write-up now distinguishes probabilistic risk estimation from deterministic crime prediction.

2. **Negative construction explained more carefully**
   - The documentation now describes synthetic negatives as a label-construction device required for supervised binary learning because the source data contains only observed crimes.
   - Model-side imbalance handling is now described separately through built-in weighting such as `class_weight = balanced`.

3. **Temporal windows justified**
   - The `7d`, `14d`, `30d`, and `90d` history windows are now explained as weekly, biweekly, monthly, and quarter-scale recency horizons.

4. **Model family rationale strengthened**
   - The training pipeline now includes a `Decision Tree` baseline in addition to Logistic Regression, Random Forest, and HistGradientBoosting.
   - This allows the report to compare a single tree against its bagged ensemble counterpart instead of jumping directly from Logistic Regression to Random Forest.

5. **Hyperparameter descriptions clarified**
   - The write-up now explains parameters in plain language, for example `C` in Logistic Regression as the inverse of L2 regularization strength.
   - Tree-model tuning is described in terms of depth control, leaf-size control, and pruning or regularization.

6. **Conclusions narrowed**
   - The evaluation draft now avoids claiming that HistGradientBoosting is uniformly best on every dimension.
   - The revised interpretation emphasizes that record-level gains are modest and that district-level spatial metrics remain weak across models.

7. **Evaluation coverage extended**
   - Spatial-temporal evaluation was rerun for the added `Decision Tree` baseline.
   - The revised comparison now makes it explicit that aggregate spatial performance does not follow the same ranking as record-level AUROC/AUPRC.

8. **Experiment defaults refactored**
   - The default modeling scripts now target the course-recommended temporal setup: train on `2015-2024` and reserve `2025` as a holdout year.
   - The dataset builder no longer silently shrinks the scope when requested yearly archives are missing.

## Remaining follow-up

- Regenerate the modeling artifacts after the `2025` yearly archive is added locally.
- Fold the revised Phase 2 sections directly into the Phase 3 final report and presentation script.
- Run one final proofreading pass on the polished report PDF before submission.
