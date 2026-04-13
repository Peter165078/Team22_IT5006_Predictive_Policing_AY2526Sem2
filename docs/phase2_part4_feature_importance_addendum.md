# Phase 2 Part 4 Addendum: Feature Importance Analysis

This note strengthens the `Feature Importance Analysis` section with concrete artifacts generated from the saved Phase 2 models.

## Method used

Different models expose different forms of feature importance, so the analysis used the following approach:

- **Logistic Regression**: absolute coefficient magnitude
- **Decision Tree**: built-in impurity-based feature importance
- **Random Forest**: built-in impurity-based feature importance
- **HistGradientBoosting**: permutation importance on the held-out test split using `average_precision` as the scoring metric

The saved artifacts are in:

- `artifacts/metrics/feature_importance/`

Main files:

- `logistic_regression_feature_importance.csv`
- `decision_tree_feature_importance.csv`
- `random_forest_feature_importance.csv`
- `hist_gradient_boosting_feature_importance.csv`
- `hist_gradient_boosting_feature_groups.csv`
- `feature_importance_summary.json`

## Main findings

### 1. Temporal features are the strongest drivers for the best model

For the current best-performing model, `HistGradientBoosting`, permutation importance shows that temporal signals dominate the importance ranking. The top features are:

1. `hour_sin`
2. `hour`
3. `day_of_week`
4. `dow_sin`

The grouped feature summary further shows:

- temporal features: `94.27%`
- spatial features: `4.08%`
- historical features: `1.36%`
- missing indicators: `0.29%`

This suggests that, for the current benchmark setting, short-term temporal structure contributes much more to prediction quality than any single spatial or historical feature when the importance is measured on held-out data.

### 2. Spatial features still matter, but more as supporting signals

For tree-based and linear models, several spatial variables appear repeatedly among the stronger predictors, including:

- `X Coordinate`
- `Y Coordinate`
- `Latitude`
- `Beat`
- `District`
- `grid_lat_bin`

This indicates that location context still plays a meaningful supporting role, even though district-level spatial aggregation performance remains weaker than temporal alignment.

### 3. The Decision Tree baseline reinforces the same qualitative story

The added `Decision Tree` baseline is dominated by short-term temporal variables, especially:

- `hour_sin`
- `hour_cos`
- `hour`
- `day_of_week`

This is useful because it shows that even a simpler nonlinear model concentrates on timing signals first. In other words, the temporal dominance seen in HistGradientBoosting is not only a property of the boosted model; it is a recurring pattern across the tree-based family in this benchmark.

### 4. Historical crime features contribute, but less strongly than expected in the current setup

The Random Forest model ranks historical variables such as:

- `crimes_last_90d`
- `crimes_last_30d`
- `crimes_last_14d`

among its top features. However, for the current best model, the permutation-based importance of historical features is smaller than that of temporal features. This may reflect the current sampled benchmark setting, the way negatives were constructed, or the fact that short-term timing patterns are easier to recover than broad spatial hotspot patterns at the current aggregation level.

## Report-ready interpretation

A concise interpretation that can be used in the report is:

> Feature importance analysis shows that temporal signals, especially hour-of-day representations such as `hour_sin` and `hour`, are the most influential predictors for the best-performing HistGradientBoosting model. Spatial variables such as coordinates, beat, and district remain relevant as supporting features, while recent historical crime-count variables also contribute but appear less dominant in the current benchmark. Overall, the results suggest that short-term temporal patterns are the strongest driver of crime-occurrence prediction in the present experimental setting, with spatial context and local historical activity providing additional predictive value.

## Important caution

This section should still be written carefully in the final report:

- feature importance depends on the model and the importance method used
- the final benchmark should be regenerated from the refactored `2015-2024` training scope with `2025` held out
- the Decision Tree and Random Forest importances are impurity-based and should be interpreted more cautiously than held-out permutation importance
- these findings should be presented as evidence from the current experiment, not as universal causal claims about crime behavior
