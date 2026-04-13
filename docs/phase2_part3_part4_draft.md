# Phase 2 Part 3 and Part 4 Draft

This draft consolidates the current evaluation, interpretation, and business-insight materials into a report-ready form for final PDF integration.

## 3. Model Evaluation and Comparison

### 3.1 Standard classification metrics

The trained models were evaluated on both the validation and test splits using accuracy, precision, recall, F1-score, AUROC, and AUPRC. The main pattern is that the tree-based ensemble models achieve slightly stronger record-level discrimination than the linear baseline, but the gaps are modest rather than dramatic.

| Model | Split | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HistGradientBoosting | Validation | 0.565 | 0.544 | 0.757 | 0.633 | 0.589 | 0.563 |
| HistGradientBoosting | Test | 0.566 | 0.550 | 0.736 | 0.630 | 0.591 | 0.573 |
| Random Forest | Validation | 0.562 | 0.544 | 0.706 | 0.615 | 0.585 | 0.560 |
| Random Forest | Test | 0.563 | 0.551 | 0.691 | 0.613 | 0.583 | 0.567 |
| Decision Tree | Validation | 0.565 | 0.549 | 0.676 | 0.606 | 0.581 | 0.545 |
| Decision Tree | Test | 0.562 | 0.551 | 0.667 | 0.604 | 0.581 | 0.552 |
| Logistic Regression | Validation | 0.563 | 0.556 | 0.586 | 0.571 | 0.579 | 0.549 |
| Logistic Regression | Test | 0.565 | 0.563 | 0.584 | 0.573 | 0.577 | 0.550 |

Accuracy values were similar across models, which is expected because accuracy is less informative for this task. The more useful distinctions come from recall, F1-score, AUROC, and AUPRC. HistGradientBoosting achieved the strongest record-level metrics on these splits, but the margins over Random Forest are small. The newly added Decision Tree baseline performs slightly better than Logistic Regression on AUROC while remaining below the two ensemble methods, which supports the claim that bagging and boosting are adding value beyond a single tree. For that reason, the correct interpretation is not that one model dominates on every dimension, but that HistGradientBoosting offers the best record-level trade-off in this particular benchmark while the other models remain competitive.

Validation and test metrics were also close for all four models, which suggests that the current models generalize reasonably well and do not show obvious overfitting under the present sampled benchmark.

### 3.2 Spatial accuracy evaluation

To evaluate spatial alignment, the validation and test predictions were aggregated by Chicago police district after matching each saved prediction back to its aligned metadata. For each district, the analysis computed total observations, actual crime counts, predicted positives, actual positive rate, predicted positive rate, average predicted risk, and maximum predicted risk. Spatial performance was then assessed using two measures:

- Pearson correlation between actual district crime counts and average predicted district risk
- hotspot overlap between the top-ranked districts by actual crime count and the top-ranked districts by predicted average risk

The district-level results indicate that spatial alignment is weak at this aggregation level. The detailed spatial and temporal aggregation below remains focused on the three main deployment candidates from the earlier benchmark release, while the Decision Tree was added later as a targeted baseline check for the instructor feedback on standard metrics.

| Model | Split | District Correlation | Top-5 Overlap | Top-10 Overlap |
| --- | --- | ---: | ---: | ---: |
| Decision Tree | Validation | -0.386 | 0.4 | 0.3 |
| Decision Tree | Test | 0.000 | 0.4 | 0.5 |
| HistGradientBoosting | Validation | -0.186 | 0.0 | 0.4 |
| HistGradientBoosting | Test | -0.181 | 0.0 | 0.2 |
| Random Forest | Validation | -0.121 | 0.2 | 0.5 |
| Random Forest | Test | -0.116 | 0.0 | 0.3 |
| Logistic Regression | Validation | -0.144 | 0.2 | 0.4 |
| Logistic Regression | Test | -0.059 | 0.2 | 0.5 |

Across the four models, district-level correlations were mostly weak or negative and hotspot overlap remained limited. The added Decision Tree baseline is useful here because it shows that aggregate spatial results do not follow the same ranking as the record-level metrics: for example, the Decision Tree achieves stronger hotspot overlap on the test split than HistGradientBoosting. This means that even though the models can make moderately useful record-level predictions, they do not reliably recover the true ranking of high-crime districts when risk is aggregated to the district level. The spatial evaluation therefore does **not** support a strong claim that HistGradientBoosting is categorically better on every operational dimension.

Some district plots may still show partial visual trend similarity between actual rates and predicted risk. However, these plots should be treated as qualitative only. The quantitative metrics provide the more reliable conclusion here: district-level spatial discrimination remains weak in the current setup. A plausible reason is that district aggregation is too coarse to preserve finer hotspot structure, especially when the model is trained for occurrence prediction at the record level rather than for district ranking directly.

### 3.3 Temporal accuracy measures

Temporal evaluation was performed by grouping predictions by hour of day and by day of week, then comparing the average predicted risk in each time bin against the corresponding actual crime rate. Pearson correlation was used to summarize alignment between the predicted and actual temporal patterns.

| Model | Split | Hourly Correlation | Day-of-Week Correlation |
| --- | --- | ---: | ---: |
| Decision Tree | Validation | 0.932 | -0.097 |
| Decision Tree | Test | 0.931 | 0.178 |
| HistGradientBoosting | Validation | 0.974 | -0.385 |
| HistGradientBoosting | Test | 0.972 | 0.505 |
| Random Forest | Validation | 0.969 | -0.071 |
| Random Forest | Test | 0.976 | -0.170 |
| Logistic Regression | Validation | 0.865 | -0.530 |
| Logistic Regression | Test | 0.827 | 0.208 |

The hourly results are strong across all four models. Hourly correlations range from 0.827 to 0.976, showing that the models capture daily timing patterns reasonably well. HistGradientBoosting and Random Forest are especially strong on this dimension, while the Decision Tree remains materially stronger than Logistic Regression on hourly alignment. This reinforces the view that short-term temporal structure is one of the most learnable signals in the current benchmark.

By contrast, day-of-week performance is much less stable. The correlations range from negative values up to 0.505, which indicates that weekly crime patterns are not consistently recovered across models and splits. In practical terms, the models appear to be much better at identifying high-risk hours within a day than at producing reliable weekday-level ranking.

### 3.4 Cross-validation and robustness analysis

To assess stability, 5-fold stratified cross-validation was conducted on the training data using the tuned hyperparameter settings selected from the original train-validation procedure. Robustness was evaluated using the mean, standard deviation, and fold range of the main classification metrics.

Among the compared models, HistGradientBoosting delivered the strongest combination of record-level performance and fold-to-fold stability:

- Accuracy: `0.5689 +/- 0.0050`
- Precision: `0.5511 +/- 0.0043`
- Recall: `0.7425 +/- 0.0060`
- F1-score: `0.6326 +/- 0.0026`
- AUROC: `0.5976 +/- 0.0060`
- AUPRC: `0.5805 +/- 0.0047`

Random Forest performed competitively but showed slightly lower recall and F1-score, together with somewhat higher variability in recall across folds. Logistic Regression was the most interpretable baseline and remained stable across folds, but its predictive performance was consistently lower than that of the ensemble models.

Across all models, the standard deviations for most metrics remained below `0.01`, which indicates that performance is not highly sensitive to the specific fold assignment. The cross-validation evidence supports HistGradientBoosting as the strongest **record-level** benchmark in the current experimental setup, but it should still be interpreted alongside the weak district-level spatial results.

### 3.5 Comparison summary

Taken together, the evaluation results point to a cautious pattern:

- `HistGradientBoosting` is the strongest record-level benchmark on AUROC, AUPRC, recall, and F1
- `Random Forest` is a close ensemble alternative
- `Decision Tree` improves modestly over Logistic Regression but remains below the ensemble methods
- `Logistic Regression` remains useful as an interpretable benchmark and is not uniformly worse on every aggregated evaluation view

The current models are better at learning record-level occurrence risk and short-term temporal rhythms than at recovering district-level hotspot structure. This distinction is important for interpretation: the model is more reliable as a short-horizon decision-support tool than as a district-ranking system.

## 4. Model Interpretation and Business Insights

### 4.1 Feature importance analysis

Feature importance analysis was strengthened using model-specific importance methods:

- Logistic Regression: absolute coefficient magnitude
- Decision Tree: impurity-based feature importance
- Random Forest: impurity-based feature importance
- HistGradientBoosting: permutation importance on the held-out test split using average precision

The best-performing model, HistGradientBoosting, is dominated by temporal features. Its top features include `hour_sin`, `hour`, `day_of_week`, and `dow_sin`. Grouped permutation importance shows the following contribution shares:

- temporal features: `94.27%`
- spatial features: `4.08%`
- historical features: `1.36%`
- missing indicators: `0.29%`

These results suggest that short-term temporal structure is the strongest driver of predictive performance in the current benchmark. Spatial variables such as `X Coordinate`, `Y Coordinate`, `Latitude`, `Beat`, and `Community Area` still appear among the stronger supporting features, while recent-history variables such as `crimes_last_30d` remain relevant but less dominant for the best model.

This interpretation should be kept modest. Feature importance here reflects the current sampled experiment and the selected importance method; it should not be treated as a direct causal explanation of crime behavior.

### 4.2 Actionable insights for law enforcement

The strongest practical insight from the current results is that crime-occurrence prediction in this setup is much more informative over time than across coarse district boundaries. Because hourly temporal alignment is strong, the model can support time-sensitive operational planning such as prioritizing patrol attention during high-risk periods of the day.

More concretely, the current benchmark supports three modest operational uses:

- **timing support**: use predicted risk to highlight higher-risk hours for patrol scheduling, shift briefing, or staffing review;
- **localized triage**: treat the prediction as a neighborhood-level cue that can be combined with recent incidents and officer knowledge, rather than as a district-wide hotspot ranking tool;
- **human-in-the-loop prioritization**: use thresholds to trigger review queues or additional monitoring, while keeping final deployment decisions with human operators.

Spatial signals are present but weaker when aggregated at the district level. This means the model is better used for localized risk awareness and scheduling support than for claiming strong district-wide hotspot ranking. In practice, the outputs should be used as one decision-support signal among others, especially when combined with local operational knowledge, recent incident context, and existing policing workflows.

### 4.3 Limitations and constraints

Several limitations should be acknowledged clearly:

- the refactored final benchmark should be rerun on the full `2015-2024` training window with `2025` held out for evaluation
- the binary task depends on synthetic negative generation, which is a design choice for label construction rather than a naturally observed label structure
- district-level spatial aggregation may be too coarse to reflect finer hotspot patterns
- the model relies on historical recorded crime data, which may contain reporting bias or incomplete coverage
- predictive outputs do not capture all real-world drivers such as policing strategy changes, environmental changes, or broader social conditions
- any operational use would require policy review, fairness review, and threshold calibration beyond what was done in this milestone

For these reasons, the model should be presented as a decision-support tool rather than a standalone operational decision system. The current results are useful for structured comparison and early forecasting analysis, but they do not justify deterministic or fully automated deployment claims.
