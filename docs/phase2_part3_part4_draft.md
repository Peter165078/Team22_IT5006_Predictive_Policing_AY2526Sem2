# Phase 2 Part 3 and Part 4 Draft

This draft consolidates the current evaluation, interpretation, and business-insight materials into a report-ready form for final PDF integration.

## 3. Model Evaluation and Comparison

### 3.1 Standard classification metrics

The three trained models, Logistic Regression, Random Forest, and HistGradientBoosting, were evaluated on both the validation and test splits using accuracy, precision, recall, F1-score, AUROC, and AUPRC. Overall, the tree-based ensemble models outperformed the linear baseline, with HistGradientBoosting achieving the strongest overall performance.

| Model | Split | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HistGradientBoosting | Validation | 0.565 | 0.544 | 0.757 | 0.633 | 0.589 | 0.563 |
| HistGradientBoosting | Test | 0.566 | 0.550 | 0.736 | 0.630 | 0.591 | 0.573 |
| Random Forest | Validation | 0.561 | 0.544 | 0.709 | 0.615 | 0.586 | 0.560 |
| Random Forest | Test | 0.563 | 0.551 | 0.685 | 0.611 | 0.584 | 0.568 |
| Logistic Regression | Validation | 0.563 | 0.556 | 0.586 | 0.571 | 0.579 | 0.549 |
| Logistic Regression | Test | 0.565 | 0.563 | 0.584 | 0.573 | 0.577 | 0.550 |

Accuracy values were similar across models, which is expected because accuracy is less informative under class imbalance. The more useful distinctions come from recall, F1-score, AUROC, and AUPRC. HistGradientBoosting achieved the highest recall on both splits and also produced the best F1, AUROC, and AUPRC, indicating the strongest balance between capturing positive crime events and preserving overall discrimination ability. Random Forest performed competitively and consistently ranked second, while Logistic Regression provided a simpler but weaker baseline.

Validation and test metrics were also close for all three models, which suggests that the current models generalize reasonably well and do not show obvious overfitting under the present sampled benchmark.

### 3.2 Spatial accuracy evaluation

To evaluate spatial alignment, the validation and test predictions were aggregated by Chicago police district after matching each saved prediction back to its aligned metadata. For each district, the analysis computed total observations, actual crime counts, predicted positives, actual positive rate, predicted positive rate, average predicted risk, and maximum predicted risk. Spatial performance was then assessed using two measures:

- Pearson correlation between actual district crime counts and average predicted district risk
- hotspot overlap between the top-ranked districts by actual crime count and the top-ranked districts by predicted average risk

The district-level results indicate that spatial alignment is weak at this aggregation level.

| Model | Split | District Correlation | Top-5 Overlap | Top-10 Overlap |
| --- | --- | ---: | ---: | ---: |
| HistGradientBoosting | Validation | -0.186 | 0.0 | 0.4 |
| HistGradientBoosting | Test | -0.181 | 0.0 | 0.2 |
| Random Forest | Validation | -0.119 | 0.2 | 0.4 |
| Random Forest | Test | -0.116 | 0.0 | 0.3 |
| Logistic Regression | Validation | -0.144 | 0.2 | 0.4 |
| Logistic Regression | Test | -0.059 | 0.2 | 0.5 |

Across all models, district-level correlations were negative and hotspot overlap remained limited. This means that even though the models can make moderately useful record-level predictions, they do not reliably recover the true ranking of high-crime districts when risk is aggregated to the district level.

Some district plots may still show partial visual trend similarity between actual rates and predicted risk. However, these plots should be treated as qualitative only. The quantitative metrics provide the more reliable conclusion here: district-level spatial discrimination remains weak in the current setup. A plausible reason is that district aggregation is too coarse to preserve finer hotspot structure, especially when the model is trained for occurrence prediction at the record level rather than for district ranking directly.

### 3.3 Temporal accuracy measures

Temporal evaluation was performed by grouping predictions by hour of day and by day of week, then comparing the average predicted risk in each time bin against the corresponding actual crime rate. Pearson correlation was used to summarize alignment between the predicted and actual temporal patterns.

| Model | Split | Hourly Correlation | Day-of-Week Correlation |
| --- | --- | ---: | ---: |
| HistGradientBoosting | Validation | 0.974 | -0.385 |
| HistGradientBoosting | Test | 0.972 | 0.505 |
| Random Forest | Validation | 0.969 | -0.071 |
| Random Forest | Test | 0.976 | -0.359 |
| Logistic Regression | Validation | 0.865 | -0.530 |
| Logistic Regression | Test | 0.827 | 0.208 |

The hourly results are strong across all three models. Hourly correlations range from 0.827 to 0.976, showing that the models capture daily timing patterns reasonably well. HistGradientBoosting and Random Forest are especially strong on this dimension, suggesting that short-term temporal structure is one of the most learnable signals in the current benchmark.

By contrast, day-of-week performance is much less stable. The correlations range from negative values up to 0.505, which indicates that weekly crime patterns are not consistently recovered across models and splits. In practical terms, the models appear to be much better at identifying high-risk hours within a day than at producing reliable weekday-level ranking.

### 3.4 Cross-validation and robustness analysis

To assess stability, 5-fold stratified cross-validation was conducted on the training data using the tuned hyperparameter settings selected from the original train-validation procedure. Robustness was evaluated using the mean, standard deviation, and fold range of the main classification metrics.

Among the three models, HistGradientBoosting again delivered the best combination of performance and stability:

- Accuracy: `0.5689 +/- 0.0050`
- Precision: `0.5511 +/- 0.0043`
- Recall: `0.7425 +/- 0.0060`
- F1-score: `0.6326 +/- 0.0026`
- AUROC: `0.5976 +/- 0.0060`
- AUPRC: `0.5805 +/- 0.0047`

Random Forest performed competitively but showed slightly lower recall and F1-score, together with somewhat higher variability in recall across folds. Logistic Regression was the most interpretable baseline and remained stable across folds, but its predictive performance was consistently lower than that of the ensemble models.

Across all three models, the standard deviations for most metrics remained below `0.01`, which indicates that performance is not highly sensitive to the specific fold assignment. The consistent model ranking across the folds strengthens the conclusion that HistGradientBoosting is the most reliable model under the current experimental setup.

### 3.5 Comparison summary

Taken together, the evaluation results point to a clear pattern:

- `HistGradientBoosting` is the strongest overall model
- `Random Forest` is a competitive second-best ensemble baseline
- `Logistic Regression` is useful as an interpretable benchmark, but underperforms the tree-based models

The current models are better at learning record-level occurrence risk and short-term temporal rhythms than at recovering district-level hotspot structure. This distinction is important for interpretation: the model is more reliable as a short-horizon decision-support tool than as a district-ranking system.

## 4. Model Interpretation and Business Insights

### 4.1 Feature importance analysis

Feature importance analysis was strengthened using model-specific importance methods:

- Logistic Regression: absolute coefficient magnitude
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

Spatial signals are present but weaker when aggregated at the district level. This means the model is better used for localized risk awareness and scheduling support than for claiming strong district-wide hotspot ranking. In practice, the outputs could be used as one decision-support signal among others, especially when combined with local operational knowledge, recent incident context, and existing policing workflows.

### 4.3 Limitations and constraints

Several limitations should be acknowledged clearly:

- the benchmark uses a sampled `2022-2024` dataset for iteration speed and reproducibility
- the binary task depends on synthetic negative generation, which is a design choice rather than a naturally observed label structure
- district-level spatial aggregation may be too coarse to reflect finer hotspot patterns
- the model relies on historical recorded crime data, which may contain reporting bias or incomplete coverage
- predictive outputs do not capture all real-world drivers such as policing strategy changes, environmental changes, or broader social conditions

For these reasons, the model should be presented as a decision-support tool rather than a standalone operational decision system. The current results are useful for structured comparison and early forecasting analysis, but they do not justify deterministic or fully automated deployment claims.
