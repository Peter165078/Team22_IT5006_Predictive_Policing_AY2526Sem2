# Phase 2 Report Draft: Problem Definition, Data Preparation, and Model Implementation

This write-up only covers the sections assigned to the modeling team in Milestone 2:

1. `Problem Definition & Data Preparation`
2. `Model Implementation & Training`

## 1. Problem Definition & Data Preparation

### 1.1 Clear problem statement and analytical approach

The objective of this phase is to formulate predictive policing as a supervised learning problem that can be trained and compared systematically. In our current setup, we define the task as a **binary classification problem**: given a location-time instance, predict whether at least one crime incident is likely to occur (`target = 1`) or not (`target = 0`). This framing is practical for police resource planning because it focuses on whether a district-grid and time combination should be flagged for higher attention, instead of attempting to predict a full incident record in advance.

We selected this formulation for two reasons. First, the available Chicago crime data provides rich temporal and spatial information, which makes occurrence forecasting more realistic than predicting post-event attributes such as detailed crime descriptions. Second, the binary setup allows us to compare multiple classical machine learning models under the same feature space, which fits the Milestone 2 requirement for model building and training.

The analytical pipeline follows a strict pre-event prediction logic. Only information that could plausibly be known before a future crime event is used during training. As a result, post-incident fields such as `Primary Type`, `Description`, `IUCR`, `Arrest`, and `Domestic` are excluded from the model features because they would leak the target and make the prediction task unrealistic. The remaining feature space is based on temporal context, spatial context, and prior crime history.

### 1.2 Data preprocessing and cleaning methodology

#### Data source and local experimental scope

The raw data source is the Chicago crime archive that has already been split by year in the repository under `apps/dashboard/split_data_by_year/`. For the current reproducible Phase 2 training run, we constructed a consolidated local dataset from the years **2022 to 2024**. To keep iteration time manageable on the development machine, we sampled **20,000 rows per year**, giving **60,000 positive incident rows** before negative-sample construction. This choice should be described as a practical experimentation setup rather than the final upper bound of the project.

After synthetic negative generation with a 1:1 ratio, the effective modeling dataset contains **120,000 rows** with a balanced class distribution. The current summary saved in `artifacts/metrics/phase2_data_summary.json` reports the following:

| Item | Value |
| --- | ---: |
| Positive rows | 60,000 |
| Negative rows | 60,000 |
| Total rows after negative construction | 120,000 |
| Overall positive rate | 0.500 |
| Feature count after preprocessing | 37 |
| Active historical windows | 7d, 14d, 30d, 90d |
| Train rows | 83,820 |
| Validation rows | 17,999 |
| Test rows | 17,999 |

#### Label construction and negative sampling

The original Chicago dataset only records observed crimes, so it does not contain explicit negative examples. To convert the problem into binary classification, we construct synthetic negatives by sampling district-day combinations with no recorded crime event in the positive set. For each accepted negative instance, we reuse a realistic historical template from the same district and only modify the timestamp and label. This design is important because an earlier median-based negative construction strategy introduced artificial patterns that tree-based models could exploit too easily. The revised approach generates negatives that are closer to the real feature distribution of the city while still enforcing the no-crime label.

#### Leakage control and train/validation/test split

The full dataset is sorted chronologically and then split into **70% training**, **15% validation**, and **15% test** partitions. This chronological split reduces temporal leakage and better matches the intended deployment scenario, where models should forecast future periods using patterns learned from earlier periods. During preprocessing, rows with no available prior history in the minimum active window are dropped as cold-start observations. In the current run, this removed **180 rows** from the training partition and **1 row** each from the validation and test partitions.

Leakage control is treated as a first-class preprocessing objective. In particular:

- identifier and administrative fields such as `ID`, `Case Number`, and `Updated On` are removed;
- post-incident descriptive fields are excluded because they are unavailable at prediction time;
- fitting operations such as scaling and category handling are learned on the training set only and then reused for validation and test data.

### 1.3 Feature engineering strategy

The final feature set combines temporal, spatial, and historical signals:

- **Temporal features**: hour, day of week, month, day of year, week of year, plus cyclic encodings for periodic patterns.
- **Behavioral indicators**: weekend, night-time, and rush-hour flags.
- **Spatial features**: district, ward, beat, community area, coordinates, and grid-cell representation.
- **Historical features**: prior crime counts in the same district and grid cell over multiple rolling windows (`7d`, `14d`, `30d`, `90d`).

This feature design reflects the operational logic of predictive policing: the model is asked to learn how recent temporal rhythms and local historical concentration patterns influence the probability of a future crime occurrence.

## 2. Model Implementation & Training

### 2.1 Implement multiple models for the proposed problem

To satisfy the project requirement for multiple models, we implemented three tabular baselines in the current benchmark:

1. **Logistic Regression**
2. **Random Forest**
3. **HistGradientBoosting**

These models were selected to cover different modeling assumptions. Logistic Regression serves as a transparent linear baseline, Random Forest captures nonlinear feature interactions through bagged decision trees, and HistGradientBoosting provides a stronger boosted-tree benchmark for structured tabular data. This model mix also satisfies the requirement to implement multiple models for the proposed predictive policing problem.

### 2.2 Analysis and choice of models

The choice of models goes beyond using a single classroom baseline. We intentionally combined:

- a **linear model** to test whether the engineered spatial-temporal features already contain useful separable signal;
- a **bagged tree ensemble** to capture nonlinear interactions without strong parametric assumptions;
- a **boosted tree model** to provide a stronger tabular benchmark with iterative error correction.

This progression allows the report to demonstrate both analytical reasoning and model diversity, rather than only reporting one algorithm.

### 2.3 Documentation of model architectures

The implemented model architectures are summarized below.

| Model | Architecture summary | Why it was included |
| --- | --- | --- |
| Logistic Regression | Linear classifier with balanced class weights and L2-style regularized optimization | Transparent baseline for binary occurrence prediction |
| Random Forest | Ensemble of decision trees trained with bootstrap sampling and feature subsampling | Captures nonlinear interactions and is robust for tabular data |
| HistGradientBoosting | Gradient-boosted tree ensemble trained in stages | Stronger benchmark for structured tabular prediction |

Although the repository also contains a PyTorch MLP scaffold in the training script, the current reproducible benchmark and saved Phase 2 artifacts focus on the three models listed above.

### 2.4 Documentation of training process

The training code has been consolidated into a reproducible script in `src/scripts/train.py`. The workflow is:

1. Build a single raw Phase 2 CSV from yearly ZIP archives if needed.
2. Run the shared preprocessing pipeline in `src/data/processor.py`.
3. Train each model on the training split.
4. Use the validation split to choose the better hyperparameter setting from a small manual grid.
5. Save the selected models, prediction outputs, and metric tables under `artifacts/models/` and `artifacts/metrics/`.

This setup is appropriate for Milestone 2 because it makes the training procedure easy to rerun, easy to compare, and easy to document in a reproducible way.

The training process can be summarized as follows:

| Step | Description |
| --- | --- |
| Data assembly | Merge yearly ZIP files into one Phase 2 CSV |
| Preprocessing | Apply the shared `DataProcessor` pipeline |
| Splitting | Use chronological 70/15/15 train-validation-test split |
| Training | Fit each candidate model on the training split |
| Selection | Choose the stronger configuration using validation AUROC |
| Persistence | Save selected model files, prediction files, and metric tables |

### 2.5 Hyperparameter tuning methodology

Instead of reporting a single untuned run, each model is trained with a small validation-oriented parameter search:

- **Logistic Regression**: tuned regularization strength `C` and optimization length.
- **Random Forest**: tuned tree depth, leaf size, estimator count, and bootstrap sample fraction.
- **HistGradientBoosting**: tuned learning rate, depth, boosting iterations, minimum leaf size, and regularization.

The validation split is used to select the configuration with the strongest AUROC before saving the final artifact. This is a lightweight but defensible tuning strategy for the current milestone.

### 2.6 Selected configurations and current training outputs

The best validation configurations in the current run are:

| Model | Selected configuration |
| --- | --- |
| Logistic Regression | `C = 0.5`, `max_iter = 400` |
| Random Forest | `n_estimators = 200`, `max_depth = 16`, `min_samples_leaf = 4`, `max_samples = 0.35`, `max_features = sqrt` |
| HistGradientBoosting | `learning_rate = 0.05`, `max_depth = 10`, `max_iter = 300`, `min_samples_leaf = 80`, `l2_regularization = 0.1` |

The model artifacts and prediction files have been saved to:

- `artifacts/models/logistic_regression.pkl`
- `artifacts/models/random_forest.pkl`
- `artifacts/models/hist_gradient_boosting.pkl`
- `artifacts/metrics/predictions/`

The current summary metrics are shown below. These numbers provide evidence that the training pipeline is working end-to-end and that the selected configurations produce stable, reproducible outputs for Milestone 2.

| Model | Validation AUROC | Validation AUPRC | Test AUROC | Test AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.579 | 0.549 | 0.577 | 0.550 |
| Random Forest | 0.586 | 0.560 | 0.584 | 0.568 |
| HistGradientBoosting | 0.589 | 0.563 | 0.591 | 0.573 |

At this stage, HistGradientBoosting is the strongest of the three implemented baselines on both validation and test data. However, these results should still be treated as intermediate Milestone 2 outputs rather than the final conclusion of the project.
