# Phase 2 Report Draft: Problem Definition, Data Preparation, and Model Implementation

This write-up only covers the sections assigned to the modeling team in Milestone 2:

1. `Problem Definition & Data Preparation`
2. `Model Implementation & Training`

## 1. Problem Definition & Data Preparation

### 1.1 Clear problem statement and analytical approach

The objective of this phase is to formulate predictive policing as a supervised learning problem that can be trained and compared systematically. In our current setup, we define the task as a **binary classification problem**: given an observed feature vector `X` for a location-time instance, estimate `P(Y = 1 | X)`, where `Y = 1` means that at least one crime incident occurs in that space-time unit and `Y = 0` means no recorded incident occurs. This framing is practical for police resource planning because it focuses on whether a district-grid and time combination should be flagged for higher attention, instead of attempting to predict a full incident record in advance.

We selected this formulation for two reasons. First, the available Chicago crime data provides rich temporal and spatial information, which makes occurrence forecasting more realistic than predicting post-event attributes such as detailed crime descriptions. Second, the binary setup allows us to compare multiple classical machine learning models under the same feature space, which fits the Milestone 2 requirement for model building and training.

The analytical pipeline follows a strict pre-event prediction logic. Only information that could plausibly be known before a future crime event is used during training. As a result, post-incident fields such as `Primary Type`, `Description`, `IUCR`, `Arrest`, and `Domestic` are excluded from the model features because they would leak the target and make the prediction task unrealistic. The remaining feature space is based on temporal context, spatial context, and prior crime history, and all models are trained to output a risk estimate rather than a deterministic claim about crime occurrence.

### 1.2 Data preprocessing and cleaning methodology

#### Data source and refactored modeling scope

The raw data source is the Chicago crime archive that has already been split by year in the repository under `apps/dashboard/split_data_by_year/`. In the refactored pipeline, the intended modeling scope follows the course recommendation more closely: yearly archives from **2015 to 2024** are used for model development and training, while **2025** is reserved as an explicit chronological holdout year for validation and test. The dataset builder now expects the full requested year range and raises an explicit error if any required yearly archive is missing, instead of silently shrinking the modeling scope.

Synthetic negative-label construction is still applied at a 1:1 ratio after merging the yearly positive records, but the final row counts should now be regenerated from the full `2015-2025` dataset before the final report is frozen. Any legacy `2022-2024` artifact summaries should be treated as superseded benchmark outputs rather than as the final modeling setup.

#### Label construction, negative construction, and imbalance handling

The original Chicago dataset only records observed crimes, so it does not contain explicit negative examples. To convert the problem into binary classification, we construct synthetic `Y = 0` rows by sampling district-day combinations with no recorded crime event in the positive set. For each accepted negative instance, we reuse a realistic historical template from the same district and only modify the timestamp and label. This preserves the district-level spatial context while ensuring that the sampled day does not collide with a known positive event in that district.

It is important to describe this step precisely. The synthetic rows are used as a **label-construction device** to make supervised binary learning possible, not as a claim that these are naturally observed negatives. Residual imbalance handling is treated separately at model-training time through built-in weighting mechanisms such as `class_weight = balanced` for Logistic Regression and Decision Tree, and `class_weight = balanced_subsample` for Random Forest. This keeps the modeling story clearer:

- synthetic negatives provide explicit `0/1` labels for occurrence forecasting;
- model-side weighting handles imbalance during fitting.

The current construction also includes several safeguards against unrealistic duplication:

- negatives are only accepted on district-day combinations with no observed crime in the sampled positive set;
- spatial fields come from a same-district template rather than from a city-wide average row;
- post-incident fields are removed before modeling, so the model cannot memorize crime-specific labels from the reused template;
- evaluation is performed on chronologically later data, which helps detect whether the constructed negatives are leading to brittle overfitting.

Because this construction remains an approximation, it should be presented transparently as a pragmatic design choice and revisited in the limitations section. A more formal space-time grid formulation would be a natural extension for future work.

#### Leakage control and train/validation/test split

The full dataset is sorted chronologically before preprocessing, but the refactored training pipeline no longer relies on a simple percentage-based split for model evaluation. Instead, the intended evaluation setup is a **year-based temporal holdout**:

- training data: **2015-2024**
- holdout year: **2025**
- validation/test split: the 2025 holdout is split chronologically into two consecutive parts

This design better matches the deployment scenario, where models should learn from past years and then be evaluated on a genuinely future year. During preprocessing, rows with no available prior history in the minimum active window are dropped as cold-start observations.

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

These rolling windows were chosen to represent interpretable recency horizons rather than arbitrary cutoffs: `7d` captures weekly rhythm, `14d` captures short-term carryover across two weeks, `30d` captures monthly seasonality, and `90d` captures broader quarter-scale persistence. This feature design reflects the operational logic of predictive policing: the model is asked to learn how recent temporal rhythms and local historical concentration patterns influence the probability of a future crime occurrence.

## 2. Model Implementation & Training

### 2.1 Implement multiple models for the proposed problem

To satisfy the project requirement for multiple models, we implemented four classical tabular baselines in the current benchmark:

1. **Logistic Regression**
2. **Decision Tree**
3. **Random Forest**
4. **HistGradientBoosting**

These models were selected to cover different modeling assumptions. Logistic Regression serves as a transparent linear baseline, Decision Tree provides the simplest nonlinear tree baseline, Random Forest captures nonlinear feature interactions through bagged decision trees, and HistGradientBoosting provides a stronger boosted-tree benchmark for structured tabular data. This model mix makes it possible to test whether ensemble methods materially improve over a single-tree baseline instead of comparing only unrelated model families.

### 2.2 Analysis and choice of models

The choice of models goes beyond using a single classroom baseline. We intentionally combined:

- a **linear model** to test whether the engineered spatial-temporal features already contain useful separable signal;
- a **single-tree model** to provide an interpretable nonlinear baseline;
- a **bagged tree ensemble** to capture nonlinear interactions without strong parametric assumptions;
- a **boosted tree model** to provide a stronger tabular benchmark with iterative error correction.

This progression allows the report to demonstrate both analytical reasoning and model diversity, rather than only reporting one algorithm.

### 2.3 Documentation of model architectures

The implemented model architectures are summarized below.

| Model | Architecture summary | Why it was included |
| --- | --- | --- |
| Logistic Regression | Linear classifier with balanced class weights and L2-regularized optimization | Transparent baseline for binary occurrence prediction |
| Decision Tree | Single classification tree with balanced class weights and depth/leaf regularization | Interpretable nonlinear baseline for testing whether ensembling is beneficial |
| Random Forest | Ensemble of decision trees trained with bootstrap sampling and feature subsampling | Captures nonlinear interactions and stabilizes a single-tree baseline |
| HistGradientBoosting | Gradient-boosted tree ensemble trained in stages | Stronger benchmark for structured tabular prediction |

Although the repository also contains a PyTorch MLP scaffold in the training script, the current reproducible benchmark and saved Phase 2 artifacts focus on the four classical baselines listed above.

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
| Splitting | Train on 2015-2024 and split the 2025 holdout chronologically into validation and test |
| Training | Fit each candidate model on the training split |
| Selection | Choose the stronger configuration using validation AUROC |
| Persistence | Save selected model files, prediction files, and metric tables |

### 2.5 Hyperparameter tuning methodology

Instead of reporting a single untuned run, each model is trained with a small validation-oriented parameter search:

- **Logistic Regression**: tuned regularization strength `C` and optimization length. Here, `C` is the inverse of the L2 regularization strength, so smaller `C` means stronger shrinkage.
- **Decision Tree**: tuned maximum tree depth, minimum leaf size, and post-pruning strength (`ccp_alpha`) to reduce overfitting.
- **Random Forest**: tuned tree depth, leaf size, estimator count, and bootstrap sample fraction.
- **HistGradientBoosting**: tuned learning rate, depth, boosting iterations, minimum leaf size, and regularization.

The validation split is used to select the configuration with the strongest AUROC before saving the final artifact. This is a lightweight but defensible tuning strategy for the current milestone.

### 2.6 Selected configurations and current training outputs

The parameter grids below remain the active benchmark search space in the refactored pipeline:

| Model | Selected configuration |
| --- | --- |
| Logistic Regression | `C = 0.5`, `max_iter = 400` |
| Decision Tree | `max_depth = 14`, `min_samples_leaf = 25`, `ccp_alpha = 0.0005` |
| Random Forest | `n_estimators = 200`, `max_depth = 16`, `min_samples_leaf = 4`, `max_samples = 0.35`, `max_features = sqrt` |
| HistGradientBoosting | `learning_rate = 0.05`, `max_depth = 10`, `max_iter = 300`, `min_samples_leaf = 80`, `l2_regularization = 0.1` |

The model artifacts and prediction files are written to:

- `artifacts/models/logistic_regression.pkl`
- `artifacts/models/decision_tree.pkl`
- `artifacts/models/random_forest.pkl`
- `artifacts/models/hist_gradient_boosting.pkl`
- `artifacts/metrics/predictions/`

Any summary metrics from the earlier sampled `2022-2024` benchmark should now be treated as legacy artifacts. The final report should cite only the rerun metrics generated from the refactored `2015-2024` training scope plus `2025` holdout evaluation.
