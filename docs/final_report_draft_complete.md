# Team 22 Predictive Policing Final Report Draft

## Cover Page

**Module:** IT5006 Fundamentals of Data Analytics  
**Project Title:** Chicago Crime Intel: District-Hour Crime Risk Modeling, External Generalization, and Proof-of-Concept Deployment  
**Team:** Team 22  
**Semester:** AY 2025/26 Semester 2  
**Course Instructors:** DISA, School of Computing, National University of Singapore  
**Submission Type:** Final Report (Phase 3)  
**GitHub Repository:** <https://github.com/Peter165078/Team22_IT5006_Predictive_Policing_AY2526Sem2.git>  
**Application Link:** <https://team22it5006predictivepolicingay2526sem2-5xjmr8cwbappeurrspssw.streamlit.app/>

---

## Executive Summary

This project studies predictive policing as a data analytics problem and frames it explicitly as a **decision-support** task rather than an automated enforcement system. Following the IT5006 project requirements, we built an end-to-end workflow that integrates historical crime analysis, machine learning model development, external generalization testing, and proof-of-concept deployment. The final system uses the Chicago Crime Dataset for model development and applies FBI NIBRS data from Texas and Colorado to examine robustness outside the original training jurisdiction.

Our final modeling definition predicts the probability that at least one crime will occur in a **district-hour** cell. This district-hour formulation is important because it replaces the earlier event-level perspective with a more operationally meaningful unit and yields natural positive and negative labels. Instead of asking whether a particular individual event belongs to a certain class, the project asks whether a location-time unit shows crime occurrence risk. This framing is more aligned with planning support, resource awareness, and spatiotemporal deployment use cases discussed in predictive policing literature and in the course brief.

For the final benchmark, we used Chicago yearly archives from **2015 to 2024** for model development and reserved **2025** as a chronological holdout year for validation and test evaluation. The resulting district-hour modeling table contains **2,217,936** rows, of which **1,409,469** are positive and **808,467** are negative. The final feature space contains **37** engineered variables after preprocessing and cold-start filtering. These features are organized into temporal, spatial, and rolling historical groups.

We compared four benchmark models: Logistic Regression, Decision Tree, Random Forest, and HistGradientBoosting. HistGradientBoosting achieved the strongest overall holdout performance, with **test AUROC = 0.7381** and **test AUPRC = 0.8063**, although the margin over Random Forest was small. Rather than overselling this result, we interpret it as evidence of a modest but real performance advantage under a fair and consistent evaluation setup.

Beyond record-level discrimination metrics, we also evaluated spatial and temporal alignment on the 2025 holdout. HistGradientBoosting achieved **district correlation = 0.9996**, **top-5 overlap = 1.0**, and **hourly correlation = 0.9900**, indicating that the final district-hour benchmark is materially more learnable and more operationally interpretable than our earlier reduced event-level benchmark. Feature importance analysis further shows that temporal signals dominate the predictive structure, followed by historical crime windows and then spatial inputs.

To satisfy the course requirement on generalization testing, we applied the Chicago-trained benchmark to externally prepared **NIBRS county-hour datasets** from Texas and Colorado. On 2024 NIBRS evaluation data, the model achieved **AUROC = 0.8229** in Texas and **AUROC = 0.8203** in Colorado. External hourly alignment remained strong, while spatial transfer weakened more noticeably, especially in Texas. This supports a cautious interpretation: the model learns meaningful timing-related structure that transfers across jurisdictions more readily than exact spatial ranking.

Finally, we deployed the project as a Streamlit-based proof-of-concept application, `Chicago Crime Intel`, which includes a historical dashboard, prediction demo, high-risk place ranking, group pattern analysis, and a NIBRS generalization results page. The app provides a working demonstration of model outputs with basic validation and user interaction while remaining faithful to the course instruction to prioritize functionality over aesthetics. Overall, the project demonstrates integrated technical work across data preparation, analytics, interpretation, deployment, and stakeholder communication.

---

## 1. Introduction and Business Context

### 1.1 Background

Predictive policing broadly refers to the use of analytical methods to identify likely targets for intervention or heightened attention based on historical data, recurring patterns, and environmental context [1], [2], [9], [11]. In practice, predictive policing systems may focus on places, offenders, groups, or victims, but the common technical foundation is the attempt to transform past observations into structured forecasts that support proactive decision-making. The IT5006 project brief specifically frames predictive policing as a practical analytics task involving historical crime data, model development, dashboarding, and proof-of-concept deployment [8].

At the same time, predictive policing is a controversial and high-stakes application area. Crime data are shaped by reporting processes, policing practices, social inequality, and institutional incentives. As a result, predictive models in this space must be interpreted with caution. A technically accurate result does not automatically imply fair or appropriate real-world deployment. This is particularly relevant when model outputs could influence patrol intensity, surveillance attention, or resource allocation in already over-policed communities [2], [5], [10].

Accordingly, our project adopts a deliberately modest and responsible framing. We do not present the system as a tool for automated police action. Instead, we position it as a **decision-support proof-of-concept** that helps explore temporal and spatial crime patterns, compare predictive benchmarks, and demonstrate how a machine learning model can be integrated into an interactive application.

### 1.2 Problem Statement

The final project problem can be stated as follows:

> Given historical crime records and spatiotemporal context, estimate the probability that at least one crime occurs in a district-hour cell.

This formulation reflects several design choices:

1. It focuses on **crime occurrence risk**, not exact offense details.
2. It uses a **district-hour** space-time unit rather than individual incident records.
3. It supports operational planning tasks better than post hoc event-level classification.
4. It naturally aligns with the temporal and spatial patterns identified during EDA.

The prediction task is therefore a binary classification problem:

\[
P(Y = 1 \mid X)
\]

where:

- \(Y = 1\) means at least one recorded crime occurs in a district-hour cell;
- \(Y = 0\) means no crime occurs in that district-hour cell.

### 1.3 Objectives

The project objectives are aligned with the course learning goals and final deliverable requirements [8]:

1. Explore the Chicago crime dataset to identify temporal, spatial, and category-level patterns.
2. Engineer a reproducible modeling pipeline that transforms raw data into district-hour inputs.
3. Train and compare up to four benchmark models using chronological evaluation.
4. Evaluate performance using both standard classification metrics and aggregate spatial-temporal measures.
5. Test generalization on external NIBRS data from other jurisdictions.
6. Deploy the resulting system as a proof-of-concept application with a live interface.
7. Translate technical results into responsible business and operational recommendations.

### 1.4 Stakeholders and Business Relevance

Although this is an academic project, the system is easiest to interpret when tied to realistic stakeholders:

- **Crime analysts** who need historical exploration and trend monitoring tools.
- **Shift planners or district supervisors** who may benefit from timing-sensitive awareness signals.
- **Operations managers** who need concise summaries of risk patterns rather than raw record tables.
- **Policy or research stakeholders** who need to see model strengths, limitations, and transfer behavior clearly documented.

In this context, the most realistic business value is not “crime prevention automation,” but improved **situational awareness**, better communication of risk patterns, and a clearer understanding of what machine learning can and cannot extract from available crime-reporting data.

---

## 2. Dataset and Preprocessing

### 2.1 Dataset Sources

Following the course requirement, our project uses two primary data sources [8]:

1. **Chicago Crime Dataset (2001–Present)**  
   Official dataset from the Chicago Police Department’s CLEAR system [12], [13].

2. **FBI NIBRS (National Incident-Based Reporting System)**  
   Used to test generalizability across jurisdictions and contexts [8], [14].

The course brief recommends using the last 10 years ending in 2024 for Chicago model development, using 2025 for holdout evaluation, and using NIBRS data specifically for robustness testing beyond the original city [8]. Our final workflow follows this structure.

### 2.2 Chicago Data Scope in the Final Project

The local repository contains yearly Chicago crime ZIP archives under:

```text
apps/dashboard/split_data_by_year/
```

The dashboard and EDA can browse yearly data from **2014 to 2025**, while the final modeling pipeline uses the following windows:

- **2015–2024** for model development and training
- **2025** for holdout validation and test

The final modeling table is stored as:

```text
data/raw/chicago_crime_district_hour_2015_2025_phase2.csv
```

### 2.3 Final Modeling Dataset Summary

The refactored district-hour dataset has the following properties:

- Total rows: **2,217,936**
- Positive rows: **1,409,469**
- Negative rows: **808,467**
- Overall positive rate: **0.6355**
- Distinct districts: **23**

After preprocessing and removal of historical-feature cold-start rows:

- Train rows: **2,015,280**
- Validation rows: **100,740**
- Test rows: **100,740**
- Final feature count: **37**

These statistics reflect the final full-year benchmark and replace older local-subset configurations that were based on sampled event rows.

### 2.4 NIBRS Data Scope

To test cross-jurisdiction generalization, we downloaded NIBRS data for:

- **Texas (2023–2024)**
- **Colorado (2023–2024)**

The 2023 data act as warm-up history for rolling features, while the 2024 data are used as the external evaluation year. We prepared NIBRS into county-hour modeling tables using project-specific scripts that aggregate incidents and align them with the existing district-hour logic as closely as possible. The resulting prepared files are:

```text
data/raw/nibrs_county_hour_tx_2023_2024.csv
data/raw/nibrs_county_hour_co_2023_2024.csv
```

### 2.5 Data Preparation Workflow

The final workflow can be summarized as:

1. Load yearly Chicago archives.
2. Normalize schema and parse timestamps.
3. Construct explicit district-hour cells.
4. Assign natural labels based on whether a crime occurred in that cell.
5. Apply shared preprocessing and feature engineering.
6. Split chronologically into train / validation / test windows.

This workflow is implemented across the following modules:

- `src/data/dataset_builder.py`
- `src/data/processor.py`
- `src/data/split_strategy.py`
- `src/scripts/prepare_phase2_data.py`

### 2.6 Missing Values and Cleaning Logic

Crime data contain multiple forms of incompleteness, including missing coordinates, missing or sparse district-level spatial fields, and inconsistent representations across years. Our preprocessing handles this by:

- coercing timestamps into a unified datetime type;
- clipping or validating spatial fields under the expected Chicago configuration;
- creating missingness indicators for relevant spatial features;
- imputing selected columns using training-set medians or modes;
- applying district-level fallback values where appropriate for spatial fields.

These steps are deliberately performed within the shared processor so that the same logic is reused during training, holdout inference, and app deployment.

### 2.7 Temporal Split Strategy

Because this is a time-dependent problem, random splitting would leak future information into training. We therefore use explicit year-based chronological splits:

- Training window: 2015–2024
- Holdout year: 2025
- Validation / test subsets: chronological partitions within the 2025 holdout

This choice directly addresses the course emphasis on rigorous evaluation and is more appropriate than random k-fold cross-validation for forecasting-style tasks.

---

## 3. Exploratory Analysis Highlights

Our exploratory data analysis served two functions: first, to understand the structure of the Chicago crime dataset; second, to motivate the subsequent feature engineering and model design.

### 3.1 Overall Trend

The yearly crime trend indicates that Chicago crime volume is not constant over time. One particularly visible pattern is the decline around 2020–2021, followed by a rebound in later years. While such variation may reflect multiple forces, including reporting conditions and pandemic-era disruptions, the key modeling implication is that temporal context matters. The crime process is not stationary in the trivial sense of showing identical volume every year.

### 3.2 Crime Type Distribution

The dataset is highly imbalanced across offense categories. Theft and battery dominate volume, while other offense classes appear less frequently. This matters for two reasons:

1. It supports the decision to define the final task as district-hour occurrence prediction rather than full multiclass offense prediction.
2. It reminds us that descriptive distributions can be driven by reporting, legal categorization, and enforcement practices, not only by underlying latent crime risk.

### 3.3 Temporal Patterns

The EDA reveals strong hourly and weekly structure:

- low activity in early morning hours,
- higher volume in late afternoon and evening,
- modest weekly differences, with some increase near the weekend.

These findings directly motivate the use of temporal features such as:

- hour,
- day of week,
- cyclical sine/cosine encodings,
- week-of-year or seasonal indicators.

### 3.4 Spatial Distribution

Crime is not evenly distributed across Chicago police districts. Some districts show consistently higher incident counts, suggesting the presence of stable spatial concentration and localized hotspots. This motivates inclusion of district identity and related spatial variables in the benchmark.

Importantly, spatial concentration in EDA does not guarantee that a model can perfectly rank districts operationally. It only shows that there is meaningful spatial structure to learn from.

### 3.5 Correlation and Arrest-Related Observations

Our exploratory analysis also examined arrest-related summaries by hour and by crime type. These analyses help characterize the dataset and may indicate enforcement-related patterns, but they are not treated as direct predictive inputs in the main benchmark. This is a crucial methodological decision, because arrest is a post-incident outcome and would create leakage if used naively in a forward-looking prediction task.

### 3.6 EDA-to-Modeling Bridge

The EDA stage supports the modeling stage in a concrete way:

- temporal rhythm motivates temporal features;
- spatial concentration motivates district-level spatial features;
- recurring crime counts motivate rolling-history windows;
- category imbalance reinforces the choice of binary occurrence prediction.

Thus, the EDA is not a disconnected descriptive exercise. It directly informs the downstream analytical design.

---

## 4. Problem Formulation and Feature Engineering

### 4.1 Final Problem Definition

The final benchmark treats the task as district-hour crime occurrence prediction. Each row in the modeling table represents a unique location-time unit rather than a unique incident.

This differs from the earlier local subset benchmark, which depended on event rows and approximated negatives synthetically. In the final refactored system, negatives arise naturally whenever a district-hour cell contains zero observed incidents. This is a major methodological improvement because it reduces label construction ambiguity and aligns more naturally with planning-oriented use cases.

### 4.2 Why District-Hour?

We chose district-hour as the modeling unit for several reasons:

1. It produces natural positive and negative labels.
2. It is directly aligned with time-sensitive and location-sensitive resource awareness.
3. It preserves operational interpretability.
4. It reduces dependence on copied-row synthetic negative construction.

The project therefore models **risk in a space-time cell**, not characteristics of an already observed incident.

### 4.3 Feature Groups

Our engineered features fall into three broad groups.

#### 4.3.1 Temporal Features

These features capture periodic and contextual time structure:

- hour
- day of week
- week of year
- month signals
- sine/cosine encodings such as `hour_sin` and `hour_cos`

Temporal features are especially important because EDA showed clear intraday and weekly rhythm.

#### 4.3.2 Spatial Features

These features capture district-level and coordinate-level variation:

- District
- Ward
- Community Area
- Beat
- Latitude / Longitude
- X / Y coordinates where available

Spatial missingness indicators are also created to preserve information about incomplete geocoding.

#### 4.3.3 Historical Features

These features summarize recent crime activity prior to the current district-hour cell:

- crimes in the last 7 days
- crimes in the last 14 days
- crimes in the last 30 days
- crimes in the last 90 days

These windows were chosen because they are operationally interpretable and correspond to weekly, biweekly, monthly, and quarterly recency patterns.

### 4.4 Leakage Controls

A central design priority in this project is leakage avoidance. We apply the following rules:

- use only features available before prediction time;
- avoid post-incident fields such as arrest outcome as direct predictive inputs;
- fit preprocessing statistics only on the training partition;
- maintain chronological separation between train and holdout periods.

This makes the benchmark more realistic and analytically defensible.

### 4.5 Cold-Start Filtering

Historical features require prior history. As a result, the processor removes cold-start rows that do not yet have sufficient lookback context. This is reported explicitly in logs and is part of the final train/validation/test row counts.

### 4.6 NIBRS Feature Alignment

For external evaluation, NIBRS data do not match Chicago one-to-one in schema or geography. To bridge this gap, we:

- aggregate incidents into **county-hour** units;
- use 2023 data as history warm-up;
- preserve compatible temporal logic;
- apply a passthrough spatial mode so external geographic identifiers are not forcibly clipped into Chicago-only bounds.

This allows a meaningful, though imperfect, cross-jurisdiction stress test.

---

## 5. Modeling Approaches

### 5.1 Why These Four Models?

We selected four models to create a structured and interpretable benchmark:

1. **Logistic Regression**  
   A linear baseline used to test whether the engineered features contain separable signal under a simple assumption.

2. **Decision Tree**  
   A nonlinear single-tree baseline used to examine whether basic nonlinear structure improves on the linear benchmark.

3. **Random Forest**  
   A bagged ensemble used to capture nonlinear interactions more robustly and reduce variance relative to a single tree.

4. **HistGradientBoosting**  
   A boosted ensemble that serves as a strong benchmark for structured tabular data.

This set spans increasing model complexity while remaining computationally manageable and explainable.

### 5.2 Decision Tree as a Deliberate Revision

One important project revision was the explicit inclusion of a Decision Tree. This strengthened the benchmark design by giving us a single-tree nonlinear point of comparison between Logistic Regression and ensemble methods. Without this model, the comparison would have jumped directly from linear methods to ensembles, making the story less structured.

### 5.3 Training Workflow

The training pipeline follows these steps:

1. Prepare the district-hour dataset from yearly archives.
2. Apply shared preprocessing using the `DataProcessor`.
3. Train the candidate models on the 2015–2024 window.
4. Use the validation partition within the 2025 holdout for selection and comparison.
5. Evaluate the final models on the 2025 test partition.
6. Save model artifacts, metrics, predictions, and summaries under `artifacts/`.

### 5.4 Hyperparameter Settings

The selected benchmark configurations were:

- **Logistic Regression:** `C = 0.5`, `max_iter = 200`
- **Decision Tree:** `max_depth = 10`, `min_samples_leaf = 50`, `ccp_alpha = 0.0`
- **Random Forest:** `n_estimators = 96`, `max_depth = 16`, `max_features = sqrt`, `max_samples = 0.12`, `min_samples_leaf = 10`
- **HistGradientBoosting:** `learning_rate = 0.08`, `max_depth = 8`, `max_iter = 120`, `min_samples_leaf = 200`, `l2_regularization = 0.0`

These settings balance model expressiveness and runtime while keeping the benchmark reproducible.

### 5.5 Why No Random Cross-Validation?

The course brief mentions cross-validation as part of evaluation rigor [8]. In our final setup, we intentionally prioritize **chronological holdout evaluation** over random k-fold cross-validation because this is a forecasting-style problem. Random cross-validation would violate temporal ordering and introduce leakage. Our validation strategy is therefore time-aware rather than fold-based.

---

## 6. Results and Evaluation

### 6.1 Standard Classification Metrics

Table 1 reports the main validation and test results on the Chicago 2025 holdout.

**Table 1. Holdout Classification Metrics**

| Model | Val AUROC | Test AUROC | Val AUPRC | Test AUPRC | Test Accuracy | Test Precision | Test Recall | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7186 | 0.7177 | 0.7792 | 0.7867 | 0.6649 | 0.7590 | 0.6874 | 0.7214 |
| Decision Tree | 0.7362 | 0.7334 | 0.7950 | 0.8003 | 0.6835 | 0.7590 | 0.7306 | 0.7445 |
| Random Forest | 0.7402 | 0.7370 | 0.8001 | 0.8056 | 0.6891 | 0.7575 | 0.7463 | 0.7519 |
| HistGradientBoosting | **0.7412** | **0.7381** | **0.8011** | **0.8063** | **0.7097** | 0.7244 | **0.8717** | **0.7913** |

### 6.2 Benchmark Interpretation

The main benchmark findings are:

1. HistGradientBoosting performs best overall on AUROC and AUPRC.
2. Random Forest remains very close, so the margin is not dramatic.
3. Decision Tree substantially improves over Logistic Regression, confirming that nonlinear structure matters.

Our interpretation is intentionally cautious: HistGradientBoosting is the strongest benchmark, but the result should not be oversold as a dramatic leap over all alternatives.

### 6.3 Spatial and Temporal Evaluation

Following the project requirement to go beyond standard classification metrics [8], we also evaluated aggregate spatial and temporal alignment.

**Table 2. Spatial-Temporal Evaluation on Chicago Holdout**

| Model | Split | District Corr. | Top-5 Overlap | Top-10 Overlap | Hourly Corr. | Day-of-Week Corr. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | Test | 0.9942 | 1.0 | 1.0 | 0.8542 | 0.7218 |
| Decision Tree | Test | 0.9929 | 0.8 | 1.0 | 0.9882 | 0.4949 |
| Random Forest | Test | 0.9936 | 0.8 | 1.0 | **0.9911** | 0.5932 |
| HistGradientBoosting | Test | **0.9996** | **1.0** | **1.0** | 0.9900 | **0.7311** |

Several observations follow:

- Temporal alignment is extremely strong for the tree-based models.
- District-level aggregate agreement is also strong in the final district-hour benchmark.
- The refactored benchmark is materially more learnable than the earlier event-row subset.

This is an important update relative to earlier project stages: once the task is reformulated around district-hour cells, both temporal and district-level aggregate behavior become much more stable and interpretable.

### 6.4 External NIBRS Generalization

To satisfy the course requirement on generalization testing [8], we evaluated the trained Chicago benchmark on prepared NIBRS county-hour datasets for Texas and Colorado.

**Table 3. External NIBRS Generalization Results (2024)**

| Dataset | Rows | Positive Rate | AUROC | AUPRC | Accuracy | Precision | Recall | F1 | County Corr. | Hourly Corr. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Colorado 2024 | 561,744 | 0.2041 | 0.8203 | 0.6243 | 0.7703 | 0.4493 | 0.5567 | 0.4973 | 0.5237 | 0.9728 |
| Texas 2024 | 2,368,296 | 0.1804 | 0.8229 | 0.4327 | 0.7949 | 0.2558 | 0.0716 | 0.1119 | 0.0801 | 0.9209 |

These results support three conclusions:

1. The model retains meaningful discrimination outside Chicago.
2. Temporal transfer is much stronger than exact spatial ranking transfer.
3. External robustness is real, but uneven across jurisdictions.

The Texas result is particularly informative. Although AUROC remains strong, county-level correlation is much weaker than hourly correlation. This suggests that the model generalizes more reliably as a **timing-aware signal** than as a universal geographic ranking engine.

### 6.5 Robustness Discussion

From an evaluation perspective, robustness in this project is demonstrated in three ways:

1. Holdout performance on unseen 2025 Chicago data
2. Aggregate spatial-temporal consistency within the holdout year
3. External transfer to Texas and Colorado NIBRS datasets

This is a stronger evaluation story than relying on one scalar metric alone.

---

## 7. Model Interpretation and Business Insights

### 7.1 Feature Importance

Permutation-importance results for HistGradientBoosting show that the strongest features are:

- `hour`
- `hour_sin`
- `District`
- `crimes_last_30d`
- `crimes_last_14d`
- `day_of_week`

Grouped importance shares are:

- Temporal: **65.9%**
- Historical: **20.1%**
- Spatial: **14.0%**

### 7.2 Interpretation

The interpretation is straightforward:

- **Timing is the dominant signal.**  
  Crime occurrence risk has clear intraday and weekly rhythm.

- **Recent history adds meaningful context.**  
  Rolling counts in the last 7–90 days improve prediction beyond purely periodic signals.

- **Spatial identity still matters.**  
  District-level structure contributes to performance, especially within Chicago.

These findings are descriptive, not causal. Feature importance tells us what helps prediction; it does not prove why crime happens.

### 7.3 Business Insights

For a planning-support stakeholder, the business value lies in:

1. **Timing-aware awareness**  
   The system can highlight hours and days where risk tends to rise.

2. **District-level contextualization**  
   Historical district identity and recent local activity provide interpretable context.

3. **Workload prioritization**  
   Analysts can use ranked probabilities or daily outlook pages to focus review effort.

4. **Communication support**  
   Dashboard and app pages provide visuals and summaries that are more accessible than raw model output tables.

### 7.4 Law-Enforcement Framing

In line with the course brief, the system can generate actionable insights for relevant stakeholders [8], but those insights should remain modest and human-in-the-loop. Appropriate report wording is:

- planning aid
- timing-sensitive risk awareness
- exploratory and analytical support

Inappropriate wording would imply:

- automated patrol deployment
- automated intervention
- deterministic forecasts of specific incidents

---

## 8. Model Deployment and Application

### 8.1 Deployment Overview

The final project includes a deployed proof-of-concept application built with **Streamlit Cloud**, which is also recommended in the course brief for quick dashboards and MVP-style deployment [8].

**Live URL:**  
<https://team22it5006predictivepolicingay2526sem2-5xjmr8cwbappeurrspssw.streamlit.app/>

### 8.2 Why Streamlit Cloud?

We selected Streamlit Cloud because it satisfies the course deployment requirements efficiently:

- live accessible URL,
- lightweight interactive UI,
- straightforward integration with Python analytics code,
- fast demonstration of model outputs without heavy backend engineering overhead.

This choice is aligned with the course note that a simple, functional proof-of-concept is preferred over a more complex but fragile system [8].

### 8.3 Application Modules

The deployed application, `Chicago Crime Intel`, includes five main modules:

1. **Dashboard**  
   Historical exploration of crime patterns by year, district, type, and arrest status.

2. **Prediction Demo**  
   Single-instance prediction for a chosen district, date, and hour.

3. **High-Risk Places**  
   Ranking of districts for a selected time point.

4. **Group Pattern Analysis**  
   Aggregate daily risk patterns across district-hour blocks.

5. **NIBRS Generalization**  
   Visualization of external transfer results from Texas and Colorado.

### 8.4 Core Prediction Workflow

The core prediction flow is:

1. user selects district, date, and hour;
2. system reconstructs the corresponding feature vector using the shared processor;
3. saved HistGradientBoosting model produces a probability;
4. probability is mapped to a risk band;
5. recent district-level history counts are displayed as additional explanation.

### 8.5 Input Validation and Error Handling

The application includes basic POC-level safeguards:

- bounded date range selection,
- district and spatial field range validation,
- prevention of impossible values,
- handling of cold-start or insufficient-history cases.

These checks satisfy the course expectation for basic validation and error handling while remaining lightweight.

### 8.6 Usage Guide

The application can be demonstrated in the following sequence:

1. open the home page,
2. browse historical context in the dashboard,
3. run a single district-hour prediction,
4. inspect ranked high-risk places,
5. review external NIBRS generalization results.

For the final PDF, screenshots should ideally be inserted for:

- the landing page,
- dashboard page,
- prediction demo result page,
- NIBRS generalization page.

### 8.7 Repository Integration

The deployment is backed by a documented GitHub repository containing:

- project code,
- model scripts,
- documentation,
- dashboard files,
- setup instructions in `README.md`.

This helps satisfy the final requirement that the report, deployment, and repository function as an integrated deliverable.

---

## 9. Business Recommendations and Expected Impact

### 9.1 Recommendations

Based on our results, we recommend the following cautious use cases:

1. **Use the model for timing-aware planning support**  
   The benchmark is especially strong at learning when risk tends to rise.

2. **Use district-level predictions as review signals, not deterministic decisions**  
   Even with strong within-Chicago aggregate performance, outputs should be reviewed alongside local context.

3. **Use the dashboard as a communication and monitoring interface**  
   It supports historical interpretation and stakeholder discussion.

4. **Use external NIBRS evaluation to calibrate expectations about transfer**  
   External generalization is meaningful, but spatial portability is weaker than temporal portability.

### 9.2 Expected Impact

If used appropriately, the system can improve:

- awareness of recurring time-based crime patterns,
- prioritization of analyst attention,
- communication of risk patterns to stakeholders,
- reproducibility of analytics work within a small team workflow.

### 9.3 What the System Should Not Be Used For

The system should not be presented as:

- an automatic patrol assignment engine,
- a fairness-validated operational system,
- a causal explanation of crime drivers,
- a guaranteed hotspot-ranking solution across jurisdictions.

This boundary is essential for both technical honesty and ethical responsibility.

---

## 10. Limitations and Ethical Considerations

### 10.1 Data Limitations

Crime data are administrative data, not perfect reflections of underlying crime prevalence. They are influenced by:

- reporting behavior,
- enforcement intensity,
- data entry practices,
- jurisdictional differences.

This means model outputs reflect patterns in recorded crime, not necessarily ground-truth victimization or harm.

### 10.2 Feature Limitations

The available features are limited mainly to:

- time,
- location,
- offense-related fields.

The course brief explicitly warns against evaluation claims requiring unavailable fields such as response times or patrol allocation [8]. Our report therefore avoids such claims.

### 10.3 Transfer Limitations

NIBRS external evaluation is informative but imperfect:

- geography is aligned only approximately through county-hour units;
- schema differs from Chicago;
- spatial transfer is weaker than temporal transfer.

Thus, external robustness should be interpreted as partial rather than universal.

### 10.4 Ethical and Policy Concerns

Predictive policing raises fairness, surveillance, and accountability concerns [2], [5], [10]. Our project does not perform a full fairness audit, calibration-by-group analysis, or policy validation study. Therefore, it should not be treated as deployment-ready in real operational settings.

### 10.5 Technical Scope Limitation

The final app is a proof-of-concept. It demonstrates:

- model inference,
- structured visualization,
- basic validation,
- external evaluation summaries.

It does not include:

- advanced authentication,
- secure role-based access,
- audit logging,
- production MLOps infrastructure,
- ongoing model retraining pipelines.

---

## 11. Conclusion and Future Work

This project demonstrates an end-to-end predictive policing proof-of-concept that integrates historical analysis, machine learning benchmarking, external generalization testing, and interactive deployment. The most important technical advance in the final system is the shift to a **district-hour** modeling formulation. This change produces natural labels, aligns more cleanly with planning-oriented use cases, and yields a stronger and more interpretable benchmark than earlier event-level approximations.

Across four benchmark models, HistGradientBoosting emerged as the strongest overall performer, though its margin over Random Forest remained small. Feature importance analysis shows that temporal structure is the dominant predictive signal, followed by recent historical activity and then spatial context. External NIBRS evaluation further shows that the learned signal is not confined to Chicago alone, though transfer is much stronger temporally than spatially.

The deployed Streamlit application extends the work beyond offline analytics by packaging dashboarding, prediction, hotspot ranking, aggregate pattern analysis, and external generalization into one accessible interface. This satisfies the course requirement for practical application while also showcasing the full workload of the project: exploratory analysis, data engineering, model comparison, interpretation, deployment, and communication.

Future work could extend this project in several directions:

1. add richer contextual variables such as weather, land use, mobility, or socioeconomic covariates;
2. perform fairness and subgroup calibration analysis;
3. improve cross-jurisdiction alignment beyond county-hour aggregation;
4. evaluate calibration and uncertainty, not only discrimination;
5. explore multi-task or hierarchical prediction settings.

In summary, the project’s strongest contribution is not a claim of perfect prediction, but a technically grounded and responsibly framed demonstration of how crime data can be turned into a transparent, reproducible, and interactively deployable analytics workflow.

---

## References

[1] Brantingham, P. J., Valasik, M., & Mohler, G. O. (2018). Does predictive policing lead to biased arrests? Results from a randomized controlled trial. *Statistics and Public Policy*, 5(1), 1-6.

[2] Ferguson, A. G. (2017). *The Rise of Big Data Policing: Surveillance, Race, and the Future of Law Enforcement*. New York University Press.

[3] Mohler, G. O., Short, M. B., Malinowski, S., Johnson, M., Tita, G. E., Bertozzi, A. L., & Brantingham, P. J. (2015). Randomized controlled field trials of predictive policing. *Journal of the American Statistical Association*, 110(512), 1399-1411.

[4] Caplan, J. M., Kennedy, L. W., & Miller, J. (2011). Risk terrain modeling: Brokering criminological theory and GIS methods for crime forecasting. *Justice Quarterly*, 28(2), 360-381.

[5] Brayne, S. (2020). *Predict and Surveil: Data, Discretion, and the Future of Policing*. Oxford University Press.

[6] Perry, W. L. (2013). *Predictive Policing: The Role of Crime Forecasting in Law Enforcement Operations*. RAND Corporation.

[7] Richardson, R. (2021). Racial segregation and the data-driven society: How our failure to reckon with root causes perpetuates separate and unequal realities. *Berkeley Technology Law Journal*, 36, 1051.

[8] Sukhwal, P. (2026). *IT5006 Project Description - AY 2025/26 Semester 2*. National University of Singapore. Available at: <https://prakashsukhwal.github.io/IT5006/IT5006_Project_Description_2026Jan_V2.html>

[9] Chicago Police Department. *Crimes - 2001 to Present*. City of Chicago Open Data. Available at: <https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2>

[10] Chicago Police Department. *Crimes - 2001 to Present: About Data*. City of Chicago Open Data. Available at: <https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data>

[11] FBI Crime Data Explorer. *NIBRS Data Downloads*. Available at: <https://cde.ucr.cjis.gov/LATEST/webapp/#/pages/downloads>

[12] Streamlit. *Streamlit Documentation*. Available at: <https://streamlit.io/>

---

## Appendix A. Reproducibility and Key Project Assets

### A.1 Repository Structure

Key directories used in the final project:

```text
data/
apps/
src/
docs/
artifacts/
slides/
```

### A.2 Main Scripts

- `src/scripts/prepare_phase2_data.py`
- `src/scripts/train.py`
- `src/scripts/evaluate_phase2.py`
- `src/scripts/feature_importance.py`
- `src/scripts/prepare_nibrs_generalization_data.py`
- `src/scripts/evaluate_nibrs_generalization.py`

### A.3 Main Application File

- `apps/dashboard/app.py`

### A.4 Key Artifacts

- `artifacts/metrics/phase2_model_metrics.csv`
- `artifacts/metrics/phase2_spatiotemporal_metrics.csv`
- `artifacts/metrics/feature_importance/feature_importance_summary.json`
- `artifacts/metrics/nibrs_generalization/nibrs_generalization_metrics_2024.csv`
- `artifacts/models/hist_gradient_boosting.pkl`
