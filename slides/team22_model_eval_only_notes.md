# Team 22 Model & Evaluation Only Notes

This note matches `slides/team22_model_eval_only.html`.

If you are only responsible for model building and evaluation, this 5-slide deck is enough.

## Suggested pacing

1. Slide 1: `20-30s`
2. Slide 2: `35-45s`
3. Slide 3: `45-55s`
4. Slide 4: `40-50s`
5. Slide 5: `40-50s`

Total: about `3 to 4 minutes`

## Slide 1 — From Raw Records to Model Inputs

- Bridge from EDA into modeling.
- Say clearly that raw records were not fed directly into the models.
- Mention the three feature groups:
  - temporal
  - spatial
  - historical
- This slide exists to complete the logic from EDA to benchmarking.

## Slide 2 — Why These Four Models?

- Say the comparison is structured, not random.
- Logistic Regression is the linear baseline.
- Decision Tree is the nonlinear single-tree baseline.
- Random Forest shows what bagging adds.
- HistGradientBoosting is the strong tabular ensemble benchmark.
- Mention that Decision Tree was added after feedback.

## Slide 3 — Benchmark Results

- State the main result clearly:
  - HistGradientBoosting is best on both AUROC and AUPRC.
- Immediately add the nuance:
  - the margin over Random Forest is small.
- This helps you sound honest and technically mature.

## Slide 4 — What Drives Predictions?

- Use the two figures together:
  - feature importance shows what the model uses
  - hourly trend explains why temporal features are dominant
- Main message:
  - timing is the dominant signal
  - recent local history helps
  - spatial context still contributes

## Slide 5 — NIBRS Generalization

- This is your strongest evaluation slide beyond internal Chicago metrics.
- Say:
  - we trained on Chicago
  - then tested on Texas and Colorado NIBRS 2024
- Main takeaway:
  - discrimination remains meaningful externally
  - temporal transfer is stronger than spatial transfer

## Short closing sentence

“Overall, our modeling results show a modest but consistent benchmark signal, and the NIBRS transfer test suggests the model learns timing-related structure that generalizes beyond Chicago.”
