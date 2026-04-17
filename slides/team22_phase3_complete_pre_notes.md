# Team 22 Complete Pre Notes

This note matches `slides/team22_phase3_complete_pre.html`.

Recommended total:
- `10 minutes` presentation
- `5 minutes` Q&A

## Slide 1 — Title and Agenda

- Time: `45 seconds`
- Open with one clean sentence: this is an end-to-end predictive policing proof-of-concept.
- Say the three outcomes clearly:
  - reproducible pipeline
  - benchmark comparison
  - deployed demo with NIBRS generalization

## Slide 2 — Final Task Definition

- Time: `55 seconds`
- Define the target in plain language: predict whether at least one crime happens in a district-hour cell.
- Stress why district-hour is better than event rows:
  - natural negatives
  - more interpretable
  - easier to defend methodologically

## Slide 3 — Data Strategy

- Time: `60 seconds`
- State the split policy very explicitly:
  - Chicago `2015-2024` for development
  - Chicago `2025` as holdout
  - NIBRS `2023-2024` for external transfer, with `2023` as warm-up and `2024` as test
- Mention Texas and Colorado only once here, then save metric details for the transfer slide.

## Slide 4 — Pipeline

- Time: `65 seconds`
- Walk left to right through the five boxes.
- Emphasize that the same code path powers training, evaluation, and the demo.
- Mention leakage control before the audience asks:
  - chronological split
  - pre-time-T rolling windows only
  - no post-incident leakage

## Slide 5 — EDA

- Time: `55 seconds`
- Use the visuals to ground the story:
  - hourly rhythm exists
  - incidents cluster spatially
  - this motivates temporal + spatial + history features
- Do not over-explain every chart.

## Slide 6 — Benchmark Results

- Time: `65 seconds`
- Say HistGradientBoosting is the strongest overall benchmark.
- Immediately add the nuance that the gap over Random Forest is small.
- This helps you sound honest and confident.

## Slide 7 — Interpretation

- Time: `70 seconds`
- Use this slide to explain what the benchmark is actually learning.
- Main line:
  - temporal features dominate
  - historical windows help
  - district identity still matters
- Important update to say accurately:
  - within Chicago, aggregate district alignment is now strong
  - cross-jurisdiction transfer is where spatial weakness becomes clearer

## Slide 8 — NIBRS Generalization

- Time: `70 seconds`
- This is the slide that answers the course requirement on generalizability.
- Core message:
  - the Chicago-trained model still transfers meaningfully on external NIBRS data
  - hourly alignment remains stronger than county-level alignment
  - Texas is harder spatially than Colorado
- Keep the conclusion careful: robust timing signal, weaker spatial portability

## Slide 9 — Demo Plan

- Time: `55 seconds`
- Use this as the bridge into live demo or, if time is tight, as a stand-in for the demo.
- Recommended click order:
  - welcome
  - dashboard
  - prediction demo
  - NIBRS generalization page

## Slide 10 — Closing and Q&A

- Time: `45 seconds`
- End with three takeaways:
  - reproducible benchmark
  - modest but real predictive signal
  - external transfer plus deployed proof-of-concept
- Finish with the responsible-use sentence:
  - this is a decision-support proof-of-concept, not a production policing system

## Likely Q&A

### Why district-hour instead of event rows?

Because the district-hour setup gives natural positive and negative labels and is easier to interpret operationally.

### Why chronological split instead of random split?

Because this is a time-dependent prediction task. Random split would leak future behavior into training.

### Why use 7, 14, 30, and 90-day windows?

They capture weekly, biweekly, monthly, and quarterly recency effects while remaining easy to explain.

### Why is HistGradientBoosting the deployed benchmark?

Because it is strongest on both AUROC and AUPRC, even though the margin over Random Forest is small.

### What does NIBRS generalization prove?

It shows that the model learns structure that survives outside Chicago, especially temporal risk patterns. It does not prove universal spatial hotspot ranking.

### Can this be used directly in the real world?

Not yet. It would still need fairness review, policy validation, threshold calibration, and governance safeguards.
