# Team 22 Phase 3 Talk Track

This note is a presenter guide for the 10-minute English presentation in `slides/team22_phase3_presentation.html`.

## Suggested pacing

1. Slide 1, 45 seconds
   - Introduce the project as an end-to-end predictive-policing decision-support system.
   - State that the work combines EDA, dashboard development, and reproducible machine-learning benchmarks.

2. Slide 2, 50 seconds
   - Explain the motivation and stress that the project is for decision support, not automated policing.
   - Define the prediction target in plain language.

3. Slide 3, 60 seconds
   - Summarize the data scope.
   - Explain why the project used 20,000 sampled rows per year.
   - Mention negative-label construction and rolling windows briefly.

4. Slide 4, 60 seconds
   - Walk through the four models.
   - Explain that Decision Tree was added after milestone feedback to make the comparison more defensible.

5. Slide 5, 75 seconds
   - Focus on the record-level results table.
   - Say HistGradientBoosting is best, but the performance gap is modest.
   - Emphasize that the team intentionally avoids overselling the result.

6. Slide 6, 80 seconds
   - This is one of the most important slides.
   - Explain the difference between strong hourly alignment and weak district-level hotspot ranking.
   - Use this to show honest interpretation of the model.

7. Slide 7, 55 seconds
   - Summarize feature-importance findings.
   - State that temporal structure is dominant, while spatial and historical features still contribute.

8. Slide 8, 65 seconds
   - Show what users actually get: the dashboard.
   - Mention map filtering, temporal trend charts, and the live Streamlit app.

9. Slide 9, 55 seconds
   - Present limitations clearly.
   - Make the responsible-use framing explicit.

10. Slide 10, 15 seconds
   - Invite questions.

## Likely Q&A points

- Why did you construct negative labels?
  - Because the source data contains observed crimes only, so supervised binary learning needs explicit negative examples.
  - We treated the construction as a pragmatic approximation and wrote that limitation clearly.

- Why is HistGradientBoosting called the best model if spatial performance is weak?
  - It is best only on record-level discrimination metrics.
  - We do not claim it is best on every operational dimension.

- Why did you use those historical windows?
  - They represent weekly, biweekly, monthly, and quarter-scale recency effects.

- Can this be deployed in the real world?
  - Not directly.
  - It needs policy review, fairness review, threshold calibration, and domain validation.
