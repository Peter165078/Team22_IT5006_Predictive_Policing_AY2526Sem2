from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

DATE_COL = "Date"


@dataclass
class SplitSummary:
    split_strategy: str
    train_start_year: int
    train_end_year: int
    holdout_year: int
    holdout_val_fraction: float
    train_rows_before_hist_drop: int
    val_rows_before_hist_drop: int
    test_rows_before_hist_drop: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_year_holdout_splits(
    raw_data: pd.DataFrame,
    *,
    train_start_year: int,
    train_end_year: int,
    holdout_year: int,
    holdout_val_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, SplitSummary]:
    if not 0 < holdout_val_fraction < 1:
        raise ValueError("holdout_val_fraction must be strictly between 0 and 1.")

    dates = pd.to_datetime(raw_data[DATE_COL], errors="coerce")
    years = dates.dt.year

    train_mask = (years >= train_start_year) & (years <= train_end_year)
    holdout_mask = years == holdout_year

    train_idx = np.flatnonzero(train_mask.to_numpy())
    holdout_idx = np.flatnonzero(holdout_mask.to_numpy())

    if len(train_idx) == 0:
        raise ValueError(
            f"No rows found between training years {train_start_year}-{train_end_year}."
        )
    if len(holdout_idx) < 2:
        raise ValueError(
            f"Not enough rows found for holdout year {holdout_year}. "
            "At least two rows are required to create validation and test splits."
        )

    split_point = int(len(holdout_idx) * holdout_val_fraction)
    split_point = max(1, min(len(holdout_idx) - 1, split_point))

    val_idx = holdout_idx[:split_point]
    test_idx = holdout_idx[split_point:]

    summary = SplitSummary(
        split_strategy="year_holdout",
        train_start_year=train_start_year,
        train_end_year=train_end_year,
        holdout_year=holdout_year,
        holdout_val_fraction=holdout_val_fraction,
        train_rows_before_hist_drop=int(len(train_idx)),
        val_rows_before_hist_drop=int(len(val_idx)),
        test_rows_before_hist_drop=int(len(test_idx)),
    )
    return train_idx, val_idx, test_idx, summary
