"""
src/data/processor.py

DataProcessor for Chicago Crime Prediction
===========================================

Problem
-------
Binary classification: given a (location, time_window) instance, predict
whether a crime will occur (target = 1) or not (target = 0).

Key design decisions
--------------------
1. Negative sample construction
   Raw data contains only reported crimes (all positive).  Negatives are
   synthesised by sampling (district, timestamp) combinations that do NOT
   appear in the positive set.

2. Chronological split  — no data leakage
   Records sorted by date; split 70 / 15 / 15 chronologically.

3. Historical features computed for ALL splits (train, val, test)
   For every sample at time T we count crimes strictly before T in the same
   district / grid-cell.  This is NOT leakage: we only use past events.
   The cumulative-sum table is built from the FULL dataset (all positives
   before T), which is available at inference time in production.
   Rows where ANY historical window has no prior data are DROPPED
   (typically the first ~90 days of the dataset).

4. Fit-on-train-only
   All statistics (medians, modes, scaler, vocab) are learned from train
   only and stored in fit_dict; applied identically to val / test.

5. Output is guaranteed all-numeric
   Any remaining object columns (e.g. grid_cell helper string) are dropped
   before returning, so X can be directly cast to torch.float32.

Pipeline order
--------------
    load_and_split()
        → fit_transform_train()  → (X_train, y_train)
        → transform(val_idx)     → (X_val,   y_val)
        → transform(test_idx)    → (X_test,  y_test)
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TARGET_COL = "target"
DATE_COL   = "Date"

_DROP_COLS = [
    "ID", "Case Number", "Block", "FBI Code",
    "Updated On", "Location", "Year",
    "Primary Type", "Description", "Location Description", "IUCR",
    "Arrest", "Domestic",
]

RARE_THRESH       = 0.005   # categories with freq < this → "OTHER"
MISSING_IND_THRESH = 0.05   # columns with > this fraction missing → add indicator

NEG_RATIO    = 1.0
GRID_CELL_DEG = 0.005       # ≈500 m at Chicago latitude

# Candidate historical windows (days).  At runtime the processor automatically
# drops any window longer than the dataset's actual time span, so this works
# correctly for both 38-day slices and multi-year full datasets.
HIST_WINDOW_CANDIDATES = {"7d": 7, "14d": 14, "30d": 30, "90d": 90}

# Minimum window required to keep a row.  Rows where even the shortest
# active window has no prior data are dropped (cold-start rows).
HIST_MIN_WINDOW_KEY = "7d"    # must be a key in HIST_WINDOW_CANDIDATES

TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────────────
# Stateless helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cyclic_encode(series: pd.Series, period: float) -> Tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def _build_grid_cell(lat: pd.Series, lon: pd.Series,
                     cell: float = GRID_CELL_DEG) -> pd.Series:
    lat_bin = (lat / cell).astype(int)
    lon_bin = (lon / cell).astype(int)
    return lat_bin.astype(str) + "_" + lon_bin.astype(str)


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor
# ──────────────────────────────────────────────────────────────────────────────

class DataProcessor:
    """
    End-to-end preprocessing pipeline for Chicago crime prediction.

    Usage
    -----
    >>> proc = DataProcessor("data/raw/Crimes_....csv")
    >>> proc.load_and_split()
    >>> X_train, y_train = proc.fit_transform_train()
    >>> X_val,   y_val   = proc.transform(proc.val_idx)
    >>> X_test,  y_test  = proc.transform(proc.test_idx)
    """

    def __init__(
        self,
        data_path:    str,
        neg_ratio:    float = NEG_RATIO,
        rare_thresh:  float = RARE_THRESH,
        random_state: int   = RANDOM_STATE,
    ) -> None:
        self.data_path    = data_path
        self.neg_ratio    = neg_ratio
        self.rare_thresh  = rare_thresh
        self.random_state = random_state

        self.raw_data:    Optional[pd.DataFrame] = None
        self.labels:      Optional[pd.Series]    = None
        self.train_idx:   Optional[np.ndarray]   = None
        self.val_idx:     Optional[np.ndarray]   = None
        self.test_idx:    Optional[np.ndarray]   = None
        self.fit_dict:    Dict                   = {}
        self.hist_windows: Dict[str, int]        = {}   # set in load_and_split

        # Full-dataset daily aggregation tables (built once, used by all splits)
        self._full_dist_daily: Optional[pd.DataFrame] = None
        self._full_grid_daily: Optional[pd.DataFrame] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Step 0: Load, build negatives, split
    # ──────────────────────────────────────────────────────────────────────────

    def load_and_split(self) -> None:
        print("  Loading raw data …")
        df = pd.read_csv(self.data_path, low_memory=False)

        df[DATE_COL] = pd.to_datetime(
            df[DATE_COL], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
        )
        df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)
        print(f"  Positives: {len(df):,}  "
              f"({df[DATE_COL].min().date()} → {df[DATE_COL].max().date()})")

        df = self._build_negatives(df)
        df = df.sort_values(DATE_COL).reset_index(drop=True)

        # ── Build full-dataset daily aggregation tables ───────────────────────
        # These are used by ALL splits for historical feature lookup.
        # Only positive rows (real crimes) contribute to the history counts;
        # negatives are synthetic and should not inflate the crime counts.
        pos_mask   = df[TARGET_COL] == 1
        dates_full = pd.to_datetime(df.loc[pos_mask, DATE_COL])
        dist_full  = df.loc[pos_mask, "District"].fillna(-1).astype(int).astype(str)
        lat_full   = df.loc[pos_mask, "Latitude"].fillna(0)
        lon_full   = df.loc[pos_mask, "Longitude"].fillna(0)
        grid_full  = _build_grid_cell(lat_full, lon_full)

        self._full_dist_daily = (
            pd.DataFrame({"_date": dates_full.dt.normalize(), "_district": dist_full})
            .groupby(["_district", "_date"])
            .size()
            .reset_index(name="count")
        )
        self._full_grid_daily = (
            pd.DataFrame({"_date": dates_full.dt.normalize(), "_grid": grid_full})
            .groupby(["_grid", "_date"])
            .size()
            .reset_index(name="count")
        )

        n    = len(df)
        n_tr = int(n * TRAIN_FRAC)
        n_va = int(n * VAL_FRAC)

        self.train_idx = np.arange(0, n_tr)
        self.val_idx   = np.arange(n_tr, n_tr + n_va)
        self.test_idx  = np.arange(n_tr + n_va, n)

        self.labels   = df[TARGET_COL].copy()
        self.raw_data = df.drop(columns=[TARGET_COL]).copy()

        # ── Adaptive historical windows ───────────────────────────────────────
        # Only keep windows shorter than the total time span of the dataset.
        # For a 38-day dataset, 90d and 30d windows would cause ALL rows to be
        # dropped (no prior history exists).  We keep only windows <= span,
        # capped so at least a few days of history are possible.
        date_col_vals   = pd.to_datetime(self.raw_data[DATE_COL])
        total_days      = (date_col_vals.max() - date_col_vals.min()).days
        usable_days     = max(total_days - 1, 1)   # need at least 1 day before T

        self.hist_windows = {
            k: v for k, v in HIST_WINDOW_CANDIDATES.items()
            if v <= usable_days
        }
        # Always include the minimum window (7d) even if data is very short;
        # rows without ANY prior history will still be dropped.
        if not self.hist_windows:
            min_key = min(HIST_WINDOW_CANDIDATES, key=HIST_WINDOW_CANDIDATES.get)
            self.hist_windows = {min_key: HIST_WINDOW_CANDIDATES[min_key]}

        print(f"  Total  : {n:,}  "
              f"train {len(self.train_idx):,} | "
              f"val {len(self.val_idx):,} | "
              f"test {len(self.test_idx):,}")
        print(f"  Date span     : {total_days} days")
        print(f"  Active windows: {list(self.hist_windows.keys())}")
        print(f"  Pos rate: {self.labels.mean():.3f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Negative sample construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_negatives(self, pos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct synthetic negative samples.

        For small datasets (< 500k rows) the (district, hour_window) collision
        space is much denser, so we use a larger attempt budget and relax the
        collision key to (district, DAY) instead of (district, HOUR).
        This still avoids fabricating negatives on days/districts with real
        crime events while dramatically reducing rejection rate.

        If the budget is exhausted before reaching n_neg, a warning is printed
        and the available negatives are used (slightly imbalanced but workable).
        """
        rng   = np.random.default_rng(self.random_state)
        n_neg = int(len(pos_df) * self.neg_ratio)
        print(f"  Building {n_neg:,} negative samples …")

        pos_df        = pos_df.copy()
        pos_df["_hw"] = pos_df[DATE_COL].dt.floor("h")
        pos_df["_day"]= pos_df[DATE_COL].dt.normalize()          # day-level key
        pos_df["_di"] = pos_df["District"].fillna(-1).astype(int).astype(str)

        # Collision set at DAY granularity (less dense, more negatives pass)
        crime_keys = set(
            pos_df["_di"].astype(str) + "_" + pos_df["_day"].astype(str)
        )

        districts  = pos_df["_di"].values
        timestamps = pos_df["_day"].values          # sample day-level timestamps
        n_pos      = len(pos_df)
        negatives: List[Dict] = []
        district_to_indices = {
            district: idx.to_numpy(copy=False)
            for district, idx in pos_df.groupby("_di").groups.items()
        }

        # For small datasets use a much larger attempt multiplier
        max_attempts = max(n_neg * 100, 500_000)
        attempts     = 0

        while len(negatives) < n_neg and attempts < max_attempts:
            batch = min(n_neg * 5, max_attempts - attempts)
            attempts += batch

            idx_d   = rng.integers(0, n_pos, size=batch)
            idx_t   = rng.integers(0, n_pos, size=batch)
            s_di    = districts[idx_d]
            s_days  = pd.to_datetime(timestamps[idx_t])

            # Add random hour + minute so samples span all times of day
            rand_hours   = rng.integers(0, 24, size=batch)
            rand_minutes = rng.integers(0, 60, size=batch)
            s_dt = s_days + pd.to_timedelta(rand_hours, unit="h") \
                          + pd.to_timedelta(rand_minutes, unit="m")

            for dist, dt in zip(s_di, s_dt):
                if len(negatives) >= n_neg:
                    break
                day_key = str(dist) + "_" + str(pd.Timestamp(dt).normalize())
                if day_key not in crime_keys:
                    template_idx = district_to_indices.get(dist)
                    if template_idx is None or len(template_idx) == 0:
                        template_row = pos_df.iloc[rng.integers(0, n_pos)]
                    else:
                        template_row = pos_df.iloc[rng.choice(template_idx)]
                    negatives.append({
                        DATE_COL:               dt,
                        "District":             template_row.get("District", np.nan),
                        "Ward":                 template_row.get("Ward", np.nan),
                        "Community Area":       template_row.get("Community Area", np.nan),
                        "Beat":                 template_row.get("Beat", np.nan),
                        "Latitude":             template_row.get("Latitude", np.nan),
                        "Longitude":            template_row.get("Longitude", np.nan),
                        "X Coordinate":         template_row.get("X Coordinate", np.nan),
                        "Y Coordinate":         template_row.get("Y Coordinate", np.nan),
                        "Primary Type":         "NO_CRIME",
                        "Description":          "NO_CRIME",
                        "Location Description": "UNKNOWN",
                        "IUCR":                 "0000",
                        "Arrest":               False,
                        "Domestic":             False,
                        TARGET_COL:             0,
                    })

        if len(negatives) < n_neg:
            print(f"  ⚠  Only {len(negatives):,} / {n_neg:,} negatives generated "
                  f"after {max_attempts:,} attempts.  "
                  f"Consider lowering neg_ratio or relaxing collision rules.")

        neg_df = pd.DataFrame(negatives)
        pos_df = pos_df.drop(columns=["_hw", "_day", "_di"])
        pos_df[TARGET_COL] = 1

        for col in pos_df.columns:
            if col not in neg_df.columns:
                neg_df[col] = np.nan

        combined = pd.concat([pos_df[neg_df.columns], neg_df], ignore_index=True)
        print(f"  Combined: {len(combined):,}  "
              f"(pos={int(pos_df[TARGET_COL].sum()):,}  neg={len(neg_df):,})")
        return combined

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def fit_transform_train(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit statistics on train split, transform, drop no-history rows.
        Returns (X_train, y_train) already aligned.
        """
        self._check_split()
        raw = self.raw_data.iloc[self.train_idx].copy()
        y   = self.labels.iloc[self.train_idx].reset_index(drop=True)

        X, mask, self.fit_dict = self._process(raw, y, is_train=True)
        y_out = pd.Series(y.values[mask], name="target").reset_index(drop=True)

        n_dropped = mask.size - mask.sum()
        print(f"  train after hist-drop : {len(X):,} rows  "
              f"(dropped {n_dropped:,} cold-start rows)")
        return X, y_out

    def transform(self, idx: np.ndarray) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform val/test split using train-fitted statistics.
        Returns (X, y) already aligned after row drops.
        """
        self._check_fitted()
        raw = self.raw_data.iloc[idx].copy()
        y   = self.labels.iloc[idx].reset_index(drop=True)

        X, mask, _ = self._process(raw, y, is_train=False)
        y_out = pd.Series(y.values[mask], name="target").reset_index(drop=True)

        n_dropped = mask.size - mask.sum()
        print(f"  split after hist-drop : {len(X):,} rows  "
              f"(dropped {n_dropped:,} cold-start rows)")
        return X, y_out

    def fit_transform_fold(
        self, fold_train_idx: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """CV fold variant — resets fit_dict each call."""
        self.fit_dict = {}
        raw = self.raw_data.iloc[fold_train_idx].copy()
        y   = self.labels.iloc[fold_train_idx].reset_index(drop=True)
        X, mask, self.fit_dict = self._process(raw, y, is_train=True)
        y_out = pd.Series(y.values[mask], name="target").reset_index(drop=True)
        return X, y_out

    # ──────────────────────────────────────────────────────────────────────────
    # Core orchestrator
    # ──────────────────────────────────────────────────────────────────────────

    def _process(
        self,
        data:     pd.DataFrame,
        y_unused: pd.Series,        # kept for signature consistency
        is_train: bool,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        """
        Returns (X, surviving_bool_mask, fit_dict).
        surviving_bool_mask is aligned to the INPUT row order.
        """
        fd = self.fit_dict
        df = data.copy()

        # 1. Drop leakage columns
        df = self._drop_leakage(df)

        # 2. Temporal
        df = self._process_temporal(df)

        # 3. Spatial numeric (adds grid_cell string col — dropped later)
        df, fd = self._process_spatial_numeric(df, is_train, fd)

        # 4. Historical features (uses full-dataset tables; drops cold rows)
        df, surviving_mask = self._process_historical(df)

        # 5. Scale numeric columns (fit on surviving train rows only)
        df, fd = self._scale_numerics(df, is_train, fd)

        # 6. Drop all remaining object / string columns (grid_cell etc.)
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if obj_cols:
            df = df.drop(columns=obj_cols)

        # 7. Drop residual date column
        df = df.drop(columns=[DATE_COL], errors="ignore")

        # Paranoia check: ensure everything is numeric
        still_obj = df.select_dtypes(include=["object"]).columns.tolist()
        if still_obj:
            print(f"  ⚠  Dropping unexpected object columns: {still_obj}")
            df = df.drop(columns=still_obj)

        return df.reset_index(drop=True), surviving_mask, fd

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 — Drop leakage columns
    # ──────────────────────────────────────────────────────────────────────────

    def _drop_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ID / Case Number  → unique identifier
        Block             → too granular, memorises addresses
        FBI Code          → derived from Primary Type (label leakage)
        Updated On        → admin metadata, not available at crime time
        Location (string) → duplicate of lat/lon
        Year              → redundant with Date
        Primary Type / Description / IUCR / Arrest / Domestic
                          → only known after an incident has occurred,
                            so they leak the binary target for occurrence
                            prediction
        """
        return df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 — Temporal features
    # ──────────────────────────────────────────────────────────────────────────

    def _process_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rich temporal features from Date.

        Raw + cyclic encodings: cyclic (sin/cos) prevents the model treating
        23→0 as a large discontinuity.  Both forms are kept so the model can
        learn both absolute position and periodicity.
        """
        dt = df[DATE_COL]

        df["hour"]         = dt.dt.hour
        df["day_of_week"]  = dt.dt.dayofweek
        df["month"]        = dt.dt.month
        df["day_of_year"]  = dt.dt.dayofyear
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

        s_h, c_h = _cyclic_encode(df["hour"],        24)
        s_d, c_d = _cyclic_encode(df["day_of_week"],  7)
        s_m, c_m = _cyclic_encode(df["month"],        12)
        df["hour_sin"],  df["hour_cos"]  = s_h, c_h
        df["dow_sin"],   df["dow_cos"]   = s_d, c_d
        df["month_sin"], df["month_cos"] = s_m, c_m

        df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
        df["is_night"]     = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
        df["is_rush_hour"] = (
            ((df["hour"] >= 7)  & (df["hour"] <= 9)) |
            ((df["hour"] >= 16) & (df["hour"] <= 18))
        ).astype(int)

        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 — Boolean flags
    # ──────────────────────────────────────────────────────────────────────────

    def _process_boolean_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["Arrest", "Domestic"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4 — Spatial numeric features
    # ──────────────────────────────────────────────────────────────────────────

    def _process_spatial_numeric(
        self, df: pd.DataFrame, is_train: bool, fd: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Per-column treatment using domain-knowledge hard bounds.

        District (1–31) / Ward (1–50) / Community Area (1–77)
        -------------------------------------------------------
        Integer administrative IDs.  Values outside the valid range are
        data-entry errors → hard-clip.  Fill missing with mode (not median,
        because the median of an ID with gaps can land between valid values).

        Beat (111–2535)
        ---------------
        Police patrol beat (district-prefixed numbering).  Hard-clip.
        Fill missing with per-district median beat (more informative than
        the global median across 1 000+ beats).

        Latitude (36–42) / Longitude (-92 to -88)
        ------------------------------------------
        Chicago metro bounding box.  Anything outside is a geocoding error
        (e.g. (0, 0) sentinel).  Fill missing with per-district median
        coordinate from train.

        X Coordinate (0–1 205 119) / Y Coordinate (0–1 951 622)
        ----------------------------------------------------------
        Illinois State Plane East (feet).  Zero = un-geocoded sentinel →
        treated as NaN before clipping.  Fill with per-district median.

        Missing indicators are added unconditionally for spatial columns
        because spatial missingness is itself a meaningful signal
        (un-geocoded records cluster in certain crime types / districts).
        """
        HARD_BOUNDS: Dict[str, Tuple[float, float]] = {
            "District":       (1.0,   31.0),
            "Ward":           (1.0,   50.0),
            "Community Area": (1.0,   77.0),
            "Beat":           (111.0, 2535.0),
            "Latitude":       (36.0,  42.0),
            "Longitude":      (-92.0, -88.0),
            "X Coordinate":   (1.0,   1_205_119.0),
            "Y Coordinate":   (1.0,   1_951_622.0),
        }
        SENTINEL_ZERO  = {"X Coordinate", "Y Coordinate"}
        MODE_FILL_COLS = {"District", "Ward", "Community Area"}

        # sentinel zeros → NaN
        for col in SENTINEL_ZERO:
            if col in df.columns:
                df[col] = df[col].replace(0.0, np.nan)

        for col, (lo, hi) in HARD_BOUNDS.items():
            if col not in df.columns:
                continue
            missing_rate = df[col].isna().mean()
            df[col] = df[col].clip(lo, hi)

            if is_train:
                fv = float(df[col].mode().iloc[0]) if col in MODE_FILL_COLS \
                     else float(df[col].median())
                fd[f"fill_{col}"]         = fv
                fd[f"missing_rate_{col}"] = missing_rate
            else:
                fv = fd.get(f"fill_{col}", float(df[col].median()))

            df[f"{col}_missing"] = df[col].isna().astype(int)
            df[col] = df[col].fillna(fv)

        # Beat: override global fill with per-district median
        if "Beat" in df.columns and "District" in df.columns:
            beat_miss = df["Beat_missing"] == 1
            if beat_miss.any():
                if is_train:
                    fd["dist_beat_med"] = df.groupby("District")["Beat"].median().to_dict()
                dm = fd.get("dist_beat_med", {})
                gf = fd.get("fill_Beat", float(df["Beat"].median()))
                df.loc[beat_miss, "Beat"] = (
                    df.loc[beat_miss, "District"].map(dm).fillna(gf)
                )

        # Lat/Lon: override with per-district median
        for coord, key in [("Latitude", "dist_lat_med"), ("Longitude", "dist_lon_med")]:
            if coord not in df.columns or "District" not in df.columns:
                continue
            cmiss = df[f"{coord}_missing"] == 1
            if cmiss.any():
                if is_train:
                    fd[key] = df.groupby("District")[coord].median().to_dict()
                dm = fd.get(key, {})
                gf = fd.get(f"fill_{coord}", float(df[coord].median()))
                df.loc[cmiss, coord] = df.loc[cmiss, "District"].map(dm).fillna(gf)

        # Grid cell features (string helper — dropped at end of _process)
        if "Latitude" in df.columns and "Longitude" in df.columns:
            df["grid_cell"]    = _build_grid_cell(df["Latitude"], df["Longitude"])
            df["grid_lat_bin"] = (df["Latitude"]  / GRID_CELL_DEG).astype(int)
            df["grid_lon_bin"] = (df["Longitude"] / GRID_CELL_DEG).astype(int)

        return df, fd

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5 — Categorical features
    # ──────────────────────────────────────────────────────────────────────────

    def _encode_categorical(
        self,
        df:       pd.DataFrame,
        col:      str,
        is_train: bool,
        fd:       Dict,
        *,
        top_n: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode a categorical column with rare-grouping and unknown handling.

        NaN → "UNKNOWN"  (missing is different from rare)
        rare / unseen → "OTHER"
        One-hot encode the cleaned column; align val/test to train schema.
        """
        key_keep = f"cat_keep_{col}"
        prefix   = col.lower().replace(" ", "_")

        df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()

        if is_train:
            freq = df[col].value_counts(normalize=True)
            keep = set(freq.head(top_n).index) if top_n \
                   else set(freq[freq >= self.rare_thresh].index)
            keep.update({"UNKNOWN", "OTHER"})
            fd[key_keep] = keep
        else:
            keep = fd.get(key_keep, set())

        df[col] = df[col].apply(lambda x: x if x in keep else "OTHER")

        dummies = pd.get_dummies(df[col], prefix=prefix, dtype=int)

        if not is_train:
            train_cols = fd.get(f"cat_cols_{col}", dummies.columns.tolist())
            for c in train_cols:
                if c not in dummies.columns:
                    dummies[c] = 0
            dummies = dummies[train_cols]
        else:
            fd[f"cat_cols_{col}"] = dummies.columns.tolist()

        df = df.drop(columns=[col])
        return pd.concat([df, dummies], axis=1), fd

    def _process_primary_type(self, df, is_train, fd):
        return self._encode_categorical(df, "Primary Type",         is_train, fd, top_n=30)

    def _process_location_desc(self, df, is_train, fd):
        return self._encode_categorical(df, "Location Description", is_train, fd, top_n=40)

    def _process_iucr(self, df, is_train, fd):
        return self._encode_categorical(df, "IUCR",                 is_train, fd, top_n=50)

    def _process_description(self, df, is_train, fd):
        return self._encode_categorical(df, "Description",          is_train, fd, top_n=60)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6 — Historical features
    # ──────────────────────────────────────────────────────────────────────────

    def _process_historical(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Compute rolling historical crime features using the FULL-dataset
        aggregation tables built during load_and_split().

        Adaptive windows
        ----------------
        Only windows whose length <= dataset time span are computed
        (set by load_and_split → self.hist_windows).  For a 38-day dataset
        with total span 37 days, only "7d" and "14d" would be active;
        "30d" and "90d" are skipped entirely and no column is added for them.

        Drop rule (corrected)
        ---------------------
        A row is dropped ONLY if the SHORTEST active window has no prior
        history — meaning there is literally no data before this row's date
        at all.  Rows where longer windows are empty but the shortest window
        has data are KEPT; longer-window columns are filled with 0 (the
        genuine interpretation: "0 crimes in the available look-back period").

        This correctly handles dense, short-span datasets where most rows
        will have 7-day history but not 30-day history.
        """
        if DATE_COL not in df.columns:
            for w in self.hist_windows:
                df[f"crimes_last_{w}"] = 0
            df["crime_density_500m"] = 0
            return df, np.ones(len(df), dtype=bool)

        dist_daily = self._full_dist_daily
        grid_daily = self._full_grid_daily

        if dist_daily is None or dist_daily.empty:
            for w in self.hist_windows:
                df[f"crimes_last_{w}"] = 0
            df["crime_density_500m"] = 0
            return df, np.ones(len(df), dtype=bool)

        dates     = pd.to_datetime(df[DATE_COL])
        districts = (
            df["District"].fillna(-1).astype(int).astype(str)
            if "District" in df.columns
            else pd.Series(["UNKNOWN"] * len(df), index=df.index)
        )
        grid_cells = (
            df["grid_cell"]
            if "grid_cell" in df.columns
            else pd.Series(["UNKNOWN"] * len(df), index=df.index)
        )

        dist_cumsum = self._build_cumsum(dist_daily, "_district")
        grid_cumsum = self._build_cumsum(grid_daily, "_grid")

        n           = len(df)
        dates_arr   = dates.values
        dist_arr    = districts.values
        grid_arr    = grid_cells.values

        # Find the shortest active window — its "has" flag determines row drop
        min_days    = min(self.hist_windows.values())
        cold_start  = np.zeros(n, dtype=bool)   # True = no history at all → drop

        # ── District-level windows ────────────────────────────────────────────
        for wname, days in self.hist_windows.items():
            counts = []
            for i, (dt, di) in enumerate(zip(dates_arr, dist_arr)):
                c, has = self._window_count(dist_cumsum.get(str(di)), dt, days)
                counts.append(c)
                # Mark cold-start only for the shortest window
                if days == min_days and not has:
                    cold_start[i] = True
            df[f"crimes_last_{wname}"] = counts

        # ── Grid-cell density (shortest active window ≤ 30d, else 7d) ────────
        density_days = min(30, max(self.hist_windows.values()))
        density = []
        for i, (dt, gc) in enumerate(zip(dates_arr, grid_arr)):
            c, has = self._window_count(grid_cumsum.get(str(gc)), dt, density_days)
            density.append(c)
            # cold_start already set by district windows above
        df["crime_density_500m"] = density

        surviving_mask = ~cold_start
        n_dropped = cold_start.sum()
        if n_dropped > 0:
            print(f"  hist-drop: {n_dropped:,} cold-start rows removed "
                  f"(no history in {min_days}-day window)")
        return df[surviving_mask].reset_index(drop=True), surviving_mask

    @staticmethod
    def _build_cumsum(daily: pd.DataFrame, group_col: str) -> Dict:
        result = {}
        for grp, sub in daily.groupby(group_col):
            sub = sub.sort_values("_date").set_index("_date")
            result[str(grp)] = sub["count"].cumsum()
        return result

    @staticmethod
    def _window_count(
        cum:        Optional[pd.Series],
        query_date,
        days:       int,
    ) -> Tuple[int, bool]:
        """
        Returns (count, has_history).

        has_history=False means no cumsum data exists within the window
        [T-days, T-1].  The caller uses this to drop the row entirely.
        """
        if cum is None or len(cum) == 0:
            return 0, False
        end   = pd.Timestamp(query_date).normalize() - pd.Timedelta(days=1)
        start = end - pd.Timedelta(days=days - 1)
        idx   = cum.index
        if not ((idx >= start) & (idx <= end)).any():
            return 0, False
        v_end   = cum[idx <= end].iloc[-1]   if (idx <= end).any()   else 0
        v_start = cum[idx <  start].iloc[-1] if (idx <  start).any() else 0
        return max(0, int(v_end - v_start)), True

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7 — StandardScaler
    # ──────────────────────────────────────────────────────────────────────────

    def _scale_numerics(
        self, df: pd.DataFrame, is_train: bool, fd: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        StandardScaler on non-binary, non-cyclic numeric columns.
        Fit on train only; transform applied to val/test identically.
        """
        def _is_binary(col: pd.Series) -> bool:
            return set(col.dropna().unique()).issubset({0, 1})

        cyclic = [c for c in df.columns if c.endswith("_sin") or c.endswith("_cos")]
        num_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in cyclic and not _is_binary(df[c])
        ]
        if not num_cols:
            return df, fd

        if is_train:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            fd["scaler"]      = scaler
            fd["scaler_cols"] = num_cols
        else:
            scaler = fd.get("scaler")
            cols   = fd.get("scaler_cols", num_cols)
            if scaler is not None:
                present = [c for c in cols if c in df.columns]
                df[present] = scaler.transform(df[present])

        return df, fd

    # ──────────────────────────────────────────────────────────────────────────
    # Guards
    # ──────────────────────────────────────────────────────────────────────────

    def _check_split(self) -> None:
        if self.train_idx is None:
            raise RuntimeError("Call load_and_split() first.")

    def _check_fitted(self) -> None:
        if not self.fit_dict:
            raise RuntimeError("Call fit_transform_train() before transform().")
