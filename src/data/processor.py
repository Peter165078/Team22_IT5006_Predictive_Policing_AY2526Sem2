"""
src/data/processor.py

DataProcessor for Chicago Crime Prediction
===========================================

Problem
-------
Binary classification: given a (location, time_window) instance, predict
whether a crime will occur (target = 1) or not (target = 0).

Key design challenges addressed
--------------------------------
1. Negative sample construction
   Raw data contains only reported crimes (all positive).  Negative samples are
   constructed by randomly drawing (district, time_window) combinations that do
   NOT appear in the positive set.  Negatives are drawn from the same
   spatial-temporal distribution to avoid trivially easy separation.

2. Temporal split (no data leakage)
   Records are sorted by date and split chronologically:
       train  →  first 70 %
       val    →  next  15 %
       test   →  last  15 %
   This mirrors real deployment: the model only sees past data at inference.

3. Historical feature leakage prevention
   Aggregation windows (crimes_last_7d, crimes_last_30d, crime_density_500m, …)
   are computed using only records that precede each sample's timestamp.
   Statistics are anchored to the training period and frozen before
   transforming val / test.

4. Fit-on-train-only
   All statistics (medians, rare-category sets, scalers, encoder vocabularies)
   are learned exclusively from the training split and stored in `fit_dict`,
   then applied identically to val and test.

Pipeline order (mirrors the ICU task pattern)
----------------------------------------------
    temporal_split()
        → fit_transform_train()
        → transform(val_idx)
        → transform(test_idx)

Each of the above calls _process(data, is_train).

Column groups
-------------
    Drop / leakage  : ID, Case Number, Block, FBI Code, Updated On,
                      Location (string duplicate of lat/lon), Year
    Target          : target  (1 = crime occurred, 0 = negative sample)
    Temporal        : Date → hour, hour_sin, hour_cos, day_of_week,
                             day_of_week_sin, day_of_week_cos,
                             month, month_sin, month_cos,
                             day_of_year, week_of_year,
                             is_weekend, is_night, is_rush_hour
    Spatial         : District, Ward, Community Area → cleaned + encoded
                      Beat                            → ordinal / numeric
                      Latitude, Longitude             → kept + cyclic encoding
                      X/Y Coordinate                  → kept (Chicago local CRS)
    Crime type      : Primary Type, Description, Location Description,
                      IUCR                            → categorical encoding
    Boolean flags   : Arrest, Domestic                → int
    Historical       : crimes_last_7d, crimes_last_30d, crimes_last_90d
                       crime_density_500m
                       (computed per district × time-window; approximated with
                        a spatial grid cell for speed on 8 M rows)
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

TARGET_COL = "target"
DATE_COL   = "Date"

# Columns to drop before any processing
_DROP_COLS = [
    "ID", "Case Number", "Block", "FBI Code",
    "Updated On", "Location", "Year",
]

# Rare-category threshold: categories whose frequency < this share of train
# rows are collapsed into "Other"
RARE_THRESH = 0.005

# Missing-indicator threshold: columns with more than this fraction missing
# get an extra <col>_missing binary indicator
MISSING_IND_THRESH = 0.05

# Negative sample ratio (negatives per positive in the final dataset)
NEG_RATIO = 1.0          # 1:1 balanced dataset by default

# Spatial grid cell size for density features (degrees ≈ 500 m at Chicago lat)
GRID_CELL_DEG = 0.005    # ~500 m

# Historical look-back windows (days)
HIST_WINDOWS = {"7d": 7, "30d": 30, "90d": 90}

# Chronological split fractions
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# Test = 1 - TRAIN_FRAC - VAL_FRAC = 0.15

RANDOM_STATE = 42


# ──────────────────────────────────────────────────────────────────────────────
# Stateless utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _cyclic_encode(series: pd.Series, period: float) -> Tuple[pd.Series, pd.Series]:
    """Return (sin, cos) cyclic encoding for a periodic numerical series."""
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


def _clamp_outliers_iqr(series: pd.Series,
                        lower_q: float = 0.01,
                        upper_q: float = 0.99) -> pd.Series:
    """Winsorise to [lower_q, upper_q] percentile range."""
    lo = series.quantile(lower_q)
    hi = series.quantile(upper_q)
    return series.clip(lo, hi)


def _build_grid_cell(lat: pd.Series, lon: pd.Series,
                     cell: float = GRID_CELL_DEG) -> pd.Series:
    """Map (lat, lon) to a string grid-cell identifier for density binning."""
    lat_bin = (lat / cell).astype(int)
    lon_bin = (lon / cell).astype(int)
    return lat_bin.astype(str) + "_" + lon_bin.astype(str)


# ──────────────────────────────────────────────────────────────────────────────
# DataProcessor
# ──────────────────────────────────────────────────────────────────────────────

class DataProcessor:
    """
    End-to-end preprocessing pipeline for the Chicago crime prediction task.

    Usage
    -----
    >>> proc = DataProcessor(data_path="data/raw/Crimes_-_2001_to_Present.csv")
    >>> proc.load_and_split()
    >>> X_train = proc.fit_transform_train()
    >>> X_val   = proc.transform(proc.val_idx)
    >>> X_test  = proc.transform(proc.test_idx)
    >>> y_train = proc.labels.iloc[proc.train_idx].reset_index(drop=True)
    """

    def __init__(
        self,
        data_path:     str,
        neg_ratio:     float = NEG_RATIO,
        rare_thresh:   float = RARE_THRESH,
        random_state:  int   = RANDOM_STATE,
    ) -> None:
        self.data_path    = data_path
        self.neg_ratio    = neg_ratio
        self.rare_thresh  = rare_thresh
        self.random_state = random_state

        self.raw_data:  Optional[pd.DataFrame] = None
        self.labels:    Optional[pd.Series]    = None
        self.train_idx: Optional[np.ndarray]   = None
        self.val_idx:   Optional[np.ndarray]   = None
        self.test_idx:  Optional[np.ndarray]   = None
        self.fit_dict:  Dict                   = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Step 0: Load, construct negatives, temporal split
    # ──────────────────────────────────────────────────────────────────────────

    def load_and_split(self) -> None:
        """
        Load raw CSV, parse dates, construct negative samples, then perform a
        chronological train / val / test split.

        Chronological split is mandatory for time-series data: random split
        would leak future crime patterns into the training set.
        """
        print("  Loading raw data …")
        df = pd.read_csv(self.data_path, low_memory=False)

        # ── Parse date ───────────────────────────────────────────────────────
        df[DATE_COL] = pd.to_datetime(df[DATE_COL],
                                      format="%m/%d/%Y %I:%M:%S %p",
                                      errors="coerce")
        df = df.dropna(subset=[DATE_COL])
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        print(f"  Positive samples: {len(df):,}  "
              f"({df[DATE_COL].min().date()} → {df[DATE_COL].max().date()})")

        # ── Construct negative samples ────────────────────────────────────────
        df = self._build_negatives(df)

        # ── Chronological split ───────────────────────────────────────────────
        # Sort entire frame again (negatives inherit timestamps from positives)
        df = df.sort_values(DATE_COL).reset_index(drop=True)

        n     = len(df)
        n_tr  = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)

        self.train_idx = np.arange(0, n_tr)
        self.val_idx   = np.arange(n_tr, n_tr + n_val)
        self.test_idx  = np.arange(n_tr + n_val, n)

        self.labels   = df[TARGET_COL].copy()
        self.raw_data = df.drop(columns=[TARGET_COL]).copy()

        print(f"  Total samples   : {n:,}  "
              f"(train {len(self.train_idx):,} | "
              f"val {len(self.val_idx):,} | "
              f"test {len(self.test_idx):,})")
        print(f"  Positive rate   : {self.labels.mean():.3f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Negative sample construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_negatives(self, pos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct synthetic negative samples (no crime occurred).

        Strategy
        --------
        For each positive record we need a matching "no crime" instance.
        We sample from the empirical distribution of (district, hour_bin,
        day_of_week) to keep negatives in the realistic spatial-temporal space
        while ensuring they do NOT overlap with a real crime on that exact
        (district, date_hour).

        Negative samples inherit the timestamp from their matched positive so
        that historical features can still be computed consistently.

        Steps
        -----
        1. Round timestamps to the nearest hour to define time windows.
        2. Build a set of (district, hour_window) that DID have crime.
        3. Draw random (district, timestamp) pairs from the marginal
           distributions observed in the positive set.
        4. Reject any draw that collides with a real (district, hour_window)
           crime event.
        5. Attach spatial coordinates by sampling from district-level medians
           so negatives have plausible lat/lon values.
        """
        rng = np.random.default_rng(self.random_state)
        n_neg = int(len(pos_df) * self.neg_ratio)

        print(f"  Building {n_neg:,} negative samples …")

        # ── Existing (district, hour_window) crime keys ──────────────────────
        pos_df = pos_df.copy()
        pos_df["_hour_window"] = pos_df[DATE_COL].dt.floor("h")
        pos_df["_district"]    = pos_df["District"].fillna(-1).astype(int).astype(str)
        crime_keys = set(
            pos_df["_district"].astype(str) + "_" +
            pos_df["_hour_window"].astype(str)
        )

        # ── District-level spatial medians for realistic lat/lon ─────────────
        district_coords = (
            pos_df.groupby("_district")[["Latitude", "Longitude",
                                         "X Coordinate", "Y Coordinate",
                                         "Ward", "Community Area", "Beat"]]
            .median()
        )

        # ── Marginal distributions to sample from ────────────────────────────
        districts   = pos_df["_district"].values
        timestamps  = pos_df["_hour_window"].values
        n_pos       = len(pos_df)

        negatives   = []
        attempts    = 0
        max_attempts = n_neg * 10

        while len(negatives) < n_neg and attempts < max_attempts:
            batch = min(n_neg * 2, max_attempts - attempts)
            attempts += batch

            # Sample from empirical joint distribution of districts + times
            idx_d = rng.integers(0, n_pos, size=batch)
            idx_t = rng.integers(0, n_pos, size=batch)
            s_districts  = districts[idx_d]
            s_timestamps = timestamps[idx_t]

            # Random offset within the hour window (0–59 min)
            offset_minutes = rng.integers(0, 60, size=batch)
            s_datetimes = (
                pd.to_datetime(s_timestamps) +
                pd.to_timedelta(offset_minutes, unit="m")
            )

            for dist, dt in zip(s_districts, s_datetimes):
                if len(negatives) >= n_neg:
                    break
                key = str(dist) + "_" + str(pd.Timestamp(dt).floor("h"))
                if key not in crime_keys:
                    # Look up spatial coords from district median
                    row = district_coords.loc[dist] \
                        if dist in district_coords.index \
                        else district_coords.iloc[0]
                    negatives.append({
                        DATE_COL:              dt,
                        "District":            float(dist) if dist != "-1" else np.nan,
                        "Ward":                row.get("Ward", np.nan),
                        "Community Area":      row.get("Community Area", np.nan),
                        "Beat":                row.get("Beat", np.nan),
                        "Latitude":            row.get("Latitude", np.nan),
                        "Longitude":           row.get("Longitude", np.nan),
                        "X Coordinate":        row.get("X Coordinate", np.nan),
                        "Y Coordinate":        row.get("Y Coordinate", np.nan),
                        "Primary Type":        "NO_CRIME",
                        "Description":         "NO_CRIME",
                        "Location Description":"UNKNOWN",
                        "IUCR":                "0000",
                        "Arrest":              False,
                        "Domestic":            False,
                        TARGET_COL:            0,
                    })

        if len(negatives) < n_neg:
            print(f"  ⚠  Only {len(negatives):,} negatives generated "
                  f"(target was {n_neg:,}); increase max_attempts if needed.")

        neg_df = pd.DataFrame(negatives)

        # ── Tag positives, concatenate ────────────────────────────────────────
        pos_df = pos_df.drop(columns=["_hour_window", "_district"])
        pos_df[TARGET_COL] = 1

        # Align columns: neg_df may miss some columns → fill with NaN
        for col in pos_df.columns:
            if col not in neg_df.columns:
                neg_df[col] = np.nan

        combined = pd.concat(
            [pos_df[neg_df.columns], neg_df],
            ignore_index=True,
        )
        print(f"  Combined dataset: {len(combined):,} rows  "
              f"(pos={pos_df[TARGET_COL].sum():,}  "
              f"neg={len(neg_df):,})")
        return combined

    # ──────────────────────────────────────────────────────────────────────────
    # Public transform interface
    # ──────────────────────────────────────────────────────────────────────────

    def fit_transform_train(self) -> pd.DataFrame:
        """Fit all statistics on train split and return transformed X_train."""
        self._check_split()
        train_raw       = self.raw_data.iloc[self.train_idx].copy()
        X_train, self.fit_dict = self._process(train_raw, is_train=True)
        return X_train

    def transform(self, idx: np.ndarray) -> pd.DataFrame:
        """Transform a split (val / test) using statistics fitted on train."""
        self._check_fitted()
        subset_raw = self.raw_data.iloc[idx].copy()
        X, _       = self._process(subset_raw, is_train=False)
        return X

    def fit_transform_fold(self, fold_train_idx: np.ndarray) -> pd.DataFrame:
        """
        Re-fit on a CV fold's train rows and return transformed X.
        Resets fit_dict so each fold is independent (leakage-free).
        """
        self.fit_dict = {}
        fold_raw = self.raw_data.iloc[fold_train_idx].copy()
        X, self.fit_dict = self._process(fold_raw, is_train=True)
        return X

    # ──────────────────────────────────────────────────────────────────────────
    # Core processing orchestrator
    # ──────────────────────────────────────────────────────────────────────────

    def _process(
        self,
        data: pd.DataFrame,
        is_train: bool,
    ) -> Tuple[pd.DataFrame, Dict]:
        fd = self.fit_dict

        # ── Drop leakage / ID columns ─────────────────────────────────────────
        df = self._drop_leakage(data)

        # ── Temporal features ─────────────────────────────────────────────────
        df = self._process_temporal(df)

        # ── Boolean flags ─────────────────────────────────────────────────────
        df = self._process_boolean_flags(df)

        # ── Spatial numerical features ────────────────────────────────────────
        df, fd = self._process_spatial_numeric(df, is_train, fd)

        # ── Categorical features ──────────────────────────────────────────────
        df, fd = self._process_primary_type(df, is_train, fd)
        df, fd = self._process_location_desc(df, is_train, fd)
        df, fd = self._process_iucr(df, is_train, fd)
        df, fd = self._process_description(df, is_train, fd)

        # ── Historical / density features ─────────────────────────────────────
        df, fd = self._process_historical(df, is_train, fd)

        # ── Scale numerical features ──────────────────────────────────────────
        df, fd = self._scale_numerics(df, is_train, fd)

        # ── Drop residual date column ─────────────────────────────────────────
        df = df.drop(columns=[DATE_COL], errors="ignore")

        return df.reset_index(drop=True), fd

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1 — Drop leakage columns
    # ──────────────────────────────────────────────────────────────────────────

    def _drop_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove columns that are either identifiers or constitute data leakage.

        Leakage column rationale:
          ID / Case Number  → unique identifier, no predictive signal
          Block             → too granular; would memorise specific addresses
          FBI Code          → derived directly from Primary Type (label leakage)
          Updated On        → administrative metadata, not available at crime time
          Location (string) → duplicate of lat/lon already kept
          Year              → redundant with Date; can induce year-level leakage
        """
        return df.drop(columns=[c for c in _DROP_COLS if c in df.columns])

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2 — Temporal features
    # ──────────────────────────────────────────────────────────────────────────

    def _process_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract rich temporal features from the Date column.

        Cyclic encodings (sin/cos) prevent the model from treating 23 → 0
        (midnight) as a large discontinuity.  Both raw and cyclic forms are
        kept to allow the model to learn both absolute position and periodicity.

        Feature list
        ------------
        hour, hour_sin, hour_cos           — time of day (period 24)
        day_of_week, dow_sin, dow_cos       — day of week  (period 7)
        month, month_sin, month_cos         — month of year (period 12)
        day_of_year                         — seasonal position (1–366)
        week_of_year                        — ISO week (1–53)
        is_weekend                          — Sat/Sun indicator
        is_night                            — 22:00–05:59 indicator
        is_rush_hour                        — 07:00–09:59 or 16:00–18:59
        """
        dt = df[DATE_COL]

        df["hour"]        = dt.dt.hour
        df["day_of_week"] = dt.dt.dayofweek          # 0 = Mon … 6 = Sun
        df["month"]       = dt.dt.month
        df["day_of_year"] = dt.dt.dayofyear
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

        # Cyclic encodings
        s_h, c_h = _cyclic_encode(df["hour"], 24)
        df["hour_sin"], df["hour_cos"] = s_h, c_h

        s_d, c_d = _cyclic_encode(df["day_of_week"], 7)
        df["dow_sin"], df["dow_cos"] = s_d, c_d

        s_m, c_m = _cyclic_encode(df["month"], 12)
        df["month_sin"], df["month_cos"] = s_m, c_m

        # Indicator flags
        df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
        df["is_night"]     = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
        df["is_rush_hour"] = (
            ((df["hour"] >= 7) & (df["hour"] <= 9)) |
            ((df["hour"] >= 16) & (df["hour"] <= 18))
        ).astype(int)

        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3 — Boolean flags
    # ──────────────────────────────────────────────────────────────────────────

    def _process_boolean_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert Arrest and Domestic (bool) to integer 0/1.

        Note: for negative samples these are set to False in construction,
        which correctly maps to 0 (no arrest possible if no crime occurred).
        """
        for col in ["Arrest", "Domestic"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4 — Spatial numerical features
    # ──────────────────────────────────────────────────────────────────────────

    def _process_spatial_numeric(
        self, df: pd.DataFrame, is_train: bool, fd: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Per-column spatial numeric processing with domain-knowledge hard bounds.

        Each column has its own outlier strategy based on its known valid range:

        District (1–31)
        ---------------
        Administrative police district ID.  Values outside [1, 31] are data
        entry errors → hard-clip, not IQR.  Treated as integer category kept
        as numeric for tree/linear models.  Missing → mode (most common district
        in train), plus missing indicator (Chicago has a small but consistent
        fraction of un-geocoded records).

        Ward (1–50)
        -----------
        City council ward.  Hard valid range [1, 50].  Same treatment as
        District.  Missing → mode.

        Community Area (1–77)
        ---------------------
        Chicago's 77 official community areas.  Hard clip to [1, 77].
        Missing → mode.  Community area is a stronger spatial signal than ward
        for crime patterns (stable neighbourhoods).

        Beat (111–2535)
        ---------------
        Police patrol beat — finest administrative granularity.  The range
        [111, 2535] is the documented Chicago PD beat numbering system.
        Values outside this range are coercion errors → hard-clip.
        Beats have large numeric gaps by design (districts prefix the beat number),
        so raw numeric value is informative; kept as-is after clipping.
        Missing → median beat within the same district (if available), else
        global median.

        Latitude (36–42)
        ----------------
        Chicago metro area spans roughly 41.6–42.1°N.  The broader [36, 42]
        bound is a generous Chicago-state envelope; anything outside is clearly
        erroneous (e.g. (0, 0) sentinel values).  Clip, then fill missing with
        district-level median lat from train.

        Longitude (-92 to -88)
        ----------------------
        Chicago metro spans roughly -87.9 to -87.5°W.  Hard clip [-92, -88].
        Same fill strategy as Latitude.

        X Coordinate (0–1,205,119)  /  Y Coordinate (0–1,951,622)
        -----------------------------------------------------------
        Illinois State Plane East projection (feet).  Values of exactly 0
        indicate missing geocoding (sentinel zero) → treat 0 as NaN.
        Hard clip to documented ranges after zero-masking.
        Fill missing with district-level median from train.
        """

        # ── Domain hard bounds: (lower_valid, upper_valid) ───────────────────
        HARD_BOUNDS: Dict[str, Tuple[float, float]] = {
            "District":      (1.0,  31.0),
            "Ward":          (1.0,  50.0),
            "Community Area":(1.0,  77.0),
            "Beat":          (111.0, 2535.0),
            "Latitude":      (36.0,  42.0),
            "Longitude":     (-92.0, -88.0),
            "X Coordinate":  (1.0,   1_205_119.0),   # 0 = sentinel missing
            "Y Coordinate":  (1.0,   1_951_622.0),   # 0 = sentinel missing
        }

        # ── Columns where 0 is a sentinel for "no geocode" ───────────────────
        SENTINEL_ZERO_COLS = {"X Coordinate", "Y Coordinate"}

        # ── Columns where mode is preferred over median ───────────────────────
        # (integer IDs with a dominant value; median can land between integers)
        MODE_FILL_COLS = {"District", "Ward", "Community Area"}

        # ── Step 0: sentinel zero → NaN ──────────────────────────────────────
        for col in SENTINEL_ZERO_COLS:
            if col in df.columns:
                df[col] = df[col].replace(0.0, np.nan)

        # ── Step 1: per-column processing ────────────────────────────────────
        for col, (lo_hard, hi_hard) in HARD_BOUNDS.items():
            if col not in df.columns:
                continue

            missing_rate = df[col].isna().mean()

            # Hard-clip values outside the valid domain before any statistics
            df[col] = df[col].clip(lo_hard, hi_hard)

            if is_train:
                if col in MODE_FILL_COLS:
                    fill_val = float(df[col].mode().iloc[0])
                else:
                    fill_val = float(df[col].median())
                fd[f"fill_{col}"]         = fill_val
                fd[f"missing_rate_{col}"] = missing_rate
            else:
                fill_val = fd.get(f"fill_{col}", float(df[col].median()))

            # Always add missing indicator (spatial missingness is informative)
            df[f"{col}_missing"] = df[col].isna().astype(int)

            df[col] = df[col].fillna(fill_val)

        # ── Step 2: Beat — district-aware median fill ─────────────────────────
        # Override the global median fill for Beat with per-district median
        # (beats are district-prefixed; a district-level median is far more
        # representative than the global median of 1,000+ beats).
        if "Beat" in df.columns and "District" in df.columns:
            beat_missing_mask = df["Beat_missing"] == 1   # created above
            if beat_missing_mask.any():
                if is_train:
                    dist_beat_median = df.groupby("District")["Beat"].median()
                    fd["dist_beat_median"] = dist_beat_median.to_dict()
                dist_beat = fd.get("dist_beat_median", {})
                global_beat_fill = fd.get("fill_Beat",
                                          float(df["Beat"].median()))
                df.loc[beat_missing_mask, "Beat"] = (
                    df.loc[beat_missing_mask, "District"]
                    .map(dist_beat)
                    .fillna(global_beat_fill)
                )

        # ── Step 3: Lat/Lon — district-aware median fill ─────────────────────
        for coord_col, dist_key in [("Latitude", "dist_lat_median"),
                                    ("Longitude", "dist_lon_median")]:
            if coord_col not in df.columns or "District" not in df.columns:
                continue
            coord_missing = df[f"{coord_col}_missing"] == 1
            if coord_missing.any():
                if is_train:
                    dist_med = df.groupby("District")[coord_col].median()
                    fd[dist_key] = dist_med.to_dict()
                dist_map   = fd.get(dist_key, {})
                global_fill = fd.get(f"fill_{coord_col}",
                                     float(df[coord_col].median()))
                df.loc[coord_missing, coord_col] = (
                    df.loc[coord_missing, "District"]
                    .map(dist_map)
                    .fillna(global_fill)
                )

        # ── Step 4: Grid cell features for spatial density lookup ─────────────
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
        Generic categorical encoder with rare-grouping and unknown handling.

        Strategy
        --------
        1. Fill NaN → "UNKNOWN"
        2. On train: identify rare categories (freq < rare_thresh) → "OTHER"
           Optionally keep only top_n most frequent categories.
        3. On val/test: map unseen categories → "OTHER"
        4. One-hot encode the resulting clean column.
           Prefix = column name (lowercased, spaces replaced with _).

        Rationale for "UNKNOWN" vs "OTHER" distinction:
          - UNKNOWN  = value was actually missing in the source data
          - OTHER    = value was present but too rare to model reliably
        This lets the model learn separate behaviour for both situations
        rather than collapsing them, while still preventing unseen-category
        errors at inference time.
        """
        key_keep = f"cat_keep_{col}"
        prefix   = col.lower().replace(" ", "_")

        df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()

        if is_train:
            freq = df[col].value_counts(normalize=True)
            if top_n:
                keep = set(freq.head(top_n).index)
            else:
                keep = set(freq[freq >= self.rare_thresh].index)
            keep.discard("NO_CRIME")    # let negatives keep their own label
            keep.add("UNKNOWN")
            keep.add("OTHER")
            fd[key_keep] = keep
        else:
            keep = fd.get(key_keep, set())

        # Map rare / unseen values
        df[col] = df[col].apply(lambda x: x if x in keep else "OTHER")

        # One-hot encode
        dummies = pd.get_dummies(df[col], prefix=prefix, dtype=int)

        # On test/val, align columns to training schema
        if not is_train:
            train_cols = fd.get(f"cat_cols_{col}", dummies.columns.tolist())
            for c in train_cols:
                if c not in dummies.columns:
                    dummies[c] = 0
            dummies = dummies[train_cols]
        else:
            fd[f"cat_cols_{col}"] = dummies.columns.tolist()

        df = df.drop(columns=[col])
        df = pd.concat([df, dummies], axis=1)
        return df, fd

    def _process_primary_type(
        self, df: pd.DataFrame, is_train: bool, fd: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Primary Type: high-level crime category.

        Top-30 types are kept; the long tail and "NO_CRIME" (negatives) are
        handled naturally via the generic encoder's OTHER bucket.

        For negative samples Primary Type = "NO_CRIME" → will map to
        "OTHER" which acts as the correct "baseline / no event" signal.
        """
        return self._encode_categorical(
            df, "Primary Type", is_train, fd, top_n=30
        )

    def _process_location_desc(
        self, df: pd.DataFrame, is_train: bool, fd: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Location Description: where the crime occurred (e.g. STREET, RESIDENCE).
        ~170 unique values; keep top-40, collapse the rest → "OTHER".
        """
        return self._encode_categorical(
            df, "Location Description", is_train, fd, top_n=40
        )

    def _process_iucr(
        self, df: pd.DataFrame, is_train: bool, fd: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        IUCR (Illinois Uniform Crime Reporting) code.
        Fine-grained crime code (~300+ unique); keep top-50 by frequency.
        """
        return self._encode_categorical(
            df, "IUCR", is_train, fd, top_n=50
        )

    def _process_description(
        self, df: pd.DataFrame, is_train: bool, fd: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Description: sub-type within each Primary Type (~300 unique).
        Keep top-60 by frequency.
        """
        return self._encode_categorical(
            df, "Description", is_train, fd, top_n=60
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6 — Historical / density features
    # ──────────────────────────────────────────────────────────────────────────

    def _process_historical(
        self, df: pd.DataFrame, is_train: bool, fd: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Compute rolling historical crime counts and spatial density features,
        each paired with a data-availability indicator column.

        Features produced (per window W ∈ {7d, 30d, 90d})
        ---------------------------------------------------
        crimes_last_{W}           : crime count in the same district, past W days
        crimes_last_{W}_has_data  : 1 if the lookup table contained at least one
                                    day of data in the window; 0 if no history
                                    was available (e.g. very first days in the
                                    dataset).  When has_data = 0, the count
                                    column is set to 0 — NOT as a legitimate
                                    "zero crimes" signal but as a placeholder.
                                    Downstream models should use the indicator
                                    to distinguish the two cases.

        crime_density_500m        : crime count in the same ≈500 m grid cell,
                                    past 30 days
        crime_density_500m_has_data : same availability indicator

        Leakage prevention
        ------------------
        Lookups are STRICTLY backward-looking (only dates < query date).

        On train : build district-level and grid-cell-level daily aggregation
                   tables from the training rows; store in fd.
        On val/test : use the frozen train-period tables.  Records whose query
                   date falls before the earliest train date will have
                   has_data = 0.

        Implementation
        --------------
        Pre-aggregated daily count table + per-group cumulative sum enables
        O(1) per-row lookup, making the 8 M-row dataset tractable.
        """
        if DATE_COL not in df.columns:
            for w in HIST_WINDOWS:
                df[f"crimes_last_{w}"]          = 0
                df[f"crimes_last_{w}_has_data"]  = 0
            df["crime_density_500m"]          = 0
            df["crime_density_500m_has_data"] = 0
            return df, fd

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

        # ── Build or retrieve lookup tables ───────────────────────────────────
        if is_train:
            dist_daily = (
                df.assign(_date=dates.dt.normalize(), _district=districts)
                .groupby(["_district", "_date"])
                .size()
                .reset_index(name="count")
            )
            fd["hist_district_daily"] = dist_daily

            grid_daily = (
                df.assign(_date=dates.dt.normalize(), _grid=grid_cells)
                .groupby(["_grid", "_date"])
                .size()
                .reset_index(name="count")
            )
            fd["hist_grid_daily"] = grid_daily

        dist_daily = fd.get(
            "hist_district_daily",
            pd.DataFrame(columns=["_district", "_date", "count"])
        )
        grid_daily = fd.get(
            "hist_grid_daily",
            pd.DataFrame(columns=["_grid", "_date", "count"])
        )

        df = self._attach_hist_features(
            df, dates, districts, grid_cells, dist_daily, grid_daily,
        )
        return df, fd

    @staticmethod
    def _attach_hist_features(
        df:         pd.DataFrame,
        dates:      pd.Series,
        districts:  pd.Series,
        grid_cells: pd.Series,
        dist_daily: pd.DataFrame,
        grid_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Attach historical count features + has_data indicators for each row.

        has_data = 1  iff the cumulative-sum series for the group contains at
                      least one date that falls within [query_date - W, query_date - 1].
        has_data = 0  means no training history was available for this
                      (group, window) combination; the count is set to 0 as a
                      neutral placeholder — not a meaningful "zero crimes".

        The explicit separation of "no history" (has_data=0, count=0) from
        "genuinely zero crimes in window" (has_data=1, count=0) prevents the
        model from treating cold-start records as low-crime locations.
        """
        def _build_cumsum(daily: pd.DataFrame, group_col: str) -> Dict:
            result = {}
            for grp, sub in daily.groupby(group_col):
                sub = sub.sort_values("_date").set_index("_date")
                result[str(grp)] = sub["count"].cumsum()
            return result

        dist_cumsum = _build_cumsum(dist_daily, "_district") \
                      if not dist_daily.empty else {}
        grid_cumsum = _build_cumsum(grid_daily, "_grid") \
                      if not grid_daily.empty else {}

        def _window_lookup(
            cum: Optional[pd.Series],
            query_date,
            days: int,
        ) -> Tuple[int, int]:
            """
            Returns (count, has_data).

            has_data = 1  if the cum-sum series has ANY entry within
                          [query_date - days, query_date - 1].
            count    = number of crimes in that window (0 if has_data=0).
            """
            if cum is None or len(cum) == 0:
                return 0, 0

            end   = pd.Timestamp(query_date).normalize() - pd.Timedelta(days=1)
            start = end - pd.Timedelta(days=days - 1)
            idx   = cum.index

            # Check if ANY historical data falls within the window
            in_window = (idx >= start) & (idx <= end)
            if not in_window.any():
                return 0, 0

            # Use cumsum difference to get window count
            val_end   = cum[idx <= end].iloc[-1]   if (idx <= end).any()   else 0
            val_start = cum[idx < start].iloc[-1]  if (idx < start).any() else 0
            count = max(0, int(val_end - val_start))
            return count, 1

        dates_arr      = dates.values
        districts_arr  = districts.values
        grid_cells_arr = grid_cells.values

        # ── District-level windows ─────────────────────────────────────────────
        for window_name, days in HIST_WINDOWS.items():
            counts    = []
            has_datas = []
            for dt, dist in zip(dates_arr, districts_arr):
                cum = dist_cumsum.get(str(dist))
                c, h = _window_lookup(cum, dt, days)
                counts.append(c)
                has_datas.append(h)
            df[f"crimes_last_{window_name}"]         = counts
            df[f"crimes_last_{window_name}_has_data"] = has_datas

        # ── Grid-cell density (30-day window) ─────────────────────────────────
        density    = []
        has_datas  = []
        for dt, grid in zip(dates_arr, grid_cells_arr):
            cum = grid_cumsum.get(str(grid))
            c, h = _window_lookup(cum, dt, 30)
            density.append(c)
            has_datas.append(h)
        df["crime_density_500m"]          = density
        df["crime_density_500m_has_data"] = has_datas

        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7 — StandardScaler on numerical columns
    # ──────────────────────────────────────────────────────────────────────────

    def _scale_numerics(
        self, df: pd.DataFrame, is_train: bool, fd: Dict
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply StandardScaler to all remaining float64 / int64 columns
        (excluding binary indicators and one-hot columns which are already 0/1).

        Columns excluded from scaling:
          - Binary indicators (values in {0, 1} only)
          - Cyclic features (already in [-1, 1])
          - One-hot columns (prefixed with known category names)

        Fit on train only; transform applied identically to val/test.
        """
        # Identify columns that are already 0/1 (no need to scale)
        def _is_binary(col: pd.Series) -> bool:
            uniq = col.dropna().unique()
            return set(uniq).issubset({0, 1})

        # Cyclic features are already bounded
        cyclic_cols = [c for c in df.columns
                       if c.endswith("_sin") or c.endswith("_cos")]

        numeric_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in cyclic_cols and not _is_binary(df[c])
        ]

        if not numeric_cols:
            return df, fd

        if is_train:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            fd["scaler"]          = scaler
            fd["scaler_cols"]     = numeric_cols
        else:
            scaler = fd.get("scaler")
            cols   = fd.get("scaler_cols", numeric_cols)
            if scaler is not None:
                # Only scale columns present in both train schema and current df
                cols_present = [c for c in cols if c in df.columns]
                df[cols_present] = scaler.transform(df[cols_present])

        return df, fd

    # ──────────────────────────────────────────────────────────────────────────
    # Guard checks
    # ──────────────────────────────────────────────────────────────────────────

    def _check_split(self) -> None:
        if self.train_idx is None:
            raise RuntimeError("Call load_and_split() before fit_transform_train().")

    def _check_fitted(self) -> None:
        if not self.fit_dict:
            raise RuntimeError(
                "fit_dict is empty. Call fit_transform_train() before transform()."
            )