"""
Build county-hour natural-label datasets from FBI NIBRS state/year downloads.

The public NIBRS extracts are distributed as multiple related tables.  This
script converts the minimum subset we need for external generalization testing
into a single CSV compatible with the existing DataProcessor natural-label path.

Design choices
--------------
- Geographic unit: county-like reporting region derived from `agencies.csv`.
  This keeps the space-time grid tractable while still testing transfer beyond
  Chicago districts.
- Temporal unit: hour.
- Label: `target = 1` if at least one incident was reported in a county-hour
  cell, otherwise `0`.
- Availability mask: hour cells are created only for county-months where at
  least one agency filed incident data (`I`) or an explicit zero report (`Z`).

The output schema intentionally mirrors the Chicago pipeline's expected numeric
columns.  Only `District` is populated with a county identifier; the remaining
Chicago-specific spatial columns are left null so the processor can treat them
as unavailable external features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NIBRS_ROOT = PROJECT_ROOT / "NIBRS data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_STATES = ["TX", "CO"]
DEFAULT_YEARS = [2023, 2024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare county-hour natural-label datasets from downloaded NIBRS tables."
    )
    parser.add_argument(
        "--nibrs-root",
        type=Path,
        default=DEFAULT_NIBRS_ROOT,
        help="Root directory containing the downloaded NIBRS folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the prepared CSVs will be written.",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=DEFAULT_STATES,
        help="State postal abbreviations to prepare, e.g. TX CO.",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=DEFAULT_YEARS,
        help="Years to include in each state's combined dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prepared CSVs.",
    )
    return parser.parse_args()


def _discover_state_year_dirs(root: Path) -> dict[tuple[str, int], Path]:
    mapping: dict[tuple[str, int], Path] = {}
    for incident_path in root.rglob("NIBRS_incident.csv"):
        base = incident_path.parent
        agencies_path = base / "agencies.csv"
        month_path = base / "NIBRS_month.csv"
        if not agencies_path.exists() or not month_path.exists():
            continue

        incident_sample = pd.read_csv(
            incident_path,
            usecols=["data_year"],
            nrows=1,
        )
        agencies_sample = pd.read_csv(
            agencies_path,
            usecols=["state_postal_abbr"],
            nrows=1,
        )
        if incident_sample.empty or agencies_sample.empty:
            continue

        year = int(incident_sample.iloc[0]["data_year"])
        state = str(agencies_sample.iloc[0]["state_postal_abbr"]).strip().upper()
        mapping[(state, year)] = base
    return mapping


def _normalize_county_name(series: pd.Series) -> pd.Series:
    clean = series.fillna("").astype(str).str.strip().str.upper()
    clean = clean.replace({"": "UNKNOWN"})
    return clean


def _load_year_tables(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    incident = pd.read_csv(
        base_dir / "NIBRS_incident.csv",
        usecols=[
            "data_year",
            "agency_id",
            "incident_id",
            "incident_date",
            "incident_hour",
            "incident_status",
        ],
        low_memory=False,
    )
    agencies = pd.read_csv(
        base_dir / "agencies.csv",
        usecols=[
            "agency_id",
            "state_name",
            "state_postal_abbr",
            "county_name",
            "population",
            "agency_status",
        ],
        low_memory=False,
    )
    month = pd.read_csv(
        base_dir / "NIBRS_month.csv",
        usecols=[
            "agency_id",
            "inc_data_year",
            "month_num",
            "reported_status",
        ],
        low_memory=False,
    )
    return incident, agencies, month


def _hourly_month_grid(year: int, month: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    return pd.date_range(start=start, end=end - pd.Timedelta(hours=1), freq="h")


def _scaffold_reported_hours(reported: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    grouped = (
        reported[["county_key", "data_year", "month_num"]]
        .drop_duplicates()
        .sort_values(["data_year", "month_num", "county_key"])
    )

    for (year, month), block in grouped.groupby(["data_year", "month_num"], sort=True):
        hours = _hourly_month_grid(int(year), int(month))
        month_frame = (
            pd.MultiIndex.from_product(
                [block["county_key"].tolist(), hours],
                names=["county_key", "Date"],
            )
            .to_frame(index=False)
        )
        frames.append(month_frame)

    if not frames:
        return pd.DataFrame(columns=["county_key", "Date"])
    return pd.concat(frames, ignore_index=True)


def _aggregate_state_year(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    incident, agencies, month = _load_year_tables(base_dir)

    agencies = agencies.loc[agencies["agency_status"].fillna("A") == "A"].copy()
    agencies["state_abbr"] = agencies["state_postal_abbr"].fillna("UNK").astype(str).str.upper()
    agencies["county_name_clean"] = _normalize_county_name(agencies["county_name"])
    agencies["county_key"] = agencies["state_abbr"] + "::" + agencies["county_name_clean"]

    county_ref = (
        agencies.groupby("county_key", as_index=False)
        .agg(
            state_abbr=("state_abbr", "first"),
            state_name=("state_name", "first"),
            county_name=("county_name_clean", "first"),
            agency_count=("agency_id", "nunique"),
            county_population=("population", "sum"),
        )
    )

    incident = incident.loc[
        incident["incident_status"].fillna("ACCEPTED").isin(["ACCEPTED", "WARNINGS"])
    ].copy()
    incident["incident_date"] = pd.to_datetime(incident["incident_date"], errors="coerce")
    incident = incident.dropna(subset=["incident_date", "incident_hour"]).copy()
    incident = incident.merge(
        agencies[["agency_id", "county_key"]],
        on="agency_id",
        how="inner",
    )
    incident["Date"] = (
        incident["incident_date"].dt.normalize()
        + pd.to_timedelta(pd.to_numeric(incident["incident_hour"], errors="coerce").fillna(0).astype(int), unit="h")
    )
    positives = (
        incident.groupby(["county_key", "Date"], as_index=False)
        .size()
        .rename(columns={"size": "incident_count"})
    )

    month = month.loc[month["reported_status"].isin(["I", "Z"])].copy()
    month = month.merge(
        agencies[["agency_id", "county_key"]],
        on="agency_id",
        how="inner",
    )
    month = month.rename(columns={"inc_data_year": "data_year"})
    scaffold = _scaffold_reported_hours(month)
    return scaffold, positives.merge(county_ref, on="county_key", how="left")


def build_state_dataset(
    discovered: dict[tuple[str, int], Path],
    *,
    state: str,
    years: Iterable[int],
    output_path: Path,
    overwrite: bool,
) -> dict:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Pass --overwrite to rebuild it."
        )

    state = state.upper()
    selected_dirs: list[Path] = []
    for year in years:
        key = (state, int(year))
        if key not in discovered:
            raise FileNotFoundError(
                f"Could not locate NIBRS tables for {state}-{year} under {DEFAULT_NIBRS_ROOT}."
            )
        selected_dirs.append(discovered[key])

    scaffold_parts: list[pd.DataFrame] = []
    positive_parts: list[pd.DataFrame] = []
    for base_dir in selected_dirs:
        scaffold, positives = _aggregate_state_year(base_dir)
        scaffold_parts.append(scaffold)
        positive_parts.append(positives)

    scaffold = pd.concat(scaffold_parts, ignore_index=True)
    positives = pd.concat(positive_parts, ignore_index=True)

    county_ref = (
        positives[
            ["county_key", "state_abbr", "state_name", "county_name", "agency_count", "county_population"]
        ]
        .drop_duplicates("county_key")
        .sort_values("county_key")
        .reset_index(drop=True)
    )
    county_ref["District"] = np.arange(1, len(county_ref) + 1)

    dataset = scaffold.merge(
        county_ref[["county_key", "District", "state_abbr", "state_name", "county_name"]],
        on="county_key",
        how="left",
    )
    dataset = dataset.merge(
        positives[["county_key", "Date", "incident_count"]],
        on=["county_key", "Date"],
        how="left",
    )
    dataset["target"] = dataset["incident_count"].fillna(0).gt(0).astype(int)

    # Keep only the minimum schema needed by the existing Chicago processor.
    dataset["Ward"] = np.nan
    dataset["Community Area"] = np.nan
    dataset["Beat"] = np.nan
    dataset["Latitude"] = np.nan
    dataset["Longitude"] = np.nan
    dataset["X Coordinate"] = np.nan
    dataset["Y Coordinate"] = np.nan

    output_cols = [
        "Date",
        "District",
        "Ward",
        "Community Area",
        "Beat",
        "Latitude",
        "Longitude",
        "X Coordinate",
        "Y Coordinate",
        "target",
        "state_abbr",
        "state_name",
        "county_name",
    ]
    dataset = (
        dataset[output_cols]
        .sort_values(["Date", "District"])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    return {
        "state": state,
        "years": list(map(int, years)),
        "output_path": str(output_path),
        "rows": int(len(dataset)),
        "positive_rows": int(dataset["target"].sum()),
        "negative_rows": int((1 - dataset["target"]).sum()),
        "distinct_counties": int(dataset["District"].nunique()),
        "date_min": str(pd.to_datetime(dataset["Date"]).min()),
        "date_max": str(pd.to_datetime(dataset["Date"]).max()),
    }


def main() -> None:
    args = parse_args()
    discovered = _discover_state_year_dirs(args.nibrs_root)
    if not discovered:
        raise FileNotFoundError(f"No NIBRS table folders found under {args.nibrs_root}")

    summaries: list[dict] = []
    for state in args.states:
        output_path = args.output_dir / f"nibrs_county_hour_{state.lower()}_{min(args.years)}_{max(args.years)}.csv"
        summary = build_state_dataset(
            discovered,
            state=state,
            years=args.years,
            output_path=output_path,
            overwrite=args.overwrite,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    summary_path = args.output_dir / "nibrs_generalization_build_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(f"\nSaved build summary to: {summary_path}")


if __name__ == "__main__":
    main()
