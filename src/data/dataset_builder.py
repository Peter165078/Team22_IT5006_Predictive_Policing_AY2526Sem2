from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "ID",
    "Case Number",
    "Date",
    "Block",
    "IUCR",
    "Primary Type",
    "Description",
    "Location Description",
    "Arrest",
    "Domestic",
    "Beat",
    "District",
    "Ward",
    "Community Area",
    "FBI Code",
    "X Coordinate",
    "Y Coordinate",
    "Year",
    "Updated On",
    "Latitude",
    "Longitude",
    "Location",
]


@dataclass
class BuildSummary:
    output_path: str
    years: list[int]
    rows_by_year: dict[int, int]
    total_rows: int
    sampled_rows_per_year: int | None
    columns: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_year(path: Path) -> int:
    match = re.search(r"(20\d{2})", path.name)
    if not match:
        raise ValueError(f"Could not infer year from archive name: {path.name}")
    return int(match.group(1))


def list_year_archives(source_dir: str | Path) -> list[Path]:
    base = Path(source_dir)
    return sorted(base.glob("chicago_crime_*.csv.zip"), key=_extract_year)


def _resolve_selected_archives(
    archives: list[Path],
    *,
    start_year: int | None,
    end_year: int | None,
    years: list[int] | None,
) -> list[Path]:
    available_by_year = {_extract_year(path): path for path in archives}

    if years is not None:
        requested_years = sorted(set(years))
    else:
        min_available = min(available_by_year)
        max_available = max(available_by_year)
        start = start_year if start_year is not None else min_available
        end = end_year if end_year is not None else max_available
        requested_years = list(range(start, end + 1))

    missing_years = [year for year in requested_years if year not in available_by_year]
    if missing_years:
        raise FileNotFoundError(
            "Missing yearly archives for requested years: "
            f"{missing_years}. Available years are: {sorted(available_by_year)}"
        )

    return [available_by_year[year] for year in requested_years]


def _parse_mixed_datetime(series: pd.Series) -> pd.Series:
    legacy = pd.to_datetime(
        series,
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    if legacy.notna().all():
        return legacy
    fallback = pd.to_datetime(series, errors="coerce")
    return legacy.fillna(fallback)


def _read_archives(selected: list[Path], max_rows_per_year: int | None) -> tuple[list[pd.DataFrame], dict[int, int], list[int]]:
    frames: list[pd.DataFrame] = []
    rows_by_year: dict[int, int] = {}
    selected_years: list[int] = []

    for archive in selected:
        year = _extract_year(archive)
        with zipfile.ZipFile(archive) as zipped:
            csv_members = [
                member for member in zipped.namelist()
                if member.lower().endswith(".csv") and not member.startswith("__MACOSX/")
            ]
            if len(csv_members) != 1:
                raise ValueError(
                    f"Expected exactly one CSV inside {archive.name}, found {csv_members}"
                )
            with zipped.open(csv_members[0]) as handle:
                frame = pd.read_csv(
                    handle,
                    usecols=REQUIRED_COLUMNS,
                    low_memory=False,
                )
        if max_rows_per_year is not None and len(frame) > max_rows_per_year:
            frame = frame.sample(max_rows_per_year, random_state=year).reset_index(drop=True)
        frames.append(frame)
        rows_by_year[year] = int(len(frame))
        selected_years.append(year)

    return frames, rows_by_year, selected_years


def _mode_or_nan(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.mode().iloc[0])


def build_phase2_dataset(
    source_dir: str | Path,
    output_path: str | Path,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    years: list[int] | None = None,
    max_rows_per_year: int | None = None,
    overwrite: bool = False,
) -> BuildSummary:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"{output} already exists. Pass overwrite=True to rebuild it."
        )

    archives = list_year_archives(source_dir)
    if not archives:
        raise FileNotFoundError(f"No yearly Chicago crime archives found in {source_dir}")

    selected = _resolve_selected_archives(
        archives,
        start_year=start_year,
        end_year=end_year,
        years=years,
    )

    if not selected:
        raise ValueError("No yearly archives matched the requested year filter.")

    frames, rows_by_year, selected_years = _read_archives(selected, max_rows_per_year)

    combined = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)

    return BuildSummary(
        output_path=str(output),
        years=selected_years,
        rows_by_year=rows_by_year,
        total_rows=int(len(combined)),
        sampled_rows_per_year=max_rows_per_year,
        columns=REQUIRED_COLUMNS,
    )


def build_district_hour_dataset(
    source_dir: str | Path,
    output_path: str | Path,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    years: list[int] | None = None,
    max_rows_per_year: int | None = None,
    overwrite: bool = False,
) -> BuildSummary:
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"{output} already exists. Pass overwrite=True to rebuild it."
        )

    archives = list_year_archives(source_dir)
    if not archives:
        raise FileNotFoundError(f"No yearly Chicago crime archives found in {source_dir}")

    selected = _resolve_selected_archives(
        archives,
        start_year=start_year,
        end_year=end_year,
        years=years,
    )
    frames, _, selected_years = _read_archives(selected, max_rows_per_year)
    combined = pd.concat(frames, ignore_index=True)

    combined["Date"] = _parse_mixed_datetime(combined["Date"])
    combined = combined.dropna(subset=["Date"]).copy()
    combined["District"] = pd.to_numeric(combined["District"], errors="coerce")
    combined = combined.dropna(subset=["District"]).copy()
    combined["District"] = combined["District"].astype(int)
    combined["_hour"] = combined["Date"].dt.floor("h")

    district_profiles = (
        combined.groupby("District", as_index=False)
        .agg(
            **{
                "Ward": ("Ward", _mode_or_nan),
                "Community Area": ("Community Area", _mode_or_nan),
                "Beat": ("Beat", _mode_or_nan),
                "Latitude": ("Latitude", "median"),
                "Longitude": ("Longitude", "median"),
                "X Coordinate": ("X Coordinate", "median"),
                "Y Coordinate": ("Y Coordinate", "median"),
            }
        )
    )

    hourly_counts = (
        combined.groupby(["District", "_hour"], as_index=False)
        .size()
        .rename(columns={"_hour": "Date", "size": "incident_count"})
    )

    all_hours = pd.date_range(
        start=hourly_counts["Date"].min(),
        end=hourly_counts["Date"].max(),
        freq="h",
    )
    all_districts = np.sort(district_profiles["District"].unique())
    unit_grid = pd.MultiIndex.from_product(
        [all_districts, all_hours],
        names=["District", "Date"],
    ).to_frame(index=False)

    modeling_df = unit_grid.merge(hourly_counts, on=["District", "Date"], how="left")
    modeling_df["target"] = modeling_df["incident_count"].fillna(0).gt(0).astype(int)
    modeling_df = modeling_df.drop(columns=["incident_count"])
    modeling_df = modeling_df.merge(district_profiles, on="District", how="left")
    modeling_df["Year"] = modeling_df["Date"].dt.year
    modeling_df["Date"] = modeling_df["Date"].dt.strftime("%m/%d/%Y %I:%M:%S %p")

    output_columns = [
        "Date",
        "District",
        "Ward",
        "Community Area",
        "Beat",
        "Latitude",
        "Longitude",
        "X Coordinate",
        "Y Coordinate",
        "Year",
        "target",
    ]
    modeling_df = modeling_df[output_columns].sort_values(["Date", "District"]).reset_index(drop=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    modeling_df.to_csv(output, index=False)

    rows_by_year = (
        modeling_df.groupby("Year")
        .size()
        .astype(int)
        .to_dict()
    )

    return BuildSummary(
        output_path=str(output),
        years=selected_years,
        rows_by_year=rows_by_year,
        total_rows=int(len(modeling_df)),
        sampled_rows_per_year=max_rows_per_year,
        columns=output_columns,
    )
