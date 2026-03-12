from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
import zipfile

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

    if years is None:
        selected = []
        for archive in archives:
            year = _extract_year(archive)
            if start_year is not None and year < start_year:
                continue
            if end_year is not None and year > end_year:
                continue
            selected.append(archive)
    else:
        year_set = set(years)
        selected = [archive for archive in archives if _extract_year(archive) in year_set]

    if not selected:
        raise ValueError("No yearly archives matched the requested year filter.")

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
