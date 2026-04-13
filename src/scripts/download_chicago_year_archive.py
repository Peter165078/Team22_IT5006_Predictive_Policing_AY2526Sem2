"""
Download a single-year Chicago crime archive and package it into the project's
expected ZIP format.

The script queries the official City of Chicago Socrata dataset and aliases the
columns back to the same headers used by the existing yearly ZIP archives.

Example:
    python src/scripts/download_chicago_year_archive.py --year 2025
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import urlopen
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATASET_ID = "ijzp-q8t2"
BASE_URL = f"https://data.cityofchicago.org/resource/{DATASET_ID}.csv"

COLUMN_ALIASES = [
    ("id", "ID"),
    ("case_number", "Case Number"),
    ("date", "Date"),
    ("block", "Block"),
    ("iucr", "IUCR"),
    ("primary_type", "Primary Type"),
    ("description", "Description"),
    ("location_description", "Location Description"),
    ("arrest", "Arrest"),
    ("domestic", "Domestic"),
    ("beat", "Beat"),
    ("district", "District"),
    ("ward", "Ward"),
    ("community_area", "Community Area"),
    ("fbi_code", "FBI Code"),
    ("x_coordinate", "X Coordinate"),
    ("y_coordinate", "Y Coordinate"),
    ("year", "Year"),
    ("updated_on", "Updated On"),
    ("latitude", "Latitude"),
    ("longitude", "Longitude"),
    ("location", "Location"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, help="Year to download.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "apps" / "dashboard" / "split_data_by_year",
        help="Directory where chicago_crime_<year>.csv.zip will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2000000,
        help="Safety limit passed to the SoQL query.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing yearly ZIP if it is already present.",
    )
    return parser.parse_args()


def build_url(year: int, limit: int) -> str:
    select_clause = ",".join(machine for machine, _ in COLUMN_ALIASES)
    return (
        f"{BASE_URL}?"
        f"$select={quote_plus(select_clause)}&"
        f"$where={quote_plus(f'year = {year}')}&"
        f"$order={quote_plus('date')}&"
        f"$limit={limit}"
    )


def download_year_csv(year: int, destination: Path, limit: int) -> None:
    url = build_url(year, limit)
    with urlopen(url) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def rewrite_headers(raw_csv_path: Path, normalized_csv_path: Path) -> None:
    header_map = {machine: human for machine, human in COLUMN_ALIASES}
    output_headers = [human for _, human in COLUMN_ALIASES]

    with raw_csv_path.open("r", encoding="utf-8", newline="") as src_handle, normalized_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst_handle:
        reader = csv.DictReader(src_handle)
        writer = csv.DictWriter(dst_handle, fieldnames=output_headers)
        writer.writeheader()
        for row in reader:
            normalized_row = {}
            for machine, human in COLUMN_ALIASES:
                value = row.get(machine, "")
                if human in {"Date", "Updated On"}:
                    value = normalize_datetime_text(value)
                normalized_row[human] = value
            writer.writerow(normalized_row)


def normalize_datetime_text(value: str) -> str:
    if not value:
        return value
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.strftime("%m/%d/%Y %I:%M:%S %p")
    except ValueError:
        return value


def package_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(csv_path, arcname=csv_path.name)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / f"chicago_crime_{args.year}.csv.zip"
    if zip_path.exists() and not args.overwrite:
        raise FileExistsError(f"{zip_path} already exists. Pass --overwrite to replace it.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_csv_path = Path(tmp_dir) / f"chicago_crime_{args.year}_raw.csv"
        csv_path = Path(tmp_dir) / f"chicago_crime_{args.year}.csv"
        print(f"Downloading Chicago crime data for {args.year} ...")
        download_year_csv(args.year, raw_csv_path, args.limit)

        if raw_csv_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded CSV for {args.year} is empty.")

        rewrite_headers(raw_csv_path, csv_path)
        package_zip(csv_path, zip_path)
        print(f"Saved yearly archive to {zip_path}")


if __name__ == "__main__":
    main()
