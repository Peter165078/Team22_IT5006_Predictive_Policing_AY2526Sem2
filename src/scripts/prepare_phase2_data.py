"""
Build a single raw modeling CSV from the yearly Chicago archives.

Default temporal coverage:
    - include 2015 through 2025
    - keep full-year data unless an explicit debug cap is passed

Example:
    python src/scripts/prepare_phase2_data.py --start-year 2015 --end-year 2025
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import build_district_hour_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "apps" / "dashboard" / "split_data_by_year",
        help="Directory containing yearly chicago_crime_YYYY.csv.zip files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "chicago_crime_district_hour_2015_2025_phase2.csv",
        help="Destination CSV used by the training pipeline.",
    )
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--max-rows-per-year",
        type=int,
        default=0,
        help="Optional cap per source year for debugging only. Use 0 for the full year.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild the output file even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_district_hour_dataset(
        source_dir=args.source_dir,
        output_path=args.output_path,
        start_year=args.start_year,
        end_year=args.end_year,
        max_rows_per_year=None if args.max_rows_per_year == 0 else args.max_rows_per_year,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
