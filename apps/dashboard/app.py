from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import zipfile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_NAME = "HistGradientBoosting"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "hist_gradient_boosting.pkl"
PREDICTION_SOURCE_DIR = PROJECT_ROOT / "apps" / "dashboard" / "split_data_by_year"
PREDICTION_DATA_PATH = Path(tempfile.gettempdir()) / "team22_district_hour_prediction_data.csv"
MIN_PREDICTION_DATE = pd.Timestamp("2025-01-01")
MAX_PREDICTION_DATE = pd.Timestamp("2025-12-31")
NIBRS_DEMO_BASE_DIR = PROJECT_ROOT / "NIBRS data"
NIBRS_METRICS_PATH = (
    PROJECT_ROOT / "artifacts" / "metrics" / "nibrs_generalization" / "nibrs_generalization_metrics_2024.csv"
)
NIBRS_BUILD_SUMMARY_PATH = PROJECT_ROOT / "data" / "raw" / "nibrs_generalization_build_summary.json"

STATE_CENTROIDS = {
    "CO": (39.55, -105.78),
    "TX": (31.05, -97.56),
}

NIBRS_OFFENSE_MAP = {
    "Larceny/Theft Offenses": "THEFT",
    "Assault Offenses": "BATTERY",
}

st.set_page_config(
    page_title="Chicago Crime Intel",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Welcome"
if "selected_year" not in st.session_state:
    st.session_state.selected_year = 2024

st.markdown(
    """
    <style>
    .main { background-color: #f4f6f9; }
    .launch-card {
        background: linear-gradient(160deg, #ffffff 0%, #f8fafc 100%);
        padding: 38px;
        border-radius: 24px;
        box-shadow: 0 22px 50px rgba(15, 23, 42, 0.10);
        text-align: center;
        max-width: 720px;
        margin: 0 auto;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    .sub-card {
        background-color: white;
        padding: 20px 22px;
        border-radius: 18px;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        border: 1px solid rgba(148, 163, 184, 0.14);
        height: 100%;
    }
    div.metric-container {
        background-color: white;
        padding: 15px 20px;
        border-radius: 14px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.06);
    }
    .prediction-card {
        background: linear-gradient(160deg, #ffffff 0%, #eff6ff 100%);
        border-radius: 18px;
        padding: 22px 24px;
        border: 1px solid rgba(59, 130, 246, 0.18);
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.10);
    }
    .risk-chip {
        display: inline-block;
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .risk-low { background: rgba(16, 185, 129, 0.12); color: #047857; }
    .risk-medium { background: rgba(245, 158, 11, 0.14); color: #b45309; }
    .risk-high { background: rgba(239, 68, 68, 0.14); color: #b91c1c; }
    .js-plotly-plot .plotly .modebar { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


def current_file_dir() -> Path:
    return Path(__file__).resolve().parent


def get_file_path(year: int) -> str | None:
    possible_paths = [
        current_file_dir() / "split_data_by_year" / f"chicago_crime_{year}.csv.zip",
        current_file_dir() / "split_data_by_year" / f"chicago_crime_{year}.zip",
        current_file_dir() / f"chicago_crime_{year}.csv.zip",
        current_file_dir() / f"chicago_crime_{year}.zip",
    ]
    for path in possible_paths:
        if path.exists():
            return str(path)
    return None


@st.cache_data(show_spinner=False)
def load_dashboard_data(year: int) -> pd.DataFrame | None:
    found_path = get_file_path(year)
    if not found_path:
        return None
    try:
        with zipfile.ZipFile(found_path, "r") as zipped:
            csv_files = [
                name for name in zipped.namelist()
                if name.endswith(".csv") and not name.startswith("__MACOSX")
            ]
            if not csv_files:
                return None
            with zipped.open(csv_files[0]) as handle:
                columns = [
                    "Date",
                    "Primary Type",
                    "Description",
                    "Arrest",
                    "District",
                    "Latitude",
                    "Longitude",
                    "Location Description",
                ]
                df = pd.read_csv(handle, usecols=columns)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month_Num"] = df["Date"].dt.month
        df["Hour"] = df["Date"].dt.hour
        df["DayOfWeek"] = df["Date"].dt.day_name()
        return df
    except Exception as exc:
        st.error(f"Error reading dashboard data: {exc}")
        return None


def metric_card(title: str, value: str, sub: str, color: str) -> None:
    st.markdown(
        f"""
        <div class="metric-container" style="border-left: 5px solid {color};">
            <p style="font-size:14px; color:#6b7280; margin:0;">{title}</p>
            <p style="font-size:26px; font-weight:700; color:#111827; margin:5px 0;">{value}</p>
            <p style="font-size:12px; color:{color}; margin:0;">{sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_nibrs_generalization_results() -> tuple[pd.DataFrame | None, dict[str, dict]]:
    if not NIBRS_METRICS_PATH.exists():
        return None, {}

    df = pd.read_csv(NIBRS_METRICS_PATH)
    if df.empty:
        return df, {}

    df["state"] = df["dataset"].str.extract(r"_([a-z]{2})_").iloc[:, 0].str.upper()
    df["label"] = df["state"].map({"TX": "Texas", "CO": "Colorado"}).fillna(df["state"])
    df["auroc_pct"] = (df["auroc"] * 100).round(1)
    df["auprc_pct"] = (df["auprc"] * 100).round(1)
    df["positive_rate_pct"] = (df["positive_rate"] * 100).round(1)

    build_summary: dict[str, dict] = {}
    if NIBRS_BUILD_SUMMARY_PATH.exists():
        summary_df = pd.read_json(NIBRS_BUILD_SUMMARY_PATH)
        for row in summary_df.to_dict(orient="records"):
            build_summary[str(row["state"]).upper()] = row
    return df, build_summary


def _dataset_display_name(path: Path) -> str:
    rel = path.relative_to(NIBRS_DEMO_BASE_DIR)
    return str(rel).replace("\\", "/")


@st.cache_data(show_spinner=False)
def discover_nibrs_demo_datasets() -> list[str]:
    if not NIBRS_DEMO_BASE_DIR.exists():
        return []

    datasets: list[str] = []
    for child in sorted(NIBRS_DEMO_BASE_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if (child / "NIBRS_incident.csv").exists():
            datasets.append(_dataset_display_name(child))
            continue
        for nested in sorted(child.iterdir()):
            if nested.is_dir() and (nested / "NIBRS_incident.csv").exists():
                datasets.append(_dataset_display_name(nested))
    return datasets


def _resolve_nibrs_demo_dataset_path(dataset_name: str) -> Path:
    return NIBRS_DEMO_BASE_DIR / Path(dataset_name)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to decode {path} using utf-8/latin-1/cp1252")


@st.cache_data(show_spinner="Loading NIBRS agency dataset...")
def load_nibrs_demo_bundle(dataset_name: str) -> dict:
    dataset_dir = _resolve_nibrs_demo_dataset_path(dataset_name)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"NIBRS dataset folder not found: {dataset_dir}")

    file_map = {
        "incident": "NIBRS_incident.csv",
        "offense": "NIBRS_OFFENSE.csv",
        "offense_type": "NIBRS_OFFENSE_TYPE.csv",
        "agencies": "agencies.csv",
    }

    tables: dict[str, pd.DataFrame] = {}
    for key, filename in file_map.items():
        file_path = dataset_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing {filename} in {dataset_dir}")
        tables[key] = _read_csv_safe(file_path)

    offense = tables["offense"][["offense_id", "incident_id", "offense_code"]].copy()
    incident = tables["incident"][["incident_id", "agency_id", "incident_date"]].copy()
    offense_type = tables["offense_type"][["offense_code", "offense_category_name"]].copy()
    agencies = tables["agencies"][["agency_id", "pub_agency_name", "state_name", "state_abbr"]].drop_duplicates(
        subset=["agency_id"]
    ).copy()

    joined = offense.merge(incident, on="incident_id", how="inner")
    joined = joined.merge(offense_type, on="offense_code", how="left")
    joined = joined.merge(agencies, on="agency_id", how="left")
    joined["Primary Type"] = joined["offense_category_name"].map(NIBRS_OFFENSE_MAP).fillna("OTHER")
    joined["Date"] = pd.to_datetime(joined["incident_date"], format="mixed", errors="coerce")
    joined.dropna(subset=["Date", "agency_id", "pub_agency_name"], inplace=True)
    joined["agency_id"] = joined["agency_id"].astype(int)

    agency_counts = joined.groupby("agency_id").size().reset_index(name="count").sort_values(
        "count", ascending=False
    )
    top_agencies = agency_counts.head(77)["agency_id"].tolist()
    joined = joined[joined["agency_id"].isin(top_agencies)].copy()

    agency_to_region = {agency_id: idx + 1 for idx, agency_id in enumerate(top_agencies)}
    joined["Community Area"] = joined["agency_id"].map(agency_to_region)

    agency_meta = (
        joined[["agency_id", "pub_agency_name", "state_name", "state_abbr"]]
        .drop_duplicates(subset=["agency_id"])
        .copy()
    )
    agency_meta["region_id"] = agency_meta["agency_id"].map(agency_to_region)
    agency_meta.sort_values("region_id", inplace=True)
    agency_meta.reset_index(drop=True, inplace=True)

    state_abbr = (
        str(agency_meta["state_abbr"].dropna().iloc[0]).upper()
        if "state_abbr" in agency_meta.columns and not agency_meta["state_abbr"].dropna().empty
        else dataset_name.split("/")[0].split("-")[0].upper()
    )
    center_lat, center_lon = STATE_CENTROIDS.get(state_abbr, (39.83, -98.58))
    agency_meta["latitude"] = [
        center_lat + ((hash(f"{name}|lat") % 1000 - 500) / 500) * 2.0
        for name in agency_meta["pub_agency_name"].astype(str)
    ]
    agency_meta["longitude"] = [
        center_lon + ((hash(f"{name}|lon") % 1000 - 500) / 500) * 3.0
        for name in agency_meta["pub_agency_name"].astype(str)
    ]

    aligned = pd.DataFrame(
        {
            "ID": joined["offense_id"].values,
            "Date": joined["Date"].values,
            "Community Area": joined["Community Area"].values,
            "Primary Type": joined["Primary Type"].values,
        }
    )

    return {
        "aligned": aligned,
        "agency_meta": agency_meta,
        "state_abbr": state_abbr,
        "center": {"lat": center_lat, "lon": center_lon},
    }


def get_nibrs_available_dates(aligned_df: pd.DataFrame, window_size: int = 7) -> list[str]:
    working = aligned_df.copy()
    working["_date"] = pd.to_datetime(working["Date"]).dt.date
    all_dates = sorted(working["_date"].unique())
    if len(all_dates) <= window_size:
        return []
    return [str(day) for day in all_dates[window_size:]]


def preview_predict_nibrs(bundle: dict, target_date: str) -> pd.DataFrame:
    aligned = bundle["aligned"].copy()
    aligned["Date"] = pd.to_datetime(aligned["Date"]).dt.date
    agency_meta = bundle["agency_meta"].copy()
    target = pd.Timestamp(target_date).date()
    history_dates = [target - pd.Timedelta(days=offset) for offset in range(7, 0, -1)]

    daily_counts = aligned.groupby(["Community Area", "Date"]).size().rename("count").reset_index()

    rows: list[dict] = []
    for _, row in agency_meta.iterrows():
        region_id = int(row["region_id"])
        history_values: list[float] = []
        for history_date in history_dates:
            match = daily_counts.loc[
                (daily_counts["Community Area"] == region_id) & (daily_counts["Date"] == history_date),
                "count",
            ]
            history_values.append(float(match.iloc[0]) if not match.empty else 0.0)

        actual_match = daily_counts.loc[
            (daily_counts["Community Area"] == region_id) & (daily_counts["Date"] == target),
            "count",
        ]
        predicted = round(sum(history_values) / len(history_values), 2) if history_values else 0.0
        actual = float(actual_match.iloc[0]) if not actual_match.empty else 0.0
        rows.append(
            {
                "region_id": region_id,
                "agency_name": str(row["pub_agency_name"]),
                "state": str(row.get("state_name", "")),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "predicted": predicted,
                "actual": actual,
            }
        )

    return pd.DataFrame(rows).sort_values("predicted", ascending=False).reset_index(drop=True)


def render_agency_leaflet_map(results: pd.DataFrame, center: dict[str, float]) -> None:
    agencies = results[
        ["agency_name", "predicted", "actual", "region_id", "latitude", "longitude"]
    ].to_dict(orient="records")
    payload = json.dumps(agencies)
    center_json = json.dumps(center)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <style>
        html, body, #map {{
          margin: 0;
          padding: 0;
          width: 100%;
          height: 560px;
          background: #faf9f7;
        }}
        .leaflet-container {{
          font-family: "DM Sans", system-ui, sans-serif;
        }}
        .bar-marker {{
          position: relative;
          display: flex;
          align-items: flex-end;
          gap: 3px;
        }}
        .bar-col {{
          width: 10px;
          border-radius: 2px 2px 0 0;
          min-height: 2px;
          box-shadow: 0 1px 2px rgba(0,0,0,0.18);
        }}
        .bar-pred {{ background: #c07a50; }}
        .bar-actual {{ background: #6b8eae; opacity: 0.82; }}
        .leaflet-popup-content-wrapper {{
          border-radius: 12px;
          box-shadow: 0 6px 18px rgba(15, 23, 42, 0.12);
        }}
        .leaflet-popup-content {{
          margin: 12px 14px;
          font-size: 13px;
          line-height: 1.55;
        }}
        .popup-name {{
          font-weight: 700;
          margin-bottom: 6px;
          color: #1f2937;
        }}
        .popup-row {{
          display: flex;
          justify-content: space-between;
          gap: 18px;
        }}
        .popup-label {{ color: #6b7280; }}
        .popup-pred {{ color: #c07a50; font-weight: 700; }}
        .popup-actual {{ color: #6b8eae; font-weight: 700; }}
        .map-legend {{
          position: absolute;
          right: 18px;
          bottom: 18px;
          z-index: 999;
          background: rgba(255,255,255,0.94);
          border: 1px solid #e5e7eb;
          border-radius: 14px;
          padding: 12px 14px;
          box-shadow: 0 8px 18px rgba(15, 23, 42, 0.10);
          font-size: 12px;
          color: #334155;
        }}
        .legend-item {{
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 4px;
        }}
        .legend-item:last-child {{ margin-bottom: 0; }}
        .legend-swatch {{
          width: 10px;
          height: 10px;
          border-radius: 2px;
        }}
      </style>
    </head>
    <body>
      <div id="map"></div>
      <div class="map-legend">
        <div class="legend-item"><span class="legend-swatch" style="background:#c07a50"></span><span>Predicted crime count</span></div>
        <div class="legend-item"><span class="legend-swatch" style="background:#6b8eae"></span><span>Actual crime count</span></div>
      </div>
      <script>
        const agencies = {payload};
        const center = {center_json};

        const map = L.map('map', {{
          zoomControl: true,
          attributionControl: true
        }}).setView([center.lat, center.lon], 5.5);

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
          maxZoom: 18
        }}).addTo(map);

        const maxVal = Math.max(1, ...agencies.map(a => Math.max(a.predicted || 0, a.actual || 0)));
        const markerLayer = L.layerGroup().addTo(map);

        agencies.forEach((agency) => {{
          const predH = Math.max(2, (agency.predicted / maxVal) * 48);
          const actH = Math.max(2, (agency.actual / maxVal) * 48);
          const html = `
            <div class="bar-marker" style="height:48px; width:24px;">
              <div class="bar-col bar-pred" style="height:${{predH}}px;"></div>
              <div class="bar-col bar-actual" style="height:${{actH}}px;"></div>
            </div>`;

          const icon = L.divIcon({{
            html,
            className: '',
            iconSize: [24, 48],
            iconAnchor: [12, 48]
          }});

          const marker = L.marker([agency.latitude, agency.longitude], {{ icon }});
          const popup = `
            <div class="popup-name">${{agency.agency_name}}</div>
            <div class="popup-row"><span class="popup-label">Predicted</span><span class="popup-pred">${{agency.predicted.toFixed(2)}}</span></div>
            <div class="popup-row"><span class="popup-label">Actual</span><span class="popup-actual">${{agency.actual.toFixed(2)}}</span></div>
            <div class="popup-row"><span class="popup-label">Region</span><span>#${{agency.region_id}}</span></div>
          `;
          marker.bindPopup(popup, {{ maxWidth: 220 }});
          markerLayer.addLayer(marker);
        }});
      </script>
    </body>
    </html>
    """

    components.html(html, height=560)


def build_prediction_dataset() -> None:
    from src.data.dataset_builder import build_district_hour_dataset

    if PREDICTION_DATA_PATH.exists():
        return
    build_district_hour_dataset(
        source_dir=PREDICTION_SOURCE_DIR,
        output_path=PREDICTION_DATA_PATH,
        start_year=2015,
        end_year=2025,
        max_rows_per_year=None,
        overwrite=True,
    )


@st.cache_resource(show_spinner="Preparing prediction engine...")
def load_prediction_engine() -> dict:
    from src.data.processor import DataProcessor

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing hist_gradient_boosting.pkl. Keep the model artifact in artifacts/models/ "
            "for the prediction demo to run."
        )

    build_prediction_dataset()
    processor = DataProcessor(str(PREDICTION_DATA_PATH), neg_ratio=1.0, random_state=42)
    processor.load_and_split()
    X_train, _ = processor.fit_transform_train()
    X_val, _ = processor.transform(processor.val_idx)

    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)

    val_prob = model.predict_proba(X_val)[:, 1]
    low_threshold = float(pd.Series(val_prob).quantile(0.33))
    high_threshold = float(pd.Series(val_prob).quantile(0.66))

    raw = processor.raw_data.copy()
    positive_raw = raw.loc[processor.labels == 1].copy()
    district_reference = (
        raw.groupby("District", dropna=True)
        .agg(
            Ward=("Ward", "median"),
            Community_Area=("Community Area", "median"),
            Beat=("Beat", "median"),
            Latitude=("Latitude", "median"),
            Longitude=("Longitude", "median"),
            X_Coordinate=("X Coordinate", "median"),
            Y_Coordinate=("Y Coordinate", "median"),
        )
        .reset_index()
    )
    district_reference["District"] = district_reference["District"].astype(int)

    return {
        "model": model,
        "processor": processor,
        "feature_columns": X_train.columns.tolist(),
        "thresholds": (low_threshold, high_threshold),
        "district_reference": district_reference,
        "positive_raw": positive_raw,
    }


def get_district_defaults(engine: dict, district: int) -> dict:
    reference = engine["district_reference"]
    match = reference.loc[reference["District"] == district]
    if match.empty:
        return {
            "Ward": None,
            "Community Area": None,
            "Beat": None,
            "Latitude": None,
            "Longitude": None,
            "X Coordinate": None,
            "Y Coordinate": None,
        }
    row = match.iloc[0]
    return {
        "Ward": None if pd.isna(row["Ward"]) else int(round(row["Ward"])),
        "Community Area": None if pd.isna(row["Community_Area"]) else int(round(row["Community_Area"])),
        "Beat": None if pd.isna(row["Beat"]) else int(round(row["Beat"])),
        "Latitude": None if pd.isna(row["Latitude"]) else float(row["Latitude"]),
        "Longitude": None if pd.isna(row["Longitude"]) else float(row["Longitude"]),
        "X Coordinate": None if pd.isna(row["X_Coordinate"]) else float(row["X_Coordinate"]),
        "Y Coordinate": None if pd.isna(row["Y_Coordinate"]) else float(row["Y_Coordinate"]),
    }


def classify_risk(probability: float, thresholds: tuple[float, float]) -> tuple[str, str]:
    low_threshold, high_threshold = thresholds
    if probability < low_threshold:
        return "Low Risk", "risk-low"
    if probability < high_threshold:
        return "Medium Risk", "risk-medium"
    return "High Risk", "risk-high"


def compute_recent_district_counts(positive_raw: pd.DataFrame, target_dt: pd.Timestamp, district: int) -> dict[str, int]:
    district_rows = positive_raw.loc[
        positive_raw["District"].fillna(-1).astype(int) == int(district)
    ].copy()
    if district_rows.empty:
        return {"7d": 0, "30d": 0, "90d": 0}

    district_rows["Date"] = pd.to_datetime(district_rows["Date"])
    before = district_rows.loc[district_rows["Date"] < target_dt]

    def count_within(days: int) -> int:
        start = target_dt - pd.Timedelta(days=days)
        return int(before.loc[before["Date"] >= start].shape[0])

    return {
        "7d": count_within(7),
        "30d": count_within(30),
        "90d": count_within(90),
    }


def build_district_prediction_payload(
    engine: dict,
    target_dt: pd.Timestamp,
    *,
    hours: list[int] | None = None,
) -> pd.DataFrame:
    reference = engine["district_reference"].copy()
    if hours is None:
        timestamps = pd.DataFrame({"Date": [target_dt]})
    else:
        base_day = pd.Timestamp(target_dt).normalize()
        timestamps = pd.DataFrame(
            {"Date": [base_day + pd.Timedelta(hours=int(hour)) for hour in hours]}
        )

    reference["_key"] = 1
    timestamps["_key"] = 1
    payload = reference.merge(timestamps, on="_key", how="inner").drop(columns="_key")
    payload = payload.rename(columns={"Community_Area": "Community Area"})
    payload["Ward"] = payload["Ward"].round().astype(int)
    payload["Community Area"] = payload["Community Area"].round().astype(int)
    payload["Beat"] = payload["Beat"].round().astype(int)
    return payload[
        [
            "Date",
            "District",
            "Ward",
            "Community Area",
            "Beat",
            "Latitude",
            "Longitude",
            "X_Coordinate",
            "Y_Coordinate",
        ]
    ].rename(
        columns={
            "X_Coordinate": "X Coordinate",
            "Y_Coordinate": "Y Coordinate",
        }
    )


def score_prediction_payload(engine: dict, payload: pd.DataFrame) -> pd.DataFrame:
    X_demo, surviving_mask, _ = engine["processor"]._process(
        payload.copy(),
        pd.Series([0] * len(payload)),
        is_train=False,
    )
    scored = payload.loc[surviving_mask].reset_index(drop=True)
    if scored.empty:
        return scored

    X_demo = X_demo.reindex(columns=engine["feature_columns"], fill_value=0)
    probabilities = engine["model"].predict_proba(X_demo)[:, 1]
    scored["probability"] = probabilities

    risk_labels: list[str] = []
    risk_classes: list[str] = []
    for probability in probabilities:
        label, css_class = classify_risk(float(probability), engine["thresholds"])
        risk_labels.append(label)
        risk_classes.append(css_class)
    scored["risk_label"] = risk_labels
    scored["risk_class"] = risk_classes
    return scored


def render_welcome() -> None:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div class="launch-card">
                <h1 style="font-size: 3em; margin-bottom: 12px;">🚔 Chicago Crime Intel</h1>
                <p style="color: #6b7280; font-size: 1.15em; margin-bottom: 18px;">
                    IT5006 Predictive Policing Project
                </p>
                <p style="color: #475569; font-size: 1.02em; margin-bottom: 0;">
                    Explore historical crime patterns, then test a prediction-oriented
                    demo built on our best-performing benchmark model.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        row_one_left, row_one_right = st.columns(2)
        if row_one_left.button("📊 Open Dashboard", type="primary", use_container_width=True):
            st.session_state.app_mode = "Dashboard"
            st.rerun()
        if row_one_right.button("🤖 Prediction Demo", use_container_width=True):
            st.session_state.app_mode = "Prediction Demo"
            st.rerun()

        row_two_left, row_two_right = st.columns(2)
        if row_two_left.button("📍 High-Risk Places", use_container_width=True):
            st.session_state.app_mode = "High-Risk Places"
            st.rerun()
        if row_two_right.button("👥 Group Pattern Analysis", use_container_width=True):
            st.session_state.app_mode = "Group Pattern Analysis"
            st.rerun()

        row_three_left, row_three_right = st.columns(2)
        if row_three_left.button("🌎 NIBRS Generalization", use_container_width=True):
            st.session_state.app_mode = "NIBRS Generalization"
            st.rerun()
        if row_three_right.button("🗺️ Agency Map Demo", use_container_width=True):
            st.session_state.app_mode = "Agency Map Demo"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sub-card">
                <strong>What this app demonstrates</strong>
                <ul>
                    <li>Interactive crime analytics dashboard</li>
                    <li>Model-backed risk prediction workflow</li>
                    <li>District-level hotspot ranking for a chosen hour</li>
                    <li>Aggregate daily risk-pattern analysis across districts</li>
                    <li>NIBRS agency-map demo inspired by the JIaDLu MVP</li>
                    <li>Basic user input validation and interpretable output</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<br><br><p style='text-align: center; color: #94a3b8;'>© Team 22 | Streamlit demo for Phase 3</p>",
        unsafe_allow_html=True,
    )


def render_dashboard() -> None:
    year = st.session_state.selected_year
    df = load_dashboard_data(year)
    if df is None or df.empty:
        st.error(f"Unable to load {year} dashboard data.")
        if st.button("← Back to Home"):
            st.session_state.app_mode = "Welcome"
            st.rerun()
        st.stop()

    with st.sidebar:
        st.title("Navigation")
        if st.button("← Home", use_container_width=True):
            st.session_state.app_mode = "Welcome"
            st.rerun()
        if st.button("🤖 Prediction Demo", use_container_width=True):
            st.session_state.app_mode = "Prediction Demo"
            st.rerun()
        if st.button("📍 High-Risk Places", use_container_width=True):
            st.session_state.app_mode = "High-Risk Places"
            st.rerun()
        if st.button("👥 Group Pattern Analysis", use_container_width=True):
            st.session_state.app_mode = "Group Pattern Analysis"
            st.rerun()
        if st.button("🌎 NIBRS Generalization", use_container_width=True):
            st.session_state.app_mode = "NIBRS Generalization"
            st.rerun()
        if st.button("🗺️ Agency Map Demo", use_container_width=True):
            st.session_state.app_mode = "Agency Map Demo"
            st.rerun()
        st.divider()
        st.title(f"Controls ({year})")
        st.success(f"Loaded {len(df):,} rows")
        available_years = [y for y in range(2014, 2026) if get_file_path(y)]
        selected_year = st.selectbox("Dashboard year", options=available_years, index=available_years.index(year))
        if selected_year != year:
            st.session_state.selected_year = selected_year
            st.rerun()

        all_types = sorted(df["Primary Type"].dropna().unique())
        default_types = ["THEFT", "BATTERY", "CRIMINAL DAMAGE", "ASSAULT"]
        selected_types = st.multiselect(
            "Crime type",
            all_types,
            default=[crime for crime in default_types if crime in all_types],
        )
        districts = sorted([int(item) for item in df["District"].dropna().unique()])
        selected_districts = st.multiselect("Police district", districts, default=[])
        arrest_status = st.radio("Arrest status", ["All", "Yes", "No"], horizontal=True)

    mask = df["Primary Type"].isin(selected_types)
    if selected_districts:
        mask &= df["District"].isin(selected_districts)
    if arrest_status == "Yes":
        mask &= df["Arrest"] == True
    if arrest_status == "No":
        mask &= df["Arrest"] == False
    filtered_df = df.loc[mask].copy()

    st.title(f"Chicago Crime Dashboard | {year}")
    st.caption("Interactive historical exploration for the Phase 3 presentation demo.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Incidents", f"{len(filtered_df):,}", "Filtered volume", "#3b82f6")
    with col2:
        arrest_rate = filtered_df["Arrest"].mean() * 100 if not filtered_df.empty else 0
        metric_card("Arrest Rate", f"{arrest_rate:.1f}%", "Observed enforcement rate", "#10b981")
    with col3:
        top_location = filtered_df["Location Description"].mode()[0] if not filtered_df.empty else "N/A"
        metric_card("Top Location", f"{top_location[:16]}...", "Most common location", "#f59e0b")
    with col4:
        peak_hour = filtered_df["Hour"].mode()[0] if not filtered_df.empty else "N/A"
        metric_card("Peak Hour", f"{peak_hour}:00", "Most frequent hour", "#ef4444")

    st.markdown("---")
    map_col, chart_col = st.columns([1.8, 1])
    with map_col:
        st.subheader("Spatial Distribution")
        if not filtered_df.empty:
            map_data = filtered_df.dropna(subset=["Latitude", "Longitude"]).copy()
            if len(map_data) > 20_000:
                map_data = map_data.sample(20_000, random_state=42)
                st.caption(f"Displaying a random 20,000 points out of {len(filtered_df):,} incidents for performance.")
            st.pydeck_chart(
                pdk.Deck(
                    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                    initial_view_state=pdk.ViewState(latitude=41.85, longitude=-87.65, zoom=10),
                    layers=[
                        pdk.Layer(
                            "ScatterplotLayer",
                            data=map_data,
                            get_position="[Longitude, Latitude]",
                            get_color="[200, 30, 0, 150]",
                            get_radius=40,
                            pickable=True,
                        )
                    ],
                    tooltip={"text": "{Primary Type}\n{Description}"},
                )
            )
        else:
            st.warning("No data available for the selected filters.")

    with chart_col:
        st.subheader("Monthly Trend")
        if not filtered_df.empty:
            trend = filtered_df.groupby("Month_Num").size().reset_index(name="Count")
            trend["Month"] = trend["Month_Num"].apply(lambda month: pd.Timestamp(2024, int(month), 1).strftime("%b"))
            st.plotly_chart(
                px.area(trend, x="Month", y="Count", markers=True).update_layout(
                    height=250, margin=dict(l=0, r=0, t=10, b=0)
                ),
                use_container_width=True,
            )

        st.subheader("Top Crime Types")
        if not filtered_df.empty:
            top_types = filtered_df["Primary Type"].value_counts().head(5).reset_index()
            top_types.columns = ["Type", "Count"]
            st.plotly_chart(
                px.bar(top_types, x="Count", y="Type", orientation="h", color="Count").update_layout(
                    height=250, margin=dict(l=0, r=0, t=0, b=0), showlegend=False
                ),
                use_container_width=True,
            )

    st.markdown("---")
    st.subheader("Temporal Heatmap")
    if not filtered_df.empty:
        heat = filtered_df.groupby(["DayOfWeek", "Hour"]).size().reset_index(name="Counts")
        ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap = px.density_heatmap(
            heat,
            x="Hour",
            y="DayOfWeek",
            z="Counts",
            category_orders={"DayOfWeek": ordered_days},
            color_continuous_scale="Reds",
            nbinsx=24,
            nbinsy=7,
        )
        heatmap.update_layout(height=360, margin=dict(l=0, r=0, t=24, b=0), xaxis=dict(dtick=1))
        st.plotly_chart(heatmap, use_container_width=True)


def render_prediction_demo() -> None:
    st.title("Prediction Demo")
    st.caption(
        "Minimal deployment-style workflow using the best-performing HistGradientBoosting model. "
        "Users provide a location and time, then the app returns a risk estimate with basic validation."
    )

    try:
        engine = load_prediction_engine()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.title("Navigation")
        if st.button("← Home", use_container_width=True):
            st.session_state.app_mode = "Welcome"
            st.rerun()
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.app_mode = "Dashboard"
            st.rerun()
        if st.button("📍 High-Risk Places", use_container_width=True):
            st.session_state.app_mode = "High-Risk Places"
            st.rerun()
        if st.button("👥 Group Pattern Analysis", use_container_width=True):
            st.session_state.app_mode = "Group Pattern Analysis"
            st.rerun()
        if st.button("🌎 NIBRS Generalization", use_container_width=True):
            st.session_state.app_mode = "NIBRS Generalization"
            st.rerun()
        if st.button("🗺️ Agency Map Demo", use_container_width=True):
            st.session_state.app_mode = "Agency Map Demo"
            st.rerun()
        st.divider()
        st.info(
            f"Model in use: {MODEL_NAME}\n\n"
            f"Prediction year: {MIN_PREDICTION_DATE.year} holdout window"
        )

    district_options = sorted(engine["district_reference"]["District"].dropna().astype(int).unique().tolist())
    default_district = 11 if 11 in district_options else district_options[0]
    district = st.selectbox("Police district", district_options, index=district_options.index(default_district))
    defaults = get_district_defaults(engine, district)

    def _bounded_default(value: object, fallback: float, min_value: float, max_value: float, as_int: bool = False):
        if value is None or pd.isna(value):
            numeric = float(fallback)
        else:
            numeric = float(value)
        numeric = max(min_value, min(max_value, numeric))
        return int(round(numeric)) if as_int else numeric

    left, right = st.columns([1.1, 0.9])
    with left:
        with st.form("prediction_form"):
            selected_date = st.date_input(
                "Incident date",
                value=pd.Timestamp("2025-10-01").date(),
                min_value=MIN_PREDICTION_DATE.date(),
                max_value=MAX_PREDICTION_DATE.date(),
            )
            selected_hour = st.slider("Hour of day", min_value=0, max_value=23, value=18)

            st.markdown("**Optional location refinement**")
            expander = st.expander("Edit advanced spatial fields", expanded=False)
            with expander:
                ward = st.number_input(
                    "Ward",
                    min_value=1,
                    max_value=50,
                    value=_bounded_default(defaults.get("Ward"), 1, 1, 50, as_int=True),
                    step=1,
                )
                community_area = st.number_input(
                    "Community Area",
                    min_value=1,
                    max_value=77,
                    value=_bounded_default(defaults.get("Community Area"), 1, 1, 77, as_int=True),
                    step=1,
                )
                beat = st.number_input(
                    "Beat",
                    min_value=111,
                    max_value=2535,
                    value=_bounded_default(defaults.get("Beat"), 111, 111, 2535, as_int=True),
                    step=1,
                )
                latitude = st.number_input(
                    "Latitude",
                    min_value=36.0,
                    max_value=42.0,
                    value=_bounded_default(defaults.get("Latitude"), 41.85, 36.0, 42.0),
                    format="%.6f",
                )
                longitude = st.number_input(
                    "Longitude",
                    min_value=-92.0,
                    max_value=-87.0,
                    value=_bounded_default(defaults.get("Longitude"), -87.65, -92.0, -87.0),
                    format="%.6f",
                )

            submit = st.form_submit_button("Predict Risk", type="primary", use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="sub-card">
                <strong>Validation built into the demo</strong>
                <ul>
                    <li>Date range restricted to the modeling horizon</li>
                    <li>District, ward, beat, and community area range checks</li>
                    <li>Latitude and longitude bounds checks</li>
                    <li>Cold-start cases are rejected if historical context is unavailable</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sub-card">
                <strong>Demo framing</strong>
                <p style="margin-top: 10px; color: #475569;">
                    This app demonstrates model predictions for decision support.
                    It should be presented as a planning aid, not as an automated enforcement system.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not submit:
        return

    prediction_dt = pd.Timestamp(selected_date).replace(hour=int(selected_hour), minute=0, second=0)
    payload = pd.DataFrame(
        [
            {
                "Date": prediction_dt,
                "District": district,
                "Ward": ward,
                "Community Area": community_area,
                "Beat": beat,
                "Latitude": latitude,
                "Longitude": longitude,
                "X Coordinate": defaults["X Coordinate"],
                "Y Coordinate": defaults["Y Coordinate"],
            }
        ]
    )

    try:
        X_demo, surviving_mask, _ = engine["processor"]._process(payload, pd.Series([0]), is_train=False)
        if not bool(surviving_mask[0]):
            st.error("This input falls into a cold-start case with insufficient history. Please choose a later date.")
            return

        X_demo = X_demo.reindex(columns=engine["feature_columns"], fill_value=0)
        probability = float(engine["model"].predict_proba(X_demo)[0, 1])
        risk_label, risk_class = classify_risk(probability, engine["thresholds"])
        recent_counts = compute_recent_district_counts(engine["positive_raw"], prediction_dt, district)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.markdown("---")
    chip_html = f'<span class="risk-chip {risk_class}">{risk_label}</span>'
    st.markdown(
        f"""
        <div class="prediction-card">
            <h3 style="margin-top: 0; margin-bottom: 8px;">Prediction Result</h3>
            <div style="margin-bottom: 14px;">{chip_html}</div>
            <p style="font-size: 1.05rem; color: #334155; margin-bottom: 6px;">
                Estimated crime-occurrence probability:
                <strong style="font-size: 1.35rem; color: #0f172a;">{probability:.3f}</strong>
            </p>
            <p style="color: #475569; margin-bottom: 0;">
                Input: district {district}, {prediction_dt.strftime("%Y-%m-%d %H:%M")}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        metric_card("Positive hours in last 7d", f"{recent_counts['7d']}", "Same district", "#3b82f6")
    with metric_col2:
        metric_card("Positive hours in last 30d", f"{recent_counts['30d']}", "Same district", "#8b5cf6")
    with metric_col3:
        metric_card("Positive hours in last 90d", f"{recent_counts['90d']}", "Same district", "#ef4444")

    st.markdown("---")
    st.subheader("How to interpret this output")
    st.write(
        "- Use the score as a relative risk indicator, not as a deterministic forecast.\n"
        "- Temporal patterns are the strongest signal in the current benchmark.\n"
        "- District-level hotspot ranking remains weak, so this is best used for timing support and localized review."
    )


def render_high_risk_places() -> None:
    st.title("High-Risk Places")
    st.caption(
        "Predicting places of increased crime risk by scoring every police district for a selected hour."
    )

    try:
        engine = load_prediction_engine()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.title("Navigation")
        if st.button("← Home", use_container_width=True):
            st.session_state.app_mode = "Welcome"
            st.rerun()
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.app_mode = "Dashboard"
            st.rerun()
        if st.button("🤖 Prediction Demo", use_container_width=True):
            st.session_state.app_mode = "Prediction Demo"
            st.rerun()
        if st.button("👥 Group Pattern Analysis", use_container_width=True):
            st.session_state.app_mode = "Group Pattern Analysis"
            st.rerun()
        if st.button("🌎 NIBRS Generalization", use_container_width=True):
            st.session_state.app_mode = "NIBRS Generalization"
            st.rerun()
        if st.button("🗺️ Agency Map Demo", use_container_width=True):
            st.session_state.app_mode = "Agency Map Demo"
            st.rerun()
        st.divider()
        top_k = st.slider("Top districts to highlight", min_value=3, max_value=10, value=5)

    control_left, control_right = st.columns([1, 1])
    with control_left:
        selected_date = st.date_input(
            "Target date",
            value=pd.Timestamp("2025-10-01").date(),
            min_value=MIN_PREDICTION_DATE.date(),
            max_value=MAX_PREDICTION_DATE.date(),
            key="places_date",
        )
    with control_right:
        selected_hour = st.slider(
            "Target hour",
            min_value=0,
            max_value=23,
            value=18,
            key="places_hour",
        )

    target_dt = pd.Timestamp(selected_date).replace(hour=int(selected_hour), minute=0, second=0)
    with st.spinner("Scoring districts for the selected hour..."):
        payload = build_district_prediction_payload(engine, target_dt)
        scored = score_prediction_payload(engine, payload)

    if scored.empty:
        st.error("No valid district scores were produced for the selected hour.")
        return

    scored = scored.sort_values("probability", ascending=False).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    top_places = scored.head(top_k).copy()
    top_places["probability_pct"] = (top_places["probability"] * 100).round(1)

    top_one, top_two, top_three = st.columns(3)
    with top_one:
        metric_card(
            "Highest-risk district",
            str(int(top_places.iloc[0]["District"])),
            top_places.iloc[0]["risk_label"],
            "#ef4444",
        )
    with top_two:
        metric_card(
            "Top risk probability",
            f"{top_places.iloc[0]['probability']:.3f}",
            target_dt.strftime("%Y-%m-%d %H:%M"),
            "#f59e0b",
        )
    with top_three:
        metric_card(
            "Districts scored",
            f"{len(scored):,}",
            "All available districts",
            "#3b82f6",
        )

    st.markdown("---")
    map_col, chart_col = st.columns([1.5, 1])

    with map_col:
        st.subheader("Predicted hotspot map")
        map_df = scored.copy()
        map_df["red"] = (120 + map_df["probability"] * 135).clip(0, 255).astype(int)
        map_df["green"] = (210 - map_df["probability"] * 160).clip(40, 210).astype(int)
        map_df["blue"] = 75
        map_df["alpha"] = 180
        map_df["radius"] = (250 + map_df["probability"] * 900).astype(int)
        st.pydeck_chart(
            pdk.Deck(
                map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
                initial_view_state=pdk.ViewState(latitude=41.85, longitude=-87.65, zoom=9.8),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_df,
                        get_position="[Longitude, Latitude]",
                        get_fill_color="[red, green, blue, alpha]",
                        get_radius="radius",
                        pickable=True,
                    )
                ],
                tooltip={
                    "text": "District {District}\nRisk {probability:.3f}\nLabel {risk_label}"
                },
            )
        )

    with chart_col:
        st.subheader("Top districts")
        bar = px.bar(
            top_places.sort_values("probability"),
            x="probability",
            y=top_places.sort_values("probability")["District"].astype(str),
            orientation="h",
            color="probability",
            color_continuous_scale="Reds",
            labels={"probability": "Predicted probability", "y": "District"},
        )
        bar.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(bar, use_container_width=True)

    st.markdown("---")
    st.subheader("Top-ranked districts")
    st.dataframe(
        top_places[["rank", "District", "risk_label", "probability_pct"]].rename(
            columns={
                "rank": "Rank",
                "District": "District",
                "risk_label": "Risk band",
                "probability_pct": "Probability (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "This view operationalizes the current model as a place-risk ranking tool: "
        "for a chosen hour, every district is scored and ranked by predicted risk."
    )


def render_group_pattern_analysis() -> None:
    st.title("Group Pattern Analysis")
    st.caption(
        "Predicting group/population crime patterns by aggregating district-hour risk across a full day."
    )

    try:
        engine = load_prediction_engine()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.title("Navigation")
        if st.button("← Home", use_container_width=True):
            st.session_state.app_mode = "Welcome"
            st.rerun()
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.app_mode = "Dashboard"
            st.rerun()
        if st.button("🤖 Prediction Demo", use_container_width=True):
            st.session_state.app_mode = "Prediction Demo"
            st.rerun()
        if st.button("📍 High-Risk Places", use_container_width=True):
            st.session_state.app_mode = "High-Risk Places"
            st.rerun()
        if st.button("🌎 NIBRS Generalization", use_container_width=True):
            st.session_state.app_mode = "NIBRS Generalization"
            st.rerun()
        if st.button("🗺️ Agency Map Demo", use_container_width=True):
            st.session_state.app_mode = "Agency Map Demo"
            st.rerun()
        st.divider()
        top_block_count = st.slider("Highlighted district-hour blocks", min_value=5, max_value=15, value=8)

    selected_date = st.date_input(
        "Daily outlook date",
        value=pd.Timestamp("2025-10-01").date(),
        min_value=MIN_PREDICTION_DATE.date(),
        max_value=MAX_PREDICTION_DATE.date(),
        key="group_patterns_date",
    )

    with st.spinner("Generating district-hour outlook for the full day..."):
        payload = build_district_prediction_payload(
            engine,
            pd.Timestamp(selected_date),
            hours=list(range(24)),
        )
        scored = score_prediction_payload(engine, payload)

    if scored.empty:
        st.error("No aggregate pattern outlook could be generated for the selected date.")
        return

    scored["Hour"] = pd.to_datetime(scored["Date"]).dt.hour
    scored["DistrictLabel"] = scored["District"].astype(int).astype(str)

    citywide_hourly = scored.groupby("Hour", as_index=False)["probability"].mean()
    district_daily = scored.groupby("District", as_index=False)["probability"].mean().sort_values(
        "probability", ascending=False
    )
    top_blocks = scored.sort_values("probability", ascending=False).head(top_block_count).copy()
    top_blocks["Date"] = pd.to_datetime(top_blocks["Date"]).dt.strftime("%Y-%m-%d %H:%M")
    top_blocks["probability_pct"] = (top_blocks["probability"] * 100).round(1)

    top_left, top_mid, top_right = st.columns(3)
    with top_left:
        metric_card(
            "Peak citywide hour",
            f"{int(citywide_hourly.loc[citywide_hourly['probability'].idxmax(), 'Hour'])}:00",
            "Highest mean risk",
            "#ef4444",
        )
    with top_mid:
        metric_card(
            "Top district (daily avg)",
            str(int(district_daily.iloc[0]["District"])),
            f"{district_daily.iloc[0]['probability']:.3f}",
            "#8b5cf6",
        )
    with top_right:
        metric_card(
            "District-hour blocks scored",
            f"{len(scored):,}",
            "24 hours x all districts",
            "#3b82f6",
        )

    st.markdown("---")
    chart_left, chart_right = st.columns([1.1, 1])
    with chart_left:
        st.subheader("Citywide hourly risk outlook")
        line = px.line(
            citywide_hourly,
            x="Hour",
            y="probability",
            markers=True,
            labels={"probability": "Mean predicted probability"},
        )
        line.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(dtick=1))
        st.plotly_chart(line, use_container_width=True)

    with chart_right:
        st.subheader("District daily average risk")
        top_daily = district_daily.head(10).copy()
        daily_bar = px.bar(
            top_daily.sort_values("probability"),
            x="probability",
            y=top_daily.sort_values("probability")["District"].astype(int).astype(str),
            orientation="h",
            color="probability",
            color_continuous_scale="Blues",
            labels={"probability": "Mean predicted probability", "y": "District"},
        )
        daily_bar.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(daily_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("District-hour heatmap")
    heat_df = scored.copy()
    heatmap = px.density_heatmap(
        heat_df,
        x="Hour",
        y="DistrictLabel",
        z="probability",
        color_continuous_scale="YlOrRd",
        histfunc="avg",
        nbinsx=24,
        nbinsy=len(heat_df["DistrictLabel"].unique()),
        labels={"DistrictLabel": "District", "probability": "Mean probability"},
    )
    heatmap.update_layout(height=460, margin=dict(l=0, r=0, t=24, b=0), xaxis=dict(dtick=1))
    st.plotly_chart(heatmap, use_container_width=True)

    st.subheader("Highest-risk district-hour blocks")
    st.dataframe(
        top_blocks[["Date", "District", "risk_label", "probability_pct"]].rename(
            columns={
                "Date": "DateTime",
                "District": "District",
                "risk_label": "Risk band",
                "probability_pct": "Probability (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "This page reframes the model as an aggregate-pattern tool: instead of scoring a single input, "
        "it predicts how risk is distributed across districts and hours for a full day."
    )


def render_nibrs_generalization() -> None:
    st.title("NIBRS Generalization Results")
    st.caption(
        "External validation of the Chicago-trained HistGradientBoosting benchmark on FBI NIBRS county-hour datasets."
    )

    with st.sidebar:
        st.title("Navigation")
        if st.button("← Home", use_container_width=True):
            st.session_state.app_mode = "Welcome"
            st.rerun()
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.app_mode = "Dashboard"
            st.rerun()
        if st.button("🤖 Prediction Demo", use_container_width=True):
            st.session_state.app_mode = "Prediction Demo"
            st.rerun()
        if st.button("📍 High-Risk Places", use_container_width=True):
            st.session_state.app_mode = "High-Risk Places"
            st.rerun()
        if st.button("👥 Group Pattern Analysis", use_container_width=True):
            st.session_state.app_mode = "Group Pattern Analysis"
            st.rerun()
        st.divider()
        st.info(
            "Evaluation setting:\n\n"
            "- Train fit: Chicago 2015-2024\n"
            "- External test: NIBRS 2024\n"
            "- Warm-up history: NIBRS 2023"
        )

    metrics_df, build_summary = load_nibrs_generalization_results()
    if metrics_df is None or metrics_df.empty:
        st.error(
            "Missing NIBRS generalization metrics. Run "
            "`python3 src/scripts/evaluate_nibrs_generalization.py --eval-year 2024` first."
        )
        return

    best_row = metrics_df.sort_values("auroc", ascending=False).iloc[0]
    coverage_label = ", ".join(metrics_df["label"].tolist())

    top_left, top_mid, top_right = st.columns(3)
    with top_left:
        metric_card(
            "Best external AUROC",
            f"{best_row['auroc']:.3f}",
            f"{best_row['label']} | {int(best_row['eval_year'])}",
            "#3b82f6",
        )
    with top_mid:
        metric_card(
            "Best external AUPRC",
            f"{metrics_df['auprc'].max():.3f}",
            coverage_label,
            "#8b5cf6",
        )
    with top_right:
        metric_card(
            "External datasets",
            f"{len(metrics_df)}",
            coverage_label,
            "#10b981",
        )

    st.markdown("---")
    compare_left, compare_right = st.columns([1.05, 0.95])

    with compare_left:
        st.subheader("External benchmark comparison")
        compare = metrics_df[["label", "auroc", "auprc"]].melt(
            id_vars="label",
            value_vars=["auroc", "auprc"],
            var_name="metric",
            value_name="score",
        )
        compare["metric"] = compare["metric"].map({"auroc": "AUROC", "auprc": "AUPRC"})
        chart = px.bar(
            compare,
            x="label",
            y="score",
            color="metric",
            barmode="group",
            color_discrete_sequence=["#2563eb", "#c2410c"],
            labels={"label": "State", "score": "Score", "metric": "Metric"},
        )
        chart.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(chart, use_container_width=True)

    with compare_right:
        st.subheader("Temporal vs spatial transfer")
        align = metrics_df[["label", "hourly_correlation", "county_correlation"]].melt(
            id_vars="label",
            value_vars=["hourly_correlation", "county_correlation"],
            var_name="metric",
            value_name="score",
        )
        align["metric"] = align["metric"].map(
            {
                "hourly_correlation": "Hourly correlation",
                "county_correlation": "County correlation",
            }
        )
        corr_chart = px.bar(
            align,
            x="label",
            y="score",
            color="metric",
            barmode="group",
            color_discrete_sequence=["#f59e0b", "#0f766e"],
            labels={"label": "State", "score": "Correlation", "metric": "Alignment"},
        )
        corr_chart.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(corr_chart, use_container_width=True)

    st.markdown("---")
    st.subheader("Dataset coverage")
    coverage_rows: list[dict] = []
    for _, row in metrics_df.iterrows():
        summary = build_summary.get(str(row["state"]).upper(), {})
        coverage_rows.append(
            {
                "State": row["label"],
                "Eval Year": int(row["eval_year"]),
                "Rows Scored": f"{int(row['rows']):,}",
                "Positive Rate (%)": f"{row['positive_rate_pct']:.1f}",
                "Distinct Counties": int(row.get("distinct_counties", 0)),
                "Prepared Rows": f"{int(summary.get('rows', 0)):,}" if summary else "N/A",
                "Prepared Positives": f"{int(summary.get('positive_rows', 0)):,}" if summary else "N/A",
            }
        )
    st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)

    st.subheader("External metrics table")
    display_df = metrics_df[
        [
            "label",
            "eval_year",
            "rows",
            "positive_rate_pct",
            "auroc",
            "auprc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "hourly_correlation",
            "county_correlation",
        ]
    ].rename(
        columns={
            "label": "State",
            "eval_year": "Eval Year",
            "rows": "Rows",
            "positive_rate_pct": "Positive Rate (%)",
            "auroc": "AUROC",
            "auprc": "AUPRC",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "hourly_correlation": "Hourly Corr",
            "county_correlation": "County Corr",
        }
    ).copy()
    display_df["Rows"] = display_df["Rows"].map(lambda value: f"{int(value):,}")
    for col in ["AUROC", "AUPRC", "Accuracy", "Precision", "Recall", "F1", "Hourly Corr", "County Corr"]:
        display_df[col] = display_df[col].map(lambda value: f"{float(value):.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    insight_left, insight_right = st.columns([1.05, 0.95])
    with insight_left:
        st.markdown(
            """
            <div class="sub-card">
                <strong>What this says about transfer</strong>
                <ul>
                    <li>The Chicago-trained model still discriminates reasonably well on external NIBRS data.</li>
                    <li>Hourly correlation remains much stronger than county-level correlation.</li>
                    <li>This supports using the benchmark as a timing-oriented signal rather than a precise hotspot locator.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with insight_right:
        state_names = " and ".join(metrics_df["label"].tolist())
        st.markdown(
            f"""
            <div class="sub-card">
                <strong>Recommended presentation wording</strong>
                <p style="margin-top: 10px; color: #475569;">
                    We trained the model on Chicago historical data, then tested external
                    generalization on {state_names} NIBRS county-hour datasets for 2024.
                    Transfer performance stays meaningful on discrimination metrics, while
                    spatial ranking remains weaker than temporal alignment.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_agency_map_demo() -> None:
    st.title("Agency Map Demo")
    st.caption(
        "A Streamlit-integrated version of the JIaDLu NIBRS agency-level demo, merged into the main Team22 app."
    )

    with st.sidebar:
        st.title("Navigation")
        if st.button("← Home", use_container_width=True):
            st.session_state.app_mode = "Welcome"
            st.rerun()
        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.app_mode = "Dashboard"
            st.rerun()
        if st.button("🤖 Prediction Demo", use_container_width=True):
            st.session_state.app_mode = "Prediction Demo"
            st.rerun()
        if st.button("📍 High-Risk Places", use_container_width=True):
            st.session_state.app_mode = "High-Risk Places"
            st.rerun()
        if st.button("👥 Group Pattern Analysis", use_container_width=True):
            st.session_state.app_mode = "Group Pattern Analysis"
            st.rerun()
        if st.button("🌎 NIBRS Generalization", use_container_width=True):
            st.session_state.app_mode = "NIBRS Generalization"
            st.rerun()
        st.divider()
        st.info(
            "Current local mode uses a 7-day average preview because the original JIaDLu GRU "
            "checkpoint assets are not present in this repository."
        )

    datasets = discover_nibrs_demo_datasets()
    if not datasets:
        st.error(
            f"No NIBRS datasets were found under `{NIBRS_DEMO_BASE_DIR}`. "
            "Keep the local NIBRS data folder beside the repository to use this page."
        )
        return

    selected_dataset = st.selectbox("NIBRS dataset", datasets, index=min(1, len(datasets) - 1))
    try:
        bundle = load_nibrs_demo_bundle(selected_dataset)
    except Exception as exc:
        st.error(f"Unable to load dataset `{selected_dataset}`: {exc}")
        return

    available_dates = get_nibrs_available_dates(bundle["aligned"])
    if not available_dates:
        st.error("This dataset does not have enough history for a 7-day preview window.")
        return

    default_index = max(0, len(available_dates) - 90)
    selected_date = st.selectbox("Target date", available_dates, index=default_index)
    run_prediction = st.button("Run Agency Prediction", type="primary", use_container_width=True)

    if not run_prediction:
        st.info("Choose a dataset and date, then run the integrated agency prediction map.")
        return

    with st.spinner("Scoring agencies for the selected day..."):
        results = preview_predict_nibrs(bundle, selected_date)

    if results.empty:
        st.warning("No agencies were available for the selected dataset/date.")
        return

    top_row = results.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Agencies", f"{len(results):,}", selected_dataset, "#c07a50")
    with c2:
        metric_card("Total Predicted", f"{results['predicted'].sum():.0f}", "Preview sum", "#c07a50")
    with c3:
        metric_card("Max Agency", f"{top_row['predicted']:.1f}", str(top_row["agency_name"])[:18], "#c07a50")
    with c4:
        metric_card("Target Date", selected_date, "7-day preview", "#c07a50")

    st.warning(
        "Preview mode: this integrated page preserves the JIaDLu agency-map interaction, but uses a 7-day "
        "average fallback because the original Baseline-GRU assets are unavailable locally."
    )

    map_col, table_col = st.columns([1.5, 1])
    with map_col:
        st.subheader("Agency prediction map")
        render_agency_leaflet_map(results, bundle["center"])

    with table_col:
        st.subheader("Top agencies")
        st.dataframe(
            results.head(15)[["agency_name", "predicted", "actual", "region_id"]].rename(
                columns={
                    "agency_name": "Agency",
                    "predicted": "Predicted",
                    "actual": "Actual",
                    "region_id": "Region",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Full result table")
    st.dataframe(
        results.rename(
            columns={
                "agency_name": "Agency",
                "state": "State",
                "predicted": "Predicted",
                "actual": "Actual",
                "region_id": "Region",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


if st.session_state.app_mode == "Welcome":
    render_welcome()
elif st.session_state.app_mode == "Dashboard":
    render_dashboard()
elif st.session_state.app_mode == "Prediction Demo":
    render_prediction_demo()
elif st.session_state.app_mode == "High-Risk Places":
    render_high_risk_places()
elif st.session_state.app_mode == "Group Pattern Analysis":
    render_group_pattern_analysis()
elif st.session_state.app_mode == "NIBRS Generalization":
    render_nibrs_generalization()
elif st.session_state.app_mode == "Agency Map Demo":
    render_agency_map_demo()
