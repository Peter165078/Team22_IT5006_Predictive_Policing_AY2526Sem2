from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_builder import build_phase2_dataset
from src.data.processor import DataProcessor

MODEL_NAME = "HistGradientBoosting"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "hist_gradient_boosting.pkl"
PREDICTION_SOURCE_DIR = PROJECT_ROOT / "apps" / "dashboard" / "split_data_by_year"
PREDICTION_DATA_PATH = Path(tempfile.gettempdir()) / "team22_phase2_prediction_data.csv"
MIN_PREDICTION_DATE = pd.Timestamp("2022-01-08")
MAX_PREDICTION_DATE = pd.Timestamp("2024-12-31")

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


def build_prediction_dataset() -> None:
    if PREDICTION_DATA_PATH.exists():
        return
    build_phase2_dataset(
        source_dir=PREDICTION_SOURCE_DIR,
        output_path=PREDICTION_DATA_PATH,
        start_year=2022,
        end_year=2024,
        max_rows_per_year=20_000,
        overwrite=True,
    )


@st.cache_resource(show_spinner="Preparing prediction engine...")
def load_prediction_engine() -> dict:
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
        left, right = st.columns(2)
        if left.button("📊 Open Dashboard", type="primary", use_container_width=True):
            st.session_state.app_mode = "Dashboard"
            st.rerun()
        if right.button("🤖 Try Prediction Demo", use_container_width=True):
            st.session_state.app_mode = "Prediction Demo"
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sub-card">
                <strong>What this app demonstrates</strong>
                <ul>
                    <li>Interactive crime analytics dashboard</li>
                    <li>Model-backed risk prediction workflow</li>
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
        st.divider()
        st.title(f"Controls ({year})")
        st.success(f"Loaded {len(df):,} rows")
        available_years = [y for y in range(2014, 2025) if get_file_path(y)]
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
        st.divider()
        st.info(
            f"Model in use: {MODEL_NAME}\n\n"
            f"Prediction date range: {MIN_PREDICTION_DATE.date()} to {MAX_PREDICTION_DATE.date()}"
        )

    district_options = sorted(engine["district_reference"]["District"].dropna().astype(int).unique().tolist())
    default_district = 11 if 11 in district_options else district_options[0]
    district = st.selectbox("Police district", district_options, index=district_options.index(default_district))
    defaults = get_district_defaults(engine, district)

    left, right = st.columns([1.1, 0.9])
    with left:
        with st.form("prediction_form"):
            selected_date = st.date_input(
                "Incident date",
                value=pd.Timestamp("2024-10-01").date(),
                min_value=MIN_PREDICTION_DATE.date(),
                max_value=MAX_PREDICTION_DATE.date(),
            )
            selected_hour = st.slider("Hour of day", min_value=0, max_value=23, value=18)

            st.markdown("**Optional location refinement**")
            expander = st.expander("Edit advanced spatial fields", expanded=False)
            with expander:
                ward = st.number_input("Ward", min_value=1, max_value=50, value=defaults["Ward"] or 1, step=1)
                community_area = st.number_input(
                    "Community Area", min_value=1, max_value=77, value=defaults["Community Area"] or 1, step=1
                )
                beat = st.number_input("Beat", min_value=111, max_value=2535, value=defaults["Beat"] or 111, step=1)
                latitude = st.number_input(
                    "Latitude", min_value=36.0, max_value=42.0, value=float(defaults["Latitude"] or 41.85), format="%.6f"
                )
                longitude = st.number_input(
                    "Longitude", min_value=-92.0, max_value=-88.0, value=float(defaults["Longitude"] or -87.65), format="%.6f"
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
        metric_card("Crimes in last 7d", f"{recent_counts['7d']}", "Same district", "#3b82f6")
    with metric_col2:
        metric_card("Crimes in last 30d", f"{recent_counts['30d']}", "Same district", "#8b5cf6")
    with metric_col3:
        metric_card("Crimes in last 90d", f"{recent_counts['90d']}", "Same district", "#ef4444")

    st.markdown("---")
    st.subheader("How to interpret this output")
    st.write(
        "- Use the score as a relative risk indicator, not as a deterministic forecast.\n"
        "- Temporal patterns are the strongest signal in the current benchmark.\n"
        "- District-level hotspot ranking remains weak, so this is best used for timing support and localized review."
    )


if st.session_state.app_mode == "Welcome":
    render_welcome()
elif st.session_state.app_mode == "Dashboard":
    render_dashboard()
elif st.session_state.app_mode == "Prediction Demo":
    render_prediction_demo()
