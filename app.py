
from __future__ import annotations
import copy
import io
import json
import warnings
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

PROJECT_DIR = Path(__file__).resolve().parent
PRODUCTION_PATH = PROJECT_DIR / "data" / "processed_data" / "crop_production_preprocessed.csv"
WEATHER_PATH = PROJECT_DIR / "data" / "processed_data" / "crop_weather_merged.csv"
STATE_WEATHER_PATH = PROJECT_DIR / "data" / "processed_data" / "india_states_yearly_weather_preprocessed.csv"
SOIL_PATH = PROJECT_DIR / "data" / "processed_data" / "crop_recommendation_preprocessed.csv"
GEOJSON_PATH = PROJECT_DIR / "assets" / "india_state.geojson"
MODEL_PATH = PROJECT_DIR / "models" / "random_forest_weather.joblib"
WEATHER_FEATURES = [
    "Mean_Temperature",
    "Avg_Max_Temp",
    "Avg_Min_Temp",
    "Total_Precipitation_mm",
    "Max_Wind_Speed",
]
REQUIRED_PRODUCTION_COLUMNS = [
    "State",
    "District",
    "Year",
    "Season",
    "Crop",
    "Area",
    "Production",
]

st.set_page_config(
    page_title="AgriPulse | Agricultural Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

def normalize_state(value: Any) -> str:
    """Normalize state names for joins to the bundled GeoJSON boundaries."""
    key = " ".join(str(value).strip().casefold().split())
    aliases = {
        "andaman and nicobar islands": "andaman and nicobar",
        "andaman & nicobar islands": "andaman and nicobar",
        "odisha": "orissa",
        "uttarakhand": "uttaranchal",
    }
    return aliases.get(key, key)

@st.cache_data(show_spinner="Loading crop production records...")
def load_production_data() -> pd.DataFrame:
    if not PRODUCTION_PATH.exists():
        raise FileNotFoundError(f"Required production dataset not found: {PRODUCTION_PATH}")
    frame = pd.read_csv(PRODUCTION_PATH)
    frame = frame.rename(
        columns={
            "State_Name": "State",
            "District_Name": "District",
            "Crop_Year": "Year",
        }
    )
    missing = [column for column in REQUIRED_PRODUCTION_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Production dataset is missing required columns: {missing}")
    for column in ["Year", "Area", "Production"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "Yield" in frame:
        frame["Yield"] = pd.to_numeric(frame["Yield"], errors="coerce")
    else:
        frame["Yield"] = np.nan
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=REQUIRED_PRODUCTION_COLUMNS).copy()
    frame = frame[(frame["Area"] > 0) & (frame["Production"] >= 0)]
    derived_yield = frame["Production"] / frame["Area"]
    frame["Yield"] = frame["Yield"].where(frame["Yield"].notna(), derived_yield)
    frame = frame.dropna(subset=["Yield"])
    frame["Year"] = frame["Year"].astype(int)
    for column in ["State", "District", "Season", "Crop"]:
        frame[column] = frame[column].astype(str).str.strip()
    return frame

@st.cache_data(show_spinner="Loading matched crop and weather records...")
def load_weather_data() -> tuple[pd.DataFrame, str | None]:
    if not WEATHER_PATH.exists():
        return pd.DataFrame(), f"Optional weather dataset was not found: {WEATHER_PATH}"
    frame = pd.read_csv(WEATHER_PATH)
    required = REQUIRED_PRODUCTION_COLUMNS + ["Yield"] + WEATHER_FEATURES
    missing = [column for column in required if column not in frame]
    if missing:
        return pd.DataFrame(), f"Weather analysis is unavailable; missing columns: {missing}"
    for column in ["Year", "Area", "Production", "Yield"] + WEATHER_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=required).copy()
    frame = frame[(frame["Area"] > 0) & (frame["Production"] >= 0)]
    frame["Year"] = frame["Year"].astype(int)
    return frame, None

@st.cache_data(show_spinner="Loading state/year weather observations...")
def load_state_weather_data() -> tuple[pd.DataFrame, str | None]:
    if not STATE_WEATHER_PATH.exists():
        return pd.DataFrame(), f"Optional state/year weather dataset was not found: {STATE_WEATHER_PATH}"
    frame = pd.read_csv(STATE_WEATHER_PATH)
    required = ["State", "Year"] + WEATHER_FEATURES
    missing = [column for column in required if column not in frame]
    if missing:
        return pd.DataFrame(), f"State/year weather context is unavailable; missing columns: {missing}"
    for column in ["Year"] + WEATHER_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    frame["Year"] = frame["Year"].astype(int)
    return frame, None

@st.cache_data(show_spinner="Loading the soil sample dataset...")
def load_soil_data() -> tuple[pd.DataFrame, str | None]:
    if not SOIL_PATH.exists():
        return pd.DataFrame(), f"Optional crop-recommendation dataset was not found: {SOIL_PATH}"
    frame = pd.read_csv(SOIL_PATH)
    required = ["N", "P", "K", "label"]
    missing = [column for column in required if column not in frame]
    if missing:
        return pd.DataFrame(), f"Soil sample analysis is unavailable; missing columns: {missing}"
    for column in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, None

@st.cache_data(show_spinner="Loading India state boundaries...")
def load_geojson() -> tuple[dict[str, Any] | None, str | None]:
    if not GEOJSON_PATH.exists():
        return None, f"State map boundaries were not found: {GEOJSON_PATH}"
    with GEOJSON_PATH.open(encoding="utf-8") as source:
        geojson = json.load(source)
    if not isinstance(geojson, dict) or not geojson.get("features"):
        return None, "The state boundary file contains no geographic features."
    geojson = copy.deepcopy(geojson)
    for feature in geojson["features"]:
        properties = feature.get("properties", {})
        if "NAME_1" not in properties:
            return None, "The state boundary file does not contain the expected NAME_1 identifiers."
        properties["map_key"] = normalize_state(properties["NAME_1"])
    return geojson, None

@st.cache_resource(show_spinner="Loading the trained weather-aware yield model...")
def load_yield_model() -> tuple[Any | None, str | None, list[str]]:
    if not MODEL_PATH.exists():
        return None, f"Compatible model not found: {MODEL_PATH}", []
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", InconsistentVersionWarning)
            model = joblib.load(MODEL_PATH)
        mismatch_messages = [str(item.message) for item in captured]
    except (FileNotFoundError, ModuleNotFoundError, ImportError, OSError, ValueError, AttributeError, TypeError) as error:
        return None, f"The trained model could not be loaded: {type(error).__name__}: {error}", []
    if not callable(getattr(model, "predict", None)):
        return None, "The model artifact does not provide a prediction interface.", mismatch_messages
    if not hasattr(model, "feature_names_in_"):
        return None, "The model artifact does not declare its input feature names.", mismatch_messages
    return model, None, mismatch_messages

def format_value(value: float) -> str:
    if not np.isfinite(value):
        return "—"
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if absolute >= 10_000:
        return f"{value / 1_000:,.1f}K"
    return f"{value:,.2f}"

def plotly_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e4eee7", "family": "Inter, Segoe UI, sans-serif"},
        margin={"l": 12, "r": 12, "t": 48, "b": 12},
        colorway=["#86d39b", "#b7e4c7", "#4ca66b", "#d7f2df", "#2b7650"],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor="rgba(191, 220, 199, 0.12)", zeroline=False, automargin=True)
    return fig

def show_empty(message: str) -> None:
    st.info(message)

def make_pdf_report(
    filtered: pd.DataFrame,
    crop_summary: pd.DataFrame,
    region_summary: pd.DataFrame,
    filters: dict[str, str],
) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    total_area = float(filtered["Area"].sum())
    total_production = float(filtered["Production"].sum())
    weighted_yield = total_production / total_area if total_area > 0 else float("nan")
    story = [
        Paragraph("Agricultural Production Summary", styles["Title"]),
        Paragraph("Crop Yield Prediction & Smart Irrigation Advisory", styles["Heading2"]),
        Spacer(1, 10),
        Paragraph("Applied filters: " + "; ".join(f"{escape(key)}: {escape(value)}" for key, value in filters.items()), styles["BodyText"]),
        Spacer(1, 10),
    ]
    metrics = [
        ["Measure", "Value"],
        ["Production total (source units)", format_value(total_production)],
        ["Cultivated area total (source units)", format_value(total_area)],
        ["Weighted yield (production / area)", format_value(weighted_yield)],
        ["Included records", f"{len(filtered):,}"],
    ]
    table = Table(metrics, colWidths=[3.5 * inch, 2.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174b32")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d9cc")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f7f3")]),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([table, Spacer(1, 16), Paragraph("Leading crops by production", styles["Heading2"])])
    for _, row in crop_summary.head(5).iterrows():
        story.append(Paragraph(f"{escape(str(row['Crop']))}: {format_value(float(row['Production']))}", styles["BodyText"]))
    story.extend([Spacer(1, 10), Paragraph("Leading regions by production", styles["Heading2"])])
    for _, row in region_summary.head(5).iterrows():
        story.append(Paragraph(f"{escape(str(row['State']))}: {format_value(float(row['Production']))}", styles["BodyText"]))
    story.extend(
        [
            Spacer(1, 14),
            Paragraph(
                "Data note: Weather values are historical state/year observations joined to crop records. "
                "Observed associations do not establish causation. This report does not provide a forecast "
                "or a validated irrigation prescription.",
                styles["BodyText"],
            ),
        ]
    )
    document.build(story)
    return output.getvalue()

try:
    production = load_production_data()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Unable to load the primary crop dataset. {error}")
    st.stop()
weather, weather_message = load_weather_data()
state_weather, state_weather_message = load_state_weather_data()
soil_data, soil_message = load_soil_data()
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp { background: #0b1711; color: #edf4ee; }
    [data-testid="stSidebar"] { background: #10271b; border-right: 1px solid #244a33; }
    [data-testid="stSidebar"] * { color: #e6f1e8; }
    .block-container { padding-top: 1.8rem; padding-bottom: 2.8rem; max-width: 1500px; }
    h1, h2, h3 { color: #edf7ef !important; letter-spacing: -0.02em; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #173b29, #10271b);
        border: 1px solid #2a5a3c; border-radius: 16px; padding: 18px 20px;
    }
    [data-testid="stMetricLabel"] { color: #b9d8c1 !important; }
    [data-testid="stMetricValue"] { color: #f4fbf5 !important; }
    div[data-testid="stAlert"] { border-radius: 12px; }
    .hero {
        border: 1px solid #2c5d3e; border-radius: 20px; padding: 26px 30px;
        background: radial-gradient(circle at 90% 5%, #28623c 0, transparent 37%),
                    linear-gradient(120deg, #143623, #10251a);
        margin-bottom: 1.3rem;
    }
    .hero-kicker { color: #9bd9a9; font-size: .8rem; letter-spacing: .15em; text-transform: uppercase; }
    .hero h1 { font-size: clamp(1.9rem, 4vw, 3rem); margin: .35rem 0; }
    .hero p { color: #c6d9ca; margin: 0; max-width: 780px; }
    .section-note { color: #b5c9b9; font-size: .92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Agricultural intelligence • India</div>
      <h1>AgriPulse</h1>
      <p>Explore production patterns, historical weather associations, regional performance,
      and transparent decision-support tools grounded in the project datasets.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
with st.sidebar:
    st.markdown("## 🌿 AgriPulse")
    st.caption("Crop yield & smart irrigation intelligence")
    page = st.radio(
        "Workspace",
        [
            "Overview",
            "Weather analytics",
            "Regional map",
            "Soil nutrients",
            "Irrigation advisory",
            "Yield prediction",
            "Reports",
        ],
        key="page_navigation",
    )
    st.divider()
    st.markdown("### Data filters")
    crop_options = sorted(production["Crop"].dropna().unique().tolist())
    state_options = sorted(production["State"].dropna().unique().tolist())
    crop_filter = st.selectbox("Crop", ["All crops"] + crop_options)
    state_filter = st.selectbox("State / region", ["All states"] + state_options)
    district_values = production
    if state_filter != "All states":
        district_values = district_values[district_values["State"] == state_filter]
    district_options = sorted(district_values["District"].dropna().unique().tolist())
    district_filter = st.selectbox("District", ["All districts"] + district_options)
    season_options = sorted(production["Season"].dropna().unique().tolist())
    season_filter = st.selectbox("Season", ["All seasons"] + season_options)
    year_min = int(production["Year"].min())
    year_max = int(production["Year"].max())
    year_range = st.slider(
        "Crop year",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )
    st.caption(f"Production coverage: {year_min}–{year_max}")
filtered = production[
    production["Year"].between(year_range[0], year_range[1])
].copy()
if crop_filter != "All crops":
    filtered = filtered[filtered["Crop"] == crop_filter]
if state_filter != "All states":
    filtered = filtered[filtered["State"] == state_filter]
if district_filter != "All districts":
    filtered = filtered[filtered["District"] == district_filter]
if season_filter != "All seasons":
    filtered = filtered[filtered["Season"] == season_filter]
weather_filtered = weather
if not weather.empty:
    weather_filtered = weather[weather["Year"].between(year_range[0], year_range[1])]
    if crop_filter != "All crops":
        weather_filtered = weather_filtered[weather_filtered["Crop"] == crop_filter]
    if state_filter != "All states":
        weather_filtered = weather_filtered[weather_filtered["State"] == state_filter]
    if district_filter != "All districts":
        weather_filtered = weather_filtered[weather_filtered["District"] == district_filter]
    if season_filter != "All seasons":
        weather_filtered = weather_filtered[weather_filtered["Season"] == season_filter]
if page == "Overview":
    st.header("Production overview")
    if filtered.empty:
        show_empty("No crop records match these filters. Adjust the sidebar filters to continue.")
    else:
        total_production = float(filtered["Production"].sum())
        total_area = float(filtered["Area"].sum())
        weighted_yield = total_production / total_area if total_area > 0 else float("nan")
        crops = filtered.groupby("Crop", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
        regions = filtered.groupby("State", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total production", format_value(total_production), help="Sum of production in the source dataset's units.")
        c2.metric("Cultivated area", format_value(total_area), help="Sum of area in the source dataset's units.")
        c3.metric("Weighted average yield", format_value(weighted_yield), help="Total production divided by total area.")
        c4.metric("Records analyzed", f"{len(filtered):,}", help="Valid rows remaining after the selected filters.")
        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Production by crop")
            chart_data = crops.head(15).sort_values("Production")
            fig = px.bar(chart_data, x="Production", y="Crop", orientation="h", color="Production", color_continuous_scale=["#244b32", "#4a9d63", "#a7ddb2"])
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Production (source units)", yaxis_title="")
            st.plotly_chart(plotly_theme(fig), width="stretch")
        with right:
            st.subheader("Historical production")
            yearly = filtered.groupby("Year", as_index=False)["Production"].sum()
            fig = px.line(yearly, x="Year", y="Production", markers=True)
            fig.update_traces(line={"color": "#86d39b", "width": 3}, marker={"size": 7})
            fig.update_layout(xaxis_title="Year", yaxis_title="Production (source units)")
            st.plotly_chart(plotly_theme(fig), width="stretch")
        left, right = st.columns(2)
        with left:
            st.subheader("Leading crops")
            st.dataframe(crops.head(8).rename(columns={"Production": "Total production"}), hide_index=True, width="stretch")
        with right:
            st.subheader("Leading regions")
            st.dataframe(regions.head(8).rename(columns={"Production": "Total production"}), hide_index=True, width="stretch")
        leader_crop = crops.iloc[0]
        leader_region = regions.iloc[0]
        st.success(
            f"{leader_crop['Crop']} leads filtered crop production; "
            f"{leader_region['State']} is the leading state/territory. "
            "These rankings describe the selected historical records, not current production."
        )
    st.caption("Yield summary uses total production divided by total cultivated area; units follow the source data.")
elif page == "Weather analytics":
    st.header("Weather & yield analytics")
    if weather_message:
        st.warning(weather_message)
    elif weather_filtered.empty:
        show_empty("No matched crop/weather rows are available under these filters.")
    else:
        st.markdown(
            '<p class="section-note">Weather is recorded at state/year level and repeated across matched crop records. '
            "Patterns are observational and do not establish that weather caused yield changes.</p>",
            unsafe_allow_html=True,
        )
        feature_labels = {
            "Total_Precipitation_mm": "Total precipitation (mm)",
            "Mean_Temperature": "Mean temperature",
            "Avg_Max_Temp": "Average maximum temperature",
            "Avg_Min_Temp": "Average minimum temperature",
            "Max_Wind_Speed": "Maximum wind speed",
        }
        feature = st.selectbox("Weather measure", list(feature_labels), format_func=lambda key: feature_labels[key])
        points = weather_filtered[[feature, "Yield", "Crop"]].dropna()
        if len(points) > 12000:
            points = points.sample(12000, random_state=42)
        if points.empty or points[feature].nunique() < 2 or points["Yield"].nunique() < 2:
            show_empty("There is not enough varying paired data to draw a weather/yield comparison.")
        else:
            fig = px.scatter(
                points,
                x=feature,
                y="Yield",
                color="Crop" if crop_filter == "All crops" else None,
                opacity=0.52,
                labels={feature: feature_labels[feature], "Yield": "Yield (source units)"},
                hover_data=["Crop"],
            )
            fig.update_layout(title=f"{feature_labels[feature]} vs observed yield")
            st.plotly_chart(plotly_theme(fig), width="stretch")
            paired = weather_filtered[[feature, "Yield"]].dropna()
            correlation = paired[feature].corr(paired["Yield"]) if len(paired) > 1 else np.nan
            if np.isfinite(correlation):
                st.info(f"Pearson correlation in the filtered paired rows: **{correlation:.3f}**. This is an association, not evidence of causation.")
        st.subheader("Crop-wise weather comparison")
        crop_weather = weather_filtered.groupby("Crop", as_index=False).agg(
            Mean_Weather=(feature, "mean"),
            Mean_Yield=("Yield", "mean"),
            Records=("Yield", "size"),
        ).sort_values("Mean_Weather", ascending=False)
        fig = px.bar(
            crop_weather.head(15).sort_values("Mean_Weather"),
            x="Mean_Weather",
            y="Crop",
            orientation="h",
            labels={"Mean_Weather": feature_labels[feature], "Crop": ""},
            hover_data=["Mean_Yield", "Records"],
        )
        st.plotly_chart(plotly_theme(fig), width="stretch")
        st.subheader("Interactive numeric correlation")
        numeric_candidates = ["Year", "Area", "Production", "Yield"] + WEATHER_FEATURES
        selected_numeric = st.multiselect(
            "Numeric measures",
            numeric_candidates,
            default=["Yield", "Total_Precipitation_mm", "Mean_Temperature", "Avg_Max_Temp"],
        )
        if len(selected_numeric) < 2:
            show_empty("Select at least two numeric measures.")
        else:
            numeric_frame = weather_filtered[selected_numeric].replace([np.inf, -np.inf], np.nan)
            numeric_frame = numeric_frame.dropna(axis=1, how="all")
            numeric_frame = numeric_frame.loc[:, numeric_frame.nunique(dropna=True) > 1]
            if numeric_frame.shape[1] < 2:
                show_empty("The selected data has fewer than two varying numeric columns.")
            else:
                corr = numeric_frame.corr(min_periods=2).dropna(axis=0, how="all").dropna(axis=1, how="all")
                if corr.empty:
                    show_empty("No valid pairwise correlations can be calculated for this selection.")
                else:
                    fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdYlGn", aspect="auto")
                    fig.update_layout(title="Pairwise Pearson correlations")
                    st.plotly_chart(plotly_theme(fig), width="stretch")
        st.warning("Humidity is not present in the matched crop/weather data, so humidity-versus-yield analysis is unavailable.")
elif page == "Regional map":
    st.header("Regional production map")
    geojson, geojson_message = load_geojson()
    if geojson_message:
        st.warning(geojson_message)
    elif filtered.empty:
        show_empty("No production records match the selected filters.")
    else:
        map_years = sorted(filtered["Year"].unique().tolist())
        map_year = st.selectbox("Map year", map_years, index=len(map_years) - 1)
        map_data = filtered[filtered["Year"] == map_year].copy()
        state_totals = map_data.groupby("State", as_index=False).agg(
            Production=("Production", "sum"),
            Area=("Area", "sum"),
        )
        state_totals["Yield"] = state_totals["Production"] / state_totals["Area"]
        state_totals["map_key"] = state_totals["State"].map(normalize_state)
        geo_keys = {feature["properties"]["map_key"] for feature in geojson["features"]}
        mapped = state_totals[state_totals["map_key"].isin(geo_keys)].copy()
        unmatched = sorted(set(state_totals["State"]) - set(mapped["State"]))
        metric = st.radio("Map measure", ["Production", "Yield"], horizontal=True, key="map_metric")
        if mapped.empty:
            show_empty("No selected state names match the bundled India boundary identifiers.")
        else:
            fig = go.Figure(
                go.Choropleth(
                    geojson=geojson,
                    locations=mapped["map_key"],
                    z=mapped[metric],
                    featureidkey="properties.map_key",
                    text=mapped["State"],
                    customdata=mapped[["Production", "Area", "Yield"]].to_numpy(),
                    colorscale=[[0, "#d9eee0"], [0.5, "#65b77a"], [1, "#14532d"]],
                    marker_line_color="#52685a",
                    marker_line_width=0.55,
                    colorbar_title="Production" if metric == "Production" else "Yield",
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        + metric
                        + ": %{z:,.2f}<br>Production: %{customdata[0]:,.2f}"
                        + "<br>Area: %{customdata[1]:,.2f}<br>Yield: %{customdata[2]:,.3f}<extra></extra>"
                    ),
                )
            )
            fig.update_geos(fitbounds="locations", visible=False, bgcolor="rgba(0,0,0,0)")
            fig.update_layout(title=f"{metric} by state/territory · {map_year}", height=610)
            st.plotly_chart(plotly_theme(fig), width="stretch")
            leaders = mapped.sort_values(metric, ascending=False)
            left, right = st.columns(2)
            left.metric("Highest mapped region", str(leaders.iloc[0]["State"]), format_value(float(leaders.iloc[0][metric])))
            right.metric("Lowest mapped region", str(leaders.iloc[-1]["State"]), format_value(float(leaders.iloc[-1][metric])))
            if unmatched:
                st.caption("Not displayed because names do not match the supplied boundaries: " + ", ".join(unmatched))
            st.caption("The supplied boundaries are state/territory-level only; no district polygons are included.")
elif page == "Soil nutrients":
    st.header("Soil nutrient sample analysis")
    if soil_message:
        st.warning(soil_message)
    elif soil_data.empty:
        show_empty("The soil sample dataset has no usable rows.")
    else:
        st.warning(
            "The available NPK/pH/humidity values are from a standalone crop-recommendation sample. "
            "They have no state, district, or year keys and are not linked to crop production records."
        )
        nutrient = st.selectbox("Nutrient", ["N", "P", "K"], format_func=lambda key: {"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)"}[key])
        valid_nutrient = soil_data[[nutrient, "label"]].dropna()
        left, right = st.columns([1.2, 1])
        with left:
            fig = px.histogram(valid_nutrient, x=nutrient, color="label", marginal="box", barmode="overlay", opacity=0.72)
            fig.update_layout(title=f"{nutrient} sample distribution", xaxis_title=f"{nutrient} (dataset units)", yaxis_title="Sample count")
            st.plotly_chart(plotly_theme(fig), width="stretch")
        with right:
            crop_nutrients = soil_data.groupby("label", as_index=False)[["N", "P", "K"]].mean().sort_values(nutrient, ascending=False)
            fig = px.bar(crop_nutrients.head(15).sort_values(nutrient), x=nutrient, y="label", orientation="h", labels={"label": "Sample crop label"})
            fig.update_layout(title="Mean sample nutrient by crop label")
            st.plotly_chart(plotly_theme(fig), width="stretch")
        st.subheader("Sample dataset summary")
        st.dataframe(soil_data[["N", "P", "K"]].describe().round(2), width="stretch")
        st.info("Regional soil nutrient comparisons and nutrient-to-yield relationships cannot be calculated from these unlinked sample records.")
elif page == "Irrigation advisory":
    st.header("Demonstration irrigation advisory")
    st.warning(
        "This is a historical-weather screening demonstration, not a forecast or an irrigation prescription. "
        "It does not estimate irrigation depth. Real recommendations require crop-stage water requirements, "
        "soil moisture, local management data, and reliable forecast information."
    )
    if state_weather.empty:
        if state_weather_message:
            st.warning(state_weather_message)
        else:
            show_empty("Matched weather observations are unavailable.")
    else:
        state_year = state_weather[["State", "Year"] + WEATHER_FEATURES].copy()
        available_advisory = state_year
        if state_filter != "All states":
            available_advisory = available_advisory[available_advisory["State"] == state_filter]
        available_advisory = available_advisory[
            available_advisory["Year"].between(year_range[0], year_range[1])
        ]
        if available_advisory.empty:
            show_empty("No state/year weather observation matches the current state and year filters.")
        else:
            advisory_state = st.selectbox("State for historical weather context", sorted(available_advisory["State"].unique()))
            state_records = state_year[state_year["State"] == advisory_state].sort_values("Year")
            advisory_years = state_records["Year"].astype(int).tolist()
            advisory_year = st.selectbox("Observed weather year", advisory_years, index=len(advisory_years) - 1)
            observation = state_records[state_records["Year"] == advisory_year].iloc[0]
            prior = state_records[state_records["Year"] != advisory_year]
            rain_values = prior["Total_Precipitation_mm"].dropna()
            temperature_values = prior["Avg_Max_Temp"].dropna()
            rainfall = float(observation["Total_Precipitation_mm"])
            max_temp = float(observation["Avg_Max_Temp"])
            st.markdown(f"**Observed annual context:** {advisory_state}, {advisory_year}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total precipitation", f"{rainfall:,.1f} mm")
            c2.metric("Average maximum temperature", f"{max_temp:,.1f} °C")
            c3.metric("Humidity", "Not available")
            notes = []
            if len(rain_values) >= 3:
                rain_threshold = float(rain_values.quantile(0.25))
                if rainfall <= rain_threshold:
                    notes.append(f"Precipitation is at or below the state's historical lower quartile ({rain_threshold:,.1f} mm).")
                else:
                    notes.append(f"Precipitation is above the state's historical lower-quartile reference ({rain_threshold:,.1f} mm).")
            else:
                notes.append("There are too few other state-year observations to establish a rainfall reference.")
            if len(temperature_values) >= 3:
                heat_threshold = float(temperature_values.quantile(0.75))
                if max_temp >= heat_threshold:
                    notes.append(f"Average maximum temperature is at or above the state's historical upper quartile ({heat_threshold:,.1f} °C).")
                else:
                    notes.append(f"Average maximum temperature is below the state's historical upper-quartile reference ({heat_threshold:,.1f} °C).")
            else:
                notes.append("There are too few other state-year observations to establish a temperature reference.")
            for note in notes:
                st.write(f"- {note}")
            st.info("Check field soil moisture and consult local crop-stage guidance before any irrigation decision. Humidity was excluded because no humidity field exists in the matched weather data.")
            st.caption("Rules compare a single observed year with other available years for the same state; they are descriptive thresholds, not agronomic validation.")
elif page == "Yield prediction":
    st.header("Yield prediction")
    st.markdown(
        "A trained weather-aware random-forest pipeline is available. The model uses state/year historical weather "
        "observations and the selected crop, district, season, year, and area. Output is a model estimate, not a guarantee."
    )
    predictor, model_error, version_warnings = load_yield_model()
    if model_error:
        st.warning(f"Model unavailable: {model_error}")
        st.info("The ML integration point remains available in models/random_forest_weather.joblib.")
    else:
        if version_warnings:
            st.warning("The model was saved with a different scikit-learn version. Inference is enabled, but retraining/loading with the matching version is recommended.")
        model_states = sorted(state_weather["State"].dropna().unique()) if not state_weather.empty else state_options
        model_crops = sorted(production["Crop"].dropna().unique())
        model_seasons = sorted(production["Season"].dropna().unique())
        model_districts = sorted(production["District"].dropna().unique())
        prediction_years = sorted(state_weather["Year"].dropna().astype(int).unique()) if not state_weather.empty else []
        with st.form("yield_prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                prediction_state = st.selectbox("State", model_states)
                prediction_crop = st.selectbox("Crop", model_crops)
                prediction_district = st.selectbox("District", model_districts)
            with c2:
                prediction_season = st.selectbox("Season", model_seasons)
                state_years = sorted(
                    state_weather.loc[state_weather["State"] == prediction_state, "Year"].astype(int).unique().tolist()
                ) if not state_weather.empty else []
                if state_years:
                    prediction_year = st.selectbox("Year with historical weather", state_years, index=len(state_years) - 1)
                else:
                    prediction_year = None
                area_value = st.number_input("Cultivated area (source units)", min_value=0.01, value=1.0, step=1.0)
            submitted = st.form_submit_button("Estimate yield", type="primary", disabled=not state_years)
        if submitted and predictor is not None and prediction_year is not None:
            weather_row = state_weather[
                (state_weather["State"] == prediction_state) & (state_weather["Year"] == prediction_year)
            ][WEATHER_FEATURES].dropna()
            if weather_row.empty:
                st.error("No complete historical weather record exists for the selected state and year.")
            else:
                input_values: dict[str, Any] = {
                    "State": prediction_state,
                    "District": prediction_district,
                    "Season": prediction_season,
                    "Crop": prediction_crop,
                    "Year": prediction_year,
                    "Area": area_value,
                }
                input_values.update(weather_row.mean().to_dict())
                model_features = list(predictor.feature_names_in_)
                missing_features = [feature for feature in model_features if feature not in input_values]
                if missing_features:
                    st.error(f"Model expects unsupported input features: {missing_features}")
                else:
                    model_input = pd.DataFrame([input_values]).reindex(columns=model_features)
                    try:
                        predicted_yield = float(np.asarray(predictor.predict(model_input)).reshape(-1)[0])
                    except (ValueError, TypeError, AttributeError, KeyError) as error:
                        st.error(f"Model inference failed: {type(error).__name__}: {error}")
                    else:
                        st.metric("Estimated yield", f"{predicted_yield:,.3f} tons/hectare")
                        st.caption(
                            f"Model: weather-aware random forest · Historical weather year: {prediction_year}. "
                            "No confidence interval, accuracy claim, or feature explanation is generated here."
                        )
elif page == "Reports":
    st.header("Download analytical results")
    if filtered.empty:
        show_empty("No data matches the current filters; adjust them before downloading a report.")
    else:
        total_production = float(filtered["Production"].sum())
        total_area = float(filtered["Area"].sum())
        top_crops = filtered.groupby("Crop", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
        top_regions = filtered.groupby("State", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
        st.write(f"Filtered records: **{len(filtered):,}**")
        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download filtered data (CSV)",
            data=csv_data,
            file_name="agripulse_filtered_crop_data.csv",
            mime="text/csv",
            type="primary",
        )
        selected_filters = {
            "Crop": crop_filter,
            "State": state_filter,
            "District": district_filter,
            "Season": season_filter,
            "Years": f"{year_range[0]}–{year_range[1]}",
        }
        try:
            pdf_data = make_pdf_report(filtered, top_crops, top_regions, selected_filters)
        except ModuleNotFoundError as error:
            st.error(f"PDF generation is unavailable because a required package is missing: {error}")
        else:
            st.download_button(
                "Download seasonal summary (PDF)",
                data=pdf_data,
                file_name="agripulse_seasonal_summary.pdf",
                mime="application/pdf",
            )
        st.subheader("Current filter summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Production", format_value(total_production))
        c2.metric("Cultivated area", format_value(total_area))
        c3.metric("Weighted yield", format_value(total_production / total_area if total_area else float("nan")))
if weather_message and page not in {"Weather analytics", "Irrigation advisory"}:
    st.caption(f"Weather-dependent features: {weather_message}")
if state_weather_message and page not in {"Irrigation advisory", "Yield prediction"}:
    st.caption(f"State/year weather context: {state_weather_message}")
if soil_message and page != "Soil nutrients":
    st.caption(f"Soil sample analysis: {soil_message}")
