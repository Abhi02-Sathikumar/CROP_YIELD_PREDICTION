from __future__ import annotations

import base64
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

from src.irrigation import (
    crop_water_stress_index,
    irrigation_advisory,
)


PROJECT_DIR = Path(__file__).resolve().parent
PRODUCTION_PATH = PROJECT_DIR / "data" / "processed_data" / "crop_production_preprocessed.csv"
WEATHER_PATH = PROJECT_DIR / "data" / "processed_data" / "crop_weather_merged.csv"
STATE_WEATHER_PATH = PROJECT_DIR / "data" / "processed_data" / "india_states_yearly_weather_preprocessed.csv"
SOIL_PATH = PROJECT_DIR / "data" / "processed_data" / "crop_recommendation_preprocessed.csv"
GEOJSON_PATH = PROJECT_DIR / "assets" / "india_state.geojson"
CROP_MODEL_PATH = PROJECT_DIR / "assets" / "maize_corn_plant.glb"
MODEL_PATH = PROJECT_DIR / "models" / "random_forest_weather.joblib"
WEATHER_FEATURES = [
    "Mean_Temperature",
    "Avg_Max_Temp",
    "Avg_Min_Temp",
    "Total_Precipitation_mm",
    "Max_Wind_Speed",
]
AREA_UNIT = "hectares"
PRODUCTION_UNIT = "metric tonnes"
YIELD_UNIT = f"{PRODUCTION_UNIT} per hectare"
HECTARES_PER_ACRE = 0.40468564224
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
    page_icon=str(PROJECT_DIR / "assets" / "logo.png"),
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


def convert_area_to_hectares(area: float, unit: str) -> float:
    if unit == "acres":
        return area * HECTARES_PER_ACRE
    if unit == AREA_UNIT:
        return area
    raise ValueError(f"Unsupported area unit: {unit}")


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


def simplify_geojson_coordinates(coordinates: Any, tolerance: float = 0.005) -> Any:
    if not coordinates:
        return coordinates

    if isinstance(coordinates[0], list) and len(coordinates[0]) >= 2 and isinstance(coordinates[0][0], (int, float)):
        if len(coordinates) <= 4:
            return coordinates

        keep = {0, len(coordinates) - 1}
        stack = [(0, len(coordinates) - 1)]
        tolerance_squared = tolerance * tolerance
        while stack:
            start, end = stack.pop()
            x1, y1 = coordinates[start][:2]
            x2, y2 = coordinates[end][:2]
            dx, dy = x2 - x1, y2 - y1
            segment_length_squared = dx * dx + dy * dy
            max_distance_squared = 0.0
            max_index = -1
            for index in range(start + 1, end):
                x, y = coordinates[index][:2]
                if segment_length_squared:
                    cross = (x - x1) * dy - (y - y1) * dx
                    distance_squared = cross * cross / segment_length_squared
                else:
                    offset_x, offset_y = x - x1, y - y1
                    distance_squared = offset_x * offset_x + offset_y * offset_y
                if distance_squared > max_distance_squared:
                    max_distance_squared = distance_squared
                    max_index = index
            if max_index >= 0 and max_distance_squared > tolerance_squared:
                keep.add(max_index)
                stack.append((start, max_index))
                stack.append((max_index, end))

        simplified = [coordinates[index] for index in sorted(keep)]
        return simplified if len(simplified) >= 4 else coordinates

    return [simplify_geojson_coordinates(part, tolerance) for part in coordinates]


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
        feature["geometry"]["coordinates"] = simplify_geojson_coordinates(
            feature["geometry"]["coordinates"]
        )
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


@st.cache_data(show_spinner=False)
def load_crop_model_data() -> str:
    if not CROP_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Required overview crop model not found: {CROP_MODEL_PATH}")
    return base64.b64encode(CROP_MODEL_PATH.read_bytes()).decode("ascii")


def render_crop_model_background(enabled: bool) -> None:
    model_source = ""
    if enabled:
        model_source = f"data:model/gltf-binary;base64,{load_crop_model_data()}"

    html = """
    <script>
    (() => {
      const enabled = __MODEL_ENABLED__;
      const modelSource = __MODEL_SOURCE__;
      const layerId = "agri-crop-model-layer";
      const app = document.querySelector(".stApp");

      if (!enabled) {
        if (app) app.classList.remove("agri-overview");
        const scrollHandler = window.__agriCropScrollHandler;
        if (scrollHandler) {
          window.__agriCropScrollTarget.removeEventListener("scroll", scrollHandler);
          delete window.__agriCropScrollHandler;
          delete window.__agriCropScrollTarget;
        }
        if (window.__agriCropScrollFrame) {
          cancelAnimationFrame(window.__agriCropScrollFrame);
          delete window.__agriCropScrollFrame;
        }
        const layer = document.getElementById(layerId);
        if (layer) layer.remove();
        return;
      }

      if (!app) return;
      app.classList.add("agri-overview");

      let layer = document.getElementById(layerId);
      if (!layer) {
        layer = document.createElement("div");
        layer.id = layerId;
        layer.setAttribute("aria-hidden", "true");
        app.prepend(layer);
      }

      const showFallback = (message) => {
        let fallback = layer.querySelector(".agri-model-fallback");
        if (!fallback) {
          fallback = document.createElement("div");
          fallback.className = "agri-model-fallback";
          layer.appendChild(fallback);
        }
        fallback.textContent = message;
      };

      const mountModel = () => {
        if (!app.classList.contains("agri-overview")) return;
        if (!customElements.get("model-viewer")) return;

        let model = layer.querySelector("#agri-crop-model");
        if (!model) {
          model = document.createElement("model-viewer");
          model.id = "agri-crop-model";
          model.src = modelSource;
          model.alt = "Three-dimensional rice plant";
          model.setAttribute("camera-orbit", "24deg 72deg auto");
          model.setAttribute("field-of-view", "30deg");
          model.setAttribute("environment-image", "neutral");
          model.setAttribute("shadow-intensity", "0.35");
          model.setAttribute("exposure", "1.1");
          model.setAttribute("loading", "eager");
          layer.appendChild(model);
          model.addEventListener("error", () => {
            model.style.display = "none";
            showFallback("The 3D crop model could not be loaded.");
          }, { once: true });
        }

        if (model.dataset.scrollRotationBound === "true") return;
        model.dataset.scrollRotationBound = "true";
        const rotateWithScroll = () => {
          if (window.__agriCropScrollFrame) return;
          window.__agriCropScrollFrame = requestAnimationFrame(() => {
            delete window.__agriCropScrollFrame;
            const scrollPosition = window.__agriCropScrollTarget === window
              ? window.scrollY
              : window.__agriCropScrollTarget.scrollTop;
            const rotation = (24 + scrollPosition * 0.14) % 360;
            model.cameraOrbit = `${rotation.toFixed(1)}deg 72deg auto`;
          });
        };
        window.__agriCropScrollHandler = rotateWithScroll;
        window.__agriCropScrollTarget = document.querySelector('[data-testid="stMain"]') || window;
        window.__agriCropScrollTarget.addEventListener("scroll", rotateWithScroll, { passive: true });
        rotateWithScroll();
      };

      if (customElements.get("model-viewer")) {
        mountModel();
        return;
      }

      let loader = document.getElementById("agri-model-viewer-loader");
      if (!loader) {
        loader = document.createElement("script");
        loader.id = "agri-model-viewer-loader";
        loader.type = "module";
        loader.src = "https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js";
        document.head.appendChild(loader);
      }
      if (loader.dataset.failed === "true") {
        showFallback("The 3D viewer could not load. Check your internet connection.");
      } else {
        loader.addEventListener("load", mountModel, { once: true });
        loader.addEventListener("error", () => {
          loader.dataset.failed = "true";
          showFallback("The 3D viewer could not load. Check your internet connection.");
        }, { once: true });
      }
    })();
    </script>
    """
    html = html.replace("__MODEL_ENABLED__", json.dumps(enabled))
    html = html.replace("__MODEL_SOURCE__", json.dumps(model_source))
    st.html(html, unsafe_allow_javascript=True)


def plotly_theme(fig: go.Figure) -> go.Figure:
    is_dark = st.session_state.get("theme_mode", "Light") == "Dark"
    font_color = "#e5f1e8" if is_dark else "#20352a"
    grid_color = "rgba(190, 219, 198, 0.18)" if is_dark else "rgba(46, 92, 61, 0.14)"
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": font_color, "family": "Aptos, Segoe UI, sans-serif"},
        margin={"l": 12, "r": 12, "t": 48, "b": 12},
        colorway=["#187343", "#42a36a", "#8bcf9b", "#246b50", "#b2dfb8"],
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=grid_color, zeroline=False, automargin=True)
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
        [f"Production total ({PRODUCTION_UNIT})", format_value(total_production)],
        [f"Cultivated area total ({AREA_UNIT})", format_value(total_area)],
        [f"Weighted yield ({YIELD_UNIT})", format_value(weighted_yield)],
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
        story.append(Paragraph(f"{escape(str(row['Crop']))}: {format_value(float(row['Production']))} {PRODUCTION_UNIT}", styles["BodyText"]))
    story.extend([Spacer(1, 10), Paragraph("Leading regions by production", styles["Heading2"])])
    for _, row in region_summary.head(5).iterrows():
        story.append(Paragraph(f"{escape(str(row['State']))}: {format_value(float(row['Production']))} {PRODUCTION_UNIT}", styles["BodyText"]))
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

st.session_state.setdefault("theme_mode", "Light")
is_dark = st.session_state["theme_mode"] == "Dark"
theme = {
    "background": "#0d1711" if is_dark else "#f4f8f4",
    "surface": "#14231a" if is_dark else "#ffffff",
    "surface_alt": "#1a2d21" if is_dark else "#eaf2eb",
    "text": "#e7f0e8" if is_dark else "#20352a",
    "muted": "#aabdaf" if is_dark else "#5d7163",
    "border": "rgba(173, 207, 181, .18)" if is_dark else "rgba(40, 91, 54, .16)",
    "sidebar": "#112219" if is_dark else "#e8f2e9",
    "hero_start": "#1e633d" if is_dark else "#d8efdc",
    "hero_end": "#173e2b" if is_dark else "#f4faf4",
    "metric_start": "#1d4e33" if is_dark else "#e3f3e6",
    "metric_end": "#173b2a" if is_dark else "#f7fbf7",
    "hero_kicker": "#9ed8aa" if is_dark else "#287647",
    "model_opacity": ".38" if is_dark else ".16",
    "model_shadow": "rgba(0, 0, 0, .34)" if is_dark else "rgba(24, 74, 42, .18)",
}

st.markdown(
    f"""
    <style>
    :root {{ color-scheme: {"dark" if is_dark else "light"}; }}
    .stApp {{
        background: linear-gradient(145deg, {theme["background"]}, {theme["surface_alt"]} 58%, {theme["background"]});
        color: {theme["text"]};
        font-family: "Aptos", "Segoe UI", sans-serif;
        font-size: 16px;
    }}
    .stApp.agri-overview [data-testid="stMain"],
    .stApp.agri-overview [data-testid="stSidebar"] {{ position: relative; z-index: 1; }}
    #agri-crop-model-layer {{
        position: fixed; z-index: 0; top: 3vh; right: -8vw;
        width: min(68vw, 980px); height: 92vh;
        pointer-events: none; opacity: {theme["model_opacity"]};
        filter: saturate(1.05) drop-shadow(0 24px 48px {theme["model_shadow"]});
    }}
    #agri-crop-model-layer model-viewer {{
        width: 100%; height: 100%; background: transparent; --poster-color: transparent;
    }}
    .agri-model-fallback {{
        position: absolute; right: 12vw; bottom: 12vh; max-width: 15rem;
        padding: .7rem 1rem; border: 1px solid rgba(43, 126, 72, .36);
        border-radius: 12px; background: {theme["surface"]};
        color: {theme["text"]}; font-size: .85rem; backdrop-filter: blur(14px);
    }}
    @media (max-width: 760px) {{
        #agri-crop-model-layer {{
            top: 8vh; right: -28vw; width: 96vw; height: 76vh; opacity: .12;
        }}
    }}
    .block-container {{ padding-top: 4rem; padding-bottom: 2.8rem; max-width: 1500px; }}
    [data-testid="stHeader"] {{
        background: {theme["background"]};
        color: {theme["text"]};
    }}
    [data-testid="stHeader"] button {{ color: {theme["text"]} !important; }}
    .agri-brand {{
        display: flex; align-items: center; gap: .2rem; color: {theme["text"]};
        font-size: 1.65rem; font-weight: 800; letter-spacing: -.04em; line-height: 1.1;
    }}
    .agri-tagline {{ color: {theme["muted"]}; font-size: .9rem; margin-top: .25rem; }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(165deg, {theme["sidebar"]}, {theme["surface"]});
        border-right: 1px solid {theme["border"]};
        box-shadow: 8px 0 30px rgba(18, 68, 37, .08);
    }}
    [data-testid="stSidebar"] * {{ color: {theme["text"]}; }}
    [data-testid="stSidebar"] hr {{ border-color: {theme["border"]}; }}
    [data-testid="stSidebar"] [role="radiogroup"] {{ gap: .2rem; }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
        border-radius: 10px; padding: .45rem .55rem; transition: background .15s ease;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: {theme["surface_alt"]}; }}
    input[type="radio"] {{ accent-color: #187343; }}
    [data-baseweb="select"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {{
        background-color: {theme["surface"]}; color: {theme["text"]};
        border-color: {theme["border"]};
    }}
    h1, h2, h3 {{
        color: {theme["text"]} !important;
        font-family: "Aptos Display", "Aptos", "Segoe UI", sans-serif;
        font-weight: 750 !important; letter-spacing: -0.025em;
    }}
    h1 {{ line-height: 1.12; }}
    h2, h3 {{ line-height: 1.25; }}
    .stApp p, .stApp label, [data-testid="stCaptionContainer"] {{
        font-family: "Aptos", "Segoe UI", sans-serif; line-height: 1.6;
    }}
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, {theme["metric_start"]}, {theme["metric_end"]});
        border: 1px solid {theme["border"]}; border-radius: 16px;
        padding: 18px 20px; box-shadow: 0 10px 26px rgba(20, 75, 40, .07);
    }}
    [data-testid="stMetricLabel"] {{ color: {theme["muted"]} !important; font-weight: 650 !important; }}
    [data-testid="stMetricValue"] {{ color: {theme["text"]} !important; font-weight: 700 !important; }}
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{
        border: 1px solid {theme["border"]}; border-radius: 14px;
        background: {theme["surface"]};
    }}
    div[data-testid="stAlert"] {{ border-radius: 12px; }}
    [data-testid="stButtonGroup"] button[role="radio"] {{
        font-family: "Aptos", "Segoe UI", sans-serif; font-weight: 650;
        transition: background-color .15s ease, border-color .15s ease, color .15s ease;
    }}
    [data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] {{
        background: linear-gradient(120deg, #1e7545, #3a9b5b) !important;
        border-color: #2b8750 !important; color: #ffffff !important;
    }}
    .hero {{
        border: 1px solid {theme["border"]}; border-radius: 18px; padding: 22px 26px;
        background: linear-gradient(120deg, {theme["hero_start"]}, {theme["hero_end"]});
        box-shadow: 0 12px 30px rgba(20, 75, 40, .08); margin-bottom: 1.2rem;
    }}
    .hero-kicker {{ color: {theme["hero_kicker"]}; font-size: .8rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }}
    .hero h1 {{ font-size: clamp(1.8rem, 3vw, 2.6rem); margin: .35rem 0 .55rem; }}
    .hero p {{ color: {theme["muted"]}; margin: 0; max-width: 900px; font-size: 1rem; }}
    .section-note {{ color: {theme["muted"]}; font-size: .98rem; line-height: 1.65; }}
    </style>
    """,
    unsafe_allow_html=True,
)

brand_columns = st.columns([0.14, 0.48, 0.38], vertical_alignment="center")
with brand_columns[0]:
    st.image(str(PROJECT_DIR / "assets" / "logo.png"), width=84)
with brand_columns[1]:
    st.markdown('<div class="agri-brand">AgriPulse</div>', unsafe_allow_html=True)
    st.markdown('<div class="agri-tagline">Agricultural intelligence for better decisions</div>', unsafe_allow_html=True)
with brand_columns[2]:
    st.segmented_control(
        "Display mode",
        ["Light", "Dark"],
        key="theme_mode",
        label_visibility="collapsed",
        width="stretch",
    )

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Agricultural intelligence · India</div>
      <h1>Crop insights, all in one place</h1>
      <p>Explore production patterns, historical weather associations, regional performance,
      and decision-support tools grounded in the project datasets.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## AgriPulse")
    st.caption("Crop yield and irrigation intelligence")
    st.divider()
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

render_crop_model_background(page == "Overview")

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
    st.caption("A quick summary of production, cultivated area, and yield across the records that match your sidebar filters.")
    if filtered.empty:
        show_empty("No crop records match these filters. Adjust the sidebar filters to continue.")
    else:
        total_production = float(filtered["Production"].sum())
        total_area = float(filtered["Area"].sum())
        weighted_yield = total_production / total_area if total_area > 0 else float("nan")
        crops = filtered.groupby("Crop", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
        regions = filtered.groupby("State", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Total production ({PRODUCTION_UNIT})", format_value(total_production), help="Sum of recorded production.")
        c2.metric(f"Cultivated area ({AREA_UNIT})", format_value(total_area), help="Sum of recorded cultivated area.")
        c3.metric(f"Weighted average yield ({YIELD_UNIT})", format_value(weighted_yield), help="Total production divided by total area.")
        c4.metric("Records analyzed", f"{len(filtered):,}", help="Valid rows remaining after the selected filters.")

        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Production by crop")
            chart_data = crops.head(15).sort_values("Production")
            fig = px.bar(chart_data, x="Production", y="Crop", orientation="h", color="Production", color_continuous_scale=["#d7efdc", "#65b77d", "#187343"])
            fig.update_layout(coloraxis_showscale=False, xaxis_title=f"Production ({PRODUCTION_UNIT})", yaxis_title="")
            st.plotly_chart(plotly_theme(fig), width="stretch")
        with right:
            st.subheader("Historical production")
            yearly = filtered.groupby("Year", as_index=False)["Production"].sum()
            fig = px.line(yearly, x="Year", y="Production", markers=True)
            fig.update_traces(line={"color": "#187343", "width": 3}, marker={"size": 7, "color": "#55b879"})
            fig.update_layout(xaxis_title="Year", yaxis_title=f"Production ({PRODUCTION_UNIT})")
            st.plotly_chart(plotly_theme(fig), width="stretch")

        left, right = st.columns(2)
        with left:
            st.subheader("Leading crops")
            st.dataframe(crops.head(8).rename(columns={"Production": f"Total production ({PRODUCTION_UNIT})"}), hide_index=True, width="stretch")
        with right:
            st.subheader("Leading regions")
            st.dataframe(regions.head(8).rename(columns={"Production": f"Total production ({PRODUCTION_UNIT})"}), hide_index=True, width="stretch")

        leader_crop = crops.iloc[0]
        leader_region = regions.iloc[0]
        st.success(
            f"{leader_crop['Crop']} leads filtered crop production; "
            f"{leader_region['State']} is the leading state/territory. "
            "These rankings describe the selected historical records, not current production."
        )
    st.caption(f"Yield is total production divided by total cultivated area ({YIELD_UNIT}).")

elif page == "Weather analytics":
    st.header("Weather and yield analytics")
    st.caption(
        "Compare historical weather conditions with recorded crop yield for the selected filters. "
        "Explore the relationship in 2D or add a third measure in 3D. These observations show association, "
        "not proof that weather caused a change in yield."
    )
    if weather_message:
        st.warning(weather_message)
    elif weather_filtered.empty:
        show_empty("No matched crop/weather rows are available under these filters.")
    else:
        st.info(
            "Weather is recorded by state and year, then repeated across matched crop records. "
            "Interpret comparisons as observational; they do not establish causation."
        )
        feature_labels = {
            "Total_Precipitation_mm": "Total precipitation (mm)",
            "Mean_Temperature": "Mean temperature (°C)",
            "Avg_Max_Temp": "Average maximum temperature (°C)",
            "Avg_Min_Temp": "Average minimum temperature (°C)",
            "Max_Wind_Speed": "Maximum wind speed",
        }
        st.subheader("Weather and yield comparison")
        st.caption(
            "Select a weather measure, then compare it with recorded yield. Each point represents a matched crop record."
        )
        feature = st.selectbox("Weather measure", list(feature_labels), format_func=lambda key: feature_labels[key])
        points = weather_filtered[[feature, "Yield", "Crop"]].dropna()
        if len(points) > 12000:
            points = points.sample(12000, random_state=42)
        chart_mode = st.segmented_control(
            "Chart view",
            ["2D comparison", "3D exploration"],
            default="2D comparison",
            key="weather_chart_mode",
            label_visibility="collapsed",
        )
        chart_mode = chart_mode or "2D comparison"
        if chart_mode == "2D comparison":
            if points.empty or points[feature].nunique() < 2 or points["Yield"].nunique() < 2:
                show_empty("There is not enough variation in the selected weather and yield values to draw a comparison.")
            else:
                fig = px.scatter(
                    points,
                    x=feature,
                    y="Yield",
                    opacity=0.52,
                    labels={
                        feature: feature_labels[feature],
                        "Yield": f"Yield ({YIELD_UNIT})",
                    },
                    hover_data=["Crop"],
                )
                fig.update_layout(
                    title=f"{feature_labels[feature]} and observed yield",
                    height=620,
                )
                st.plotly_chart(plotly_theme(fig), width="stretch")
        else:
            z_options = {
                "Crop year": "Year",
                f"Production ({PRODUCTION_UNIT})": "Production",
                f"Cultivated area ({AREA_UNIT})": "Area",
                **{label: key for key, label in feature_labels.items() if key != feature},
            }
            z_label = st.selectbox(
                "Choose the vertical axis",
                list(z_options),
                help="This adds a third measure to the weather-versus-yield chart.",
            )
            st.caption(
                "The x-axis shows the selected weather measure, the y-axis shows recorded yield, and the z-axis "
                f"shows {z_label.lower()}. Drag the chart to rotate it or use the toolbar to reset the view."
            )
            z_feature = z_options[z_label]
            points_3d = weather_filtered[[feature, "Yield", z_feature, "Crop"]].dropna()
            if len(points_3d) > 6000:
                points_3d = points_3d.sample(6000, random_state=42)
            if (
                points_3d.empty
                or points_3d[feature].nunique() < 2
                or points_3d["Yield"].nunique() < 2
                or points_3d[z_feature].nunique() < 2
            ):
                show_empty("There is not enough varying data across all three measures to draw a 3D comparison.")
            else:
                fig = px.scatter_3d(
                    points_3d,
                    x=feature,
                    y="Yield",
                    z=z_feature,
                    opacity=0.52,
                    labels={
                        feature: feature_labels[feature],
                        "Yield": f"Yield ({YIELD_UNIT})",
                        z_feature: z_label,
                    },
                    hover_data=["Crop"],
                )
                fig.update_layout(
                    title=f"Weather, yield, and {z_label.lower()}",
                    height=680,
                    scene={
                        "xaxis_title": feature_labels[feature],
                        "yaxis_title": f"Yield ({YIELD_UNIT})",
                        "zaxis_title": z_label,
                    },
                )
                st.plotly_chart(plotly_theme(fig), width="stretch")

        paired = weather_filtered[[feature, "Yield"]].dropna()
        correlation = paired[feature].corr(paired["Yield"]) if len(paired) > 1 else np.nan
        if np.isfinite(correlation):
            st.info(
                f"Pearson correlation between the selected weather measure and yield: **{correlation:.3f}** "
                f"across {len(paired):,} matched records. This describes association, not causation."
            )

        st.divider()
        st.subheader("Average weather by crop")
        st.caption(
            "Compare the average selected weather measure across crops. Hover over a bar to see average yield "
            "and the number of records."
        )
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
            labels={
                "Mean_Weather": feature_labels[feature],
                "Mean_Yield": f"Mean yield ({YIELD_UNIT})",
                "Crop": "",
            },
            hover_data=["Mean_Yield", "Records"],
        )
        st.plotly_chart(plotly_theme(fig), width="stretch")

        st.divider()
        st.subheader("Correlation between numeric measures")
        st.caption(
            "Select two or more numeric measures to view their pairwise Pearson correlations. Values range "
            "from -1 (negative association) to 1 (positive association); correlation does not imply causation."
        )
        numeric_candidates = ["Year", "Area", "Production", "Yield"] + WEATHER_FEATURES
        numeric_labels = {
            "Area": f"Area ({AREA_UNIT})",
            "Production": f"Production ({PRODUCTION_UNIT})",
            "Yield": f"Yield ({YIELD_UNIT})",
        }
        selected_numeric = st.multiselect(
            "Numeric measures",
            numeric_candidates,
            default=["Yield", "Total_Precipitation_mm", "Mean_Temperature", "Avg_Max_Temp"],
            format_func=lambda key: numeric_labels.get(key, key),
        )
        if len(selected_numeric) < 2:
            show_empty("Select at least two numeric measures.")
        else:
            numeric_frame = weather_filtered[selected_numeric].replace([np.inf, -np.inf], np.nan)
            numeric_frame = numeric_frame.dropna(axis=1, how="all")
            numeric_frame = numeric_frame.loc[:, numeric_frame.nunique(dropna=True) > 1]
            numeric_frame = numeric_frame.rename(
                columns={key: numeric_labels.get(key, key) for key in numeric_frame.columns}
            )
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
elif page == "Regional map":
    st.header("Regional production map")
    st.caption("Compare state- and territory-level production or yield for one year. Select a measure and hover over a region for its values.")
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
        metric = st.segmented_control(
            f"Map measure (production in {PRODUCTION_UNIT}; yield in {YIELD_UNIT})",
            ["Production", "Yield"],
            default="Production",
            key="map_metric",
            label_visibility="collapsed",
        )
        if metric is None:
            metric = "Production"
        metric_column = metric
        metric_unit = PRODUCTION_UNIT if metric_column == "Production" else YIELD_UNIT
        metric_label = f"{metric_column} ({metric_unit})"
        st.caption(
            f"Production is shown in {PRODUCTION_UNIT}; yield is shown in {YIELD_UNIT}. "
            f"Area in hover details is shown in {AREA_UNIT}."
        )
        if mapped.empty:
            show_empty("No selected state names match the bundled India boundary identifiers.")
        else:
            mapped_keys = set(mapped["map_key"])
            mapped_geojson = {
                **geojson,
                "features": [
                    feature
                    for feature in geojson["features"]
                    if feature["properties"]["map_key"] in mapped_keys
                ],
            }
            fig = go.Figure(
                [
                    go.Choropleth(
                        geojson=geojson,
                        locations=sorted(geo_keys),
                        z=[0] * len(geo_keys),
                        featureidkey="properties.map_key",
                        colorscale=[[0, "rgba(82, 108, 88, 0.48)"], [1, "rgba(82, 108, 88, 0.48)"]],
                        showscale=False,
                        showlegend=False,
                        hoverinfo="skip",
                        marker_line_color="#718777",
                        marker_line_width=0.55,
                    ),
                    go.Choropleth(
                        geojson=mapped_geojson,
                        locations=mapped["map_key"],
                        z=mapped[metric_column],
                        featureidkey="properties.map_key",
                        text=mapped["State"],
                        customdata=mapped[["Production", "Area", "Yield"]].to_numpy(),
                        colorscale=[[0, "#b9dfc3"], [0.48, "#54ad70"], [1, "#ee9a3c"]],
                        marker_line_color="#0b1711",
                        marker_line_width=1.15,
                        colorbar_title=metric_label,
                        hovertemplate=(
                            "<b>%{text}</b><br>"
                            + metric_label
                            + ": %{z:,.2f}<br>"
                            + f"Production: %{{customdata[0]:,.2f}} {PRODUCTION_UNIT}"
                            + f"<br>Area: %{{customdata[1]:,.2f}} {AREA_UNIT}"
                            + f"<br>Yield: %{{customdata[2]:,.3f}} {YIELD_UNIT}<extra></extra>"
                        ),
                    ),
                ]
            )
            fig.update_geos(
                fitbounds="geojson",
                visible=False,
                bgcolor="rgba(0,0,0,0)",
                showframe=True,
                framecolor="#86d39b",
                framewidth=1,
            )
            fig.update_layout(title=f"{metric_label} by state/territory · {map_year}", height=610)
            with st.container(border=True):
                st.plotly_chart(plotly_theme(fig), width="stretch")
            leaders = mapped.sort_values(metric_column, ascending=False)
            left, right = st.columns(2)
            left.metric(
                "Highest mapped region",
                str(leaders.iloc[0]["State"]),
                f"{format_value(float(leaders.iloc[0][metric_column]))} {metric_unit}",
            )
            right.metric(
                "Lowest mapped region",
                str(leaders.iloc[-1]["State"]),
                f"{format_value(float(leaders.iloc[-1][metric_column]))} {metric_unit}",
            )
            if unmatched:
                st.caption("Not displayed because names do not match the supplied boundaries: " + ", ".join(unmatched))
            st.caption("The supplied boundaries are state/territory-level only; no district polygons are included.")

elif page == "Soil nutrients":
    st.header("Soil nutrient sample analysis")
    st.caption("Review the sample dataset's nutrient ranges and crop-label averages. Use the selector to focus on nitrogen, phosphorus, or potassium.")
    if soil_message:
        st.warning(soil_message)
    elif soil_data.empty:
        show_empty("The soil sample dataset has no usable rows.")
    else:
        st.info(
            "These are standalone crop-recommendation samples, not field-level measurements. "
            "They have no location or year identifiers and are not linked to the production records."
        )
        nutrient_labels = {
            "Nitrogen (N)": "N",
            "Phosphorus (P)": "P",
            "Potassium (K)": "K",
        }
        selected_nutrient = st.segmented_control(
            "Choose a nutrient",
            list(nutrient_labels),
            default="Nitrogen (N)",
            key="soil_nutrient",
            label_visibility="collapsed",
        )
        selected_nutrient = selected_nutrient or "Nitrogen (N)"
        nutrient = nutrient_labels[selected_nutrient]
        valid_nutrient = soil_data[[nutrient, "label"]].dropna()
        complete_soil_samples = soil_data.dropna(subset=["N", "P", "K", "label"])
        soil_metric_columns = st.columns(3)
        soil_metric_columns[0].metric("Complete samples", f"{len(complete_soil_samples):,}")
        soil_metric_columns[1].metric("Crop labels", f"{soil_data['label'].nunique():,}")
        soil_metric_columns[2].metric("Nutrients tracked", "3")

        st.subheader(f"{selected_nutrient} exploration")
        st.caption("The distribution shows individual sample values; crop averages summarize the sample labels and are not recommendations.")
        left, right = st.columns([1.2, 1])
        with left:
            with st.container(border=True):
                st.markdown("**Sample distribution**")
                fig = px.histogram(valid_nutrient, x=nutrient, color="label", marginal="box", barmode="overlay", opacity=0.72)
                fig.update_layout(xaxis_title=f"{selected_nutrient} (dataset units)", yaxis_title="Sample count")
                st.plotly_chart(plotly_theme(fig), width="stretch")
        with right:
            crop_nutrients = soil_data.groupby("label", as_index=False)[["N", "P", "K"]].mean().sort_values(nutrient, ascending=False)
            fig = px.bar(crop_nutrients.head(15).sort_values(nutrient), x=nutrient, y="label", orientation="h", labels={"label": "Sample crop label"})
            fig.update_layout(xaxis_title=f"Mean {selected_nutrient} (dataset units)", yaxis_title="")
            with st.container(border=True):
                st.markdown("**Highest sample crop averages**")
                st.plotly_chart(plotly_theme(fig), width="stretch")

        with st.expander("View nutrient summary statistics"):
            summary = soil_data[["N", "P", "K"]].describe().loc[
                ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            ].rename(
                index={
                    "count": "Sample count",
                    "mean": "Mean",
                    "std": "Standard deviation",
                    "min": "Minimum",
                    "25%": "25th percentile",
                    "50%": "Median",
                    "75%": "75th percentile",
                    "max": "Maximum",
                },
                columns={"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)"},
            ).round(2)
            st.dataframe(summary, width="stretch")
        st.caption("Regional comparisons and nutrient-to-yield relationships are not available because these samples are not linked to production records.")

elif page == "Irrigation advisory":
    st.header("Daily irrigation advisory")
    st.caption(
        "Enter a daily forecast and reference evapotranspiration (ET₀) for a "
        "transparent crop-water estimate. Weather alone cannot confirm field need."
    )
    with st.container(border=True):
        st.subheader("Today's forecast")
        with st.form("daily_irrigation_advisory_form"):
            crop_columns = st.columns(2)
            with crop_columns[0]:
                advisory_crop = st.selectbox(
                    "Crop",
                    ["wheat", "rice", "maize", "cotton", "sugarcane", "groundnut", "soybean", "potato"],
                    key="daily_advisory_crop",
                )
                forecast_temperature = st.number_input(
                    "Forecast temperature (°C)",
                    min_value=-20.0,
                    max_value=60.0,
                    value=35.0,
                    step=1.0,
                )
                forecast_humidity = st.number_input(
                    "Forecast relative humidity (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=40.0,
                    step=1.0,
                )
            with crop_columns[1]:
                forecast_rain = st.number_input(
                    "Expected rainfall (mm/day)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                )
                reference_et0 = st.number_input(
                    "Reference ET₀ (mm/day)",
                    min_value=0.0,
                    value=5.0,
                    step=0.5,
                    help="Use a local weather service or validated ET₀ calculation when available.",
                )
                growth_stage = st.selectbox(
                    "Growth stage",
                    ["initial", "development", "mid", "late"],
                    index=2,
                )
            measured_deficit = st.checkbox(
                "I have a measured/modelled soil-water deficit"
            )
            soil_deficit = st.number_input(
                "Soil-water deficit (mm)",
                min_value=0.0,
                value=15.0,
                step=1.0,
                disabled=not measured_deficit,
            )
            advisory_submitted = st.form_submit_button(
                "Calculate daily advisory",
                type="primary",
            )

        if advisory_submitted:
            try:
                stress = crop_water_stress_index(
                    advisory_crop,
                    forecast_temperature,
                    forecast_humidity,
                )
                advisory = irrigation_advisory(
                    crop=advisory_crop,
                    et0_mm=reference_et0,
                    forecast_rain_mm=forecast_rain,
                    growth_stage=growth_stage,
                    soil_moisture_deficit_mm=soil_deficit if measured_deficit else None,
                    forecast_temperature_c=forecast_temperature,
                    forecast_humidity_pct=forecast_humidity,
                )
            except ValueError as error:
                st.error(f"Could not calculate the advisory: {error}")
            else:
                metric_columns = st.columns(3)
                metric_columns[0].metric(
                    "Weather stress proxy",
                    f"{stress['index']:.2f} / 1.00",
                    stress["status"],
                )
                metric_columns[1].metric(
                    "Vapor-pressure deficit",
                    f"{stress['vpd_kpa']:.2f} kPa",
                )
                metric_columns[2].metric(
                    "Estimated gross irrigation",
                    f"{advisory['gross_irrigation_mm']:.1f} mm",
                )
                if advisory["net_irrigation_mm"] >= 0.5:
                    st.info(
                        f"Based on today's forecast ({forecast_temperature:.0f}°C, "
                        f"{forecast_rain:.0f} mm expected rain), your {advisory_crop} "
                        f"field may need about {advisory['gross_irrigation_mm']:.1f} mm "
                        "of irrigation. Check soil moisture before applying."
                    )
                else:
                    st.success(
                        "The estimated water balance does not indicate irrigation "
                        "today; verify soil moisture in the field."
                    )
                st.caption(advisory["warning"])
                st.caption(stress["method_note"])

    st.divider()
    st.subheader("Historical weather context")
    st.caption(
        "Historical observations provide background only; they are not a live forecast."
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
            c1, c2 = st.columns(2)
            c1.metric("Total precipitation", f"{rainfall:,.1f} mm")
            c2.metric("Average maximum temperature", f"{max_temp:,.1f} °C")
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
            st.info("Check field soil moisture and consult local crop-stage guidance before making an irrigation decision.")
            st.caption("Rules compare a single observed year with other available years for the same state; they are descriptive thresholds, not agronomic validation.")

elif page == "Yield prediction":
    st.header("Yield prediction")
    st.markdown(
        f"Choose a state, crop, season, district, and cultivated area to estimate yield in {YIELD_UNIT}. "
        "The estimate uses the latest available historical weather for that state. The model was trained on "
        "historical records from 2000-2014, and the bundled weather data ends in 2014, so this is not a live "
        "forecast or a guarantee of actual yield."
    )
    with st.expander("Why are there multiple model files?"):
        st.write(
            "The project trains two algorithms—Random Forest and XGBoost—and compares each with and without "
            "weather features. This creates four artifacts for experiments, not four models that run together. "
            "This page currently uses only the weather-aware Random Forest model. The alternatives are not "
            "automatically compared or selected during prediction."
        )
        st.caption(
            "Models help estimate yield from several inputs and can capture patterns that a simple formula may miss. "
            "The estimate is only as reliable as its training data and has no confidence interval in this app."
        )
    predictor, model_error, version_warnings = load_yield_model()
    if model_error:
        st.warning(f"Model unavailable: {model_error}")
        st.info("The ML integration point remains available in models/random_forest_weather.joblib.")
    else:
        if version_warnings:
            st.warning("The model was saved with a different scikit-learn version. Inference is enabled, but retraining/loading with the matching version is recommended.")
        model_states = sorted(weather["State"].dropna().unique()) if not weather.empty else state_options
        model_crops = sorted(production["Crop"].dropna().unique())
        model_seasons = sorted(production["Season"].dropna().unique())
        state_columns = st.columns(2)
        with state_columns[0]:
            prediction_state = st.selectbox("State", model_states)
        with state_columns[1]:
            prediction_season = st.selectbox("Season", model_seasons)
        with state_columns[2]:
            prediction_year = st.selectbox("Year",list(range(2000, 2015)), index=14,)
            
        normalized_state = normalize_state(prediction_state)
        state_weather_rows = state_weather[
            state_weather["State"].map(normalize_state) == normalized_state
        ] if not state_weather.empty else state_weather
        state_years = sorted(state_weather_rows["Year"].astype(int).unique().tolist())
        prediction_year = state_years[-1] if state_years else None
        if prediction_year is not None:
            st.caption(
                f"Using the latest available historical weather for {prediction_state}: {prediction_year}. "
                "The bundled weather data ends in 2014, so this is not a current-year or live-weather prediction."
            )
        else:
            st.warning(f"No historical weather data is available for {prediction_state}.")
        district_rows = weather[
            weather["State"].map(normalize_state) == normalized_state
        ] if not weather.empty else weather
        district_options = sorted(
            district_rows["District"].dropna().astype(str).str.strip().unique()
        )
        district_choices = district_options or ["No districts available"]

        with st.form("yield_prediction_form"):
            c1, c2 = st.columns(2)
            with c1:
                prediction_crop = st.selectbox("Crop", model_crops)
                prediction_district = st.selectbox(
                    "District",
                    district_choices,
                    key=f"yield_prediction_district_{normalized_state}",
                    disabled=not district_options,
                )
            with c2:
                area_columns = st.columns([1.4, 1])
                with area_columns[0]:
                    area_value = st.number_input("Cultivated area", min_value=0.01, value=1.0, step=1.0)
                with area_columns[1]:
                    area_unit = st.selectbox("Area unit", [AREA_UNIT, "acres"])
            submitted = st.form_submit_button(
                "Estimate yield",
                type="primary",
                disabled=not state_years or not district_options,
            )
        if submitted and predictor is not None and prediction_year is not None:
            if prediction_district not in district_options:
                st.error("The selected district does not belong to the selected state.")
            else:
                weather_row = state_weather_rows[
                    state_weather_rows["Year"] == prediction_year
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
                        "Area": convert_area_to_hectares(area_value, area_unit),
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
                            area_hectares = convert_area_to_hectares(area_value, area_unit)
                            estimated_production = predicted_yield * area_hectares
                            result_columns = st.columns(2)
                            result_columns[0].metric(
                                f"Estimated yield ({YIELD_UNIT})",
                                f"{predicted_yield:,.3f}",
                            )
                            result_columns[1].metric(
                                f"Estimated production ({PRODUCTION_UNIT})",
                                f"{estimated_production:,.3f}",
                            )
                            st.caption(
                                f"Area used: {area_value:,.2f} {area_unit} ({area_hectares:,.2f} {AREA_UNIT}). "
                                "Estimated production is yield multiplied by area. "
                                f"Model: weather-aware random forest · Historical weather year used: {prediction_year}. "
                                "No confidence interval, accuracy claim, or feature explanation is generated here."
                            )

elif page == "Reports":
    st.header("Download analytical results")
    st.caption(
        f"Export filtered records as a CSV, or download a PDF summary. Units: area in {AREA_UNIT}, "
        f"production in {PRODUCTION_UNIT}, and yield in {YIELD_UNIT}."
    )
    if filtered.empty:
        show_empty("No data matches the current filters; adjust them before downloading a report.")
    else:
        total_production = float(filtered["Production"].sum())
        total_area = float(filtered["Area"].sum())
        top_crops = filtered.groupby("Crop", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
        top_regions = filtered.groupby("State", as_index=False)["Production"].sum().sort_values("Production", ascending=False)
        st.write(f"Filtered records: **{len(filtered):,}**")
        export_data = filtered.rename(
            columns={
                "Area": f"Area ({AREA_UNIT})",
                "Production": f"Production ({PRODUCTION_UNIT})",
                "Yield": f"Yield ({YIELD_UNIT})",
            }
        )
        csv_data = export_data.to_csv(index=False).encode("utf-8")
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
        c1.metric(f"Production ({PRODUCTION_UNIT})", format_value(total_production))
        c2.metric(f"Cultivated area ({AREA_UNIT})", format_value(total_area))
        c3.metric(f"Weighted yield ({YIELD_UNIT})", format_value(total_production / total_area if total_area else float("nan")))

if weather_message and page not in {"Weather analytics", "Irrigation advisory"}:
    st.caption(f"Weather-dependent features: {weather_message}")
if state_weather_message and page not in {"Irrigation advisory", "Yield prediction"}:
    st.caption(f"State/year weather context: {state_weather_message}")
if soil_message and page != "Soil nutrients":
    st.caption(f"Soil sample analysis: {soil_message}")
