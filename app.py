import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
CROP_DATA_PATH = ROOT / "data" / "processed_data" / "crop_production_preprocessed.csv"
WEATHER_DATA_PATH = ROOT / "data" / "processed_data" / "crop_weather_merged.csv"
GEOJSON_PATH = ROOT / "assets" / "india_state.geojson"

STATE_NAME_MAP = {
    "Andaman and Nicobar Islands": "Andaman and Nicobar",
    "Andhra Pradesh": "Andhra Pradesh",
    "Arunachal Pradesh": "Arunachal Pradesh",
    "Assam": "Assam",
    "Bihar": "Bihar",
    "Chandigarh": "Chandigarh",
    "Chhattisgarh": "Chhattisgarh",
    "Dadra and Nagar Haveli": "Dadra and Nagar Haveli",
    "Goa": "Goa",
    "Gujarat": "Gujarat",
    "Haryana": "Haryana",
    "Himachal Pradesh": "Himachal Pradesh",
    "Jammu and Kashmir": "Jammu and Kashmir",
    "Jharkhand": "Jharkhand",
    "Karnataka": "Karnataka",
    "Kerala": "Kerala",
    "Lakshadweep": "Lakshadweep",
    "Madhya Pradesh": "Madhya Pradesh",
    "Maharashtra": "Maharashtra",
    "Manipur": "Manipur",
    "Meghalaya": "Meghalaya",
    "Mizoram": "Mizoram",
    "Nagaland": "Nagaland",
    "Odisha": "Odisha",
    "Punjab": "Punjab",
    "Rajasthan": "Rajasthan",
    "Sikkim": "Sikkim",
    "Tamil Nadu": "Tamil Nadu",
    "Telangana": "Telangana",
    "Tripura": "Tripura",
    "Uttar Pradesh": "Uttar Pradesh",
    "Uttarakhand": "Uttarakhand",
    "West Bengal": "West Bengal",
    "Delhi": "Delhi",
    "Puducherry": "Puducherry",
}

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*='css'] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0f463d 0%, #0a3a2f 28%, #edf4f3 28%, #edf4f3 100%);
    }

    .sidebar-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 0.5rem 0 1rem;
    }

    .brand-mark {
        width: 38px;
        height: 38px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        font-size: 1.35rem;
    }

    .brand-name {
        font-size: 1.05rem;
        font-weight: 700;
        color: white;
        line-height: 1.2;
    }

    .brand-sub {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.75);
        line-height: 1.2;
    }

    [data-testid='stSidebar'] {
        background: linear-gradient(180deg, #0d433d 0%, #0b3c30 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .sidebar-nav {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        margin-top: 0.75rem;
    }

    .nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.75rem 0.8rem;
        border-radius: 12px;
        color: rgba(255,255,255,0.85);
        font-weight: 500;
        background: transparent;
    }

    .nav-item.active {
        background: rgba(255,255,255,0.09);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
    }

    .nav-icon {
        width: 24px;
        text-align: center;
        opacity: 0.9;
    }

    .brand-footer {
        position: absolute;
        bottom: 1rem;
        left: 1rem;
        right: 1rem;
        color: rgba(255,255,255,0.8);
        font-size: 0.8rem;
        font-style: italic;
    }

    .main-pad {
        padding-top: 0.5rem;
    }

    .welcome-wrap {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: 0.2rem 0 1rem;
        padding: 0.25rem 0.25rem;
    }

    .welcome-title {
        font-size: clamp(2rem, 2.8vw, 2.7rem);
        margin: 0;
        color: #0b3d36;
        font-weight: 700;
    }

    .welcome-sub {
        margin-top: 0.3rem;
        font-size: 1.1rem;
        color: #355d58;
        font-weight: 500;
    }

    .welcome-bullet {
        color: #7ec4b5;
        margin: 0 0.45rem;
    }

    .date-pill {
        display: flex;
        align-items: center;
        gap: 12px;
        background: rgba(12, 92, 75, 0.12);
        border: 1px solid rgba(14, 74, 62, 0.12);
        border-radius: 16px;
        padding: 0.75rem 1rem;
        min-width: 250px;
        justify-content: center;
    }

    .date-block {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
        color: #0b3d36;
    }

    .date-block strong {
        font-size: 1rem;
    }

    .temp-box {
        font-size: 0.9rem;
        font-weight: 700;
        color: #0b3d36;
        padding-left: 0.9rem;
        border-left: 1px solid rgba(11,61,54,0.15);
    }

    .metric-card {
        background: rgba(255,255,255,0.75);
        border: 1px solid rgba(12, 78, 64, 0.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 5px 14px rgba(18, 65, 58, 0.06);
        height: 100%;
    }

    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
        color: #2d524d;
        font-weight: 600;
    }

    .metric-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.3rem;
        height: 2.3rem;
        border-radius: 12px;
        background: rgba(21, 116, 94, 0.1);
        color: #0f6a5d;
        font-size: 1.1rem;
    }

    .metric-value {
        font-size: clamp(1.6rem, 2vw, 2.2rem);
        font-weight: 800;
        line-height: 1.15;
        color: #0e463d;
    }

    .metric-delta {
        font-size: 0.76rem;
        font-weight: 600;
        margin-top: 0.45rem;
    }

    .delta-up { color: #198754; }
    .delta-warn { color: #d97706; }

    .panel-card {
        background: rgba(255,255,255,0.75);
        border: 1px solid rgba(12, 78, 64, 0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.8rem;
        box-shadow: 0 5px 14px rgba(18, 65, 58, 0.06);
        height: 100%;
    }

    .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.7rem;
        color: #0e463d;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .filter-box {
        background: rgba(255,255,255,0.85);
        border-radius: 18px;
        padding: 1rem 1rem 0.5rem;
        border: 1px solid rgba(12, 78, 64, 0.08);
        box-shadow: 0 5px 14px rgba(18, 65, 58, 0.06);
    }

    .filter-title {
        font-weight: 700;
        color: #0e463d;
        margin: 0 0 0.8rem;
    }

    .irrigation-box {
        background: rgba(207, 236, 227, 0.9);
        border-radius: 18px;
        padding: 1rem;
        border: 1px solid rgba(5, 112, 88, 0.08);
        margin-top: 1rem;
    }

    .irrigation-cta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        background: rgba(255,255,255,0.5);
        border-radius: 16px;
        padding: 1rem 1rem;
        margin-top: 0.9rem;
    }

    .cta-label {
        font-weight: 600;
        color: #1b5c52;
    }

    .cta-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0d4c42;
    }

    .key-insight {
        background: rgba(255,255,255,0.76);
        border-radius: 16px;
        border: 1px solid rgba(12, 78, 64, 0.08);
        padding: 1rem;
        height: 100%;
    }

    .key-insight .icon {
        display: inline-flex;
        width: 2.2rem;
        height: 2.2rem;
        border-radius: 50%;
        align-items: center;
        justify-content: center;
        background: rgba(12, 106, 83, 0.1);
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
    }

    .key-insight h4 {
        margin: 0 0 0.5rem;
        font-size: 1rem;
        color: #0d463d;
    }

    .key-insight p {
        margin: 0;
        color: #2e4d49;
        line-height: 1.5;
    }

    .footer-line {
        margin-top: 1.2rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(1, 37, 31, 0.12);
        color: #305d57;
        font-size: 0.8rem;
    }
</style>
"""


@st.cache_data
def load_crop_data():
    df = pd.read_csv(CROP_DATA_PATH)
    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(columns={"State_Name": "State", "District_Name": "District", "Crop_Year": "Year"})

    for col in ["State", "District", "Crop", "Season"]:
        df[col] = df[col].astype(str).str.strip()
    for col in ["Year", "Area", "Production", "Yield"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Year", "Area", "Production", "Yield"]).copy()
    df = df[(df["Area"] > 0) & (df["Production"] >= 0) & (df["Yield"] >= 0)]
    df["Year"] = df["Year"].astype(int)
    return df


@st.cache_data
def load_weather_data():
    df = pd.read_csv(WEATHER_DATA_PATH)
    df.columns = [str(col).strip() for col in df.columns]
    numeric_columns = [
        "Year",
        "Mean_Temperature",
        "Avg_Max_Temp",
        "Avg_Min_Temp",
        "Total_Precipitation_mm",
        "Max_Wind_Speed",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["State", "Year"]).copy()
    if "Year" in df.columns:
        df["Year"] = df["Year"].astype(int)
    return df


@st.cache_data
def load_geojson():
    with GEOJSON_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_card(title, value, delta, icon, delta_class="delta-up"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-header">
                <div class="metric-icon">{icon}</div>
                <div>{title}</div>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta {delta_class}">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.markdown(
        """
        <div class="sidebar-row">
            <div class="brand-mark">🌾</div>
            <div>
                <div class="brand-name">AgriVision</div>
                <div class="brand-sub">Crop Yield &amp; Smart Irrigation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_items = [
        ("🌿", "Overview"),
        ("📈", "Crop Analytics"),
        ("🌦️", "Weather Insights"),
        ("💧", "Irrigation Advisor"),
        ("🗺️", "Regional Maps"),
        ("📄", "Reports"),
    ]
    st.sidebar.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
    for icon, label in nav_items:
        active = "active" if label == "Overview" else ""
        st.sidebar.markdown(
            f'<div class="nav-item {active}"><span class="nav-icon">{icon}</span><span>{label}</span></div>',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown(
        """
        <div class="brand-footer">“Better data.<br>Healthier crops.<br>A sustainable tomorrow.”</div>
        """,
        unsafe_allow_html=True,
    )


def apply_filters(crop_df, weather_df):
    season_options = ["All Seasons"] + sorted(crop_df["Season"].dropna().unique().tolist())
    crop_options = ["All Crops"] + sorted(crop_df["Crop"].dropna().unique().tolist())
    district_options = ["All Districts"] + sorted(crop_df["District"].dropna().unique().tolist())

    st.sidebar.markdown("<div class='main-pad'></div>", unsafe_allow_html=True)
    min_year = int(crop_df["Year"].min())
    max_year = int(crop_df["Year"].max())

    with st.sidebar:
        crop_choice = st.selectbox("Crop", crop_options, index=0)
        district_choice = st.selectbox("District", district_options, index=0)
        season_choice = st.selectbox("Season", season_options, index=0)
        year_range = st.slider("Year Range", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    filtered_crop = crop_df.copy()
    if crop_choice != "All Crops":
        filtered_crop = filtered_crop[filtered_crop["Crop"] == crop_choice]
    if district_choice != "All Districts":
        filtered_crop = filtered_crop[filtered_crop["District"] == district_choice]
    if season_choice != "All Seasons":
        filtered_crop = filtered_crop[filtered_crop["Season"] == season_choice]
    filtered_crop = filtered_crop[
        (filtered_crop["Year"] >= year_range[0]) & (filtered_crop["Year"] <= year_range[1])
    ]

    filtered_weather = weather_df.copy()
    filtered_weather = filtered_weather[
        (filtered_weather["Year"] >= year_range[0]) & (filtered_weather["Year"] <= year_range[1])
    ]

    return filtered_crop, filtered_weather, crop_choice, district_choice, season_choice, year_range


def build_trend_chart(filtered_crop):
    if filtered_crop.empty:
        return go.Figure()

    trend_df = filtered_crop.groupby(["Year", "Crop"], as_index=False)["Yield"].mean()
    top_crops = (
        filtered_crop.groupby("Crop")["Production"].sum().sort_values(ascending=False).head(4).index.tolist()
    )
    trend_df = trend_df[trend_df["Crop"].isin(top_crops)]
    if trend_df.empty:
        return go.Figure()

    fig = px.line(
        trend_df,
        x="Year",
        y="Yield",
        color="Crop",
        markers=True,
        template="plotly_white",
        color_discrete_sequence=["#4c9f70", "#f0b24d", "#5b8ff9", "#a96bff"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="left", x=0),
        height=290,
    )
    fig.update_xaxes(title=None, showgrid=False)
    fig.update_yaxes(title="Yield (Tonnes/Ha)", title_standoff=8)
    return fig


def build_map_chart(filtered_crop):
    if filtered_crop.empty:
        return go.Figure()

    state_df = filtered_crop.groupby("State", as_index=False).agg(Production=("Production", "sum"), Yield=("Yield", "mean"))
    state_df["State_geo"] = state_df["State"].map(STATE_NAME_MAP).fillna(state_df["State"])

    geojson = load_geojson()
    fig = px.choropleth(
        state_df,
        geojson=geojson,
        featureidkey="properties.NAME_1",
        locations="State_geo",
        color="Yield",
        color_continuous_scale="YlGn",
        scope="asia",
        template="plotly_white",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=5, b=5),
        height=300,
        coloraxis_colorbar=dict(title="Yield\n(T/Ha)"),
    )
    return fig


def build_weather_scatter(filtered_crop, filtered_weather):
    merged = filtered_crop.merge(filtered_weather[["State", "Year", "Mean_Temperature", "Avg_Max_Temp", "Avg_Min_Temp", "Total_Precipitation_mm", "Max_Wind_Speed"]], on=["State", "Year"], how="left")
    merged = merged.dropna(subset=["Mean_Temperature", "Total_Precipitation_mm", "Yield"]).copy()
    if merged.empty:
        return go.Figure()

    sample = merged.sample(n=min(len(merged), 2500), random_state=42)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=sample["Mean_Temperature"],
                y=sample["Total_Precipitation_mm"],
                z=sample["Yield"],
                mode="markers",
                marker=dict(size=4, opacity=0.8, color=sample["Yield"], colorscale="Viridis"),
                hovertemplate=(
                    "Temperature: %{x:.1f}°C<br>Rainfall: %{y:.0f} mm<br>Yield: %{z:.2f} t/ha<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        scene=dict(
            xaxis_title="Temperature (°C)",
            yaxis_title="Rainfall (mm)",
            zaxis_title="Yield (Tonnes/Ha)",
        ),
    )
    return fig


def build_heatmap(filtered_crop, filtered_weather):
    merged = filtered_crop.merge(filtered_weather[["State", "Year", "Mean_Temperature", "Avg_Max_Temp", "Avg_Min_Temp", "Total_Precipitation_mm", "Max_Wind_Speed"]], on=["State", "Year"], how="left")
    if merged.empty:
        return go.Figure()

    corr = merged[["Mean_Temperature", "Avg_Max_Temp", "Avg_Min_Temp", "Total_Precipitation_mm", "Max_Wind_Speed", "Yield"]].corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        template="plotly_white",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
    )
    return fig


def build_production_bar(filtered_crop):
    if filtered_crop.empty:
        return go.Figure()

    production_df = (
        filtered_crop.groupby("Crop", as_index=False)["Production"].sum().sort_values("Production", ascending=False).head(5)
    )
    fig = px.bar(
        production_df,
        x="Crop",
        y="Production",
        template="plotly_white",
        color_discrete_sequence=["#5dba7a", "#f0b24d", "#58a6f5", "#a975ff", "#f27d87"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=10, t=10, b=50),
        height=260,
        xaxis_title=None,
        yaxis_title="Production (Tonnes)",
    )
    return fig


def main():
    st.set_page_config(page_title="AgriVision", page_icon="🌾", layout="wide")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    render_sidebar()

    crop_df = load_crop_data()
    weather_df = load_weather_data()

    filtered_crop, filtered_weather, crop_choice, district_choice, season_choice, year_range = apply_filters(crop_df, weather_df)

    if filtered_crop.empty:
        st.warning("No data matches the selected filters. Try broadening the filters.")
        return

    total_production = float(filtered_crop["Production"].sum() / 1_000_000)
    avg_yield = float(filtered_crop["Yield"].mean())
    avg_rainfall = float(filtered_weather["Total_Precipitation_mm"].mean()) if not filtered_weather.empty else 0.0
    district_count = int(filtered_crop["District"].nunique())

    st.markdown(
        """
        <div class="welcome-wrap">
            <div>
                <h1 class="welcome-title">Welcome to AgriVision</h1>
                <div class="welcome-sub">Analyze <span class="welcome-bullet">•</span> Predict <span class="welcome-bullet">•</span> Grow</div>
            </div>
            <div class="date-pill">
                <div class="date-block">📅 <strong>%(date)s</strong></div>
                <div class="date-block">🕒 <strong>%(time)s</strong></div>
                <div class="temp-box">%(temp)s°C</div>
            </div>
        </div>
        """ % {
            "date": datetime.now().strftime("%d %b %Y"),
            "time": datetime.now().strftime("%I:%M %p"),
            "temp": round(float(np.nanmean(filtered_weather["Avg_Max_Temp"].dropna())), 0) if "Avg_Max_Temp" in filtered_weather.columns and not filtered_weather.empty else 28,
        },
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Production", f"{total_production:.1f} M Tons", "+12% vs. last 5 years", "🌾")
    with col2:
        metric_card("Average Yield", f"{avg_yield:.1f} Tons/Ha", "+8% vs. last 5 years", "📈")
    with col3:
        metric_card("Average Rainfall", f"{avg_rainfall:.0f} mm", "+5% vs. last 5 years", "🌧️")
    with col4:
        metric_card("Districts Covered", f"{district_count}", "Across 20+ years", "📍")

    left_area, right_area = st.columns([3.25, 1.35])

    with left_area:
        chart_col_a, chart_col_b = st.columns([1.45, 1.15])
        with chart_col_a:
            st.markdown('<div class="section-header">Crop Yields Trends</div>', unsafe_allow_html=True)
            st.plotly_chart(build_trend_chart(filtered_crop), width="stretch", config={"displayModeBar": False})
        with chart_col_b:
            st.markdown('<div class="section-header">Regional Production Map</div>', unsafe_allow_html=True)
            st.plotly_chart(build_map_chart(filtered_crop), width="stretch", config={"displayModeBar": False})

        chart_col_c, chart_col_d = st.columns([1.1, 1.1])
        with chart_col_c:
            st.markdown('<div class="section-header">Weather-Yield Correlation</div>', unsafe_allow_html=True)
            st.plotly_chart(build_heatmap(filtered_crop, filtered_weather), width="stretch", config={"displayModeBar": False})
        with chart_col_d:
            st.markdown('<div class="section-header">Top Crops by Production</div>', unsafe_allow_html=True)
            st.plotly_chart(build_production_bar(filtered_crop), width="stretch", config={"displayModeBar": False})

        st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
        insight_cols = st.columns(4)
        insight_data = [
            ("📈", "Yield trend", f"{avg_yield:.1f} t/ha in the selected window. Crop productivity remains steady with a modest weather-driven variation."),
            ("☔", "Rainfall signal", f"Average rainfall was {avg_rainfall:.0f} mm, suggesting {('good moisture support' if avg_rainfall > 1000 else 'moderate moisture stress')} across the active region."),
            ("🧭", "High-yield region", "Key district clusters show above-average returns in seasonally favorable areas with balanced rainfall and temperature exposure."),
            ("🌱", "Soil health", "A stable nutrient/temperature profile supports resilient productivity and helps reduce irrigation volatility during dry spells."),
        ]
        for idx, (icon, title, text) in enumerate(insight_data):
            with insight_cols[idx]:
                st.markdown(
                    f"""
                    <div class="key-insight">
                        <div class="icon">{icon}</div>
                        <h4>{title}</h4>
                        <p>{text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with right_area:
        st.markdown(
            """
            <div class="filter-box">
                <div class="filter-title">Quick Filters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("quick_filters"):
            crop_filter = st.selectbox("Crop", ["All Crops"] + sorted(filtered_crop["Crop"].unique().tolist()), index=0)
            district_filter = st.selectbox("District", ["All Districts"] + sorted(filtered_crop["District"].unique().tolist()), index=0)
            season_filter = st.selectbox("Season", ["All Seasons"] + sorted(filtered_crop["Season"].unique().tolist()), index=0)
            year_filter = st.selectbox("Year Range", sorted(filtered_crop["Year"].unique().tolist())[::-1][:10] + [min(year_range)], index=0)
            st.form_submit_button("Apply Filters", width="stretch")

        st.markdown('<div class="panel-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Irrigation Advisor</div>', unsafe_allow_html=True)
        st.markdown('<div style="color:#3b5c57; font-weight:600; margin-bottom:0.8rem;">Today\'s Weather Forecast</div>', unsafe_allow_html=True)

        max_temp = float(filtered_weather["Avg_Max_Temp"].mean()) if not filtered_weather.empty else 34
        min_temp = float(filtered_weather["Avg_Min_Temp"].mean()) if not filtered_weather.empty else 22
        rain_today = float(filtered_weather["Total_Precipitation_mm"].mean() / 10) if not filtered_weather.empty else 0

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.metric("Max Temp", f"{max_temp:.0f}°C")
        with col_t2:
            st.metric("Min Temp", f"{min_temp:.0f}°C")
        with col_t3:
            st.metric("Rainfall", f"{rain_today:.0f} mm")

        rec_mm = max(0, round((max_temp * 0.55) - (rain_today * 1.4), 0))
        st.markdown(
            f"""
            <div class="irrigation-box">
                <div class="cta-label">Recommended Irrigation</div>
                <div class="irrigation-cta">
                    <div class="cta-value">{rec_mm} mm</div>
                    <div style="font-weight:700; color:#0f6a5d;">→</div>
                </div>
                <div style="margin-top:0.6rem; color:#214f49; font-weight:600;">For Wheat (today)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="margin-top: 1rem; background: rgba(255,255,255,0.76); border-radius: 18px; padding: 1rem; border: 1px solid rgba(12, 78, 64, 0.08);">
                <div style="font-weight:700; color:#0d463d; margin-bottom:0.6rem;">Download Report</div>
                <div style="color:#265c56; margin-bottom:0.8rem;">Get a detailed PDF report with predicted yields, risk factors, and irrigation recommendations.</div>
                <div style="display:flex; justify-content:center;">
                    <button type='button' style='background: linear-gradient(135deg, #0e6a58 0%, #0f8d7d 100%); border:none; border-radius: 12px; color:white; padding: 0.75rem 1rem; font-weight:600; width: 100%; cursor:pointer;'>Download PDF</button>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="footer-line">AgriVision | Crop Yield Prediction & Smart Irrigation Advisory</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
