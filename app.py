from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed_data"
DATASETS = {
	"Crop + weather (merged)": PROCESSED / "crop_weather_merged.csv",
	"Crop + weather (all records)": PROCESSED / "crop_weather_merge_all.csv",
	"Crop production": PROCESSED / "crop_production_preprocessed.csv",
	"Training data": ROOT / "data" / "training_data.csv",
	"Weather summary": PROCESSED / "weather_preprocessed.csv",
	"Crop recommendation (soil)": PROCESSED / "crop_recommendation_preprocessed.csv",
}
COORDINATES_PATH = PROCESSED / "india_states_yearly_weather_2000_2014.csv"
GREEN, LIME, ORANGE = "#62d69a", "#c1e875", "#e5a766"

st.set_page_config(
	page_title="AgriVision | Smart Agriculture Intelligence",
	page_icon="🌱",
	layout="wide",
	initial_sidebar_state="auto",
)


@st.cache_data(show_spinner=False)
def read_csv(path_string):
	try:
		return pd.read_csv(path_string, low_memory=False)
	except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError):
		return pd.DataFrame()


def column_key(value):
	return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def canonicalize(frame):
	"""Map known source headers to shared names without assuming one schema."""
	aliases = {
		"State": ["state", "state_name"],
		"District": ["district", "district_name"],
		"Year": ["year", "crop_year"],
		"Crop": ["crop", "label"],
		"Season": ["season"],
		"Area": ["area"],
		"Production": ["production"],
		"Yield": ["yield"],
		"Rainfall": ["total_precipitation_mm", "precipitation_mm", "rain_mm", "rainfall"],
		"Temperature": ["mean_temperature", "temperature", "avg_max_temp", "temp_max_c"],
		"Min temperature": ["avg_min_temp", "temp_min_c"],
		"Wind speed": ["max_wind_speed", "wind_speed_max_kmh"],
	}
	lookup = {column_key(column): column for column in frame.columns}
	rename = {}
	for target, candidates in aliases.items():
		if target not in frame.columns:
			for candidate in candidates:
				if column_key(candidate) in lookup:
					rename[lookup[column_key(candidate)]] = target
					break
	result = frame.rename(columns=rename).copy()
	for column in ["Year", "Area", "Production", "Yield", "Rainfall", "Temperature"]:
		if column in result:
			result[column] = pd.to_numeric(result[column], errors="coerce")
	if "Yield" not in result and {"Production", "Area"}.issubset(result.columns):
		result["Yield"] = result["Production"] / result["Area"].where(result["Area"] > 0)
	for column in ["State", "District", "Crop", "Season"]:
		if column in result:
			result[column] = result[column].astype("string").str.strip()
	return result.replace([np.inf, -np.inf], np.nan)


def load_main_data():
	for label in ["Crop + weather (merged)", "Training data", "Crop production"]:
		path = DATASETS[label]
		if path.exists():
			frame = canonicalize(read_csv(str(path)))
			if not frame.empty:
				return frame, label
	return pd.DataFrame(), "No compatible dataset found"


def metric_card(label, value, detail, icon, accent="green"):
	st.markdown(
		f"""
		<div class="metric-card {accent}">
			<div class="metric-top"><span>{label}</span><span class="metric-icon">{icon}</span></div>
			<div class="metric-value">{value}</div>
			<div class="metric-detail">{detail}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def heading(title, caption=None):
	st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
	if caption:
		st.caption(caption)


def chart_style(fig, height=370):
	fig.update_layout(
		height=height,
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
		font={"family": "Aptos, Segoe UI, sans-serif", "color": "#edf4eb", "size": 12},
		margin={"l": 12, "r": 12, "t": 32, "b": 12},
		legend={"orientation": "h", "y": 1.12, "x": 0},
		hoverlabel={"bgcolor": "#172b21", "font_color": "#edf4eb"},
	)
	fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#304238")
	fig.update_yaxes(showgrid=True, gridcolor="#263a2e", zeroline=False)
	return fig


def weather_rows(frame):
	if {"State", "Year"}.issubset(frame.columns):
		return frame.drop_duplicates(["State", "Year"])
	return frame


def render_sidebar():
	st.sidebar.markdown(
		'<div class="brand"><span class="brand-mark">✳</span><div>'
		'<strong>AgriVision</strong><small>FIELD INTELLIGENCE</small></div></div>',
		unsafe_allow_html=True,
	)
	st.sidebar.markdown('<div class="sidebar-label">WORKSPACE</div>', unsafe_allow_html=True)
	pages = [
		"⌂  Overview", "▥  Crop Yield Analytics", "☼  Weather Insights",
		"◉  Smart Irrigation Advisor", "⌖  Regional Analysis", "▤  Dataset Explorer",
	]
	selected = st.sidebar.radio("Dashboard sections", pages, label_visibility="collapsed")
	st.sidebar.markdown(
		'<div class="sidebar-note"><span class="live-dot"></span> Historical data workspace</div>',
		unsafe_allow_html=True,
	)
	return selected.split("  ", 1)[1]


def render_overview(frame, source):
	st.markdown(
		"""
		<div class="hero"><div class="hero-copy">
			<div class="eyebrow"><span class="live-dot"></span> AGRICULTURE INTELLIGENCE</div>
			<h1>AgriVision</h1><p>Smart Agriculture Intelligence</p>
			<div class="hero-subtitle">Data-driven insights for more informed, sustainable farming.</div>
		</div><div class="hero-stamp"><span>GROW WITH</span><strong>INSIGHT</strong><i>◌</i></div></div>
		""",
		unsafe_allow_html=True,
	)
	st.caption(f"Primary analysis source: {source}")
	crop_count = frame["Crop"].nunique() if "Crop" in frame else None
	region_count = frame["State"].nunique() if "State" in frame else None
	avg_yield = frame["Yield"].mean() if "Yield" in frame else np.nan
	cols = st.columns(4, gap="medium")
	with cols[0]:
		metric_card("PRODUCTION RECORDS", f"{len(frame):,}", "Rows in analysis source", "▤")
	with cols[1]:
		metric_card("CROPS TRACKED", f"{crop_count:,}" if crop_count is not None else "—", "Distinct crop labels", "✳", "lime")
	with cols[2]:
		metric_card("REGIONS", f"{region_count:,}" if region_count is not None else "—", "States represented", "⌖", "orange")
	with cols[3]:
		metric_card("AVERAGE YIELD", f"{avg_yield:,.2f}" if pd.notna(avg_yield) else "—", "Dataset yield units; source-defined", "↗", "blue")

	left, right = st.columns([1.55, 1], gap="large")
	with left:
		heading("Production overview", "Leading crops by recorded production")
		if {"Crop", "Production"}.issubset(frame.columns):
			summary = frame.groupby("Crop", as_index=False)["Production"].sum().nlargest(10, "Production").sort_values("Production")
			fig = px.bar(summary, x="Production", y="Crop", orientation="h", color="Production",
						 color_continuous_scale=["#285e44", GREEN, LIME], labels={"Production": "Recorded production", "Crop": ""})
			fig.update_layout(coloraxis_showscale=False)
			st.plotly_chart(chart_style(fig, 390), width="stretch")
		else:
			st.info("Crop and production fields are unavailable in this source.")
	with right:
		heading("Weather snapshot", "Historical observations in the analysis source")
		weather = weather_rows(frame)
		for column, label, unit, icon, accent in [
			("Rainfall", "MEAN RAINFALL", "mm", "☂", "blue"),
			("Temperature", "MEAN TEMPERATURE", "°C", "☼", "orange"),
			("Wind speed", "MEAN MAX WIND", "km/h", "≋", "lime"),
		]:
			if column in weather:
				metric_card(label, f"{weather[column].mean():,.1f} {unit}", "Mean per stored record", icon, accent)

	if {"State", "Production"}.issubset(frame.columns):
		heading("Regional leaders", "States ranked by total recorded production")
		summary = frame.groupby("State", as_index=False)["Production"].sum().nlargest(5, "Production").sort_values("Production")
		fig = px.bar(summary, x="Production", y="State", orientation="h", color_discrete_sequence=[ORANGE])
		st.plotly_chart(chart_style(fig, 280), width="stretch")


def filtered_yield(frame):
	data = frame.copy()
	cols = st.columns([1, 1, 1.4])
	if "Crop" in data:
		with cols[0]:
			selected = st.multiselect("Crop", sorted(data["Crop"].dropna().astype(str).unique()), placeholder="All crops")
		if selected:
			data = data[data["Crop"].isin(selected)]
	if "State" in data:
		with cols[1]:
			selected = st.multiselect("State / region", sorted(data["State"].dropna().astype(str).unique()), placeholder="All regions")
		if selected:
			data = data[data["State"].isin(selected)]
	if "Year" in data and data["Year"].notna().any():
		years = data["Year"].dropna()
		low, high = int(years.min()), int(years.max())
		with cols[2]:
			if low < high:
				year_range = st.slider("Year range", low, high, (low, high))
				data = data[data["Year"].between(*year_range)]
			else:
				st.caption(f"Year: {low}")
	return data


def render_yield(frame):
	st.title("Crop Yield Analytics")
	st.caption("Compare crop performance and production across places and years.")
	data = filtered_yield(frame)
	if data.empty:
		st.warning("No records match these filters. Broaden your selection to continue.")
		return
	left, right = st.columns(2, gap="large")
	with left:
		heading("Yield by crop", "Mean source yield; units follow the dataset")
		if {"Crop", "Yield"}.issubset(data.columns):
			summary = data.groupby("Crop", as_index=False)["Yield"].mean().nlargest(10, "Yield").sort_values("Yield")
			st.plotly_chart(chart_style(px.bar(summary, x="Yield", y="Crop", orientation="h", color_discrete_sequence=[GREEN])), width="stretch")
		else:
			st.info("A crop and yield pair is not available.")
	with right:
		heading("Production by region", "Top 10 states in filtered records")
		if {"State", "Production"}.issubset(data.columns):
			summary = data.groupby("State", as_index=False)["Production"].sum().nlargest(10, "Production").sort_values("Production")
			st.plotly_chart(chart_style(px.bar(summary, x="Production", y="State", orientation="h", color_discrete_sequence=[ORANGE])), width="stretch")
		else:
			st.info("State and production fields are not available.")
	heading("Yearly trend", "Aggregate yield and production by recorded year")
	if "Year" in data and ({"Yield", "Production"} & set(data.columns)):
		aggregations = {column: "mean" if column == "Yield" else "sum" for column in ["Yield", "Production"] if column in data}
		yearly = data.groupby("Year", as_index=False).agg(aggregations).sort_values("Year")
		if len(aggregations) == 2:
			fig = go.Figure()
			fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["Yield"], name="Mean yield", mode="lines+markers", line={"color": GREEN}))
			fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["Production"], name="Production", mode="lines", yaxis="y2", line={"color": ORANGE}))
			fig.update_layout(yaxis={"title": "Mean yield"}, yaxis2={"title": "Production", "overlaying": "y", "side": "right"})
		else:
			column = next(iter(aggregations))
			fig = px.line(yearly, x="Year", y=column, markers=True)
			fig.update_traces(line_color=GREEN)
		st.plotly_chart(chart_style(fig, 340), width="stretch")
	else:
		st.info("Yearly trends need a year field and at least one production metric.")


def render_weather(frame):
	st.title("Weather Insights")
	st.caption("Historical weather summaries linked to crop records where available.")
	weather = weather_rows(frame)
	cards = st.columns(3, gap="medium")
	for slot, column, label, icon, unit, accent in [
		(cards[0], "Rainfall", "MEAN RAINFALL", "☂", "mm", "blue"),
		(cards[1], "Temperature", "MEAN TEMPERATURE", "☼", "°C", "orange"),
		(cards[2], "Wind speed", "MEAN MAX WIND", "≋", "km/h", "lime"),
	]:
		with slot:
			if column in weather:
				metric_card(label, f"{weather[column].mean():,.2f} {unit}", "Mean per stored weather record", icon, accent)
			else:
				metric_card(label, "—", "Not present in this dataset", icon, accent)
	if "Year" in weather and ({"Rainfall", "Temperature"} & set(weather.columns)):
		heading("Weather through time", "State-year weather records; each state-year counted once")
		columns = [column for column in ["Rainfall", "Temperature"] if column in weather]
		trend = weather.groupby("Year", as_index=False)[columns].mean().sort_values("Year")
		fig = go.Figure()
		for column, color in [("Rainfall", GREEN), ("Temperature", ORANGE)]:
			if column in trend:
				fig.add_trace(go.Scatter(x=trend["Year"], y=trend[column], name=column, mode="lines+markers", line={"color": color}))
		st.plotly_chart(chart_style(fig, 340), width="stretch")
	left, right = st.columns(2, gap="large")
	with left:
		heading("Rainfall vs temperature", "Each point is a crop record; weather is historical")
		if {"Rainfall", "Temperature"}.issubset(frame.columns):
			points = frame.dropna(subset=["Rainfall", "Temperature"])
			if len(points) > 5000:
				points = points.sample(5000, random_state=42)
			fig = px.scatter(points, x="Rainfall", y="Temperature", color="Yield" if "Yield" in points else None,
							 hover_data=[c for c in ["Crop", "State", "Year", "Yield"] if c in points],
							 color_continuous_scale=["#3a7154", LIME], opacity=0.72)
			st.plotly_chart(chart_style(fig, 390), width="stretch")
		else:
			st.info("One or both weather fields are missing.")
	with right:
		heading("Weather correlation", "Pairwise correlation of available numeric measures")
		numeric = [c for c in ["Rainfall", "Temperature", "Min temperature", "Wind speed", "Yield", "Production"]
				   if c in frame and pd.api.types.is_numeric_dtype(frame[c])]
		if len(numeric) >= 2:
			fig = px.imshow(frame[numeric].corr(numeric_only=True), text_auto=".2f", aspect="auto",
							color_continuous_scale="RdYlGn", zmin=-1, zmax=1, labels={"color": "Correlation"})
			st.plotly_chart(chart_style(fig, 390), width="stretch")
		else:
			st.info("At least two numeric weather or production fields are needed.")
	heading("Weather and yield in 3D", "Available only when all three measured fields coexist")
	if {"Rainfall", "Temperature", "Yield"}.issubset(frame.columns):
		points = frame.dropna(subset=["Rainfall", "Temperature", "Yield"])
		if len(points) > 3500:
			points = points.sample(3500, random_state=42)
		if not points.empty:
			fig = px.scatter_3d(points, x="Rainfall", y="Temperature", z="Yield", color="Yield",
								hover_data=[c for c in ["Crop", "State", "Year"] if c in points],
								color_continuous_scale=["#2e7652", LIME], opacity=0.75)
			fig.update_layout(scene={"xaxis_title": "Rainfall", "yaxis_title": "Temperature", "zaxis_title": "Yield"})
			st.plotly_chart(chart_style(fig, 520), width="stretch")
	else:
		st.info("Required fields are missing; no values are fabricated for the 3D chart.")


def render_irrigation(frame):
	st.title("Smart Irrigation Advisor")
	st.caption("A historical weather screening tool, not a field prescription.")
	if not {"Rainfall", "Temperature"}.issubset(frame.columns):
		st.warning("Choose a weather-linked dataset; this source lacks rainfall or temperature.")
		return
	historical = weather_rows(frame).dropna(subset=["Rainfall", "Temperature"])
	if historical.empty or "State" not in historical:
		st.info("No complete regional weather records are available for comparison.")
		return
	regions = sorted(historical["State"].dropna().astype(str).unique())
	crops = sorted(frame["Crop"].dropna().astype(str).unique()) if "Crop" in frame else []
	with st.form("irrigation_form"):
		region = st.selectbox("State / region", regions)
		crop = st.selectbox("Crop (context only)", crops) if crops else "Not available"
		region_history = historical[historical["State"].astype(str) == region]
		default_rain = float(region_history["Rainfall"].median())
		default_temp = float(region_history["Temperature"].median())
		rainfall = st.number_input("Rainfall for the period (mm)", min_value=0.0, value=max(0.0, default_rain), step=1.0)
		temperature = st.number_input("Mean temperature for the period (°C)", value=default_temp, step=0.5)
		submitted = st.form_submit_button("Review weather signals", type="primary", width="stretch")
	st.caption("Defaults are regional historical medians. Enter values for the same period as stored records; no current forecast or soil-moisture measurement is available.")
	if not submitted:
		return
	rain_threshold = float(region_history["Rainfall"].quantile(0.25))
	temp_threshold = float(region_history["Temperature"].quantile(0.75))
	low_rain, high_temp = rainfall < rain_threshold, temperature > temp_threshold
	if low_rain and high_temp:
		verdict = "Review irrigation need"
		explanation = "Both values cross regional historical screening thresholds. Check field soil moisture and a current forecast before deciding."
	elif low_rain:
		verdict = "Rainfall is unusually low"
		explanation = "Entered rainfall is below the regional historical 25th percentile. Check field soil moisture and current conditions."
	elif high_temp:
		verdict = "Temperature is unusually high"
		explanation = "Entered temperature is above the regional historical 75th percentile. Heat alone does not determine irrigation need."
	else:
		verdict = "No weather trigger detected"
		explanation = "Values are within the regional historical screening range. This does not confirm adequate soil moisture."
	st.markdown(f'<div class="advisory {"warning" if low_rain or high_temp else "good"}"><div class="eyebrow">{crop} · {region}</div><h2>{verdict}</h2><p>{explanation}</p></div>', unsafe_allow_html=True)
	cards = st.columns(4, gap="medium")
	items = [("ENTERED RAINFALL", f"{rainfall:,.1f} mm"), ("REGIONAL P25 RAINFALL", f"{rain_threshold:,.1f} mm"),
			 ("ENTERED TEMPERATURE", f"{temperature:,.1f} °C"), ("REGIONAL P75 TEMPERATURE", f"{temp_threshold:,.1f} °C")]
	for slot, (label, value) in zip(cards, items):
		with slot:
			metric_card(label, value, "Historical comparison", "◌", "orange")
	st.warning("Basic advisory only. Crop water requirement, soil moisture, growth stage, current forecast, and irrigation depth are not available. Confirm with field measurements and local agronomic guidance.", icon="⚠️")


def render_regions(frame):
	st.title("Regional Analysis")
	st.caption("Compare production by state and explore approximate geographic context.")
	if not {"State", "Production"}.issubset(frame.columns):
		st.info("State and production fields are required for regional comparisons.")
		return
	aggregations = {"Production": ("Production", "sum")}
	aggregations["Mean_yield"] = ("Yield", "mean") if "Yield" in frame else ("Production", "mean")
	summary = frame.groupby("State", as_index=False).agg(**aggregations).sort_values("Production", ascending=False)
	cards = st.columns(3, gap="medium")
	with cards[0]:
		metric_card("STATES / REGIONS", f"{len(summary):,}", "With production records", "⌖")
	with cards[1]:
		metric_card("LEADING REGION", str(summary.iloc[0]["State"]), "By total production", "↗", "lime")
	with cards[2]:
		metric_card("TOTAL PRODUCTION", f"{summary['Production'].sum():,.0f}", "Source production units", "▤", "orange")
	left, right = st.columns([1.15, 1], gap="large")
	with left:
		heading("Production leaderboard", "Top states by total recorded production")
		chart = summary.sort_values("Production").tail(20)
		fig = px.bar(chart, x="Production", y="State", orientation="h", color="Production",
					 color_continuous_scale=["#285e44", GREEN, LIME])
		fig.update_layout(coloraxis_showscale=False)
		st.plotly_chart(chart_style(fig, 560), width="stretch")
	with right:
		heading("Regional map", "State-centroid markers; not administrative boundaries")
		if COORDINATES_PATH.exists():
			original = read_csv(str(COORDINATES_PATH))
			coords = canonicalize(original)
			latitude = next((c for c in original if column_key(c) == "latitude"), None)
			longitude = next((c for c in original if column_key(c) == "longitude"), None)
			if latitude and longitude and "State" in coords:
				coords["Latitude"] = pd.to_numeric(original[latitude], errors="coerce").to_numpy()
				coords["Longitude"] = pd.to_numeric(original[longitude], errors="coerce").to_numpy()
				coords = coords.dropna(subset=["Latitude", "Longitude"]).drop_duplicates("State")
				mapped = summary.merge(coords[["State", "Latitude", "Longitude"]], on="State", how="inner")
				if not mapped.empty:
					fig = px.scatter_geo(mapped, lat="Latitude", lon="Longitude", size="Production", color="Production",
										 hover_name="State", hover_data={"Production": ":,.0f", "Mean_yield": ":.2f"},
										 color_continuous_scale=["#285e44", GREEN, LIME], projection="mercator", size_max=28)
					fig.update_geos(scope="asia", center={"lat": 22, "lon": 80}, lataxis_range=[5, 38], lonaxis_range=[67, 98],
									showcountries=True, countrycolor="#506458", showland=True, landcolor="#172b21",
									showocean=True, oceancolor="#0d1c15", bgcolor="rgba(0,0,0,0)")
					fig.update_layout(coloraxis_showscale=False)
					st.plotly_chart(chart_style(fig, 560), width="stretch")
				else:
					st.info("State names did not match the coordinate file; use the ranked comparison.")
			else:
				st.info("Usable state coordinates are not available; use the ranked comparison.")
		else:
			st.info("No coordinate file is available; use the ranked comparison.")


def render_explorer():
	st.title("Dataset Explorer")
	st.caption("Inspect, filter, and export CSV files already in this project.")
	available = {name: path for name, path in DATASETS.items() if path.exists()}
	if not available:
		st.error("No recognized CSV files were found under data/.")
		return
	selected = st.selectbox("Dataset", list(available))
	raw = read_csv(str(available[selected]))
	if raw.empty:
		st.error("This file could not be read as a non-empty CSV.")
		return
	data = canonicalize(raw)
	cards = st.columns(3, gap="medium")
	for slot, label, value, detail, icon, accent in [
		(cards[0], "ROWS", f"{len(data):,}", "Source records", "▤", "green"),
		(cards[1], "COLUMNS", f"{len(data.columns):,}", "Available fields", "▥", "lime"),
		(cards[2], "MISSING VALUES", f"{int(data.isna().sum().sum()):,}", "Across all fields", "◌", "orange"),
	]:
		with slot:
			metric_card(label, value, detail, icon, accent)
	filters = st.columns(3)
	for slot, column in zip(filters, ["Crop", "State", "Year"]):
		if column not in data:
			continue
		with slot:
			if column == "Year":
				years = pd.to_numeric(data[column], errors="coerce").dropna()
				if not years.empty and int(years.min()) < int(years.max()):
					bounds = st.slider("Year range", int(years.min()), int(years.max()),
									   (int(years.min()), int(years.max())), key="explorer_year")
					data = data[pd.to_numeric(data[column], errors="coerce").between(*bounds)]
			else:
				choices = sorted(data[column].dropna().astype(str).unique())
				values = st.multiselect(column, choices, placeholder=f"All {column.lower()}s", key=f"explorer_{column}")
				if values:
					data = data[data[column].astype(str).isin(values)]
	quality = pd.DataFrame({
		"Column": data.columns,
		"Data type": [str(data[c].dtype) for c in data.columns],
		"Missing": [int(data[c].isna().sum()) for c in data.columns],
		"Missing %": [round(float(data[c].isna().mean() * 100), 2) for c in data.columns],
		"Distinct values": [int(data[c].nunique(dropna=True)) for c in data.columns],
	})
	heading("Data quality", f"Filtered records: {len(data):,}")
	st.dataframe(quality, width="stretch", hide_index=True)
	heading("Preview", "First 100 rows of filtered data")
	st.dataframe(data.head(100), width="stretch", hide_index=True)
	st.download_button("Download filtered CSV", data=data.to_csv(index=False).encode("utf-8-sig"),
					   file_name=f"agrivision_{column_key(selected)}_filtered.csv", mime="text/csv", type="primary")


def add_styles():
	st.markdown(
		"""
		<style>
		@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
		:root { --forest:#0b1712; --line:#263a2e; --muted:#9aada0; }
		html,body,[class*="css"] { font-family:'DM Sans','Segoe UI',sans-serif; }
		.stApp { background:var(--forest); color:#edf4eb; }
		[data-testid="stHeader"] { background:rgba(11,23,18,.86); }
		[data-testid="stSidebar"] { background:#0d1d15; border-right:1px solid #203329; }
		[data-testid="stSidebar"] > div:first-child { padding-top:1.2rem; }
		[data-testid="stSidebar"] [data-testid="stRadio"] label { padding:9px 10px; border-radius:8px; transition:background .18s ease; }
		[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#172a20; }
		[data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] { color:#8de2ae; }
		h1,h2,h3 { font-family:'Manrope','DM Sans',sans-serif; letter-spacing:0; color:#f1f6ed; }
		h1 { font-size:2rem; font-weight:750; }
		.brand { display:flex; align-items:center; gap:12px; padding:6px 4px 24px; }
		.brand-mark { display:grid; place-items:center; height:42px; width:42px; border-radius:12px; background:#1c422d; color:#8fe2a8; font-size:24px; }
		.brand strong { display:block; color:#eff6ee; font:700 17px 'Manrope',sans-serif; }
		.brand small { display:block; color:#829888; font-size:9px; letter-spacing:1px; margin-top:3px; }
		.sidebar-label,.eyebrow { color:#84a28d; font-size:10px; font-weight:700; letter-spacing:1.4px; }
		.sidebar-label { padding:0 10px 10px; }
		.sidebar-note { margin:28px 8px 0; border-top:1px solid #203329; padding:18px 4px; color:#829888; font-size:11px; }
		.live-dot { display:inline-block; width:7px; height:7px; margin-right:6px; border-radius:50%; background:#69d99a; box-shadow:0 0 10px #69d99a55; }
		.hero { min-height:265px; display:flex; align-items:center; justify-content:space-between; padding:38px 44px; margin:8px 0 12px; border:1px solid #294334; border-radius:14px; overflow:hidden; background-color:#173b28; background-image:linear-gradient(90deg,rgba(8,24,15,.96),rgba(9,30,18,.83) 44%,rgba(8,25,16,.18)),url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85'); background-size:cover; background-position:center 54%; }
		.hero-copy,.hero-stamp { position:relative; z-index:1; }
		.hero h1 { margin:13px 0 0; font-size:48px; line-height:1.08; }
		.hero-copy p { color:#d4e7d5; margin:4px 0 14px; font:500 19px 'Manrope',sans-serif; }
		.hero-subtitle { color:#b0c5b4; font-size:13px; }
		.hero-stamp { display:flex; flex-direction:column; align-items:flex-end; gap:5px; color:#e5f4df; font-size:9px; letter-spacing:2px; }
		.hero-stamp strong { font:800 21px 'Manrope',sans-serif; letter-spacing:0; }
		.hero-stamp i { color:#b9e87d; font-size:26px; font-style:normal; }
		.metric-card { min-height:126px; padding:17px 18px 15px; margin:0 0 8px; border:1px solid #263c30; border-radius:10px; background:linear-gradient(145deg,#15271e,#102018); box-shadow:0 9px 22px #00000017; transition:transform .18s ease,border-color .18s ease; }
		.metric-card:hover { transform:translateY(-2px); border-color:#43694e; }
		.metric-top { display:flex; align-items:center; justify-content:space-between; color:#9db1a2; font-size:9px; font-weight:700; letter-spacing:1px; }
		.metric-icon { display:grid; place-items:center; width:27px; height:27px; border-radius:8px; color:#82dda1; background:#234531; font-size:16px; }
		.metric-card.lime .metric-icon { color:#d0ed90; background:#394523; }
		.metric-card.orange .metric-icon { color:#f0ba7d; background:#45321f; }
		.metric-card.blue .metric-icon { color:#8bd1c0; background:#1c3d39; }
		.metric-value { margin-top:11px; color:#eff6ed; font:700 25px 'Manrope',sans-serif; overflow-wrap:anywhere; }
		.metric-detail { margin-top:3px; color:#8ea295; font-size:10px; }
		.section-title { margin:15px 0 1px; color:#e7efe6; font:700 16px 'Manrope',sans-serif; }
		.advisory { padding:24px 26px; margin:18px 0; border:1px solid #445437; border-left:4px solid #c1e875; border-radius:10px; background:#17271d; }
		.advisory.warning { border-color:#594532; border-left-color:#e5a766; }
		.advisory h2 { margin:9px 0 5px; font-size:22px; }
		.advisory p { margin:0; max-width:860px; color:#bac9bb; }
		[data-testid="stDataFrame"] { border:1px solid #263a2e; border-radius:8px; overflow:hidden; }
		.stSelectbox>div>div,.stMultiSelect>div>div,.stNumberInput>div>div { background:#13241b; }
		@media(max-width:700px) { .hero { min-height:225px; padding:26px 22px; background-position:62% center; } .hero h1 { font-size:38px; } .hero-stamp { display:none; } .hero-subtitle { max-width:250px; line-height:1.5; } .metric-value { font-size:21px; } }
		@media(prefers-reduced-motion:reduce) { *,*::before,*::after { transition:none!important; scroll-behavior:auto!important; } }
		</style>
		""",
		unsafe_allow_html=True,
	)


def main():
	add_styles()
	frame, source = load_main_data()
	page = render_sidebar()
	if frame.empty:
		st.error("No readable crop dataset was found. Check data/processed_data or data/.")
		st.info("Expected: crop_weather_merged.csv, training_data.csv, or crop_production_preprocessed.csv.")
		return
	if page == "Overview":
		render_overview(frame, source)
	elif page == "Crop Yield Analytics":
		render_yield(frame)
	elif page == "Weather Insights":
		render_weather(frame)
	elif page == "Smart Irrigation Advisor":
		render_irrigation(frame)
	elif page == "Regional Analysis":
		render_regions(frame)
	else:
		render_explorer()
	st.markdown('<div style="border-top:1px solid #203329;margin-top:28px;padding:14px 0;color:#708579;font-size:10px">AGRIVISION · HISTORICAL DATA ONLY · VERIFY DECISIONS WITH LOCAL FIELD CONDITIONS</div>', unsafe_allow_html=True)


if __name__ == "__main__":
	main()
