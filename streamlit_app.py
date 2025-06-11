# --- London Crime Analysis Streamlit App (Enhanced) ---
"""
This Streamlit application provides an interactive dashboard to explore London
crime data, deprivation indices and an optimisation‑based resource allocation
schedule.  Improvements over the previous version:

• **Total Burglaries** – YoY %‑change toggle + rolling mean
• **LSOA Crimes**     – choose *All‑time* or single month, map + Top‑10 bar
• **Deprivation vs Burglaries** – fixed join, proper column names, correlation
• **Resource Allocation** – fixed columns, choropleth + bar of top‑allocated
"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Config & helpers
# ---------------------------------------------------------------------------
st.set_page_config(page_title="London Crime Dashboard", layout="wide")
st.title("\U0001F4C8 London Crime Analysis Dashboard")

DATA_DIR = "Streamlit_files"
MAPBOX_TOKEN = "pk.eyJ1Ijoia3Vib3JybyIsImEiOiJjbWJwcG93aWMwN2R6MmxxdTNxbGliamdxIn0.2R-jJjLu8pU_dvbr_vylmw"
px.set_mapbox_access_token(MAPBOX_TOKEN)

# --------------------------- Data loaders (cached) -------------------------
@st.cache_data(show_spinner="Loading …")
def load_csv(name, **kw):
    return pd.read_csv(f"{DATA_DIR}/{name}", **kw)

@st.cache_data
def load_geojson():
    with open(f"{DATA_DIR}/london_lsoa_filtered.geojson", "r") as f:
        return json.load(f)

@st.cache_data
def lsoa_name_map():
    gj = load_geojson()
    return {
        feat["properties"]["LSOA21CD"]: feat["properties"]["LSOA21NM"]
        for feat in gj["features"]
    }

# Individual datasets --------------------------------------------------------

def total_burglaries():
    df = load_csv("total_burglaries_per_month.csv", index_col=0)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    return df.sort_values("month")

def lsoa_monthly():
    df = load_csv("crimes_per_month_per_LSOA.csv", index_col=0)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "month"}, inplace=True)
    df["month"] = pd.to_datetime(df["month"])
    return df

def burglaries_lsoa():
    return load_csv("burglaries_per_LSOA.csv")

def deprivation():
    return load_csv("deprivation_LSOA.csv")

def schedule():
    return load_csv("schedule_output.csv")

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "Total Burglaries",
    "LSOA Crimes Map",
    "Deprivation vs Burglaries",
    "Resource Allocation"
]
page = st.sidebar.radio("Choose a page", PAGES)

# ---------------------------------------------------------------------------
# PAGE 1 – TOTAL BURGLARIES
# ---------------------------------------------------------------------------
if page == "Total Burglaries":
    df = total_burglaries()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Monthly burglary counts")
    with col_right:
        yoy_toggle = st.toggle("Show YoY % change")

    # Controls – convert pandas Timestamps -> python datetime for Streamlit slider
    min_dt = df["month"].min().to_pydatetime()
    max_dt = df["month"].max().to_pydatetime()

    dr = st.slider(
        "Date range",
        value=(min_dt, max_dt),
        min_value=min_dt,
        max_value=max_dt,
        format="%b %Y"
    )
    window_on = st.checkbox("3‑month rolling mean", value=False)

    # Filter & transform
    filtered = df[(df["month"] >= pd.to_datetime(dr[0])) & (df["month"] <= pd.to_datetime(dr[1]))].copy()
    if window_on:
        filtered["rolling"] = filtered["crime_count"].rolling(3).mean()
    if yoy_toggle:
        filtered["pct_change"] = filtered["crime_count"].pct_change(12) * 100

    # Choose y-axis column
    y_col = "pct_change" if yoy_toggle else ("rolling" if window_on else "crime_count")
    y_title = "% change YoY" if yoy_toggle else ("3‑month mean" if window_on else "Burglaries")

    # Plot
    fig = px.line(
        filtered, x="month", y=y_col, markers=True,
        labels={"month": "Month", y_col: y_title}
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(filtered)

# ---------------------------------------------------------------------------
if page == "Total Burglaries":
    df = total_burglaries()

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Monthly burglary counts")
    with col_right:
        yoy_toggle = st.toggle("Show YoY % change")

    # Controls
    min_dt, max_dt = df["month"].min(), df["month"].max()
    dr = st.slider("Date range", value=(min_dt, max_dt), min_value=min_dt, max_value=max_dt,
                   format="%b %Y")
    window_on = st.checkbox("3‑month rolling mean", value=False)

    # Filter & transform
    filtered = df.query("@dr[0] <= month <= @dr[1]").copy()
    if window_on:
        filtered["rolling"] = filtered["crime_count"].rolling(3).mean()
    if yoy_toggle:
        filtered["pct_change"] = filtered["crime_count"].pct_change(12) * 100

    # Plot
    y_col = "pct_change" if yoy_toggle else ("rolling" if window_on else "crime_count")
    y_title = "% change YoY" if yoy_toggle else "Burglaries"
    fig = px.line(filtered, x="month", y=y_col, markers=True,
                  labels={"month": "Month", y_col: y_title})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(filtered)

# ---------------------------------------------------------------------------
# PAGE 2 – LSOA CRIME MAP
# ---------------------------------------------------------------------------
elif page == "LSOA Crimes Map":
    df = lsoa_monthly()
    gj = load_geojson()
    name_map = lsoa_name_map()

    st.subheader("Crime density by LSOA")
    mode = st.radio("Display", ["Single month", "All time"], horizontal=True)

    if mode == "Single month":
        months = sorted(df["month"].dt.to_period("M").unique().to_timestamp())
        sel_month = st.selectbox("Month", months)
        subset = df[df["month"] == sel_month]
        title_suffix = sel_month.strftime("%b %Y")
    else:
        subset = df.copy()
        title_suffix = "All‑time"

    grouped = subset.groupby("LSOA_code", as_index=False)["crime_count"].sum()
    grouped["LSOA_name"] = grouped["LSOA_code"].map(name_map)

    fig = px.choropleth_mapbox(grouped, geojson=gj, locations="LSOA_code", featureidkey="properties.LSOA21CD",
                                color="crime_count", color_continuous_scale="OrRd", opacity=0.65,
                                hover_name="LSOA_name", hover_data={"crime_count":":,"},
                                center={"lat": 51.5074, "lon": -0.1278}, zoom=9,
                                title=f"Crime count – {title_suffix}")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Top‑10 bar
    top10 = grouped.nlargest(10, "crime_count").sort_values("crime_count")
    st.markdown("### Top 10 LSOAs by crime count")
    st.bar_chart(top10.set_index("LSOA_name")["crime_count"])

# ---------------------------------------------------------------------------
# PAGE 3 – DEPRIVATION VS BURGLARIES
# ---------------------------------------------------------------------------
elif page == "Deprivation vs Burglaries":
    crime = burglaries_lsoa()
    dep   = deprivation()
    merged = crime.merge(dep, on="LSOA_code", how="inner")

    st.subheader("Socio‑economic deprivation vs burglary")

    # Scatter with trendline
    fig = px.scatter(merged, x="deprivation", y="crime_count", trendline="ols",
                     labels={"deprivation": "Deprivation score", "crime_count": "Burglary count"},
                     hover_data=["LSOA_code"])
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation coefficient
    corr = np.corrcoef(merged["deprivation"], merged["crime_count"])[0, 1]
    st.info(f"**Pearson r = {corr:.2f}**")

    with st.expander("Raw merged data"):
        st.dataframe(merged)

# ---------------------------------------------------------------------------
# PAGE 4 – RESOURCE ALLOCATION
# ---------------------------------------------------------------------------
else:
    sched = schedule()
    gj = load_geojson()
    name_map = lsoa_name_map()

    st.subheader("Optimised police resource allocation")

    # Map
    sched["LSOA_name"] = sched["lsoa21cd"].map(name_map)
    fig = px.choropleth_mapbox(sched, geojson=gj, locations="lsoa21cd", featureidkey="properties.LSOA21CD",
                                color="officers_allocated", color_continuous_scale="Blues", opacity=0.7,
                                hover_name="LSOA_name", hover_data={"risk_score":":.2f","officers_allocated":":d"},
                                center={"lat": 51.5074, "lon": -0.1278}, zoom=9,
                                title="Allocated officers by LSOA")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Top‑N bar
    top_n = st.slider("Show Top‑N areas", 5, 20, 10)
    top_sched = sched.nlargest(top_n, "officers_allocated").sort_values("officers_allocated")
    st.bar_chart(top_sched.set_index("LSOA_name")["officers_allocated"], height=400)

    with st.expander("Resource‑allocation table"):
        st.dataframe(sched)
