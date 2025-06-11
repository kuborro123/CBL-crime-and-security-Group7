# --- London Crime Analysis Streamlit App (Enhanced & Fixed) ---
"""
Interactive dashboard for London crime data, deprivation indices and
optimised police‑resource allocation.
"""
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="London Crime Dashboard", layout="wide")
st.title("📈 London Crime Analysis Dashboard")

DATA_DIR = "Streamlit_files"
px.set_mapbox_access_token(
    "pk.eyJ1Ijoia3Vib3JybyIsImEiOiJjbWJwcG93aWMwN2R6MmxxdTNxbGliamdxIn0.2R-jJjLu8pU_dvbr_vylmw"
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers – cached loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading …")
def load_csv(name: str, **kw):
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

# Dataset‑specific wrappers ---------------------------------------------------

def total_burglaries():
    df = load_csv("total_burglaries_per_month.csv", index_col=0)
    df["month"] = pd.to_datetime(df["month"], format="MMM YYYY")
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

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
PAGES = [
    "Total Burglaries",
    "LSOA Crimes Map",
    "Deprivation vs Burglaries",
    "Resource Allocation",
]
page = st.sidebar.radio("Choose a page", PAGES)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – TOTAL BURGLARIES
# ─────────────────────────────────────────────────────────────────────────────
if page == "Total Burglaries":
    df = total_burglaries()

    st.subheader("Monthly burglary counts")

    # Controls
    min_dt = df["month"].min().to_pydatetime()
    max_dt = df["month"].max().to_pydatetime()
    dr = st.slider(
        "Date range",
        min_value=min_dt,
        max_value=max_dt,
        value=(min_dt, max_dt),
        format="MMM YYYY",
        key="date_range",
    )
    yoy_toggle = st.toggle("Show Year‑over‑Year % change", key="yoy")
    rolling_toggle = st.checkbox("3‑month rolling mean", key="rolling")

    # Filter + transform
    filt = df[(df["month"] >= pd.to_datetime(dr[0])) & (df["month"] <= pd.to_datetime(dr[1]))].copy()
    if rolling_toggle:
        filt["rolling"] = filt["crime_count"].rolling(3).mean()
    if yoy_toggle:
        filt["pct_change"] = filt["crime_count"].pct_change(12) * 100

    y_col = "pct_change" if yoy_toggle else ("rolling" if rolling_toggle else "crime_count")
    y_title = "% change YoY"  # Year‑over‑Year percentage change if yoy_toggle else ("3‑month mean" if rolling_toggle else "Burglaries")

    fig = px.line(filt, x="month", y=y_col, markers=True, labels={"month": "Month", y_col: y_title})
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(filt)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – LSOA CRIMES MAP
# ─────────────────────────────────────────────────────────────────────────────
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

    fig = px.choropleth_mapbox(
        grouped,
        geojson=gj,
        locations="LSOA_code",
        featureidkey="properties.LSOA21CD",
        color="crime_count",
        color_continuous_scale="OrRd",
        opacity=0.65,
        hover_name="LSOA_name",
        hover_data={"crime_count":":,"},
        center={"lat": 51.5074, "lon": -0.1278},
        zoom=9,
        title=f"Crime count – {title_suffix}",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    top10 = grouped.nlargest(10, "crime_count").sort_values("crime_count")
    st.markdown("### Top 10 LSOAs by crime count")
    st.bar_chart(top10.set_index("LSOA_name")["crime_count"])

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – DEPRIVATION VS BURGLARIES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Deprivation vs Burglaries":
    crime = burglaries_lsoa()
    dep = deprivation()
    merged = crime.merge(dep, on="LSOA_code", how="inner")

    st.subheader("Socio‑economic deprivation vs burglary")

    fig = px.scatter(
        merged,
        x="deprivation",
        y="crime_count",
        trendline="ols",
        labels={"deprivation": "Deprivation score", "crime_count": "Burglary count"},
        hover_data=["LSOA_code"],
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    corr = np.corrcoef(merged["deprivation"], merged["crime_count"])[0, 1]
    st.info(f"**Pearson r = {corr:.2f}**")

    with st.expander("Raw merged data"):
        st.dataframe(merged)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 – RESOURCE ALLOCATION
# ─────────────────────────────────────────────────────────────────────────────
else:
    sched = schedule()
    gj = load_geojson()
    name_map = lsoa_name_map()

    st.subheader("Optimised police resource allocation")

    sched["LSOA_name"] = sched["lsoa21cd"].map(name_map)

    fig = px.choropleth_mapbox(
        sched,
        geojson=gj,
        locations="lsoa21cd",
        featureidkey="properties.LSOA21CD",
        color="officers_allocated",
        color_continuous_scale="Blues",
        opacity=0.7,
        hover_name="LSOA_name",
        hover_data={"risk_score":":.2f", "officers_allocated":":d"},
        center={"lat": 51.5074, "lon": -0.1278},
        zoom=9,
        title="Allocated officers by LSOA",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    top_n = st.slider("Show Top‑N areas", 5, 20, 10)
    top_sched = sched.nlargest(top_n, "officers_allocated").sort_values("officers_allocated")
    st.bar_chart(top_sched.set_index("LSOA_name")["officers_allocated"], height=400)

    with st.expander("Resource‑allocation table"):
        st.dataframe(sched)
