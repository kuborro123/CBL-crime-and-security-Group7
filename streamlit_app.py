# --- London Crime Analysis Streamlit App ---
import streamlit as st
import pandas as pd
import plotly.express as px
import json

# Page configuration
st.set_page_config(page_title="Crime Dashboard", layout="wide")
st.title("London Crime Analysis Dashboard")

# Mapbox token
px.set_mapbox_access_token("pk.eyJ1Ijoia3Vib3JybyIsImEiOiJjbWJwcG93aWMwN2R6MmxxdTNxbGliamdxIn0.2R-jJjLu8pU_dvbr_vylmw")

# Data directory
DATA_DIR = "Streamlit_files"

# ---------- Data Loaders ----------
@st.cache_data(show_spinner="Loading CSV...")
def load_csv(filename, **kwargs):
    path = f"{DATA_DIR}/{filename}"
    return pd.read_csv(path, **kwargs)

@st.cache_data
def load_lsoa_geojson():
    with open(f"{DATA_DIR}/london_lsoa_filtered.geojson", "r") as f:
        return json.load(f)

@st.cache_data
def load_lsoa_name_mapping():
    with open(f"{DATA_DIR}/london_lsoa_filtered.geojson", "r") as f:
        geojson = json.load(f)
    return {
        f["properties"]["LSOA21CD"]: f["properties"]["LSOA21NM"]
        for f in geojson["features"]
    }

def load_total_burglaries():
    df = load_csv("total_burglaries_per_month.csv", index_col=0)
    df["month"] = pd.to_datetime(df["month"], format="%Y-%m")
    return df.sort_values("month")

def load_lsoa_monthly():
    df = load_csv("crimes_per_month_per_LSOA.csv", index_col=0, parse_dates=True)
    df.index.name = "month"
    df.reset_index(inplace=True)
    df["month"] = pd.to_datetime(df["month"])
    return df

def load_burglaries_lsoa():
    return load_csv("burglaries_per_LSOA.csv", index_col=0)

def load_deprivation():
    return load_csv("deprivation_LSOA.csv", index_col=0)

def load_schedule():
    return load_csv("schedule_output.csv")

# ---------- Sidebar Navigation ----------
PAGES = [
    "Total Burglaries",
    "LSOA Monthly Crimes",
    "Deprivation vs Burglaries",
    "Resource Allocation"
]

page = st.sidebar.selectbox("Select Page", PAGES)

# ---------- Page 1: Total Burglaries ----------
if page == "Total Burglaries":
    df = load_total_burglaries()

    st.subheader("\U0001F512 Total Burglaries per Month")
    st.markdown("""
    Analyze burglary trends across London over time. Adjust the date range and optionally apply a rolling average.
    """)

    date_range = st.slider("Select Date Range", min_value=df["month"].min().to_pydatetime(),
                           max_value=df["month"].max().to_pydatetime(),
                           value=(df["month"].min().to_pydatetime(), df["month"].max().to_pydatetime()),
                           format="MM/YYYY")
    df_filtered = df[(df["month"] >= date_range[0]) & (df["month"] <= date_range[1])]

    rolling_enabled = st.checkbox("Apply 3-Month Rolling Average")
    if rolling_enabled:
        df_filtered = df_filtered.copy()
        df_filtered["Rolling Average"] = df_filtered["crime_count"].rolling(window=3).mean()

    fig = px.line(df_filtered, x="month",
                  y="Rolling Average" if rolling_enabled else "crime_count",
                  labels={"month": "Month", "crime_count": "Burglaries", "Rolling Average": "3-Month Average"},
                  markers=True)
    fig.update_layout(title="Burglary Trends Over Time", xaxis_title="Month", yaxis_title="Number of Burglaries")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show Raw Data"):
        st.dataframe(df_filtered.reset_index(drop=True))

# ---------- Page 2: LSOA Monthly Crimes ----------
elif page == "LSOA Monthly Crimes":
    df = load_lsoa_monthly()
    geojson = load_lsoa_geojson()
    name_map = load_lsoa_name_mapping()

    st.subheader("\U0001F5FA️ Crime by LSOA (Geographic)")
    st.markdown("""
    View crime distribution geographically across LSOA regions. Use the dropdown to filter by month.
    """)

    available_months = sorted(df["month"].dt.to_period("M").unique().to_timestamp())
    selected_month = st.selectbox("Select Month", available_months)

    df_month = df[df["month"] == selected_month]
    df_total = df_month.groupby("LSOA_code", as_index=False)["crime_count"].sum()
    df_total["LSOA_name"] = df_total["LSOA_code"].map(name_map)

    fig = px.choropleth_mapbox(df_total,
                                geojson=geojson,
                                locations="LSOA_code",
                                color="crime_count",
                                featureidkey="properties.LSOA21CD",
                                hover_name="LSOA_name",
                                mapbox_style="carto-positron",
                                center={"lat": 51.5074, "lon": -0.1278},
                                zoom=9,
                                opacity=0.6,
                                color_continuous_scale="OrRd")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True)

# ---------- Page 3: Deprivation vs Burglaries ----------
elif page == "Deprivation vs Burglaries":
    crime_df = load_burglaries_lsoa()
    deprivation_df = load_deprivation()

    st.subheader("\U0001F4CA Deprivation Index vs. Burglary Rates")
    st.markdown("""
    Explore the relationship between socioeconomic deprivation and burglary rates.
    """)

    merged_df = pd.merge(crime_df, deprivation_df, left_index=True, right_index=True)

    fig = px.scatter(merged_df,
                     x="IMD_score",
                     y="crime_count",
                     labels={"IMD_score": "Deprivation Score", "crime_count": "Burglary Count"},
                     hover_name=merged_df.index,
                     trendline="ols")
    fig.update_layout(title="Correlation Between Deprivation and Burglaries")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(merged_df.reset_index())

# ---------- Page 4: Resource Allocation ----------
elif page == "Resource Allocation":
    schedule_df = load_schedule()

    st.subheader("\U0001F46E‍♂️ Optimized Resource Allocation")
    st.markdown("""
    View recommended police deployments across LSOAs based on crime trends and optimization model output.
    """)

    st.dataframe(schedule_df)

    top_areas = schedule_df.sort_values("officers", ascending=False).head(10)
    st.markdown("### Top 10 Areas by Allocated Officers")
    st.bar_chart(top_areas.set_index("LSOA_code")["officers"])
