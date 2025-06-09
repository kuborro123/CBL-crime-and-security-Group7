import pandas as pd
import streamlit as st
import plotly.express as px
import json

# Page configuration
st.set_page_config(page_title="Crime Dashboard", layout="wide")
st.title("London Crime Analysis")

# File directory for data
DATA_DIR = "Streamlit_files"

# ---------- Data Loaders ----------

@st.cache_data(show_spinner="Loading data...")
def load_csv(filename, **kwargs):
    path = f"{DATA_DIR}/{filename}"
    return pd.read_csv(path, **kwargs)

@st.cache_data
def load_lsoa_geojson():
    with open(f"{DATA_DIR}/london_lsoa_filtered.geojson", "r") as f:
        return json.load(f)

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

# ---------- Page Logic ----------

if page == "Total Burglaries":
    df = load_total_burglaries()
    st.subheader("Total Burglaries per Month")
    fig = px.line(df, x="month", y="crime_count", markers=True)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df)

elif page == "LSOA Monthly Crimes":
    df = load_lsoa_monthly()
    geojson = load_lsoa_geojson()

    # Month selection
    selected_month = st.selectbox("Select Month", sorted(df["month"].dt.strftime("%Y-%m").unique()))
    df_month = df[df["month"].dt.strftime("%Y-%m") == selected_month]

    st.subheader(f"Burglary Map for {selected_month}")

    fig = px.choropleth_mapbox(
        df_month,
        geojson=geojson,
        locations="LSOA_code",
        color="crime_count",
        featureidkey="properties.LSOA11CD",  # Adjust if your GeoJSON uses a different key
        mapbox_style="carto-positron",
        center={"lat": 51.5074, "lon": -0.1278},
        zoom=9,
        opacity=0.6,
        hover_name="LSOA_code"
    )
    fig.update_layout(margin={"r":0, "t":0, "l":0, "b":0})
    st.plotly_chart(fig, use_container_width=True)

    # Time series for selected LSOA
    lsoas = sorted(df["LSOA_code"].unique())
    selected = st.selectbox("Inspect Time Series for LSOA", lsoas)
    df_f = df[df["LSOA_code"] == selected]
    st.subheader(f"Monthly Burglaries for {selected}")
    fig = px.line(df_f, x="month", y="crime_count", markers=True)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df_f)

elif page == "Deprivation vs Burglaries":
    burg = load_burglaries_lsoa()
    dep = load_deprivation()
    merged = pd.merge(burg, dep, on="LSOA_code", how="inner")
    st.subheader("Burglary Count vs Deprivation Score")
    fig = px.scatter(merged, x="deprivation", y="crime_count", hover_name="LSOA_code")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(merged)

elif page == "Resource Allocation":
    df = load_schedule()
    wards = sorted(df["ward"].unique())
    ward = st.selectbox("Ward", ["All"] + wards)
    if ward != "All":
        df = df[df["ward"] == ward]
    st.subheader("Officer Allocation Schedule")
    st.dataframe(df)

# ---------- Footer ----------
st.markdown("---\nMade with ❤️ using Streamlit")
