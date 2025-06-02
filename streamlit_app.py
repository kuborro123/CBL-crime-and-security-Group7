import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Page config
st.set_page_config(page_title='Data Explorer', page_icon='📊', layout='wide')

# Dataset definitions
DATASETS = {
    'Daily Sales': {
        'file': 'data/sales_data_daily.csv',
        'date_col': 'date'
    },
    'Customers': {
        'file': 'data/customer_data.csv',
        'date_col': 'signup_date'
    }
}

# Sidebar - dataset selector
st.sidebar.title("Choose Dataset")
dataset_name = st.sidebar.selectbox("Dataset", list(DATASETS.keys()))
spec = DATASETS[dataset_name]

# Cached loader
@st.cache_data(show_spinner='Loading data …')
def load_data(path, date_col):
    df = pd.read_csv(path, parse_dates=[date_col])
    return df

df = load_data(spec['file'], spec['date_col'])
st.title(f"{dataset_name} Dashboard")

# ---------------------------------------------------------------------------
# DAILY SALES DASHBOARD
# ---------------------------------------------------------------------------
if dataset_name == 'Daily Sales':
    # Filters
    regions = st.sidebar.multiselect('Region', sorted(df['region'].unique()), default=sorted(df['region'].unique()))
    df_f = df[df['region'].isin(regions)]

    start_date = df_f['date'].min().to_pydatetime()
    end_date = df_f['date'].max().to_pydatetime()
    date_range = st.sidebar.slider('Date range', min_value=start_date, max_value=end_date, value=(start_date, end_date))
    df_f = df_f[(df_f['date'] >= date_range[0]) & (df_f['date'] <= date_range[1])]

    # Trend line
    st.subheader('📈 Daily Sales Trend')
    pivot = df_f.pivot_table(index='date', columns='region', values='sales', aggfunc='sum')
    st.line_chart(pivot)

    # Bar chart
    st.subheader('📊 Total Sales by Region')
    totals = df_f.groupby('region')['sales'].sum()
    st.bar_chart(totals)

    # Heatmap by region/month
    st.subheader("🔥 Average Sales Heatmap")
    df_f['month'] = df_f['date'].dt.strftime('%b')
    df_f['year'] = df_f['date'].dt.year
    pivot_heat = df_f.pivot_table(index='region', columns='month', values='sales', aggfunc='mean')
    fig, ax = plt.subplots()
    sns.heatmap(pivot_heat, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    # Map visualization
    st.subheader("🗺️ Sales by Region (Map)")
    region_coords = {
        'North': {'lat': 52.5, 'lon': 19.5},
        'South': {'lat': 49.5, 'lon': 19.5},
        'East':  {'lat': 51.0, 'lon': 23.5},
        'West':  {'lat': 51.0, 'lon': 15.5},
    }
    map_df = df_f.groupby("region")["sales"].sum().reset_index()
    map_df["lat"] = map_df["region"].map(lambda x: region_coords[x]["lat"])
    map_df["lon"] = map_df["region"].map(lambda x: region_coords[x]["lon"])

    fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        size="sales",
        hover_name="region",
        size_max=50,
        color="sales",
        zoom=4,
        mapbox_style="open-street-map"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Raw data
    st.subheader('🧾 Raw Data')
    st.dataframe(df_f)

# ---------------------------------------------------------------------------
# CUSTOMER DASHBOARD
# ---------------------------------------------------------------------------
else:
    # Filters
    regions = st.sidebar.multiselect('Region', sorted(df['region'].unique()), default=sorted(df['region'].unique()))
    genders = st.sidebar.multiselect('Gender', sorted(df['gender'].unique()), default=sorted(df['gender'].unique()))
    df_f = df[df['region'].isin(regions) & df['gender'].isin(genders)]

    # KPIs
    st.metric('👥 Total Customers', len(df_f))
    churn_rate = df_f['churned'].mean()
    st.metric('⚠️ Churn Rate', f'{churn_rate:.1%}')

    # Churn heatmap
    st.subheader("🔥 Churn Rate Heatmap by Gender and Region")
    churn_pivot = df_f.pivot_table(index='gender', columns='region', values='churned', aggfunc='mean')
    fig, ax = plt.subplots()
    sns.heatmap(churn_pivot, annot=True, fmt=".2f", cmap="RdPu", ax=ax)
    st.pyplot(fig)

    # Bar charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gender Distribution")
        st.bar_chart(df_f['gender'].value_counts())

    with col2:
        st.subheader("Age Distribution")
        age_counts = df_f['age'].value_counts().sort_index()
        st.bar_chart(age_counts)

    # Churn by region
    st.subheader("📉 Churn by Region")
    churn_by_region = df_f.groupby('region')['churned'].mean()
    st.bar_chart(churn_by_region)

    # Raw data
    st.subheader('🧾 Raw Data')
    st.dataframe(df_f)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('---\nMade with ❤️ using Streamlit')
