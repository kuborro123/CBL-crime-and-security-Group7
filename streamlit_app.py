import pandas as pd
import streamlit as st

st.set_page_config(page_title='Data Explorer', page_icon='📊', layout='wide')

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Sidebar: dataset selector
# ---------------------------------------------------------------------------
dataset_name = st.sidebar.selectbox('Dataset', list(DATASETS.keys()), index=0)

# ---------------------------------------------------------------------------
# Cached data loader
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner='Loading data …')
def load_data(name: str) -> pd.DataFrame:
    spec = DATASETS[name]
    df = pd.read_csv(spec['file'], parse_dates=[spec['date_col']])
    return df

df = load_data(dataset_name)

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

    # Visuals
    st.subheader('Daily sales trend')
    pivot = df_f.pivot_table(index='date', columns='region', values='sales', aggfunc='sum')
    st.line_chart(pivot)

    st.subheader('Total sales by region')
    totals = df_f.groupby('region')['sales'].sum()
    st.bar_chart(totals)

    # Raw data
    st.subheader('Raw data')
    st.dataframe(df_f)

# ---------------------------------------------------------------------------
# CUSTOMER DASHBOARD
# ---------------------------------------------------------------------------
else:
    # Filters
    regions = st.sidebar.multiselect('Region', sorted(df['region'].unique()), default=sorted(df['region'].unique()))
    genders = st.sidebar.multiselect('Gender', sorted(df['gender'].unique()), default=sorted(df['gender'].unique()))
    df_f = df[df['region'].isin(regions) & df['gender'].isin(genders)]

    # KPI tiles
    st.metric('Total customers', len(df_f))
    churn_rate = df_f['churned'].mean()
    st.metric('Churn rate', f'{churn_rate:.1%}')

    # Visuals
    st.subheader('Churn by region')
    churn_by_region = df_f.groupby('region')['churned'].mean()
    st.bar_chart(churn_by_region)

    st.subheader('Age distribution')
    age_counts = df_f['age'].value_counts().sort_index()
    st.bar_chart(age_counts)

    # Raw data
    st.subheader('Raw data')
    st.dataframe(df_f)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('---\nMade with ❤️ using Streamlit')
