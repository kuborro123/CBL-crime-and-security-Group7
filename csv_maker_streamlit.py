from Dataset_maker import burglaries_month_LSOA, get_deprivation_score, burglaries_LSOA, total_burglaries_month
from time_series_prediction import fill_missing_months
import pandas as pd

# get_deprivation_score().to_csv('Streamlit_files/deprivation_LSOA.csv')
# burglaries_LSOA().to_csv('Streamlit_files/burglaries_per_LSOA.csv')
# total_burglaries_month().to_csv('Streamlit_files/total_burglaries_per_month.csv')

burglaries_month = burglaries_month_LSOA()
burglaries_month['month'] = pd.to_datetime(burglaries_month['month'], infer_datetime_format=True)
burglaries_month = burglaries_month.set_index(['month'])
burglaries_month = fill_missing_months(burglaries_month)
burglaries_month.to_csv('Streamlit_files/crimes_per_month_per_LSOA.csv')