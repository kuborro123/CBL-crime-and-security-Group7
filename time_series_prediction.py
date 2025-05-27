from Dataset_maker import burglaries_month_LSOA, get_deprivation_score
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pylab as plt
import pmdarima as pm
from tqdm import tqdm
from joblib import Parallel, delayed


def prediction_network():
    '''
    :return: df_prediction: The final dataframe with LSOA_code and a prediction of crimes in the comming month.
    '''
    # Get the data.
    df_crime_month = burglaries_month_LSOA()

    df_crime_month['month'] = pd.to_datetime(df_crime_month['month'], infer_datetime_format=True)
    df_crime_month = df_crime_month.set_index(['month'])

    df_crime_month = fill_missing_months(df_crime_month)

    # Group the data by LSOA.
    grouped = df_crime_month.groupby('LSOA_code')

    # Run all groups in parallel how many at a time depends on your pc. Change n_jobs to a number if you do not want to
    # use all your logical processors. n_jobs=-1 for full use.
    results = Parallel(n_jobs=-1)(delayed(process_group)(code, group) for code, group in tqdm(grouped))

    # Put the prediction into a dataframe we can use.
    df_prediction = pd.DataFrame(results, columns=['LSOA_code', 'predicted_value'])
    df_prediction['predicted_value'] = df_prediction['predicted_value'].apply(lambda x: x[0])
    df_prediction['predicted_value'] = df_prediction['predicted_value'].apply(lambda x: format(x, 'f'))

    return df_prediction

def process_group(LSOA_code, group):
    '''
    :param LSOA_code: The code for the LSOA.
    :param group: The group with all months of 1 LSOA used for prediction.
    :return: LSOA_code: The code for the LSOA.
    :return: fitted.values: the values of the prediction.
    '''
    # Change the group dataframe to the right format.
    group['crime_diff'] = group['crime_count'].diff(periods=4)
    group['crime_diff'].fillna(method='backfill', inplace=True)

    group['month_index'] = group.index.month

    # Define the model.
    SARIMAX_model = pm.auto_arima(group[['crime_count']], exogenous=group[['month_index']],
                                  start_p=1, start_q=1,
                                  test='adf',
                                  max_p=3, max_q=3, m=12,
                                  start_P=0, seasonal=True,
                                  d=None, D=1,
                                  trace=False,
                                  error_action='ignore',
                                  suppress_warnings=True,
                                  stepwise=True)

    # Predict.
    fitted = sarimax_forecast(SARIMAX_model, group, periods=1)

    return LSOA_code, fitted.values

def sarimax_forecast(SARIMAX_model, prediction_df, periods):
    '''
    :param SARIMAX_model: The model.
    :param prediction_df: The dataframe that should be predicted.
    :param periods: Number of periods that should be predicted.
    :return: A dataframe with the predicted value(s).
    '''
    # Forecast
    forecast_df = pd.DataFrame(
            {'month_index': pd.date_range(prediction_df.index[-1], periods=periods, freq='MS').month},
                    index=pd.date_range(prediction_df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')
                )

    fitted, confint = SARIMAX_model.predict(n_periods=n_periods,
                                            return_conf_int=True,
                                            exogenous=forecast_df[['month_index']])

    return fitted


def fill_missing_months(df_crimes_month):
    '''
    :param df_crimes_month: The dataframe with the number of crimes per month per LSOA.
    :return: df_filled: Same dataframe but with all missing months filled with 0.
    '''
    # define the full index per LSOA
    full_index = pd.date_range(start=df_crimes_month.index.min(), end=df_crimes_month.index.max(),
                               freq='MS')
    grouped = df_crimes_month.groupby('LSOA_code')
    filled_dfs = []

    # Sort per LSOA
    for LSOA_code, group in grouped:
        group = group.sort_index()
        group_reindexed = group.reindex(full_index)
        group_reindexed['LSOA_code'] = LSOA_code
        filled_dfs.append(group_reindexed)

    # Fill the missing months
    df_filled = pd.concat(filled_dfs)
    df_filled['crime_count'] = df_filled['crime_count'].fillna(0)

    return df_filled

df_prediction = prediction_network()
print(df_prediction)