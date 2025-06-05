from Dataset_maker import burglaries_month_LSOA_complete, get_deprivation_score, burglaries_month_LSOA
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pylab as plt
import pmdarima as pm
from tqdm import tqdm
from joblib import Parallel, delayed


def prediction_network_test():
    '''
    Tests the model and gives a dataframe with 7 columns of different outputs.
    '''
    # Get the data.
    df_deprivation = get_deprivation_score()
    df_crimes_month = burglaries_month_LSOA_complete()
    df_crimes_deprivation = pd.merge(df_crimes_month, df_deprivation, on='LSOA_code', how='inner')

    df_crimes_deprivation['month'] = pd.to_datetime(df_crimes_deprivation['month'], infer_datetime_format=True)
    df_crimes_deprivation = df_crimes_deprivation.set_index(['month'])
    df_crimes_deprivation = fill_missing_months_test(df_crimes_deprivation)

    # Group the data by LSOA.
    grouped = df_crimes_deprivation.groupby('LSOA_code')

    # Run all groups in parallel how many at a time depends on your pc. Change n_jobs to a number if you do not want to
    # use all your logical processors. n_jobs=-1 for full use.
    results = Parallel(n_jobs=-1)(delayed(process_group_test)(code, group) for code, group in tqdm(grouped))

    # Aggregate the results.
    df_prediction_accuracy = pd.DataFrame(results)
    df_prediction_accuracy = df_prediction_accuracy.sum().to_frame().T  # collapse into single-row summary

    # Print the dataframe in the right way.
    with pd.option_context('display.max_columns', None):
        print(df_prediction_accuracy)



def process_group_test(LSOA_code, group):
    '''
    :param LSOA_code: The code for the LSOA.
    :param group: The group with all months of 1 LSOA used for prediction.
    :return: The code for the LSOA.
    :return: The values of the prediction.
    '''
    # Change the group dataframe to the right format.
    group['crime_diff'] = group['crime_count'].diff(periods=4)
    group['crime_diff'].fillna(method='backfill', inplace=True)
    group['month_index'] = group.index.month

    # Split into test and training groups, test: feb 2025 and train everything else.
    group_train = group[group.index != datetime(2025, 2, 1)]
    group_test = group[group.index == datetime(2025, 2, 1)]

    exog_features = group[['month_index']].copy()
    exog_train = exog_features.loc[group_train.index]


    try:
        # Define the model.
        SARIMAX_model = pm.auto_arima(group_train[['crime_count']], exogenous=exog_train,
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
        fitted, lower, high = sarimax_forecast_test(SARIMAX_model, group_train, periods=1)

        # Get the values.
        prediction = fitted.values[0]
        lower = lower.values[0]
        high = high.values[0]
        value = group_test['crime_count'].values[0]

        # Calculate which groups to increment.
        metrics = {
            'Pred_right': int(round(prediction) == value),
            'Pred_wrong_high': int(round(prediction) > value),
            'pred_wrong_low': int(round(prediction) < value),
            'In_range': int(lower <= value <= high),
            'Out_range_low': int(value < lower),
            'Out_range_high': int(value > high),
            'total': 1
        }

    except Exception as e:
        # Return all 0 if the model fails.
        return {
            'Pred_right': 0,
            'Pred_wrong_high': 0,
            'pred_wrong_low': 0,
            'In_range': 0,
            'Out_range_low': 0,
            'Out_range_high': 0,
            'total': 0
        }


    return metrics


def sarimax_forecast_test(SARIMAX_model, prediction_df, periods):
    '''
    :param SARIMAX_model: The model.
    :param prediction_df: The dataframe that should be predicted.
    :param periods: Number of periods that should be predicted.
    :return: A dataframe with the predicted value(s).
    '''
    # Forecast
    n_periods = periods

    forecast_df = pd.DataFrame(
        {
            'month_index': pd.date_range(prediction_df.index[-1], periods=periods, freq='MS').month
            # ,'deprivation': [prediction_df['deprivation'].iloc[-1]] * periods
        },
        index=pd.date_range(prediction_df.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS')
    )

    fitted, confint = SARIMAX_model.predict(n_periods=n_periods,
                                            return_conf_int=True,
                                            exogenous=forecast_df[['month_index']])
    index_of_fc = pd.date_range(prediction_df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')

    # Calculate the series.
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Calculate the bounds.
    fitted_series[fitted_series < 0] = 0
    lower_series[lower_series < 0] = 0
    upper_series[upper_series < 0] = 0

    ### Plot the graph comment out if not wanted ###
    # forcast_plot(fitted_series, lower_series, upper_series, confint, index_of_fc, prediction_df)

    return fitted_series, lower_series, upper_series

def fill_missing_months_test(df_crimes_month):
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


def forcast_plot(fitted_series, lower_series, upper_series, confint, index_of_fc, prediction_df):
    # Plot the forcast
    plt.figure(figsize=(15, 7))
    plt.plot(prediction_df['crime_count'], color='#1f76b4')
    plt.plot(fitted_series, color='darkgreen')
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)

    plt.title('SARIMAX - Forecast of Airline Passengers')
    plt.show()



def plot_trend_season_residual(LSOA_code, group):
    # Plot the trend and seasonal functions.
    result = seasonal_decompose(group['crime_count'], model='additive', period=4)
    trend = result.trend.dropna()
    seasonal = result.seasonal.dropna()
    residual = result.resid.dropna()

    # Plot the decomposed components
    plt.figure(figsize=(6, 6))
    plt.title(LSOA_code)

    plt.subplot(4, 1, 1)
    plt.plot(group['crime_count'], label='Original Series')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(trend, label='Trend')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(seasonal, label='Seasonal')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(residual, label='Residuals')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run the code.
prediction_network_test()