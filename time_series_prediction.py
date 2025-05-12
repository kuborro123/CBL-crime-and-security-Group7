from Dataset_maker import burglaries_month_LSOA, get_deprivation_score
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pylab as plt
import pmdarima as pm


def prediction_network():
    df_crime_month = burglaries_month_LSOA()

    df_crime_month['month'] = pd.to_datetime(df_crime_month['month'], infer_datetime_format=True)
    df_crime_month = df_crime_month.set_index(['month'])

    df_crime_month = fill_missing_months(df_crime_month)

    grouped = df_crime_month.groupby('LSOA_code')
    for LSOA_code, group in grouped:
        group['crime_diff'] = group['crime_count'].diff(periods=4)
        group['crime_diff'].fillna(method='backfill', inplace=True)

        ### Plot the graph comment out if not wanted ###
        plot_trend_season_residual(LSOA_code, group)

        group['month_index'] = group.index.month

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

        sarimax_forecast(SARIMAX_model, group, periods=4)
        


def sarimax_forecast(SARIMAX_model, prediction_df, periods):
    # Forecast
    n_periods = periods

    forecast_df = pd.DataFrame(
            {'month_index': pd.date_range(prediction_df.index[-1], periods=n_periods, freq='MS').month},
                    index=pd.date_range(prediction_df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')
                )

    fitted, confint = SARIMAX_model.predict(n_periods=n_periods,
                                            return_conf_int=True,
                                            exogenous=forecast_df[['month_index']])
    index_of_fc = pd.date_range(prediction_df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')

    ### Plot the graph comment out if not wanted ###
    forcast_plot(fitted, confint, index_of_fc, prediction_df)


def forcast_plot(fitted, confint, index_of_fc, prediction_df):
    # make series for plotting purpose
    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    fitted_series[fitted_series < 0] = 0
    lower_series[lower_series < 0] = 0
    upper_series[upper_series < 0] = 0
    # Plot
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


def fill_missing_months(df_crimes_deprivation):
    full_index = pd.date_range(start=df_crimes_deprivation.index.min(), end=df_crimes_deprivation.index.max(),
                               freq='MS')
    grouped = df_crimes_deprivation.groupby('LSOA_code')
    filled_dfs = []

    for LSOA_code, group in grouped:
        group = group.sort_index()
        group_reindexed = group.reindex(full_index)
        group_reindexed['LSOA_code'] = LSOA_code
        filled_dfs.append(group_reindexed)

    df_filled = pd.concat(filled_dfs)
    df_filled['crime_count'] = df_filled['crime_count'].fillna(0)

    return df_filled

prediction_network()
