from Data_loader import data_loader
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import matplotlib.pylab as plt
import pmdarima as pm
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import warnings
warnings.filterwarnings('ignore')


def burglaries_month_LSOA():
    """
    Get burglary data aggregated by LSOA and month
    """
    query = "SELECT * FROM crimes"
    df_crimes = data_loader(query)
    
    # Filter for burglary only
    burglary_df = df_crimes[df_crimes['Crime type'] == 'Burglary']
    
    # Convert Month to datetime
    burglary_df['Month'] = pd.to_datetime(burglary_df['Month'], infer_datetime_format=True)
    
    # Group by LSOA code and month, count crimes
    monthly_burglary = burglary_df.groupby(['LSOA code', 'Month']).size().reset_index(name='crime_count')
    
    return monthly_burglary


def get_deprivation_score():
    """
    Get deprivation score data
    """
    try:
        query = "SELECT * FROM deprivation"
        df_deprivation = data_loader(query)
        return df_deprivation
    except Exception as e:
        print(f"Failed to get deprivation score data: {e}")
        return None


def improved_prediction_network():
    """
    Improved prediction network using multiple time windows for training and validation
    """
    df_crime_month = burglaries_month_LSOA()
    df_crime_month['Month'] = pd.to_datetime(df_crime_month['Month'], infer_datetime_format=True)
    df_crime_month = df_crime_month.set_index(['Month'])
    df_crime_month = fill_missing_months(df_crime_month)

    # Define multiple training-testing time windows
    time_windows = define_time_windows()
    
    grouped = df_crime_month.groupby('LSOA code')
    
    print(f"Start processing {len(grouped)} LSOA regions")
    print(f"Using {len(time_windows)} time windows")
    print(datetime.now())
    
    results = Parallel(n_jobs=-1)(
        delayed(process_group_improved)(code, group, time_windows) 
        for code, group in tqdm(grouped, desc="Processing LSOA regions")
    )
    
    # Aggregate results
    aggregate_results(results, time_windows)
    print(datetime.now())


def define_time_windows():
    """
    Define multiple training-testing time windows
    """
    windows = []
    
    # Window 1: Use 2023 full year data to predict January 2024
    windows.append({
        'name': '2023_to_2024_01',
        'train_start': datetime(2023, 1, 1),
        'train_end': datetime(2023, 12, 1),
        'test_date': datetime(2024, 1, 1)
    })
    
    # Window 2: Use 2023 full year data to predict February 2024
    windows.append({
        'name': '2023_to_2024_02',
        'train_start': datetime(2023, 1, 1),
        'train_end': datetime(2023, 12, 1),
        'test_date': datetime(2024, 2, 1)
    })
    
    # Window 3: Use 2024 full year data to predict January 2025
    windows.append({
        'name': '2024_to_2025_01',
        'train_start': datetime(2024, 1, 1),
        'train_end': datetime(2024, 12, 1),
        'test_date': datetime(2025, 1, 1)
    })
    
    # Window 4: Use 2024 full year data to predict February 2025
    windows.append({
        'name': '2024_to_2025_02',
        'train_start': datetime(2024, 1, 1),
        'train_end': datetime(2024, 12, 1),
        'test_date': datetime(2025, 2, 1)
    })
    
    # Window 5: Rolling window - Use recent 24 months to predict next month
    windows.append({
        'name': 'rolling_24_months',
        'train_months': 24,
        'test_date': datetime(2025, 2, 1)
    })
    
    return windows


def process_group_improved(LSOA_code, group, time_windows):
    """
    Improved processing function for individual LSOA region using multiple time windows and parameter optimization
    """
    group = group.sort_index()
    group['crime_diff'] = group['crime_count'].diff(periods=4)
    group['crime_diff'].fillna(method='backfill', inplace=True)
    group['month_index'] = group.index.month
    
    # Store results for each time window
    window_results = {}
    all_predictions = []
    all_actuals = []
    
    for window in time_windows:
        try:
            # Split data according to time window
            if 'rolling' in window['name']:
                # Rolling window
                test_date = window['test_date']
                train_start = test_date - relativedelta(months=window['train_months'])
                train_data = group[(group.index >= train_start) & (group.index < test_date)]
                test_data = group[group.index == test_date]
            else:
                # Fixed time window
                train_data = group[(group.index >= window['train_start']) & 
                                 (group.index <= window['train_end'])]
                test_data = group[group.index == window['test_date']]
            
            if len(train_data) < 12 or len(test_data) == 0:  # Need at least 12 months of data
                continue
                
            # Find optimal parameters
            best_model, best_params = find_optimal_parameters(train_data)
            
            if best_model is not None:
                # Make prediction
                fitted, lower, upper = sarimax_forecast(best_model, train_data, periods=1)
                
                prediction = fitted.values[0]
                actual = test_data['crime_count'].values[0]
                
                all_predictions.append(prediction)
                all_actuals.append(actual)
                
                # Calculate metrics for this window
                window_metrics = calculate_detailed_metrics(prediction, actual, lower.values[0], upper.values[0])
                window_metrics['best_params'] = best_params
                window_results[window['name']] = window_metrics
                
        except Exception as e:
            print(f"LSOA {LSOA_code}, window {window['name']} processing failed: {str(e)}")
            continue
    
    # Calculate overall metrics
    overall_metrics = {}
    if all_predictions:
        overall_metrics = {
            'LSOA_code': LSOA_code,
            'total_windows': len(all_predictions),
            'mean_mae': mean_absolute_error(all_actuals, all_predictions),
            'mean_rmse': np.sqrt(mean_squared_error(all_actuals, all_predictions)),
            'mean_accuracy': np.mean([1 if round(p) == a else 0 for p, a in zip(all_predictions, all_actuals)]),
            'window_results': window_results
        }
    
    return overall_metrics


def find_optimal_parameters(train_data):
    """
    Find optimal SARIMAX parameters through grid search
    """
    # Define parameter search space
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]
    m = 12  # Seasonal period
    
    best_aic = float('inf')
    best_model = None
    best_params = None
    
    # First try auto_arima (fast method)
    try:
        auto_model = pm.auto_arima(
            train_data[['crime_count']], 
            exogenous=train_data[['month_index']],
            start_p=0, start_q=0,
            max_p=2, max_q=2, 
            seasonal=True,
            m=12,
            start_P=0, start_Q=0,
            max_P=1, max_Q=1,
            d=None, D=None,
            trace=False, 
            error_action='ignore',
            suppress_warnings=True, 
            stepwise=True,
            n_fits=50
        )
        
        if auto_model.aic() < best_aic:
            best_aic = auto_model.aic()
            best_model = auto_model
            best_params = auto_model.order + auto_model.seasonal_order
            
    except Exception as e:
        pass
    
    # If auto_arima fails or performs poorly, conduct limited grid search
    if best_model is None or best_aic > 1000:  # AIC threshold
        param_combinations = [
            ((1,1,1), (0,1,1,12)),
            ((1,0,1), (1,1,1,12)),
            ((2,1,2), (1,1,1,12)),
            ((1,1,0), (0,1,1,12)),
            ((0,1,1), (0,1,1,12))
        ]
        
        for (p,d,q), (P,D,Q,s) in param_combinations:
            try:
                model = pm.ARIMA(
                    order=(p,d,q),
                    seasonal_order=(P,D,Q,s),
                    suppress_warnings=True
                )
                model.fit(train_data[['crime_count']], exogenous=train_data[['month_index']])
                
                if model.aic() < best_aic:
                    best_aic = model.aic()
                    best_model = model
                    best_params = (p,d,q,P,D,Q,s)
                    
            except Exception as e:
                continue
    
    return best_model, best_params


def calculate_detailed_metrics(prediction, actual, lower, upper):
    """
    Calculate detailed evaluation metrics
    """
    pred_rounded = round(prediction)
    
    metrics = {
        'prediction': prediction,
        'actual': actual,
        'lower_ci': lower,
        'upper_ci': upper,
        'absolute_error': abs(prediction - actual),
        'squared_error': (prediction - actual) ** 2,
        'percentage_error': abs(prediction - actual) / max(actual, 1) * 100,
        'pred_exact': int(pred_rounded == actual),
        'pred_wrong_high': int(pred_rounded > actual),
        'pred_wrong_low': int(pred_rounded < actual),
        'in_confidence_interval': int(lower <= actual <= upper),
        'out_range_low': int(actual < lower),
        'out_range_high': int(actual > upper)
    }
    
    return metrics


def aggregate_results(results, time_windows):
    """
    Aggregate and analyze all results
    """
    # Filter out empty results
    valid_results = [r for r in results if r and 'total_windows' in r]
    
    if not valid_results:
        print("No valid prediction results")
        return
    
    print(f"\n=== Overall Prediction Performance Analysis ===")
    print(f"Successfully processed LSOA regions: {len(valid_results)}")
    
    # Calculate overall metrics
    overall_mae = np.mean([r['mean_mae'] for r in valid_results])
    overall_rmse = np.mean([r['mean_rmse'] for r in valid_results])
    overall_accuracy = np.mean([r['mean_accuracy'] for r in valid_results])
    
    print(f"Mean Absolute Error (MAE): {overall_mae:.4f}")
    print(f"Root Mean Square Error (RMSE): {overall_rmse:.4f}")
    print(f"Prediction Accuracy: {overall_accuracy:.4f}")
    
    # Analysis by time window
    print(f"\n=== Performance Analysis by Time Window ===")
    
    for window in time_windows:
        window_name = window['name']
        window_metrics = []
        
        for result in valid_results:
            if window_name in result['window_results']:
                window_metrics.append(result['window_results'][window_name])
        
        if window_metrics:
            mae = np.mean([m['absolute_error'] for m in window_metrics])
            accuracy = np.mean([m['pred_exact'] for m in window_metrics])
            ci_coverage = np.mean([m['in_confidence_interval'] for m in window_metrics])
            
            print(f"{window_name}:")
            print(f"  - Sample count: {len(window_metrics)}")
            print(f"  - Mean absolute error: {mae:.4f}")
            print(f"  - Prediction accuracy: {accuracy:.4f}")
            print(f"  - Confidence interval coverage: {ci_coverage:.4f}")
    
    # Parameter usage frequency analysis
    print(f"\n=== Optimal Parameter Usage Frequency ===")
    param_counter = {}
    
    for result in valid_results:
        for window_result in result['window_results'].values():
            if 'best_params' in window_result:
                params = str(window_result['best_params'])
                param_counter[params] = param_counter.get(params, 0) + 1
    
    # Show top 5 most used parameter combinations
    sorted_params = sorted(param_counter.items(), key=lambda x: x[1], reverse=True)[:5]
    for params, count in sorted_params:
        print(f"  {params}: {count} times")


def sarimax_forecast(SARIMAX_model, prediction_df, periods):
    """
    Improved SARIMAX forecasting function
    """
    n_periods = periods

    forecast_df = pd.DataFrame(
        {'month_index': pd.date_range(prediction_df.index[-1], periods=n_periods, freq='MS').month},
        index=pd.date_range(prediction_df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')
    )

    fitted, confint = SARIMAX_model.predict(n_periods=n_periods,
                                            return_conf_int=True,
                                            exogenous=forecast_df[['month_index']])
    index_of_fc = pd.date_range(prediction_df.index[-1] + pd.DateOffset(months=1), periods=n_periods, freq='MS')

    fitted_series = pd.Series(fitted, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Ensure predictions are non-negative
    fitted_series[fitted_series < 0] = 0
    lower_series[lower_series < 0] = 0
    upper_series[upper_series < 0] = 0

    return fitted_series, lower_series, upper_series


def fill_missing_months(df_crimes_deprivation):
    """
    Fill missing months in the data
    """
    full_index = pd.date_range(start=df_crimes_deprivation.index.min(), 
                               end=df_crimes_deprivation.index.max(),
                               freq='MS')
    grouped = df_crimes_deprivation.groupby('LSOA code')
    filled_dfs = []

    for LSOA_code, group in grouped:
        group = group.sort_index()
        group_reindexed = group.reindex(full_index)
        group_reindexed['LSOA code'] = LSOA_code
        filled_dfs.append(group_reindexed)

    df_filled = pd.concat(filled_dfs)
    df_filled['crime_count'] = df_filled['crime_count'].fillna(0)

    return df_filled


# Run the improved prediction network
if __name__ == "__main__":
    improved_prediction_network()