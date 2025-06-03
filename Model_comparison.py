import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pmdarima as pm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CrimePredictionModelComparison:
    """
    Crime prediction model comparison
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def prepare_data_for_ml(self, df):
        """
        Preparing feature data for models
        """
        # different time features
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['is_summer'] = df['month'].isin([6,7,8]).astype(int)
        df['is_winter'] = df['month'].isin([12,1,2]).astype(int)
        
        for lag in [1, 2, 3, 6, 12]:
            df[f'crime_lag_{lag}'] = df['crime_count'].shift(lag)
        
        for window in [3, 6, 12]:
            df[f'crime_mean_{window}'] = df['crime_count'].rolling(window).mean()
            df[f'crime_std_{window}'] = df['crime_count'].rolling(window).std()
        
        df['crime_diff'] = df['crime_count'].diff()
        df['crime_pct_change'] = df['crime_count'].pct_change()
        
        return df.fillna(method='bfill').fillna(0)
    
    def sarimax_model(self, train_data, test_data, exog_vars=None):
        """
        SARIMAX model
        """
        try:
            if exog_vars is None:
                exog_vars = ['month']
            
            available_exog = [var for var in exog_vars if var in train_data.columns]
            
            if available_exog:
                model = pm.auto_arima(
                    train_data['crime_count'],
                    exogenous=train_data[available_exog],
                    seasonal=True,
                    m=12,
                    suppress_warnings=True,
                    stepwise=True,
                    max_p=2, max_q=2, max_P=1, max_Q=1
                )
                
                forecast, conf_int = model.predict(
                    n_periods=len(test_data),
                    exogenous=test_data[available_exog],
                    return_conf_int=True
                )
                
                return forecast, conf_int, model
            else:
                model = pm.auto_arima(
                    train_data['crime_count'],
                    seasonal=True,
                    m=12,
                    suppress_warnings=True,
                    stepwise=True
                )
                
                forecast, conf_int = model.predict(
                    n_periods=len(test_data),
                    return_conf_int=True
                )
                
                return forecast, conf_int, model
                
        except Exception as e:
            print(f"SARIMAX fail {e}")
            return None, None, None
    
    def exponential_smoothing_model(self, train_data, test_data):
        """
        Exponential Smoothing Model
        """
        try:
            model = ExponentialSmoothing(
                train_data['crime_count'],
                seasonal='add',
                seasonal_periods=12,
                trend='add'
            ).fit()
            
            forecast = model.forecast(len(test_data))
            
            residuals = model.resid
            std_residuals = np.std(residuals)
            conf_int = np.column_stack([
                forecast - 1.96 * std_residuals,
                forecast + 1.96 * std_residuals
            ])
            
            return forecast, conf_int, model
            
        except Exception as e:
            print(f"Expo fails: {e}")
            return None, None, None
    
    def random_forest_model(self, train_data, test_data):
        """
        Random Forest Model Prediction
        """
        try:
            feature_cols = [col for col in train_data.columns 
                          if col not in ['crime_count', 'LSOA_code']]
            
            X_train = train_data[feature_cols]
            y_train = train_data['crime_count']
            X_test = test_data[feature_cols]
            
            # train the model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            forecast = model.predict(X_test)
            
            predictions = []
            for tree in model.estimators_:
                pred = tree.predict(X_test)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            conf_int = np.column_stack([
                np.percentile(predictions, 2.5, axis=0),
                np.percentile(predictions, 97.5, axis=0)
            ])
            
            return forecast, conf_int, model
            
        except Exception as e:
            print(f"Ramdom forest fails: {e}")
            return None, None, None
    
    def xgboost_model(self, train_data, test_data):
        """
        XGBoost model
        """
        try:
            feature_cols = [col for col in train_data.columns 
                          if col not in ['crime_count', 'LSOA_code']]
            
            X_train = train_data[feature_cols]
            y_train = train_data['crime_count']
            X_test = test_data[feature_cols]
            
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            forecast = model.predict(X_test)
            
            train_pred = model.predict(X_train)
            residuals = y_train - train_pred
            std_residuals = np.std(residuals)
            
            conf_int = np.column_stack([
                forecast - 1.96 * std_residuals,
                forecast + 1.96 * std_residuals
            ])
            
            return forecast, conf_int, model
            
        except Exception as e:
            print(f"XGBoost fails: {e}")
            return None, None, None
    
    def evaluate_model(self, actual, predicted, conf_int, model_name):
        """
        evaluate each models
        """
        # basic stastics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        
        exact_predictions = np.sum(np.round(predicted) == actual) / len(actual)
        
        within_ci = np.sum((actual >= conf_int[:, 0]) & (actual <= conf_int[:, 1])) / len(actual)
        
        avg_ci_width = np.mean(conf_int[:, 1] - conf_int[:, 0])
        
        if len(actual) > 1:
            actual_direction = np.diff(actual) > 0
            pred_direction = np.diff(predicted) > 0
            direction_accuracy = np.sum(actual_direction == pred_direction) / len(actual_direction)
        else:
            direction_accuracy = 0
        
        return {
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'exact_predictions': exact_predictions,
            'ci_coverage': within_ci,
            'avg_ci_width': avg_ci_width,
            'direction_accuracy': direction_accuracy
        }
    
    def run_comparison_experiment(self, df, lsoa_codes=None, test_months=3):
        """
        run the models to compare
        """
        if lsoa_codes is None:
            lsoa_codes = df['LSOA_code'].unique()[:1000]  # only choose the first 1000 LSOA
        
        all_results = []
        
        for lsoa_code in lsoa_codes:
            
            lsoa_data = df[df['LSOA_code'] == lsoa_code].sort_index()
            
            if len(lsoa_data) < 24:  # need enough data
                continue
            
            lsoa_data_ml = self.prepare_data_for_ml(lsoa_data.copy())
            
            split_idx = len(lsoa_data) - test_months
            train_data = lsoa_data_ml.iloc[:split_idx]
            test_data = lsoa_data_ml.iloc[split_idx:]
            
            if len(train_data) < 12 or len(test_data) == 0:
                continue
            
            actual_values = test_data['crime_count'].values
            
            models_to_test = {
                'SARIMAX': self.sarimax_model,
                'Exponential_Smoothing': self.exponential_smoothing_model,
                'Random_Forest': self.random_forest_model,
                'XGBoost': self.xgboost_model
            }
            
            for model_name, model_func in models_to_test.items():
                try:
                    if model_name == 'SARIMAX':
                        forecast, conf_int, model = model_func(train_data, test_data, ['month'])
                    else:
                        forecast, conf_int, model = model_func(train_data, test_data)
                    
                    if forecast is not None:
                        # Ensure the predicted value is non-negative
                        forecast = np.maximum(forecast, 0)
                        
                        # mesure models
                        result = self.evaluate_model(actual_values, forecast, conf_int, model_name)
                        result['lsoa_code'] = lsoa_code
                        all_results.append(result)
                    
                except Exception as e:
                    print(f" {model_name} failed on LSOA {lsoa_code} : {e}")
                    continue
        
        results_df = pd.DataFrame(all_results)
        
        if not results_df.empty:
            self.results = results_df
            self.generate_comparison_report()
            self.visualize_results()
        else:
            print("all fails")
        
        return results_df
    
    def generate_comparison_report(self):
        """
        give the outcmes
        """
        if self.results.empty:
            print("no result")
            return
        
        print("\n" + "="*60)
        print("model comparison outcomes")
        print("="*60)
        
        summary = self.results.groupby('model').agg({
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'r2': ['mean', 'std'],
            'exact_predictions': ['mean', 'std'],
            'ci_coverage': ['mean', 'std'],
            'avg_ci_width': ['mean', 'std'],
            'direction_accuracy': ['mean', 'std']
        }).round(4)
        
        print("\n Overall performance index comparison:")
        print("-" * 60)
        
        for model in summary.index:
            print(f"\n {model}:")
            print(f"  MAE:     {summary.loc[model, ('mae', 'mean')]:.4f} ± {summary.loc[model, ('mae', 'std')]:.4f}")
            print(f"  RMSE:      {summary.loc[model, ('rmse', 'mean')]:.4f} ± {summary.loc[model, ('rmse', 'std')]:.4f}")
            print(f"  R²:            {summary.loc[model, ('r2', 'mean')]:.4f} ± {summary.loc[model, ('r2', 'std')]:.4f}")
            print(f"  Precision prediction rate:            {summary.loc[model, ('exact_predictions', 'mean')]:.4f} ± {summary.loc[model, ('exact_predictions', 'std')]:.4f}")
            print(f"  Confidence interval coverage:        {summary.loc[model, ('ci_coverage', 'mean')]:.4f} ± {summary.loc[model, ('ci_coverage', 'std')]:.4f}")
            print(f"  Average confidence interval width:      {summary.loc[model, ('avg_ci_width', 'mean')]:.4f} ± {summary.loc[model, ('avg_ci_width', 'std')]:.4f}")
            print(f"  Direction prediction accuracy:        {summary.loc[model, ('direction_accuracy', 'mean')]:.4f} ± {summary.loc[model, ('direction_accuracy', 'std')]:.4f}")
        

# exe fucntion
def run_comprehensive_model_comparison(df):
    
    comparator = CrimePredictionModelComparison()
    
    results = comparator.run_comparison_experiment(df, test_months=3)
    
    return results, comparator

