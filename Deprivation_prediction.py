from Dataset_maker import burglaries_LSOA, get_deprivation_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def deprivation_prediction():
    df_deprivation = get_deprivation_score()
    df_crimes_month = burglaries_LSOA()
    df_crimes_deprivation = pd.merge(df_crimes_month, df_deprivation, on='LSOA_code', how='inner')

    X = df_crimes_deprivation['deprivation'].values.reshape(-1, 1)
    y = df_crimes_deprivation['crime_count'].values

    regression = LinearRegression()
    regression.fit(X, y)

    ### Plot the graph comment out if not wanted ###
    plot_deprivation(X, y, regression)

def plot_deprivation(X, y, model):
    x_range = np.linspace(float(X.min()), float(X.max()), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)

    fig, ax = plt.subplots()
    ax.scatter(X, y)
    plt.plot(x_range, y_pred, color='red', label='Regression Line')
    plt.show()


deprivation_prediction()