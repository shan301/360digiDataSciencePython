# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 16:03:10 2025

@author: Shantanu
"""

"""15. Time Series Analysis
This script introduces fundamental time series analysis concepts using pandas and Prophet. It covers time series data handling, decomposition, forecasting, and evaluation, using the `time_series.csv` dataset from the `data/` directory.

Prerequisites:
- Python 3.9+
- Libraries: pandas, numpy, matplotlib, seaborn, prophet
- Dataset: `time_series.csv` from the `data/` directory
"""

# 15.1. Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load sample dataset
ts_df = pd.read_csv('../data/time_series.csv')
print('Time Series Dataset Head:')
print(ts_df.head())

"""15.2. Introduction to Time Series
Time series data consists of observations collected over time, often with trends, seasonality, and noise. Key concepts:
- Trend: Long-term increase or decrease.
- Seasonality: Repeating patterns at fixed intervals.
- Noise: Random fluctuations.
This script focuses on time series analysis and forecasting with pandas and Prophet."""

"""15.3. Data Preparation
Convert date column to datetime and set as index for time series analysis."""
# Convert date column to datetime
ts_df['date'] = pd.to_datetime(ts_df['date'])
ts_df.set_index('date', inplace=True)
print('Dataset with Datetime Index:')
print(ts_df.head())

"""15.4. Time Series Visualization
Visualize the time series to identify trends and seasonality."""
# Plot time series
plt.figure(figsize=(10, 6))
plt.plot(ts_df['value'], label='Value')
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

"""15.5. Time Series Decomposition
Decompose the time series into trend, seasonal, and residual components."""
# Decompose time series (assuming monthly seasonality)
decomposition = seasonal_decompose(ts_df['value'], model='additive', period=12)
plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(ts_df['value'], label='Original')
plt.legend()
plt.subplot(412)
plt.plot(decomposition.trend, label='Trend')
plt.legend()
plt.subplot(413)
plt.plot(decomposition.seasonal, label='Seasonal')
plt.legend()
plt.subplot(414)
plt.plot(decomposition.resid, label='Residual')
plt.legend()
plt.tight_layout()
plt.show()

"""15.6. Stationarity Check
Stationarity is required for many time series models. Use Augmented Dickey-Fuller (ADF) test."""
# ADF test for stationarity
result = adfuller(ts_df['value'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

"""15.7. Differencing
Differencing removes trends to make the series stationary."""
# Apply first-order differencing
ts_diff = ts_df['value'].diff().dropna()
print('Differenced Series Head:')
print(ts_diff.head())

# ADF test on differenced series
result_diff = adfuller(ts_diff)
print('ADF Statistic (Differenced):', result_diff[0])
print('p-value (Differenced):', result_diff[1])

"""15.8. Forecasting with Prophet
Prophet is a forecasting tool for time series with trends and seasonality."""
# Prepare data for Prophet (requires 'ds' and 'y' columns)
prophet_df = ts_df.reset_index().rename(columns={'date': 'ds', 'value': 'y'})

# Initialize and fit Prophet model
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
model.fit(prophet_df)

# Create future dataframe for predictions (e.g., 30 days)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title('Prophet Forecast')
plt.show()

# Plot forecast components
model.plot_components(forecast)
plt.show()

"""15.9. Model Evaluation
Evaluate forecast accuracy using metrics like MAE and RMSE."""
# Evaluate on historical data
actual = prophet_df['y']
predicted = forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'yhat']
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

"""15. Time Series Analysis Exercises"""

"""Exercise 1: Load and Inspect Dataset
Load `time_series.csv` and print its first 5 rows and data types."""
def exercise_1():
    df = pd.read_csv('../data/time_series.csv')
    print('First 5 rows:')
    print(df.head())
    print('\nData Types:')
    print(df.dtypes)

exercise_1()

"""Exercise 2: Convert to Datetime
Convert the `date` column to datetime and set as index."""
def exercise_2():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    print('Dataset with Datetime Index:')
    print(df.head())

exercise_2()

"""Exercise 3: Plot Time Series
Plot the `value` column to visualize the time series."""
def exercise_3():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    plt.figure(figsize=(10, 6))
    plt.plot(df['value'], label='Value')
    plt.title('Time Series Plot')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

exercise_3()

"""Exercise 4: Resample Data
Resample the time series to monthly frequency, using the mean."""
def exercise_4():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    monthly_df = df.resample('ME').mean()
    print('Monthly Resampled Data:')
    print(monthly_df.head())

exercise_4()

"""Exercise 5: Rolling Mean
Compute and plot a 7-day rolling mean of the `value` column."""
def exercise_5():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    rolling_mean = df['value'].rolling(window=7).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(df['value'], label='Original')
    plt.plot(rolling_mean, label='7-Day Rolling Mean')
    plt.title('Rolling Mean')
    plt.legend()
    plt.show()

exercise_5()

"""Exercise 6: Seasonal Decomposition
Decompose the time series into trend, seasonal, and residual components (assume monthly seasonality)."""
def exercise_6():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    decomposition = seasonal_decompose(df['value'], model='additive', period=12)
    plt.figure(figsize=(10, 8))
    plt.subplot(411)
    plt.plot(df['value'], label='Original')
    plt.legend()
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend()
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonal')
    plt.legend()
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residual')
    plt.legend()
    plt.tight_layout()
    plt.show()

exercise_6()

"""Exercise 7: Stationarity Test
Perform an ADF test to check if the time series is stationary."""
def exercise_7():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    result = adfuller(df['value'])
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])

exercise_7()

"""Exercise 8: Differencing
Apply first-order differencing and plot the result."""
def exercise_8():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    diff = df['value'].diff().dropna()
    plt.figure(figsize=(10, 6))
    plt.plot(diff, label='Differenced Series')
    plt.title('First-Order Differencing')
    plt.legend()
    plt.show()

exercise_8()

"""Exercise 9: Autocorrelation Plot
Plot the autocorrelation of the `value` column to identify lags."""
def exercise_9():
    from pandas.plotting import autocorrelation_plot
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    plt.figure(figsize=(10, 6))
    autocorrelation_plot(df['value'])
    plt.title('Autocorrelation Plot')
    plt.show()

exercise_9()

"""Exercise 10: Prophet Forecast
Train a Prophet model and forecast 60 days into the future."""
def exercise_10():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    prophet_df = df.rename(columns={'date': 'ds', 'value': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=60)
    forecast = model.predict(future)
    model.plot(forecast)
    plt.title('Prophet Forecast (60 Days)')
    plt.show()

exercise_10()

"""Exercise 11: Evaluate Forecast
Compute MAE and RMSE for the Prophet model's historical predictions."""
def exercise_11():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    prophet_df = df.rename(columns={'date': 'ds', 'value': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    actual = prophet_df['y']
    predicted = forecast['yhat']
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    print(f'MAE: {mae:.2f}')
    print(f'RMSE: {rmse:.2f}')

exercise_11()

"""Exercise 12: Add Holiday Effects
Incorporate a holiday effect in the Prophet model for Christmas."""
def exercise_12():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    prophet_df = df.rename(columns={'date': 'ds', 'value': 'y'})
    holidays = pd.DataFrame({
        'holiday': 'Christmas',
        'ds': pd.to_datetime(['2023-12-25', '2024-12-25']),
        'lower_window': 0,
        'upper_window': 1
    })
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True, holidays=holidays)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    model.plot(forecast)
    plt.title('Prophet Forecast with Christmas Holiday')
    plt.show()

exercise_12()

"""Exercise 13: Weekly Seasonality
Extract and plot the weekly seasonality component from the Prophet model."""
def exercise_13():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    prophet_df = df.rename(columns={'date': 'ds', 'value': 'y'})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    weekly = forecast[['ds', 'weekly']].set_index('ds')
    plt.figure(figsize=(10, 6))
    plt.plot(weekly['weekly'])
    plt.title('Weekly Seasonality')
    plt.show()

exercise_13()

"""Exercise 14: Moving Average Forecast
Implement a simple moving average forecast for the next 30 days."""
def exercise_14():
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    ma = df['value'].rolling(window=7).mean().iloc[-1]
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30)
    forecast = pd.Series([ma] * 30, index=future_dates)
    plt.figure(figsize=(10, 6))
    plt.plot(df['value'], label='Original')
    plt.plot(forecast, label='Moving Average Forecast')
    plt.title('Moving Average Forecast')
    plt.legend()
    plt.show()

exercise_14()

"""Exercise 15: Lag Features
Create a lag feature (previous day's value) and use it in a linear regression model."""
def exercise_15():
    from sklearn.linear_model import LinearRegression
    df = pd.read_csv('../data/time_series.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df['lag_1'] = df['value'].shift(1)
    df.dropna(inplace=True)
    X = df[['lag_1']]
    y = df['value']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE with Lag Feature: {mse:.2f}')

exercise_15()