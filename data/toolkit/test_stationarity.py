
import pandas as pd
import numpy as np


def return_trend_variance_non_stationary_series():
    # Set a random seed for reproducibility
    np.random.seed(0)

    # Define the number of time points and the time interval
    n = 100
    time_interval = 1

    # Create a time index
    time_index = pd.date_range(start='2023-01-01', periods=n, freq=f'{time_interval}D')

    # Generate a time series with a linear trend and increasing variance
    trend = 0.1 * np.arange(n)  # Trend component

    # Generate noise with increasing variance
    noise_variance = np.linspace(0.1, 2.0, n)  # Increasing variance
    noise = np.random.normal(0, noise_variance, n)

    # Combine the trend and noise to create the time series
    time_series = trend + noise

    # Create a Pandas DataFrame
    df = pd.DataFrame({'Value': time_series}, index=time_index)

    return df


def return_trend_seasonal_and_variance_nstationary_series():
    # Set a random seed for reproducibility
    np.random.seed(0)

    # Define the number of time points, the time interval, and the number of seasons
    n = 100
    time_interval = 1
    num_seasons = 52  # Four seasons for simplicity

    # Create a time index
    time_index = pd.date_range(start='2023-01-01', periods=n, freq=f'{time_interval}D')

    # Generate a time series with a linear trend and seasonal pattern
    trend = 0.1 * np.arange(n)  # Trend component

    # Generate noise with increasing variance
    noise_variance = np.linspace(0.1, 2.0, n)  # Increasing variance
    noise = np.abs(np.random.normal(0, noise_variance, n))  # Positive noise

    # Generate a seasonal pattern with increased amplitude
    seasonal_amplitude = 10  # Increased amplitude for more prominent seasonality
    seasonal_pattern = seasonal_amplitude * np.abs(np.sin(2 * np.pi * np.arange(n) / num_seasons + np.pi / 2))  # Shift phase to make it strictly positive

    # Combine the trend, seasonal pattern, and noise to create the time series
    time_series = trend + seasonal_pattern + noise

    # Create a Pandas DataFrame
    df = pd.DataFrame({'Value': time_series}, index=time_index)

    return df

def sample_real_dataset_test(csv_filepath: str):
    real_dataset = pd.read_csv(csv_filepath, index_col=0)
    real_dataset = real_dataset.fillna(0)
    return real_dataset