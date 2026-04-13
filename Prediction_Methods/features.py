import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame, lag_start: int = 1,
                    lag_end: int = 12, rolling_window: int = 3) -> pd.DataFrame:
    """
    Build ML features from a time series DataFrame with columns ['ds', 'y'].
    Adds calendar features, lagged values, and rolling statistics.
    """
    df_features = df.copy()
    df_features['ds'] = pd.to_datetime(df_features['ds'])

    df_features['year']      = df_features['ds'].dt.year
    df_features['month']     = df_features['ds'].dt.month
    df_features['dayofweek'] = df_features['ds'].dt.dayofweek
    df_features['dayofyear'] = df_features['ds'].dt.dayofyear

    for lag in range(lag_start, lag_end + 1):
        df_features[f'lag_{lag}'] = df_features['y'].shift(lag)

    df_features['rolling_mean'] = (
        df_features['y'].shift(lag_start).rolling(window=rolling_window).mean()
    )
    df_features['rolling_std'] = (
        df_features['y'].shift(lag_start).rolling(window=rolling_window).std()
    )

    return df_features.dropna()


def series_to_sequences(series: np.ndarray, n_steps: int):
    """
    Convert a 1-D array into supervised learning sequences.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_steps)
    y : np.ndarray of shape (n_samples,)
    """
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X), np.array(y)