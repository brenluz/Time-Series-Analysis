import pandas as pd


def calculate_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Root Mean Squared Error."""
    return (((actual - predicted) ** 2).mean()) ** 0.5


def calculate_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean Absolute Percentage Error (%)."""
    return (abs((actual - predicted) / actual)).mean() * 100