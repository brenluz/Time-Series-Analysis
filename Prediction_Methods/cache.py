"""
cache.py
--------
Per-window error computation with MD5-based disk caching.

_compute_window_errors and _worker MUST remain top-level functions in this
module so ProcessPoolExecutor can pickle them by reference.
"""

import hashlib
import os
import pickle

import numpy as np
import pandas as pd

from forecast_methods import MODEL_FUNCS


def _compute_window_errors(
    state: str,
    train_values: np.ndarray,
    train_index,
    test_values: np.ndarray,
    test_index,
    test_periods: int,
    cache_dir: str,
) -> dict:
    """
    Run all models for one (state, window) pair and return per-step absolute errors.

    The cache key is an MD5 hash of (state, train data, test data, test_periods).
    This means:
    - Renaming or restructuring files does NOT invalidate the cache.
    - Changing model logic DOES require clearing the cache manually.

    Returns
    -------
    dict : model_name -> list[float] of length test_periods
           Each value is |actual_k - predicted_k| at forecast step k.
           The calling function squares and averages these to compute RMSE.
    """
    key_data  = np.concatenate([train_values, test_values])
    key_bytes = f"{state}_{key_data.tobytes()}_{test_periods}".encode()
    cache_key  = hashlib.md5(key_bytes).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    train_series = pd.Series(train_values, index=train_index)
    test_series  = pd.Series(test_values,  index=test_index)

    results = {}
    for model_name, model_func in MODEL_FUNCS.items():
        try:
            forecast, _ = model_func(train_series, n_periods=test_periods)
            if forecast is None or len(forecast) != len(test_series):
                continue
            aligned = pd.Series(forecast.values, index=test_series.index)
            results[model_name] = np.abs(
                test_series.values - aligned.values
            ).tolist()
        except Exception:
            pass

    with open(cache_path, 'wb') as f:
        pickle.dump(results, f)

    return results


def _worker(args: tuple) -> dict:
    """Top-level unpacker for ProcessPoolExecutor — must not be a lambda or closure."""
    return _compute_window_errors(*args)