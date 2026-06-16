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
import time

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
    (results, timings) : tuple
        results : dict model_name -> list[float] of length test_periods.
                  Each value is |actual_k - predicted_k| at forecast step k.
                  The calling function squares and averages these for RMSE.
        timings : dict model_name -> float, wall-clock seconds spent fitting +
                  forecasting that model on this window. Empty on a cache hit
                  (no model is actually run), so cached windows do not distort
                  the average runtime reported per model.
    """
    device_mode = os.getenv("ML_DEVICE", "auto").strip().lower()
    cpu_cap = os.getenv("ML_MAX_CPU_CORES", "").strip()
    key_data = np.concatenate([train_values, test_values])
    key_bytes = (
        f"{state}_{key_data.tobytes()}_{test_periods}_{device_mode}_{cpu_cap}"
    ).encode()
    cache_key  = hashlib.md5(key_bytes).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        # Guard against stale/failed runs that cached an empty dict:
        # those produce plots with no box traces at all.
        if isinstance(cached, dict) and len(cached) > 0:
            return cached, {}

    train_series = pd.Series(train_values, index=train_index)
    test_series  = pd.Series(test_values,  index=test_index)

    results = {}
    timings = {}
    for model_name, model_func in MODEL_FUNCS.items():
        try:
            t0 = time.perf_counter()
            forecast, _ = model_func(train_series, n_periods=test_periods)
            elapsed = time.perf_counter() - t0
            if forecast is None or len(forecast) != len(test_series):
                continue
            aligned = pd.Series(forecast.values, index=test_series.index)
            results[model_name] = np.abs(
                test_series.values - aligned.values
            ).tolist()
            timings[model_name] = elapsed
        except Exception:
            pass

    # Do not persist empty results; they are usually transient failures and
    # would otherwise poison future runs for the same window key.
    if results:
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)

    return results, timings


def _worker(args: tuple) -> tuple:
    """Top-level unpacker for ProcessPoolExecutor — must not be a lambda or closure.

    Returns (results, timings); see _compute_window_errors.
    """
    return _compute_window_errors(*args)