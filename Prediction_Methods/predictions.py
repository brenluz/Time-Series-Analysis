import hashlib
import math
import os
import pickle
import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pmdarima as pm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_rmse(actual, predicted):
    return ((( actual - predicted) ** 2).mean()) ** 0.5

def calculate_mape(actual, predicted):
    return (abs((actual - predicted) / actual)).mean() * 100


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def create_features(df, lag_start=1, lag_end=12, rolling_window=3):
    df_features = df.copy()
    df_features['ds'] = pd.to_datetime(df_features['ds'])
    df_features['year'] = df_features['ds'].dt.year
    df_features['month'] = df_features['ds'].dt.month
    df_features['dayofweek'] = df_features['ds'].dt.dayofweek
    df_features['dayofyear'] = df_features['ds'].dt.dayofyear
    for lag in range(lag_start, lag_end + 1):
        df_features[f'lag_{lag}'] = df_features['y'].shift(lag)
    df_features['rolling_mean'] = df_features['y'].shift(lag_start).rolling(window=rolling_window).mean()
    df_features['rolling_std'] = df_features['y'].shift(lag_start).rolling(window=rolling_window).std()
    return df_features.dropna()


def series_to_sequences(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Models (batch_first=True for efficiency)
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :]).squeeze(1)


class GRUmodel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_layer_size, num_layers=2,
                          batch_first=True, dropout=0.1)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :]).squeeze(1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:x.size(0), :].unsqueeze(1)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_linear = nn.Linear(d_model, 1)
        for m in [self.input_linear, self.output_linear]:
            m.weight.data.uniform_(-0.1, 0.1)
            m.bias.data.zero_()

    def forward(self, src):
        src = self.input_linear(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return self.output_linear(output[-1, :, :]).squeeze(1)


# ---------------------------------------------------------------------------
# Shared batch trainer (no epoch reduction, early stopping instead)
# ---------------------------------------------------------------------------

def _train_batch(model, X_tensor, y_tensor, epochs, lr=0.001, patience=30, clip=1.0):
    """
    Full-batch training with early stopping and best-weight restoration.
    Saves the best model state and restores it at the end, so the returned
    model is the one with lowest training loss — not the last epoch.
    """
    import copy
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience // 2, min_lr=1e-6
    )
    loss_fn = nn.MSELoss()
    best_loss   = float('inf')
    best_state  = copy.deepcopy(model.state_dict())
    counter     = 0
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_tensor), y_tensor)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step(loss.item())
        if loss.item() < best_loss - 1e-6:
            best_loss  = loss.item()
            best_state = copy.deepcopy(model.state_dict())
            counter    = 0
        else:
            counter += 1
            if counter >= patience:
                break
    # Restore best weights
    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Forecast functions (same interface as original)
# ---------------------------------------------------------------------------

def auto_arima_forecast(series, seasonal=True, m=12, n_periods=12):
    model = pm.auto_arima(
        series.dropna(), d=1, D=1, start_p=1, start_q=1, max_p=5, max_q=5,
        seasonal=True, m=m, error_action='ignore', suppress_warnings=True, stepwise=True
    )
    pred = model.predict(n_periods=n_periods)
    idx = pd.date_range(start=series.index[-1], periods=n_periods + 1, freq='MS')[1:]
    return pd.Series(pred, index=idx), model


def ets_forecast(series, m=12, n_periods=12, trend='add', seasonal_model='add'):
    clean = series.dropna()
    try:
        fit = ExponentialSmoothing(
            clean, trend=trend, seasonal=seasonal_model,
            seasonal_periods=m, initialization_method='estimated'
        ).fit()
    except Exception:
        return pd.Series(dtype='float64'), None
    idx = pd.date_range(start=clean.index[-1], periods=n_periods + 1, freq='MS')[1:]
    return pd.Series(fit.forecast(n_periods), index=idx), fit


def prophet_forecast(series, seasonal=True, m=12, n_periods=12):
    df_p = series.reset_index().dropna()
    df_p.columns = ['ds', 'y']
    model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
    if seasonal and m and m > 1:
        model.add_seasonality('custom_seasonal', period=m, fourier_order=5, mode='multiplicative')
    model.fit(df_p)
    future = model.make_future_dataframe(periods=n_periods, freq='MS')
    fc = model.predict(future).set_index('ds')['yhat'][-n_periods:]
    fc.name = series.name
    return fc, model


def random_forest(series, n_periods=12):
    df_train = series.reset_index()
    df_train.columns = ['ds', 'y']
    df_features = create_features(df_train, lag_start=1, lag_end=12, rolling_window=6).dropna(subset=['y'])
    X_cols = [c for c in df_features.columns if c not in ['ds', 'y']]
    model = RandomForestRegressor(n_estimators=100, random_state=25, n_jobs=-1)
    model.fit(df_features[X_cols], df_features['y'])

    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=n_periods + 1)[1:]
    df_full = pd.concat([df_train, pd.DataFrame({'ds': future_dates})], ignore_index=True)
    df_full['ds'] = pd.to_datetime(df_full['ds'])

    predictions = []
    for i in range(len(df_train), len(df_full)):
        df_iter = create_features(df_full[:i], lag_end=12, rolling_window=6)
        X_pred = df_iter.iloc[-1][X_cols].to_frame().T
        next_pred = model.predict(X_pred)[0]
        predictions.append(next_pred)
        df_full.loc[i, 'y'] = next_pred

    return pd.Series(predictions, index=future_dates, name='RF'), model


def lstm_forecast(series, n_periods=12):
    if len(series.dropna()) < n_periods + 2:
        return pd.Series(dtype='float64'), None
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y = series_to_sequences(scaled, n_periods)
    if X.shape[0] == 0:
        return pd.Series(dtype='float64'), None

    # [samples, seq_len, features=1]
    X_t = torch.FloatTensor(X).unsqueeze(2)
    y_t = torch.FloatTensor(y)

    model = LSTMModel()
    model = _train_batch(model, X_t, y_t, epochs=300, patience=30)

    model.eval()
    cur = scaled[-n_periods:].copy()
    preds = []
    with torch.no_grad():
        for _ in range(n_periods):
            inp = torch.FloatTensor(cur).unsqueeze(0).unsqueeze(2)  # [1, seq, 1]
            p = model(inp).item()
            preds.append(p)
            cur = np.roll(cur, -1)
            cur[-1] = p

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    idx = pd.date_range(start=series.index[-1], periods=n_periods + 1, freq='MS')[1:]
    return pd.Series(preds, index=idx, name='LSTM'), model


def gru_forecast(series, n_periods=12):
    if len(series.dropna()) < n_periods + 2:
        return pd.Series(dtype='float64'), None
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y = series_to_sequences(scaled, n_periods)
    if X.shape[0] == 0:
        return pd.Series(dtype='float64'), None

    X_t = torch.FloatTensor(X).unsqueeze(2)
    y_t = torch.FloatTensor(y)

    model = GRUmodel()
    model = _train_batch(model, X_t, y_t, epochs=300, patience=30)

    model.eval()
    cur = scaled[-n_periods:].copy()
    preds = []
    with torch.no_grad():
        for _ in range(n_periods):
            inp = torch.FloatTensor(cur).unsqueeze(0).unsqueeze(2)
            p = model(inp).item()
            preds.append(p)
            cur = np.roll(cur, -1)
            cur[-1] = p

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    idx = pd.date_range(start=series.index[-1], periods=n_periods + 1, freq='MS')[1:]
    return pd.Series(preds, index=idx, name='GRU'), model


def conv_transformer_forecast(series, n_periods=12):
    if len(series.dropna()) < n_periods + 1:
        return pd.Series(dtype='float64'), None
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y = series_to_sequences(scaled, n_periods)

    # shape for transformer: [seq_len, batch, 1]
    X_t = torch.FloatTensor(X).transpose(0, 1).unsqueeze(2)  # [seq, samples, 1]
    y_t = torch.FloatTensor(y)

    model = TimeSeriesTransformer(input_dim=1, d_model=64, nhead=8, num_layers=2, dropout=0.1)
    model = _train_batch(model, X_t, y_t, epochs=500, lr=0.001, patience=40)

    model.eval()
    cur = scaled[-n_periods:].copy()
    preds = []
    with torch.no_grad():
        for _ in range(n_periods):
            inp = torch.FloatTensor(cur).view(n_periods, 1, 1)
            p = model(inp).item()
            preds.append(p)
            cur = np.roll(cur, -1)
            cur[-1] = p

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    idx = pd.date_range(start=series.index[-1], periods=n_periods + 1, freq='MS')[1:]
    return pd.Series(preds, index=idx, name='Transformer'), model


# ---------------------------------------------------------------------------
# Wrappers with consistent signature
# ---------------------------------------------------------------------------

def _model_forecast_arima(series, n_periods=12):
    return auto_arima_forecast(series, seasonal=True, m=12, n_periods=n_periods)

def _model_forecast_ets(series, n_periods=12):
    return ets_forecast(series, n_periods=n_periods, m=12, trend='add', seasonal_model='add')

def _model_forecast_prophet(series, n_periods=12):
    return prophet_forecast(series, seasonal=True, m=12, n_periods=n_periods)


MODEL_FUNCS = {
    "ARIMA":         _model_forecast_arima,
    "ETS":           _model_forecast_ets,
    "Prophet":       _model_forecast_prophet,
    "Random Forest": random_forest,
    "LSTM":          lstm_forecast,
    "GRU":           gru_forecast,
    "Transformer":   conv_transformer_forecast,
}


# ---------------------------------------------------------------------------
# Disk-cached single window computation (top-level so it's picklable)
# ---------------------------------------------------------------------------

def _compute_window_errors(
    state: str,
    train_values: np.ndarray,
    train_index,
    test_values: np.ndarray,
    test_index,
    test_periods: int,
    cache_dir: str,
):
    """
    Runs all models for one (state, window) pair and returns per-step RMSE.

    For each model we produce one RMSE scalar per forecast step:
        rmse_at_step_k = sqrt( (actual_k - predicted_k)^2 )
                       = |actual_k - predicted_k|
    Because there is only one observation per step per window, the RMSE at
    step k equals the absolute error at that step. Aggregating many windows
    gives a distribution of RMSE values per step per model — what the
    boxplots show.

    Results are cached to disk — re-runs skip already-computed windows.

    Returns
    -------
    dict : model_name -> list of length test_periods
           each element is the RMSE (= |error|) at that forecast step
    """
    key_data  = np.concatenate([train_values, test_values])
    key_bytes = f"{state}_{key_data.tobytes()}_{test_periods}".encode()
    cache_key = hashlib.md5(key_bytes).hexdigest()
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    train_series = pd.Series(train_values, index=train_index)
    test_series  = pd.Series(test_values,  index=test_index)

    rmse_this_window = {}
    for model_name, model_func in MODEL_FUNCS.items():
        try:
            forecast, _ = model_func(train_series, n_periods=test_periods)
            if forecast is None or len(forecast) != len(test_series):
                continue
            aligned = pd.Series(forecast.values, index=test_series.index)
            # Store absolute error per step — caller aggregates with sqrt(mean(sq))
            per_step_abs = np.abs(test_series.values - aligned.values).tolist()
            rmse_this_window[model_name] = per_step_abs
        except Exception:
            pass

    with open(cache_path, 'wb') as f:
        pickle.dump(rmse_this_window, f)

    return rmse_this_window


def _worker(args):
    """Unpacks args for ProcessPoolExecutor — must be top-level."""
    return _compute_window_errors(*args)


# ---------------------------------------------------------------------------
# Main optimized function
# ---------------------------------------------------------------------------

def sliding_rmse_excel(
    df: pd.DataFrame,
    product_name: str = "",
    test_periods: int = 12,
    min_train_size: int = 36,
    max_windows: int = 12,
    output_xlsx: str = "sliding_rmse.xlsx",
    cache_dir: str = ".boxplot_cache",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Walk-forward RMSE export for all models, steps, and states.

    For every (state, sliding-window) pair and every model, computes the
    absolute error at each of the test_periods forecast steps:

        rmse_step_k = |actual_k - predicted_k|

    Results are written to an Excel file with one sheet per state. Each
    sheet has rows = windows and a MultiIndex column of (model, step).

    Additionally returns a summary sheet with the mean RMSE per
    (state, model, step) aggregated across all windows.

    Excel structure
    ---------------
    Sheet "Summary":
        Rows    = states (UF)
        Columns = MultiIndex (Model, Step 1 ... Step N)
        Values  = mean RMSE across all windows for that state/model/step

    Sheet per state (e.g. "SP", "MS", ...):
        Rows    = one per sliding window (labelled by the last training date)
        Columns = MultiIndex (Model, Step 1 ... Step N)
        Values  = per-window per-step RMSE

    Parameters
    ----------
    df            : DataFrame with DatetimeIndex, one column per state/UF.
    product_name  : Used only in the log messages.
    test_periods  : Forecast horizon (number of steps per window).
    min_train_size: Minimum training observations required per window.
    max_windows   : Maximum sliding windows evaluated per state.
    output_xlsx   : Output path for the Excel file.
    cache_dir     : Directory where per-window cache files are stored.
    n_jobs        : Parallel worker processes (-1 = all CPU cores).

    Returns
    -------
    summary_df : DataFrame with mean RMSE per (state, model, step).
    """
    os.makedirs(cache_dir, exist_ok=True)

    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()

    logger.info(
        f"sliding_rmse_excel | product={product_name!r} "
        f"| states={len(df.columns)} | max_windows={max_windows} | n_jobs={n_jobs}"
    )

    # ------------------------------------------------------------------
    # 1. Build jobs list
    # ------------------------------------------------------------------
    # Each job also carries the label for the window (last training date)
    # so we can use it as the row index in the per-state sheet.
    jobs = []          # (state, train_vals, train_idx, test_vals, test_idx, periods, cache)
    job_meta = []      # (state, window_label) — same order as jobs

    for state in df.columns:
        series = df[state].dropna()
        if len(series) < min_train_size + test_periods:
            logger.warning(f"Skipping {state}: only {len(series)} observations.")
            continue

        start_end = len(series)
        stop_end  = max(min_train_size + test_periods - 1, start_end - max_windows)

        for end_idx in range(start_end, stop_end, -1):
            if end_idx - test_periods < min_train_size:
                break
            train_s = series.iloc[:end_idx - test_periods]
            test_s  = series.iloc[end_idx - test_periods:end_idx]
            last_idx = train_s.index[-1]
            try:
                window_label = str(pd.Timestamp(last_idx).date())  # e.g. "2022-12-01"
            except Exception:
                window_label = str(last_idx)
            jobs.append((
                state,
                train_s.values,
                train_s.index,
                test_s.values,
                test_s.index,
                test_periods,
                cache_dir,
            ))
            job_meta.append((state, window_label))

    logger.info(f"Total jobs: {len(jobs)}")

    # ------------------------------------------------------------------
    # 2. Run in parallel
    # ------------------------------------------------------------------
    # raw_results[(state, window_label)] = {model: [rmse_step0, ..., rmse_stepN-1]}
    raw_results: dict = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_worker, job): idx for idx, job in enumerate(jobs)}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Computing windows", unit="window"):
            idx = futures[future]
            state, window_label = job_meta[idx]
            try:
                result = future.result()   # {model: [rmse_step0, ...]}
                raw_results[(state, window_label)] = result
            except Exception as exc:
                logger.error(f"Window ({state}, {window_label}) failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Assemble per-state DataFrames and summary
    # ------------------------------------------------------------------
    model_names = list(MODEL_FUNCS.keys())
    step_labels = [f"Step {k + 1}" for k in range(test_periods)]

    # MultiIndex columns: (model, step)
    col_tuples = [(m, s) for m in model_names for s in step_labels]
    col_index  = pd.MultiIndex.from_tuples(col_tuples, names=["Model", "Step"])

    # Group results by state
    state_groups: dict[str, dict] = {}   # state -> {window_label: {model: [rmses]}}
    for (state, window_label), result in raw_results.items():
        state_groups.setdefault(state, {})[window_label] = result

    per_state_dfs: dict[str, pd.DataFrame] = {}
    for state, windows in state_groups.items():
        rows = {}
        for window_label, model_rmses in windows.items():
            row = {}
            for model in model_names:
                rmses = model_rmses.get(model, [np.nan] * test_periods)
                for k, step_label in enumerate(step_labels):
                    row[(model, step_label)] = rmses[k] if k < len(rmses) else np.nan
            rows[window_label] = row
        per_state_dfs[state] = pd.DataFrame(rows).T
        per_state_dfs[state].index.name = "Window (last train date)"
        per_state_dfs[state].columns = pd.MultiIndex.from_tuples(
            per_state_dfs[state].columns, names=["Model", "Step"]
        )

    # Summary: mean RMSE per (state, model, step) across windows
    summary_rows = {}
    for state, df_state in per_state_dfs.items():
        summary_rows[state] = df_state.mean(axis=0)   # Series with MultiIndex

    summary_df = pd.DataFrame(summary_rows).T
    summary_df.index.name = "State"

    # ------------------------------------------------------------------
    # 4. Write Excel
    # ------------------------------------------------------------------
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary")
        for state, df_state in per_state_dfs.items():
            # Sheet names max 31 chars; state names are short UF codes
            df_state.to_excel(writer, sheet_name=str(state)[:31])

    logger.info(f"Saved Excel -> {output_xlsx}  ({len(per_state_dfs)} state sheets + Summary)")
    return summary_df



def sliding_rmse_boxplots(
    df: pd.DataFrame,
    product_name: str = "",
    test_periods: int = 12,
    min_train_size: int = 36,
    max_windows: int = 12,
    output_html: str = "sliding_rmse_boxplots.html",
    cache_dir: str = ".boxplot_cache",
    n_jobs: int = -1,
    steps_per_row: int = 4,
):
    """
    Walk-forward RMSE boxplots for all models across all states.

    Layout
    ------
    3 rows (subplots), each row contains one grouped chart with `steps_per_row`
    forecast steps on the x-axis. Within each step, 7 side-by-side boxes are
    shown — one per model — color-coded via a shared legend. No model name is
    printed under the boxes; the legend at the top identifies the colors.

    What each box shows
    -------------------
    Each data point is the per-step RMSE for one (state, window) pair:
        rmse_step_k = |actual_k - predicted_k|
    The box shows the distribution of these values across all windows and
    states: median, IQR, whiskers, and outliers.

    Parameters
    ----------
    df            : DataFrame with DatetimeIndex, one column per state/UF.
    product_name  : Label shown in the chart title.
    test_periods  : Total forecast horizon (default 12 → 3 rows of 4 steps).
    min_train_size: Minimum training observations required per window.
    max_windows   : Maximum sliding windows evaluated per state.
    output_html   : Output path for the single HTML file.
    cache_dir     : Directory where per-window cache files are stored.
    n_jobs        : Parallel worker processes (-1 = all CPU cores).
    steps_per_row : How many forecast steps appear on each row (default 4).

    Returns
    -------
    fig : Plotly Figure.
    """
    os.makedirs(cache_dir, exist_ok=True)

    if n_jobs == -1:
        import multiprocessing
        n_jobs = multiprocessing.cpu_count()

    logger.info(
        f"sliding_rmse_boxplots | product={product_name!r} "
        f"| states={len(df.columns)} | max_windows={max_windows} | n_jobs={n_jobs}"
    )

    # ------------------------------------------------------------------
    # 1. Build jobs
    # ------------------------------------------------------------------
    jobs      = []
    job_meta  = []

    for state in df.columns:
        series = df[state].dropna()
        if len(series) < min_train_size + test_periods:
            logger.warning(f"Skipping {state}: only {len(series)} observations.")
            continue

        start_end = len(series)
        stop_end  = max(min_train_size + test_periods - 1, start_end - max_windows)

        for end_idx in range(start_end, stop_end, -1):
            if end_idx - test_periods < min_train_size:
                break
            train_s = series.iloc[:end_idx - test_periods]
            test_s  = series.iloc[end_idx - test_periods:end_idx]
            jobs.append((
                state,
                train_s.values,
                train_s.index,
                test_s.values,
                test_s.index,
                test_periods,
                cache_dir,
            ))
            job_meta.append(state)

    logger.info(f"Total jobs: {len(jobs)}")

    # ------------------------------------------------------------------
    # 2. Run in parallel (cache hits are instant)
    # ------------------------------------------------------------------
    # Accumulate squared errors per (state, model, step) across windows.
    # sq_errors_by_state[state][model][step] = [se_w1, se_w2, ...]
    # After all windows are done we compute:
    #   RMSE_state_model_step = sqrt( mean(squared_errors across windows) )
    # giving one scalar per state — 26 scalars per (model, step) box.
    sq_errors_by_state = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_worker, job): (idx, job) for idx, job in enumerate(jobs)}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Computing windows", unit="window"):
            idx, _ = futures[future]
            state  = job_meta[idx]
            try:
                result = future.result()   # {model: [|e_step0|, ..., |e_stepN-1|]}
                if state not in sq_errors_by_state:
                    sq_errors_by_state[state] = {m: [[] for _ in range(test_periods)]
                                                 for m in MODEL_FUNCS}
                for model_name, abs_errors in result.items():
                    for step, ae in enumerate(abs_errors):
                        if step < test_periods:
                            # store squared error for later mean+sqrt aggregation
                            sq_errors_by_state[state][model_name][step].append(ae ** 2)
            except Exception as exc:
                logger.error(f"Window failed: {exc}")

    # ------------------------------------------------------------------
    # 3. Aggregate: one RMSE per (state, model, step)
    #    rmse_by_step[step][model] = [rmse_state1, rmse_state2, ...]  (26 values)
    # ------------------------------------------------------------------
    rmse_by_step = {step: {m: [] for m in MODEL_FUNCS} for step in range(test_periods)}
    for state, models in sq_errors_by_state.items():
        for model_name, steps_sq in models.items():
            for step, sq_list in enumerate(steps_sq):
                if sq_list:
                    rmse_val = float(np.sqrt(np.mean(sq_list)))
                    rmse_by_step[step][model_name].append(rmse_val)

    # ------------------------------------------------------------------
    # 4. Global y-axis range (same scale on every subplot)
    # ------------------------------------------------------------------
    all_vals = [v for step in range(test_periods)
                  for m in MODEL_FUNCS
                  for v in rmse_by_step[step][m]]
    y_max = float(np.max(all_vals)) * 1.05 if all_vals else 1.0
    y_min = 0.0  # RMSE is always >= 0

    # ------------------------------------------------------------------
    # 5. Build figure
    #    Layout: 6 rows (steps_per_row=2), each row is one subplot with
    #    2 step groups on the x-axis. Within each step, 7 boxes sit
    #    side-by-side — one per model — identified by color only.
    #    With only 2 steps per row each box has plenty of horizontal room.
    # ------------------------------------------------------------------
    steps_per_row = 6
    n_rows        = math.ceil(test_periods / steps_per_row)  # = 2

    palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA",
               "#FFA15A", "#19D3F3", "#FF6692"]
    model_names = list(MODEL_FUNCS.keys())
    color_map   = {m: palette[i % len(palette)] for i, m in enumerate(model_names)}

    row_titles = []
    for r in range(n_rows):
        s_start = r * steps_per_row + 1
        s_end   = min((r + 1) * steps_per_row, test_periods)
        row_titles.append(f"Steps {s_start} - {s_end}")

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=row_titles,
        vertical_spacing=0.03,
        shared_yaxes=True,
    )

    n_models   = len(model_names)
    box_width  = 0.10          # width of each individual box
    step_gap   = 0.25          # tighter gap between step groups to fit 6 per row
    step_span  = n_models * box_width   # total width occupied by one step group

    # Pre-compute x centre for every (row, step) group and each model offset
    # x_centers[row][step] = centre x position of that step group on that row
    x_centers = {}
    for r in range(n_rows):
        x_centers[r] = {}
        for s in range(steps_per_row):
            step_idx = r * steps_per_row + s
            if step_idx >= test_periods:
                break
            x_centers[r][step_idx] = s * (step_span + step_gap) + step_span / 2

    # model offset relative to the step centre: centres the 7 boxes around 0
    model_offsets = {m: (i - (n_models - 1) / 2) * box_width
                     for i, m in enumerate(model_names)}

    # Build x-axis tick positions and labels per row
    row_xticks = {}   # row -> (tick_positions, tick_labels)
    for r in range(n_rows):
        positions, labels = [], []
        for step_idx, cx in x_centers[r].items():
            positions.append(cx)
            labels.append(f"Step {step_idx + 1}")
        row_xticks[r] = (positions, labels)

    for model_idx, model_name in enumerate(model_names):
        show_legend = True
        for step in range(test_periods):
            row = step // steps_per_row
            if step not in x_centers[row]:
                continue
            vals = rmse_by_step[step][model_name]
            if not vals:
                continue

            cx     = x_centers[row][step]
            offset = model_offsets[model_name]
            x_pos  = cx + offset   # exact centre for this model in this step

            fig.add_trace(
                go.Box(
                    x=[x_pos] * len(vals),
                    y=vals,
                    name=model_name,
                    legendgroup=model_name,
                    showlegend=show_legend,
                    marker_color=color_map[model_name],
                    fillcolor=color_map[model_name],
                    width=box_width,
                    boxmean=False,
                    whiskerwidth=1.0,
                    boxpoints="outliers",
                    jitter=0.15,
                    marker=dict(
                        symbol="circle",
                        size=4,
                        opacity=0.6,
                        line=dict(width=0),
                    ),
                    line=dict(width=1.5, color="rgba(0,0,0,0.5)"),
                    opacity=0.85,
                ),
                row=row + 1, col=1,
            )
            show_legend = False

    for r in range(n_rows):
        positions, labels = row_xticks[r]
        fig.update_yaxes(
            range=[y_min, y_max],
            title_text="RMSE",
            title_font=dict(size=20),
            tickfont=dict(size=18),
            row=r + 1, col=1,
        )
        fig.update_xaxes(
            tickmode="array",
            tickvals=positions,
            ticktext=labels,
            tickfont=dict(size=18),
            row=r + 1, col=1,
        )

    title_suffix = f" - {product_name}" if product_name else ""
    fig.update_layout(
        title=dict(
            text=f"RMSE Distribution by Forecast Step and Model{title_suffix}",
            x=0.5,
            font=dict(size=22),
        ),
        # Width and height sized for a full journal page column (exported at scale=3)
        width=1400,
        height=600 * n_rows,
        template="plotly_white",
        boxmode="overlay",
        # Global font floor — catches any text not set explicitly above
        font=dict(size=18),
        # Legend at the bottom, horizontal, so it never squeezes the plot area
        legend=dict(
            title=dict(text="Model", font=dict(size=18)),
            orientation="h",
            yanchor="top",
            y=-0.12,           # below the bottom x-axis
            xanchor="center",
            x=0.5,
            font=dict(size=18),
            tracegroupgap=4,
        ),
        # Tight margins — legend is below so bottom margin makes room for it
        margin=dict(t=60, b=120, l=60, r=20),
    )

    # --- HTML (interactive, for browser preview) ---
    fig.write_html(output_html, auto_open=True)

    # --- PDF (vector, preferred for LaTeX) ---
    pdf_path = output_html.replace(".html", ".pdf")
    try:
        fig.write_image(pdf_path, format="pdf", width=1400, height=600 * n_rows, scale=1)
        logger.info(f"Saved PDF  -> {pdf_path}")
    except Exception as e:
        logger.warning(f"PDF export failed (install kaleido: pip install kaleido): {e}")

    # --- High-res PNG fallback ---
    png_path = output_html.replace(".html", ".png")
    try:
        fig.write_image(png_path, format="png", width=1400, height=600 * n_rows, scale=3)
        logger.info(f"Saved PNG  -> {png_path}")
    except Exception as e:
        logger.warning(f"PNG export failed: {e}")

    logger.info(f"Saved HTML -> {output_html}")
    return fig



# Chart helpers (unchanged interface)
# ---------------------------------------------------------------------------

def generate_forecast_chart(serie_historica, serie_previsoes, nome_produto, uf, model_name, rmse):
    import plotly.express as px

    df_hist = serie_historica.reset_index()
    df_hist.columns = ['Date', 'Value']
    df_hist['Tipo'] = 'Histórico'

    df_prev = serie_previsoes.reset_index()
    df_prev.columns = ['Date', 'Value']
    df_prev['Tipo'] = 'Previsão'

    df_long = pd.concat([df_hist, df_prev], ignore_index=True)
    df_long['Date'] = df_long['Date'].astype(str)

    fig = px.line(
        df_long, x='Date', y='Value', color='Tipo',
        color_discrete_map={'Histórico': 'blue', 'Previsão': 'red'},
        title=f"Previsão — {uf}: {nome_produto} | {model_name} (RMSE: {rmse:.2f})",
        labels={'Date': 'Data', 'Value': 'Preço', 'Tipo': 'Série'},
    )
    fig.update_traces(line=dict(dash='dash', width=3), selector=dict(name='Previsão'))
    fig.update_xaxes(rangeslider=dict(visible=True), type="date")
    fig.update_layout(title_x=0.5, title_font_size=20, legend_title="Tipo de Série")
    return fig


def error_comparison_table(df, test_periods=12, n_periods=12):
    results = {}
    for state in df.columns:
        series = df[state]
        if len(series.dropna()) <= test_periods:
            continue
        train, test = series[:-test_periods], series[-test_periods:]
        results[state] = {}
        for label, fn in [
            ('ARIMA',         lambda s: _model_forecast_arima(s, n_periods)),
            ('ETS',           lambda s: _model_forecast_ets(s, n_periods)),
            ('PROPHET',       lambda s: _model_forecast_prophet(s, n_periods)),
            ('RANDOM_FOREST', lambda s: random_forest(s, n_periods)),
            ('LSTM',          lambda s: lstm_forecast(s, n_periods)),
            ('GRU',           lambda s: gru_forecast(s, n_periods)),
            ('TRANSFORMER',   lambda s: conv_transformer_forecast(s, n_periods)),
        ]:
            fc, _ = fn(train)
            fc.index = test.index
            results[state][f'RMSE_{label}'] = calculate_rmse(test, fc)
            results[state][f'MAPE_{label}'] = calculate_mape(test, fc)

    df_out = pd.DataFrame(results).T
    df_out.index.name = "UF"
    return df_out


def sliding_error_chart(series, model_func, model_name, test_periods=12, n_periods=12):
    import plotly.express as px
    all_sq_errors = []
    start_index = len(series) + 1
    stop_index  = start_index - 12
    for end_idx in range(start_index, stop_index, -1):
        test_s  = series[end_idx - test_periods: end_idx]
        train_s = series[:end_idx - test_periods]
        try:
            fc, _ = model_func(train_s, n_periods=test_periods)
            if len(test_s) != len(fc):
                continue
            aligned = pd.Series(fc.values, index=test_s.index)
            all_sq_errors.append(((test_s - aligned) ** 2).values)
        except Exception:
            continue
    if not all_sq_errors:
        return pd.Series(dtype='float64')
    rmse_by_horizon = np.sqrt(np.mean(np.array(all_sq_errors), axis=0))
    labels = [f"Passo {i + 1}" for i in range(test_periods)]
    return pd.Series(rmse_by_horizon, index=labels, name=f'RMSE Médio - {model_name}')


def generate_sliding_chart(rmse_series, model_name, product_name, uf):
    import plotly.express as px
    if rmse_series.empty:
        return px.scatter(title="Dados insuficientes.")
    df_plot = rmse_series.reset_index()
    df_plot.columns = ['Horizonte', 'RMSE']
    fig = px.line(
        df_plot, x='Horizonte', y='RMSE', markers=True,
        title=f"RMSE por Passo — {model_name} | {product_name} | {uf}",
        labels={'Horizonte': 'Horizonte de Previsão', 'RMSE': 'RMSE Médio'},
    )
    fig.update_layout(title_x=0.5, title_font_size=20, xaxis=dict(tickmode='linear'))
    return fig


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"
    SHEET_NAME      = "ARROZ"
    OUTPUT_HTML     = "sliding_rmse_boxplots.html"
    CACHE_DIR       = ".boxplot_cache"

    TEST_PERIODS     = 12
    FORECAST_PERIODS = 12
    MODEL_TO_RUN     = "sliding_rmse_boxplots"
    UF               = "MS"

    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, index_col=0)
    df.index.name = 'Date'

    series      = df[UF]
    train_series = series[:-TEST_PERIODS]
    test_series  = series[-TEST_PERIODS:]

    # Columns to exclude from all analysis (e.g. test series left in by mistake)
    EXCLUDE_STATES = []   # e.g. ["RANDOM"]
    if EXCLUDE_STATES:
        df = df.drop(columns=EXCLUDE_STATES, errors="ignore")

    if MODEL_TO_RUN == "sliding_rmse_boxplots":
        fig = sliding_rmse_boxplots(
            df,
            product_name=SHEET_NAME,
            test_periods=TEST_PERIODS,
            min_train_size=36,
            max_windows=12,
            output_html=OUTPUT_HTML,
            cache_dir=CACHE_DIR,
            n_jobs=-1,
        )
        print(f"Done. Open {OUTPUT_HTML} in your browser.")

    elif MODEL_TO_RUN == "sliding_rmse_excel":
        summary = sliding_rmse_excel(
            df,
            product_name=SHEET_NAME,
            test_periods=TEST_PERIODS,
            min_train_size=36,
            max_windows=12,
            output_xlsx="sliding_rmse.xlsx",
            cache_dir=CACHE_DIR,
            n_jobs=-1,
        )
        print(summary)

    elif MODEL_TO_RUN == "error_comparison_table":
        df_rmse = error_comparison_table(df, test_periods=TEST_PERIODS, n_periods=FORECAST_PERIODS)
        df_rmse.to_excel(f"{MODEL_TO_RUN}_modelos.xlsx")
        print(df_rmse)

    elif MODEL_TO_RUN == "sliding_error_chart":
        errors = sliding_error_chart(series, random_forest, "Random Forest",
                                     test_periods=TEST_PERIODS, n_periods=FORECAST_PERIODS)
        fig = generate_sliding_chart(errors, "Random Forest", SHEET_NAME, UF)
        fig.write_html("sliding_error_chart.html", auto_open=True)

    else:
        # Single-model forecast
        fn_map = {
            "Auto_Arima":   lambda: auto_arima_forecast(train_series, seasonal=True, m=12, n_periods=12),
            "ETS":          lambda: ets_forecast(train_series, m=12, n_periods=12),
            "Prophet":      lambda: prophet_forecast(train_series, seasonal=True, m=12, n_periods=12),
            "Random_Forest":lambda: random_forest(train_series, n_periods=12),
            "LSTM":         lambda: lstm_forecast(train_series, n_periods=12),
            "GRU":          lambda: gru_forecast(train_series, n_periods=12),
            "Transformer":  lambda: conv_transformer_forecast(train_series, n_periods=12),
        }
        prediction, model = fn_map[MODEL_TO_RUN]()
        if len(test_series) == len(prediction):
            prediction.index = test_series.index
        rmse = calculate_rmse(test_series, prediction)
        print(f"RMSE ({MODEL_TO_RUN}): {rmse:.2f}")
        fig = generate_forecast_chart(series, prediction, SHEET_NAME, UF, MODEL_TO_RUN, rmse)
        fig.write_html("previsao_interativa.html", auto_open=True)