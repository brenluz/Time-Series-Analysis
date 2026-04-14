import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from features import create_features, series_to_sequences
from models import LSTMModel, GRUModel, TimeSeriesTransformer, Informer
from training import train_batch


# ---------------------------------------------------------------------------
# Statistical models
# ---------------------------------------------------------------------------

def auto_arima_forecast(series: pd.Series, seasonal: bool = True,
                        m: int = 12, n_periods: int = 12):
    """Fit Auto ARIMA and forecast n_periods ahead."""
    model = pm.auto_arima(
        series.dropna(), d=1, D=1,
        start_p=1, start_q=1, max_p=5, max_q=5,
        seasonal=True, m=m,
        error_action='ignore', suppress_warnings=True, stepwise=True
    )
    pred = model.predict(n_periods=n_periods)
    idx  = pd.date_range(start=series.index[-1], periods=n_periods + 1, freq='MS')[1:]
    return pd.Series(pred, index=idx), model


def ets_forecast(series: pd.Series, m: int = 12, n_periods: int = 12,
                 trend: str = 'add', seasonal_model: str = 'add'):
    """Fit Holt-Winters ETS and forecast n_periods ahead."""
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


def prophet_forecast(series: pd.Series, seasonal: bool = True,
                     m: int = 12, n_periods: int = 12):
    """Fit Facebook Prophet and forecast n_periods ahead."""
    df_p = series.reset_index().dropna()
    df_p.columns = ['ds', 'y']
    model = Prophet(
        yearly_seasonality=False,
        daily_seasonality=False,
        weekly_seasonality=False
    )
    if seasonal and m and m > 1:
        model.add_seasonality(
            'custom_seasonal', period=m, fourier_order=5, mode='multiplicative'
        )
    model.fit(df_p)
    future = model.make_future_dataframe(periods=n_periods, freq='MS')
    fc = model.predict(future).set_index('ds')['yhat'][-n_periods:]
    fc.name = series.name
    return fc, model


# ---------------------------------------------------------------------------
# Machine learning models
# ---------------------------------------------------------------------------

def random_forest(series: pd.Series, n_periods: int = 12):
    """
    Iterative walk-forward forecast using a Random Forest Regressor.
    Each future step is predicted using the previous step's prediction
    as input features.
    """
    df_train = series.reset_index()
    df_train.columns = ['ds', 'y']
    df_features = create_features(
        df_train, lag_start=1, lag_end=12, rolling_window=6
    ).dropna(subset=['y'])

    X_cols = [c for c in df_features.columns if c not in ['ds', 'y']]
    model  = RandomForestRegressor(n_estimators=100, random_state=25, n_jobs=-1)
    model.fit(df_features[X_cols], df_features['y'])

    last_date    = series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=n_periods + 1)[1:]
    df_full      = pd.concat(
        [df_train, pd.DataFrame({'ds': future_dates})], ignore_index=True
    )
    df_full['ds'] = pd.to_datetime(df_full['ds'])

    predictions = []
    for i in range(len(df_train), len(df_full)):
        df_iter  = create_features(df_full[:i], lag_end=12, rolling_window=6)
        X_pred   = df_iter.iloc[-1][X_cols].to_frame().T
        next_val = model.predict(X_pred)[0]
        predictions.append(next_val)
        df_full.loc[i, 'y'] = next_val

    return pd.Series(predictions, index=future_dates, name='RF'), model


def lstm_forecast(series: pd.Series, n_periods: int = 12):
    """Train an LSTM and forecast n_periods ahead via walk-forward inference."""
    if len(series.dropna()) < n_periods + 2:
        return pd.Series(dtype='float64'), None

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y   = series_to_sequences(scaled, n_periods)
    if X.shape[0] == 0:
        return pd.Series(dtype='float64'), None

    X_t = torch.FloatTensor(X).unsqueeze(2)   # [samples, seq_len, 1]
    y_t = torch.FloatTensor(y)

    model = LSTMModel()
    model = train_batch(model, X_t, y_t, epochs=300, patience=30)
    model.eval()

    cur   = scaled[-n_periods:].copy()
    preds = []
    with torch.no_grad():
        for _ in range(n_periods):
            inp = torch.FloatTensor(cur).unsqueeze(0).unsqueeze(2)  # [1, seq, 1]
            p   = model(inp).item()
            preds.append(p)
            cur = np.roll(cur, -1)
            cur[-1] = p

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    idx   = pd.date_range(start=series.index[-1], periods=n_periods + 1, freq='MS')[1:]
    return pd.Series(preds, index=idx, name='LSTM'), model


def gru_forecast(series: pd.Series, n_periods: int = 12):
    """Train a GRU and forecast n_periods ahead via walk-forward inference."""
    if len(series.dropna()) < n_periods + 2:
        return pd.Series(dtype='float64'), None

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y   = series_to_sequences(scaled, n_periods)
    if X.shape[0] == 0:
        return pd.Series(dtype='float64'), None

    X_t = torch.FloatTensor(X).unsqueeze(2)
    y_t = torch.FloatTensor(y)

    model = GRUModel()
    model = train_batch(model, X_t, y_t, epochs=300, patience=30)
    model.eval()

    cur   = scaled[-n_periods:].copy()
    preds = []
    with torch.no_grad():
        for _ in range(n_periods):
            inp = torch.FloatTensor(cur).unsqueeze(0).unsqueeze(2)
            p   = model(inp).item()
            preds.append(p)
            cur = np.roll(cur, -1)
            cur[-1] = p

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    idx   = pd.date_range(start=series.index[-1], periods=n_periods + 1, freq='MS')[1:]
    return pd.Series(preds, index=idx, name='GRU'), model


def transformer_forecast(series: pd.Series, n_periods: int = 12):
    """Train a Transformer encoder and forecast n_periods ahead."""
    if len(series.dropna()) < n_periods + 1:
        return pd.Series(dtype='float64'), None

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
    X, y   = series_to_sequences(scaled, n_periods)

    # Transformer expects [seq_len, batch, features]
    X_t = torch.FloatTensor(X).transpose(0, 1).unsqueeze(2)
    y_t = torch.FloatTensor(y)

    model = TimeSeriesTransformer(
        input_dim=1, d_model=64, nhead=8, num_layers=2, dropout=0.1
    )
    model = train_batch(model, X_t, y_t, epochs=500, lr=0.001, patience=40)
    model.eval()

    cur   = scaled[-n_periods:].copy()
    preds = []
    with torch.no_grad():
        for _ in range(n_periods):
            inp = torch.FloatTensor(cur).view(n_periods, 1, 1)
            p   = model(inp).item()
            preds.append(p)
            cur = np.roll(cur, -1)
            cur[-1] = p

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    idx   = pd.date_range(start=series.index[-1], periods=n_periods + 1, freq='MS')[1:]
    return pd.Series(preds, index=idx, name='Transformer'), model



def informer_forecast(series: pd.Series, n_periods: int = 12):
    """
    Train an Informer encoder-decoder and forecast n_periods ahead.

    Encoder input  : full historical window (enc_len observations)
    Decoder input  : start token (last label_len observations) + n_periods zero slots
    Inference      : single forward pass — the decoder generates all n_periods
                     future values simultaneously (generative decoding, no loop).

    Temporal features [value, sin(2pi*month/12), cos(2pi*month/12)] are built
    automatically from the series index so the model is aware of seasonality.
    """
    min_len = 2 * n_periods + 4
    if len(series.dropna()) < min_len:
        return pd.Series(dtype='float64'), None

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    enc_len   = len(scaled)       # use full history as encoder input
    label_len = n_periods         # start-token = last n_periods observations
    pred_len  = n_periods

    # Infer start month from the series index (fall back to 1 if unavailable)
    try:
        start_month = int(pd.Timestamp(series.index[0]).month)
    except Exception:
        start_month = 1

    # Build training samples --------------------------------------------------
    # Each sample: encoder sees enc_window steps, decoder predicts pred_len steps
    enc_window = max(n_periods * 2, 24)   # history window fed to encoder
    X_enc_list, X_dec_list, y_list = [], [], []

    for i in range(enc_window, len(scaled) - pred_len + 1):
        enc_vals  = torch.FloatTensor(scaled[i - enc_window: i])
        dec_start = scaled[i - label_len: i]
        dec_zeros = np.zeros(pred_len)
        dec_vals  = torch.FloatTensor(np.concatenate([dec_start, dec_zeros]))
        target    = torch.FloatTensor(scaled[i: i + pred_len])

        sm_enc = ((start_month + (i - enc_window)) - 1) % 12 + 1
        sm_dec = ((start_month + (i - label_len))  - 1) % 12 + 1

        X_enc_list.append(Informer.build_features(enc_vals,  sm_enc))
        X_dec_list.append(Informer.build_features(dec_vals,  sm_dec))
        y_list.append(target)

    if len(X_enc_list) == 0:
        return pd.Series(dtype='float64'), None

    # Stack into batch tensors: [samples, seq_len, 3]
    X_enc_t = torch.cat(X_enc_list, dim=0)                 # [N, enc_window, 3]
    X_dec_t = torch.cat(X_dec_list, dim=0)                 # [N, label+pred, 3]
    y_t     = torch.stack(y_list, dim=0)                   # [N, pred_len]

    # Build and train model ---------------------------------------------------
    model = Informer(
        d_model=64, n_heads=8, e_layers=2, d_layers=1,
        d_ff=256, factor=5, dropout=0.1,
        label_len=label_len, pred_len=pred_len,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)
    loss_fn    = torch.nn.MSELoss()
    best_loss  = float('inf')
    best_state = None
    patience_counter = 0
    patience  = 40

    import copy
    model.train()
    for epoch in range(500):
        optimizer.zero_grad()
        pred = model(X_enc_t, X_dec_t)   # [N, pred_len]
        loss = loss_fn(pred, y_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())

        if loss.item() < best_loss - 1e-6:
            best_loss        = loss.item()
            best_state       = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    # Inference — single generative pass --------------------------------------
    with torch.no_grad():
        enc_vals = torch.FloatTensor(scaled[-enc_window:])
        sm_enc   = ((start_month + len(scaled) - enc_window) - 1) % 12 + 1
        sm_dec   = ((start_month + len(scaled) - label_len)  - 1) % 12 + 1

        dec_start = scaled[-label_len:]
        dec_zeros = np.zeros(pred_len)
        dec_vals  = torch.FloatTensor(np.concatenate([dec_start, dec_zeros]))

        x_enc = Informer.build_features(enc_vals, sm_enc)   # [1, enc_window, 3]
        x_dec = Informer.build_features(dec_vals, sm_dec)   # [1, label+pred, 3]

        preds_scaled = model(x_enc, x_dec).squeeze(0).numpy()  # [pred_len]

    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    idx   = pd.date_range(start=series.index[-1], periods=n_periods + 1, freq='MS')[1:]
    return pd.Series(preds, index=idx, name='Informer'), model

# ---------------------------------------------------------------------------
# Wrappers with uniform (series, n_periods) signature
# ---------------------------------------------------------------------------

def _wrap_arima(series, n_periods=12):
    return auto_arima_forecast(series, seasonal=True, m=12, n_periods=n_periods)

def _wrap_ets(series, n_periods=12):
    return ets_forecast(series, n_periods=n_periods, m=12, trend='add', seasonal_model='add')

def _wrap_prophet(series, n_periods=12):
    return prophet_forecast(series, seasonal=True, m=12, n_periods=n_periods)


# ---------------------------------------------------------------------------
# Central registry used by cache, analysis, and evaluation functions
# ---------------------------------------------------------------------------

MODEL_FUNCS: dict = {
    "ARIMA":         _wrap_arima,
    "ETS":           _wrap_ets,
    "Prophet":       _wrap_prophet,
    "Random Forest": random_forest,
    "LSTM":          lstm_forecast,
    "GRU":           gru_forecast,
    "Transformer":   transformer_forecast,
    "Informer":      informer_forecast,
}