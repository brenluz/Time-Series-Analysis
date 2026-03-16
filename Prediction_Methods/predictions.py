import math
from functools import partial

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pmdarima as pm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, optim

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

def calculate_rmse(actual, predicted):
    """
    Calcula o Root Mean Squared Error (RMSE) entre os valores reais e previstos.

    Parâmetros:
    - actual (pd.Series): Série com os valores reais.
    - predicted (pd.Series): Série com os valores previstos.

    Retorna:
    - rmse (float): O valor do RMSE calculado.
    """
    error = actual - predicted
    mse = (error ** 2).mean()
    rmse = mse ** 0.5
    return rmse

def calculate_mape(actual, predicted):
    """
    Calcula o Mean Absolute Percentage Error (MAPE) entre os valores reais e previstos.

    Parâmetros:
    - actual (pd.Series): Série com os valores reais.
    - predicted (pd.Series): Série com os valores previstos.

    Retorna:
    - mape (float): O valor do MAPE calculado em porcentagem.
    """
    error = abs((actual - predicted) / actual)
    mape = error.mean() * 100
    return mape

def create_features(df, lag_start=1, lag_end=12,rolling_window=3):
    """
    Cria features de Machine Learning (lags e features temporais) para a série temporal.

    Parameters:
    - df (pd.DataFrame): DataFrame com 'ds' (Date) e 'y' (Value).
    - lag_start (int): Início da defasagem (lag).
    - lag_end (int): Fim da defasagem (lag).
    - rolling_window (int): Tamanho da janela para média móvel.

    Returns:
    - pd.DataFrame: DataFrame com as novas features.
    """

    df_features = df.copy()

    df_features['ds'] = pd.to_datetime(df_features['ds'])

    df_features['year'] = df_features['ds'].dt.year
    df_features['month'] = df_features['ds'].dt.month
    df_features['dayofweek'] = df_features['ds'].dt.dayofweek
    df_features['dayofyear'] = df_features['ds'].dt.dayofyear

    # 2. Lagged Values (Capturam Autocorrelação)
    for lag in range(lag_start, lag_end + 1):
        df_features[f'lag_{lag}'] = df_features['y'].shift(lag)

    # 3. Rolling Window Features (Capturam Tendência e Variação)
    df_features['rolling_mean'] = df_features['y'].shift(lag_start).rolling(window=rolling_window).mean()
    df_features['rolling_std'] = df_features['y'].shift(lag_start).rolling(window=rolling_window).std()

    return df_features.dropna()

def series_to_sequences(series, n_steps):
    X, y = list(), list()
    for i in range(len(series)):
        end_ix = i + n_steps
        if end_ix > len(series) - 1:
            break
        # seq_x: janela de n_steps (passado), seq_y: próximo valor (futuro)
        seq_x, seq_y = series[i:end_ix], series[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class GRUmodel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = torch.zeros(1, 1, self.hidden_layer_size)

    def forward(self, input_seq):
        gru_out, self.hidden_cell = self.gru(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(gru_out.view(len(input_seq), -1))
        return predictions[-1]

# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Usando math.log
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # self.pe is stored as [max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        seq_len = x.size(0)

        # 1. Slice PE: [seq_len, d_model]
        pe_slice = self.pe[:seq_len, :]

        # 2. CRÍTICO: Adiciona a dimensão do batch (tamanho 1) de volta: [seq_len, 1, d_model]
        # Isso garante que a soma por broadcasting funcione corretamente na dimensão 1 (Batch).
        pe_slice_expanded = pe_slice.unsqueeze(1)

        # 3. Adiciona PE ao input X.
        return x + pe_slice_expanded

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()

        self.input_linear = nn.Linear(input_dim, d_model)
        self.d_model = d_model

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_linear = nn.Linear(d_model, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.input_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)
        self.output_linear.bias.data.zero_()

    def forward(self, src):
        src = self.input_linear(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        prediction = self.output_linear(output[-1, :, :])

        return prediction.squeeze(1)

class ConvInputEmbedding(nn.Module):
    """
    Camada de Convolução 1D para extrair features locais antes do Transformer.
    Requer transposição da entrada de [Seq_len, Batch, Features] para [Batch, Features, Seq_len].
    """

    def __init__(self, input_dim, d_model):
        super().__init__()
        # Kernel size 3 é ideal para capturar correlações em janelas curtas
        self.conv = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)

    def forward(self, x):
        # x shape: [Seq_len, Batch_size, Features (1)]

        # 1. Transpose para Conv1D: [Batch_size, Features, Seq_len]
        x = x.transpose(0, 1).transpose(1, 2)

        # 2. Convolução: [Batch_size, d_model, Seq_len]
        x = self.conv(x)

        # 3. Transpose de volta para Transformer: [Seq_len, Batch_size, d_model]
        return x.transpose(1, 2).transpose(0, 1)

def auto_arima_forecast(series, seasonal=True, m=12, n_periods=12):
    """
    Ajusta um modelo Auto ARIMA à série temporal e faz previsões.

    Parâmetros:
    - series (pd.Series): Série temporal univariada.
    - seasonal (bool): Indica se o modelo deve considerar sazonalidade.
    - m (int): Período sazonal (ex: 12 para dados mensais com sazonalidade anual).
    - n_periods (int): Número de períodos futuros a serem previstos.
    - test_periods (int): Número de períodos a serem usados para teste/validação.

    Retorna:
    - forecast (pd.Series): Série contendo as previsões futuras.
    - model: O modelo ARIMA ajustado.
    """

    # Ajustar o modelo Auto ARIMA
    model = pm.auto_arima(
        series.dropna(),
        d=1,
        D=1,
        start_p=1,
        start_q=1,
        max_p=5,
        max_q=5,
        seasonal=True,
        m=m,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    pred_array = model.predict(n_periods=n_periods)
    last_date = series.index[-1]
    freq_inferida = 'MS'
    indice_futuro = pd.date_range(start=last_date,
                                  periods=n_periods + 1,
                                  freq=freq_inferida)[1:]

    series_predictions = pd.Series(pred_array, index=indice_futuro)

    return series_predictions, model


def ets_forecast(series, m=12, n_periods=12, trend='add', seasonal_model='add'):
    """
    Ajusta um modelo de Suavização Exponencial (Holt-Winters/ETS) à série
    temporal e faz previsões.

    Parâmetros:
    - series (pd.Series): Série temporal univariada.
    - seasonal (bool): Indica se o modelo deve considerar sazonalidade.
    - m (int, opcional): Período sazonal (ex: 12 para dados mensais).
                         Se seasonal=True e m=None, tentará inferir o período
                         pela frequência do índice, mas é **altamente recomendado** especificar.
    - n_periods (int): Número de períodos futuros a serem previstos.
    - trend (str, opcional): Tipo de componente de tendência: 'add' (aditiva),
                             'mul' (multiplicativa) ou None (sem tendência). Padrão: 'add'.
    - seasonal_model (str, opcional): Tipo de componente sazonal: 'add' (aditiva),
                                      'mul' (multiplicativa) ou None. Padrão: 'add'.

    Retorna:
    - forecast (pd.Series): Série contendo as previsões futuras.
    - model_fit: O objeto de resultados do modelo ajustado.
    """

    # 1. Preparar a série (remover NaNs)
    series_clean = series.dropna()

    # 2. Definir parâmetros para ExponentialSmoothing
    trend_param = trend if trend is not None else None

    # 3. Ajustar o modelo Exponential Smoothing (ETS/Holt-Winters)
    try:
        model = ExponentialSmoothing(
            series_clean,
            trend=trend_param,
            seasonal=seasonal_model,
            seasonal_periods=m,
            initialization_method='estimated'  # Deixa o statsmodels estimar os valores iniciais
        )
        # O .fit() realiza a otimização dos parâmetros de suavização (alpha, beta, gamma)
        model_fit = model.fit()

    except Exception as e:
        print(f"Erro ao ajustar o modelo ETS: {e}")
        # Uma alternativa seria tentar com um modelo mais simples (sem sazonalidade/tendência)
        # Mas para manter a estrutura, vamos re-lançar o erro ou retornar None
        return pd.Series(dtype='float64'), None

    # 4. Fazer a previsão
    pred_array = model_fit.forecast(steps=n_periods)

    # 5. Criar o índice futuro (igual à função auto_arima)
    last_date = series_clean.index[-1]

    # Tenta inferir a frequência do índice
    freq_inferida = "MS"
    if freq_inferida is None:
        try:
            # Tenta inferir se não estiver explicitamente definido
            freq_inferida = pd.infer_freq(series_clean.index)
        except:
            # Fallback para frequência mensal se a inferência falhar
            freq_inferida = 'MS'

    if freq_inferida is None:
        raise ValueError(
            "Não foi possível inferir a frequência da série temporal. Certifique-se de que o índice de data está configurado corretamente (ex: df.index.freq='MS').")

    indice_futuro = pd.date_range(start=last_date,
                                  periods=n_periods + 1,
                                  freq=freq_inferida)[1:]

    # 6. Criar a série de previsões com o índice futuro
    series_predictions = pd.Series(pred_array, index=indice_futuro)

    return series_predictions, model_fit

def prophet_forecast(series, seasonal=True,m=12, n_periods=12):
    """
    Ajusta um modelo Prophet à série temporal e faz previsões.

    Parâmetros:
    - series (pd.Series): Série temporal univariada.
    - seasonal (bool): Indica se o modelo deve considerar sazonalidade.
    - m (int): Período sazonal (ex: 12 para dados mensais com sazonalidade anual).
    - n_periods (int): Número de períodos futuros a serem previstos.

    Retorna:
    - forecast (pd.Series): Série contendo as previsões futuras.
    - model: O modelo Prophet ajustado.
    """

    # Preparar os dados para o Prophet
    df_prophet = series.reset_index()
    df_prophet.columns = ['ds', 'y']
    df_prophet.dropna()

    # Ajustar o modelo Prophet
    model = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
    if seasonal and m is not None and m > 1:
        model.add_seasonality(name='custom_seasonal', period=m, fourier_order=5, mode='multiplicative')
    model.fit(df_prophet)
    # Criar DataFrame para períodos futuros
    future = model.make_future_dataframe(periods=n_periods, freq='MS')

    # Fazer a previsão
    forecast = model.predict(future)

    # Extrair as previsões futuras
    forecast_series = forecast.set_index('ds')['yhat'][-n_periods:]
    forecast_series.name = series.name

    return forecast_series, model

def random_forest(series, n_periods=12):
    """
    Metodo de previsao machine learning: Random Forest Regressor.

    Parametros:
    - series (pd.Series): Série temporal univariada.
    - n_periods (int): Número de períodos futuros a serem previstos.

    Retorna:
    - forecast (pd.Series): Série contendo as previsões futuras.
    - model: O modelo Random Forest ajustado.
    """
    df_train = series.reset_index()
    df_train.columns = ['ds', 'y']
    df_train.dropna()

    df_features = create_features(df_train, lag_start=1, lag_end=12, rolling_window=6)
    df_features = df_features.dropna(subset=['y'])

    X_cols = [col for col in df_features.columns if col not in ['ds', 'y']]
    X_train = df_features[X_cols]
    y_train = df_features['y']

    model = RandomForestRegressor(n_estimators=100, random_state=25, n_jobs=-1)
    model.fit(X_train, y_train)

    last_date = series.index[-1]

    # Cria as datas futuras
    future_dates = pd.date_range(start=last_date, periods=n_periods + 1)[1:]
    df_future = pd.DataFrame({'ds': future_dates})

    # Inicializa 'y' e concatena para gerar FEATURES DE FORMA ITERATIVA
    df_full = pd.concat([df_train, df_future], ignore_index=True)
    df_full['ds'] = pd.to_datetime(df_full['ds'])

    predictions = []
    start_idx = len(df_train)
    for i in range(start_idx, len(df_full)):
        # Gera features para toda a série (treino + previsão)
        df_iter = create_features(df_full[:i], lag_end=12, rolling_window=6)

        # Seleciona a linha mais recente (o ponto a ser previsto)
        X_predict = df_iter.iloc[-1][X_cols].to_frame().T

        # Faz a previsão do próximo passo
        next_pred = model.predict(X_predict)[0]
        predictions.append(next_pred)

        # Atualiza df_full com a previsão para usá-la como input (lag) na próxima iteração
        df_full.loc[i, 'y'] = next_pred

        # 6. Formatação do Resultado
    forecast_series = pd.Series(predictions, index=future_dates, name='Previsão RF')
    forecast_series.index.name = 'Date'

    return forecast_series, model

def lstm_forecast(series, n_periods=12):
    """
    Metodo de previsao machine learning: LSTM.

    Parametros:
    - series (pd.Series): Série temporal univariada.
    - n_periods (int): Número de períodos futuros a serem previstos.

    Retorna:
    - forecast (pd.Series): Série contendo as previsões futuras.
    - model: O modelo LSTM ajustado.
    """

    if len(series.dropna()) < n_periods + 2:
        return pd.Series(dtype='float64'), None  # Não há dados suficientes para sequências

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = series_to_sequences(scaled_data.flatten(), n_periods)

    if X.shape[0] == 0:
        return pd.Series(dtype='float64'), None

    # Reshape input para ser [samples, time steps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    X = torch.from_numpy(X).float().unsqueeze(2)
    y = torch.from_numpy(y).float()

    model = LSTMModel(input_size=1, hidden_layer_size=50, output_size=1)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(25):  # Usando 25 épocas para velocidade no loop
        for seq, labels in zip(X, y):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

    predictions = []
    # Sequência de entrada atual (últimos n_steps do treino)
    current_input = scaled_data[-n_periods:].flatten()

    for _ in range(n_periods):
        # Prever o próximo passo
        seq = torch.from_numpy(current_input).float()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        with torch.no_grad():
            next_pred_scaled = model(seq).item()

        predictions.append(next_pred_scaled)

        # Atualizar o input (retira o primeiro valor e anexa a previsão)
        current_input = np.roll(current_input, -1)
        current_input[-1] = next_pred_scaled

    # 6. Escala Inversa (Converter de volta para os valores monetários originais)
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions).flatten()

    # 7. Criar índice futuro
    last_date = series.index[-1]
    freq_inferida = 'MS'
    future_dates = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq_inferida)[1:]

    forecast_series = pd.Series(predictions, index=future_dates, name='Previsão LSTM')
    forecast_series.index.name = 'Date'  # Garantindo o nome do índice

    return forecast_series, model


def gru_forecast(series, n_periods=12):
    """
    Metodo de previsao machine learning: GRU.

    Parametros:
    - series (pd.Series): Série temporal univariada.
    - n_periods (int): Número de períodos futuros a serem previstos.

    Retorna:
    - forecast (pd.Series): Série contendo as previsões futuras.
    - model: O modelo GRU ajustado.
    """
    if len(series.dropna()) < n_periods + 2:
        return pd.Series(dtype='float64'), None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = series_to_sequences(scaled_data.flatten(), n_periods)

    if X.shape[0] == 0:
        print("Dados insuficientes para criar sequências GRU.")
        return pd.Series(dtype='float64'), None

    X_train = torch.from_numpy(X).float()
    y_train = torch.from_numpy(y).float()

    model = GRUmodel(input_size=1, hidden_layer_size=50, output_size=1)
    loss_function = nn.MSELoss()
    # learning rate do modelo é passada aqui
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(25):
        # Re-inicializa o hidden state no início de cada época (boas práticas)
        model.hidden_cell = torch.zeros(1, 1, model.hidden_layer_size)

        for i in range(len(X_train)):
            optimizer.zero_grad()

            # Input para forward deve ser [seq_len, batch_size, input_size]
            input_seq = X_train[i].view(n_periods, 1, 1)

            # CRUCIAL: Detach para quebrar o histórico de gradientes da sequência anterior
            # Isso impede o RuntimeError: Trying to backward through the graph a second time.
            model.hidden_cell = model.hidden_cell.detach()

            y_pred = model(input_seq)

            loss = loss_function(y_pred, y_train[i].unsqueeze(0))
            loss.backward()
            optimizer.step()

    model.eval()
    predictions = []
    current_input_cpu = scaled_data[-n_periods:].flatten()
    current_input = torch.from_numpy(current_input_cpu).float()  #
    with torch.no_grad():
        for _ in range(n_periods):
            # Desanexar o hidden cell para que o loop de previsão use apenas o estado
            # sem tentar acumular mais histórico de gradientes.
            model.hidden_cell = model.hidden_cell.detach()

            # Prever o próximo passo
            input_seq = current_input.view(n_periods, 1, 1)

            next_pred_scaled = model(input_seq).item()
            predictions.append(next_pred_scaled)

            # Atualizar o input (retira o primeiro valor e anexa a previsão)
            new_input = np.append(current_input_cpu[1:], next_pred_scaled)
            current_input_cpu = new_input
            current_input = torch.from_numpy(new_input).float()  # Sem .to(device)

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions).flatten()

    last_date = series.index[-1]
    freq_inferida = 'MS'
    future_dates = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq_inferida)[1:]

    forecast_series = pd.Series(predictions, index=future_dates, name=f'Previsão GRU')
    forecast_series.index.name = 'Date'  # Garantindo o nome do índice

    return forecast_series, model

def transformer_forecast(series, n_periods=12):
    """
    Metodo de previsao machine learning: Transformer.

    Parametros:
    - series (pd.Series): Série temporal univariada.
    - n_periods (int): Número de períodos futuros a serem previstos.

    Retorna:
    - forecast (pd.Series): Série contendo as previsões futuras.
    - model: O modelo Transformer ajustado.
    """

    #Parametros
    input_dim = 1
    d_model = 128
    nhead = 8
    num_layers = 3
    n_epochs = 150

    if len(series.dropna()) < n_periods + 2:
        return pd.Series(dtype='float64'), None

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = series_to_sequences(scaled_data.flatten(), n_periods)

    X_train = torch.from_numpy(X).float()
    y_train = torch.from_numpy(y).float()

    X_train_transpose = X_train.transpose(0,1).unsqueeze(2)

    model = TimeSeriesTransformer(input_dim, d_model, nhead, num_layers, dropout=0.3)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        y_pred = model(X_train_transpose)
        loss = loss_function(y_pred, y_train)

        loss.backward()
        optimizer.step()

    model.eval()
    predictions = []
    current_input_cpu = scaled_data[-n_periods:].flatten()

    with torch.no_grad():
        for _ in range(n_periods):
            input_seq = torch.from_numpy(current_input_cpu).float().view(n_periods, 1, 1)

            next_pred_scaled = model(input_seq).item()
            predictions.append(next_pred_scaled)

            np.roll(current_input_cpu, -1)
            current_input_cpu[-1] = next_pred_scaled

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions).flatten()

    last_date = series.index[-1]
    freq_inferida = 'MS'
    future_dates = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq_inferida)[1:]

    forecast_series = pd.Series(predictions, index=future_dates, name='Previsão Transformer')
    forecast_series.index.name = 'Date'
    return forecast_series, model


def conv_transformer_forecast(series, n_periods=12):
    """
    Método de previsão baseado na arquitetura Transformer com Embedding Convolucional.

    Parámetros:
    - series (pd.Series): Série temporal univariada.
    - n_periods (int): Número de períodos futuros a serem previstos.

    Retorna:
    - forecast (pd.Series): Série contendo as previsões futuras.
    - model: O modelo Transformer ajustado.
    """
    # Parâmetros otimizados para estabilidade e capacidade
    INPUT_DIM = 1
    D_MODEL = 128
    NHEAD = 8
    NUM_LAYERS = 3
    N_EPOCHS = 200
    LEARNING_RATE = 0.0005
    MAX_NORM = 1.0

    # 1. Validação de Dados
    if len(series.dropna()) < n_periods + 1:
        return pd.Series(dtype='float64'), None

    # 2. Pré-processamento e Sequenciamento
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(series.values.reshape(-1, 1))

    # X: [samples, seq_len], y: [samples]
    X, y = series_to_sequences(scaled_data.flatten(), n_periods)

    X_train = torch.from_numpy(X).float()
    y_train = torch.from_numpy(y).float()

    # Redimensiona X_train para [seq_len, samples, input_dim] para o Transformer
    X_train_transposed = X_train.transpose(0, 1).unsqueeze(2)

    # 3. Inicialização do Modelo (USANDO EMBEDDING CONVOLUCIONAL)
    model = TimeSeriesTransformer(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=0.1
    )
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Treinamento
    model.train()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()

        y_pred = model(X_train_transposed)
        loss = loss_function(y_pred, y_train)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)

        optimizer.step()

    # 5. Previsão Iterativa (Walk-Forward)
    model.eval()
    predictions = []
    current_input_cpu = scaled_data[-n_periods:].flatten()  # Última sequência de treino

    with torch.no_grad():
        for _ in range(n_periods):
            # Prepara a sequência de input para o Transformer
            # [seq_len] -> [seq_len, 1, 1]
            input_seq = torch.from_numpy(current_input_cpu).float().view(n_periods, 1, 1)

            # Preve o próximo passo
            next_pred_scaled = model(input_seq).item()
            predictions.append(next_pred_scaled)

            # Atualiza o input (desliza a janela e adiciona a previsão)
            current_input_cpu = np.roll(current_input_cpu, -1)
            current_input_cpu[-1] = next_pred_scaled

    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions).flatten()


    last_date = series.index[-1]
    freq_inferida = 'MS'
    future_dates = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq_inferida)[1:]

    forecast_series = pd.Series(predictions, index=future_dates, name='Previsão Conv-Transformer')
    forecast_series.index.name = 'Date'

    return forecast_series, model


def generate_forecast_chart(serie_historica: pd.Series, serie_previsoes: pd.Series, nome_produto: str, uf: str, model_name: str, rmse):
    """
    Cria um DataFrame longo combinando histórico e previsão e gera um gráfico Plotly interativo.

    Parâmetros:
    - serie_historica (pd.Series): Série temporal histórica.
    - serie_previsoes (pd.Series): Série temporal de previsões.
    - nome_produto (str): Nome do produto para o título.
    - uf (str): Unidade Federativa (Estado) para o título.
    - model_name (str): Nome do modelo utilizado para previsão.
    - rmse (float): Valor do RMSE para exibir no título.

    Retorna:
    - fig: Objeto Plotly figure representando o gráfico de linha.

    """

    # 1. PREPARAÇÃO DO DATAFRAME LONGO

    # Adicionar coluna 'Tipo' para distinguir as séries
    df_historico = serie_historica.reset_index()
    df_historico.columns = ['Date', 'Value']
    df_historico['Tipo'] = 'Histórico'

    df_previsao = serie_previsoes.reset_index()
    df_previsao.columns = ['Date', 'Value']
    df_previsao['Tipo'] = 'Previsão'

    # Concatenar: O resultado está no formato longo pronto para o Plotly
    df_longo_final = pd.concat([df_historico, df_previsao], ignore_index=True)

    # Usar string para 'Date' (conforme seu código original, mas Plotly trata bem datetime também)
    df_longo_final['Date'] = df_longo_final['Date'].astype(str)

    # 2. CRIAR O GRÁFICO DE LINHA INTERATIVO

    cores = {'Histórico': 'blue', 'Previsão': 'red'}

    fig = px.line(
        df_longo_final,
        x='Date',
        y='Value',
        color='Tipo',  # Cor diferente para 'Histórico' e 'Previsão'
        color_discrete_map=cores,
        title=f"Previsão da Série do Estado {uf}: {nome_produto} usando o modelo: {model_name} (RMSE: {rmse:.2f}) ",
        labels={'Date': 'Data', 'Value': "Preço", 'Tipo': 'Série'}
    )

    # 3. CUSTOMIZAÇÕES

    # Linha tracejada para a previsão
    fig.update_traces(
        line=dict(dash='dash', width=3),
        selector=dict(name='Previsão')
    )

    # Customizações do seu código original
    fig.update_xaxes(
        rangeslider=dict(visible=True),
        type="date"
    )

    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        legend_title="Tipo de Série"
    )

    return fig

def error_comparison_table(df: pd.DataFrame, test_periods: int, n_periods: int):
    """
    Calcula o RSME das funcoes de previsao (ARIMA, ETS, PROPHET) e gera uma tabela comparativa.

    Parametros:
    - df (pd.DataFrame): DataFrame contendo as séries temporais para análise.
    - test_periods (int): Número de períodos a serem usados para teste/validação.
    - n_periods (int): Número de períodos futuros a serem previstos.

    Retorna:
    - Arquivo .xlsx para comparação dos modelos.
    - pd.DataFrame: DataFrame contendo os valores de RSME para cada modelo e série temporal.
    """
    results = {}
    states = df.columns.tolist()
    for state in states:
        series = df[state]
        series.name = state
        if len(series.dropna()) <= test_periods:
            print(f"AVISO: Série para {state} tem dados insuficientes. Pulando.")
            continue

        test_series = series[-test_periods:]
        train_series = series[:-test_periods]
        results[state] = {}

        forecast_arima, _ = auto_arima_forecast(train_series, seasonal=True, m=12, n_periods=n_periods)
        forecast_arima.index = test_series.index

        forecast_ets, _ = ets_forecast(train_series, n_periods=n_periods, m=12, trend='add', seasonal_model='add')
        forecast_ets.index = test_series.index


        forecast_prophet, _ = prophet_forecast(train_series, n_periods=n_periods, m=12)
        forecast_prophet.index = test_series.index

        forecast_random_forest, _ = random_forest(train_series, n_periods=n_periods)
        forecast_random_forest.index = test_series.index

        forecast_lstm, _ = lstm_forecast(train_series, n_periods=n_periods)
        forecast_lstm.index = test_series.index

        forecast_gru, _ = gru_forecast(train_series, n_periods=n_periods)
        forecast_gru.index = test_series.index

        forecast_transformer, _ = conv_transformer_forecast(train_series, n_periods=n_periods)
        forecast_transformer.index = test_series.index


        results[state]['RMSE_ARIMA'] = calculate_rmse(test_series, forecast_arima)
        results[state]['RMSE_ETS'] = calculate_rmse(test_series, forecast_ets)
        results[state]['RMSE_PROPHET'] = calculate_rmse(test_series, forecast_prophet)
        results[state]['RMSE_RANDOM_FOREST'] = calculate_rmse(test_series, forecast_random_forest)
        results[state]['RMSE_LSTM'] = calculate_rmse(test_series, forecast_lstm)
        results[state]['RMSE_GRU'] = calculate_rmse(test_series, forecast_gru)
        results[state]['RMSE_TRANSFORMER'] = calculate_rmse(test_series, forecast_transformer)
        # print("ARIMA ", forecast_arima)
        # print("RF", forecast_random_forest)
        # print("TEST", test_series)
        # print("RMSE_RANDOM_FOREST:", calculate_rmse(test_series, forecast_random_forest))


        results[state]['MAPE_ARIMA'] = calculate_mape(test_series, forecast_arima)
        results[state]['MAPE_ETS'] = calculate_mape(test_series, forecast_ets)
        results[state]['MAPE_PROPHET'] = calculate_mape(test_series, forecast_prophet)
        results[state]['MAPE_RANDOM_FOREST'] = calculate_mape(test_series, forecast_random_forest)
        results[state]['MAPE_LSTM'] = calculate_mape(test_series, forecast_lstm)
        results[state]['MAPE_GRU'] = calculate_mape(test_series, forecast_gru)
        results[state]['MAPE_TRANSFORMER'] = calculate_mape(test_series, forecast_transformer)

    df_rmse_table = pd.DataFrame(results).T
    df_rmse_table.index.name = "UF"
    return df_rmse_table

def sliding_error_chart(series, model_func, model_name, test_periods=12, n_periods=12):
    """
        Calcula o RMSE Médio para cada passo (Horizonte 1, Horizonte 2, ..., Horizonte N)
        usando a Validação Cruzada Deslizante (Walk-Forward).
        """

    # Lista para armazenar o erro ABSOLUTO (ou QUADRADO) de CADA PASSO em CADA JANELA
    all_squared_errors = []

    min_train_size = 20

    # Range: O índice onde o conjunto de teste termina
    start_index = len(series) + 1
    stop_index= start_index - 12

    for end_idx in range(start_index, stop_index, -1):
        # O conjunto de teste é sempre o último 'test_periods'
        test_series = series[end_idx - test_periods: end_idx]
        # O conjunto de treino é tudo antes do conjunto de teste
        train_series = series[: end_idx - test_periods]

        try:
            # 1. Faz a previsão
            forecast, _ = model_func(train_series, n_periods=test_periods)

            if len(test_series) != len(forecast):
                continue

            # 2. Alinha os índices para comparação ponto-a-ponto
            aligned_forecast = pd.Series(forecast.values, index=test_series.index)

            # 3. Calcula o Erro Quadrático (Squared Error) para CADA PASSO
            squared_error = (test_series - aligned_forecast) ** 2
            print("actual", test_series)
            print("predicted", aligned_forecast)
            # print("error", squared_error)

            # Adiciona o array de 12 erros quadráticos (Passo 1 ao Passo 12) à lista
            all_squared_errors.append(squared_error.values)

        except Exception as e:
            print(f"Erro no passo de validação ({end_idx}): {type(e).__name__}")
            print(e)
            continue

    if not all_squared_errors:
        return pd.Series(dtype='float64')

    # 4. Agrega os resultados
    # Converte a lista de arrays em uma matriz (n_windows x test_periods)
    error_matrix = np.array(all_squared_errors)

    # Calcula a média (mean) para CADA COLUNA (Horizonte) e depois a raiz quadrada (RMSE)

    # print("matriz", error_matrix)
    mean_squared_error_by_horizon = np.mean(error_matrix, axis=0)
    # print("medias dos erros", mean_squared_error_by_horizon)
    rmse_by_horizon = np.sqrt(mean_squared_error_by_horizon)

    # 5. Formata a saída
    horizonte_labels = [f"Passo {i + 1}" for i in range(test_periods)]
    rmse_series = pd.Series(rmse_by_horizon, index=horizonte_labels, name=f'RMSE Médio - {model_name}')

    return rmse_series

def generate_sliding_chart(rmse_series: pd.Series, model_name: str, product_name: str, uf: str):
    """
    Gera um gráfico de barras Plotly para visualizar o RMSE médio em cada horizonte
    de previsão (Passo 1, Passo 2, ..., Passo N).
    """
    if rmse_series.empty:
        print("Série de RMSE vazia, não é possível gerar o gráfico.")
        return px.scatter(title="Dados insuficientes para gerar o gráfico de erro deslizante.")

    df_plot = rmse_series.reset_index()
    df_plot.columns = ['Horizonte', 'RMSE']

    # Ordenar pelo número do horizonte
    try:
        df_plot['Horizonte_Num'] = df_plot['Horizonte'].apply(lambda x: int(x.split(' ')[1]))
        df_plot = df_plot.sort_values('Horizonte_Num')
    except:
        # Fallback se a formatação 'Horizonte N' falhar
        pass

    fig = px.line(
        df_plot,
        x='Horizonte',
        y='RMSE',
        title=f"RMSE Média por Passo de Previsão - Modelo: {model_name} - Produto: {product_name} - Estado: {uf}",
        labels={'Horizonte': 'Horizonte de Previsão', 'RMSE': 'RMSE Médio'},
        markers=True
    )

    # fig = px.bar(
    #     df_plot,
    #     x='Horizonte',
    #     y='RMSE',
    #     title=f"RMSE Médio por Passo de Previsão - Modelo: {model_name} - Produto: {product_name} - Estado: {uf}",
    #     labels={'Horizonte': 'Horizonte de Previsão', 'RMSE': 'RMSE Médio'},
    # )
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        xaxis=dict(tickmode='linear')
    )
    # fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    # fig.update_yaxes(rangemode='tozero')

    return fig


def _model_forecast_arima(series, n_periods=12):
    """Wrapper para auto_arima_forecast com parâmetros padrão."""
    return auto_arima_forecast(series, seasonal=True, m=12, n_periods=n_periods)


def _model_forecast_ets(series, n_periods=12):
    """Wrapper para ets_forecast com parâmetros padrão."""
    return ets_forecast(series, n_periods=n_periods, m=12, trend='add', seasonal_model='add')


def _model_forecast_prophet(series, n_periods=12):
    """Wrapper para prophet_forecast com parâmetros padrão."""
    return prophet_forecast(series, seasonal=True, m=12, n_periods=n_periods)


def sliding_rmse_boxplots(
    df,
    product_name="",
    test_periods=12,
    n_periods=12,
    min_train_size=36,
    max_windows=12,
    output_path=None,
):
    """
    Calcula o erro (actual - predicted) para cada passo da previsão em validação deslizante
    (walk-forward) para todos os modelos e estados, e gera 12 boxplots (um por passo).

    Em cada gráfico há 7 boxplots (um por modelo), onde cada boxplot mostra a distribuição
    dos erros de previsão naquele passo específico, agregando todos os estados e janelas
    deslizantes.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com Datas como índice e Estados (UFs) como colunas.
    - product_name (str): Nome do produto para os títulos dos gráficos.
    - test_periods (int): Número de períodos de teste por janela (horizonte de previsão).
    - n_periods (int): Mesmo que test_periods, usado na chamada dos modelos.
    - min_train_size (int): Tamanho mínimo da série de treino para cada janela.
    - max_windows (int): Número máximo de janelas deslizantes por estado (padrão 12).
    - output_path (str, opcional): Caminho base para salvar os HTMLs (ex: 'sliding_rmse_step').
                                   Se None, não salva arquivos.

    Retorna:
    - list: Lista de até 12 figuras Plotly, uma para cada passo de previsão.
    """
    MODEL_FUNCS = {
        "ARIMA": _model_forecast_arima,
        "ETS": _model_forecast_ets,
        "Prophet": _model_forecast_prophet,
        "Random Forest": random_forest,
        "LSTM": lstm_forecast,
        "GRU": gru_forecast,
        "Transformer": conv_transformer_forecast,
    }

    # Estrutura: errors_by_step[step_idx][model_name] = lista de erros (todos os estados e janelas)
    errors_by_step = {step: {m: [] for m in MODEL_FUNCS} for step in range(test_periods)}
    states = df.columns.tolist()

    for state in states:
        series = df[state].dropna()
        if len(series) < min_train_size + test_periods:
            continue

        # Janelas: end_idx indica o fim do conjunto de teste; fazemos no máximo max_windows janelas
        start_end = len(series)
        stop_end = max(min_train_size + test_periods - 1, start_end - max_windows)

        for end_idx in range(start_end, stop_end, -1):
            if end_idx - test_periods < min_train_size:
                break

            test_series = series.iloc[end_idx - test_periods : end_idx]
            train_series = series.iloc[: end_idx - test_periods]

            for model_name, model_func in MODEL_FUNCS.items():
                try:
                    forecast, _ = model_func(train_series, n_periods=test_periods)
                    if forecast is None or len(forecast) != len(test_series):
                        continue

                    aligned_forecast = pd.Series(forecast.values, index=test_series.index)
                    errors = test_series.values - aligned_forecast.values

                    for step in range(test_periods):
                        if step < len(errors):
                            errors_by_step[step][model_name].append(errors[step])
                except Exception as e:
                    continue

    # Gerar 12 figuras (uma por passo)
    figures = []
    for step in range(test_periods):
        # Preparar dados em formato longo para Plotly
        rows = []
        for model_name in MODEL_FUNCS:
            errs = errors_by_step[step][model_name]
            for e in errs:
                rows.append({"Model": model_name, "Erro": e})

        if not rows:
            continue

        df_plot = pd.DataFrame(rows)

        fig = go.Figure()
        for model_name in MODEL_FUNCS:
            vals = df_plot[df_plot["Model"] == model_name]["Erro"].values
            if len(vals) > 0:
                fig.add_trace(
                    go.Box(
                        x=[model_name] * len(vals),
                        y=vals,
                        name=model_name,
                        boxpoints="outliers",
                        jitter=0.3,
                        pointpos=-1.8,
                    )
                )

        fig.update_layout(
            title=f"Passo {step + 1} - Distribuição do Erro de Previsão por Modelo"
            + (f" ({product_name})" if product_name else ""),
            title_x=0.5,
            xaxis_title="Modelo",
            yaxis_title="Erro (Real - Previsto)",
            template="plotly_white",
            height=500,
            width=900,
            showlegend=False,
        )
        fig.update_xaxes(tickangle=45)
        figures.append(fig)

        if output_path:
            out = f"{output_path}_passo{step + 1:02d}.html"
            fig.write_html(out)

    return figures


# --- Execução Principal ---
if __name__ == "__main__":
    EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"
    SHEET_NAME = "ARROZ"
    OUTPUT_HTML_PATH = 'previsao_interativa.html'

    TEST_PERIODS = 12
    FORECAST_PERIODS = 12
    MODEL_TO_RUN = "error_comparison_table"
    # Opções: "Auto_Arima", "ETS", "Prophet", "error_comparison_table", "Random_Forest", "sliding_error_chart", "sliding_rmse_boxplots"
    # LSTM", "GRU", "Transformer"
    UF="MS"

    # 1. Carrega os dados
    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, index_col=0)

    # 2. Fixa o nome do índice (data)
    df.index.name = 'Date'

    # Seleciona uma série temporal específica (por exemplo, o primeiro estado)
    state = df.columns[5]
    series = df[UF]

    train_series = series[:-TEST_PERIODS]
    test_series = series[-TEST_PERIODS:]  # Valores reais que o modelo tentará prever

    func_map = {
        "Auto_Arima": auto_arima_forecast,
        "ETS": ets_forecast,
        "Prophet": prophet_forecast,
        "RandomForest": random_forest,
        "LSTM": lstm_forecast
    }

    if MODEL_TO_RUN == "Auto_Arima":
        model_name = "Auto ARIMA"
        prediction, model = auto_arima_forecast(train_series, seasonal=True, m=12, n_periods=12)
    elif MODEL_TO_RUN == "ETS":
        model_name = "ETS (Holt-Winters)"
        prediction, model = ets_forecast(train_series, m=12, n_periods=12, trend='add', seasonal_model='add')
    elif MODEL_TO_RUN == "Prophet":
        model_name = "Prophet"
        prediction, model = prophet_forecast(train_series, seasonal=True, m=12, n_periods=12)
    elif MODEL_TO_RUN == "error_comparison_table":
        df_rmse = error_comparison_table(df, test_periods=TEST_PERIODS, n_periods=FORECAST_PERIODS)
        print("\n--- TABELA COMPARATIVA DE RMSE DOS MODELOS ---")
        df_rmse.to_excel(f"{MODEL_TO_RUN}_modelos.xlsx")
        exit(0)
    elif MODEL_TO_RUN == "Random_Forest":
        model_name = "Random Forest"
        prediction, model = random_forest(train_series, n_periods=12)
    elif MODEL_TO_RUN == "sliding_error_chart":
        model_name = "random_forest"
        model_func = random_forest
        error_values = sliding_error_chart(series,model_func, model_name, test_periods=TEST_PERIODS, n_periods=FORECAST_PERIODS)
        fig = generate_sliding_chart(error_values, model_name, SHEET_NAME, UF)
        fig.write_html(OUTPUT_HTML_PATH, auto_open=True)
        exit(0)
    elif MODEL_TO_RUN == "sliding_rmse_boxplots":
        figures = sliding_rmse_boxplots(
            df,
            product_name=SHEET_NAME,
            test_periods=TEST_PERIODS,
            n_periods=FORECAST_PERIODS,
            min_train_size=36,
            max_windows=12,
            output_path="sliding_rmse_boxplot",
        )
        if figures:
            figures[0].write_html("sliding_rmse_boxplot_passo01.html", auto_open=True)
            for i, fig in enumerate(figures[1:], start=2):
                fig.write_html(f"sliding_rmse_boxplot_passo{i:02d}.html")
            print(f"Gerados {len(figures)} gráficos de boxplot (passo 1 a {len(figures)}).")
        exit(0)
    elif MODEL_TO_RUN == "LSTM":
        model_name = "LSTM"
        prediction, model = lstm_forecast(train_series, n_periods=12)
    elif MODEL_TO_RUN == "GRU":
        model_name = "GRU"
        prediction, model = gru_forecast(train_series, n_periods=12)
    elif MODEL_TO_RUN == "Transformer":
        model_name = "Transformer"
        prediction, model = conv_transformer_forecast(train_series, n_periods=12)
    else:
        raise ValueError(f"Modelo desconhecido: {MODEL_TO_RUN}")

    if len(test_series) == len(prediction):
        prediction.index = test_series.index

    rmse = calculate_rmse(test_series, prediction)

    print(f"\n--- VALIDAÇÃO DO MODELO {model_name} ---")
    print(f"Série Analisada: {UF} do produto {SHEET_NAME}")
    print(f"Períodos de Teste (Validação): {TEST_PERIODS}")
    print(f"RMSE (Root Mean Square Error): {rmse:.2f}")

    fig = generate_forecast_chart(series,prediction, SHEET_NAME,  uf=UF, model_name=model_name, rmse=rmse)
    fig.write_html(OUTPUT_HTML_PATH, auto_open=True)

