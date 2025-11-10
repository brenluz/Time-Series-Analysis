import pandas as pd
import plotly.express as px
import pmdarima as pm
from sklearn.ensemble import RandomForestRegressor
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
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    pred_array = model.predict(n_periods=n_periods)
    last_date = series.index[-1]
    freq_inferida = 'MS'
    if freq_inferida is None:
        raise ValueError(
            "Não foi possível inferir a frequência da série temporal. Certifique-se de que o índice de data está configurado corretamente (ex: df.index.freq='MS').")

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
    df_train = series.reset_index()
    df_train.columns = ['ds', 'y']

    df_features = create_features(df_train, lag_start=1, lag_end=12, rolling_window=6)
    df_features = df_features.dropna(subset=['y'])

    X_cols = [col for col in df_features.columns if col not in ['ds', 'y']]
    X_train = df_features[X_cols]
    y_train = df_features['y']

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    last_date = series.index[-1]
    freq_inferida = 'MS'

    # Cria as datas futuras
    future_dates = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq_inferida)[1:]
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

    return forecast_series, model

def generate_forecast_chart(serie_historica: pd.Series, serie_previsoes: pd.Series, nome_produto: str, uf: str, value_name: str, model_name: str, rmse):
    """
    Cria um DataFrame longo combinando histórico e previsão e gera um gráfico Plotly interativo.
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
        labels={'Date': 'Data', 'Value': value_name, 'Tipo': 'Série'}
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

        results[state]['RMSE_ARIMA'] = calculate_rmse(test_series, forecast_arima)
        results[state]['RMSE_ETS'] = calculate_rmse(test_series, forecast_ets)
        results[state]['RMSE_PROPHET'] = calculate_rmse(test_series, forecast_prophet)
        results[state]['RMSE_RANDOM_FOREST'] = calculate_rmse(test_series, forecast_random_forest)

        results[state]['MAPE_ARIMA'] = calculate_mape(test_series, forecast_arima)
        results[state]['MAPE_ETS'] = calculate_mape(test_series, forecast_ets)
        results[state]['MAPE_PROPHET'] = calculate_mape(test_series, forecast_prophet)
        results[state]['MAPE_RANDOM_FOREST'] = calculate_mape(test_series, forecast_random_forest)
    df_rmse_table = pd.DataFrame(results).T
    df_rmse_table.index.name = "UF"
    return df_rmse_table

def sliding_rmse_chart(series, model_func, model_name, test_periods=12, n_periods=12):

    rmse_values = {}
    dates = []

    for end_idx in range(len(series) - test_periods, len(series)):
        train_series = series[:end_idx - test_periods]
        test_series = series[end_idx - test_periods:end_idx]

        forecast, _ = model_func(train_series, n_periods=n_periods)
        forecast.index = test_series.index

        rmse = calculate_rmse(test_series, forecast)
        rmse_values.append(rmse)
        dates.append(test_series.index[-1])

    rmse_df = pd.DataFrame({'Date': dates, 'RMSE': rmse_values})

    fig = px.line(
        rmse_df,
        x='Date',
        y='RMSE',
        title=f'Evolução do RMSE ao Longo do Tempo - Modelo: {model_name}',
        labels={'Date': 'Data', 'RMSE': 'RMSE'}
    )

    return fig

# --- Execução Principal ---
if __name__ == "__main__":
    EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"
    SHEET_NAME = "ARROZ"
    OUTPUT_HTML_PATH = 'previsao_interativa.html'

    TEST_PERIODS = 12
    FORECAST_PERIODS = 12
    MODEL_TO_RUN ="sliding_rmse_chart"  # Opções: "Auto_Arima", "ETS", "Prophet", "error_comparison_table", "Random_Forest"
    UF="BA"

    # 1. Carrega os dados
    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, index_col=0)

    # 2. Fixa o nome do índice (data)
    df.index.name = 'Date'

    # Seleciona uma série temporal específica (por exemplo, o primeiro estado)
    state = df.columns[5]
    series = df[UF]

    train_series = series[:-TEST_PERIODS]
    test_series = series[-TEST_PERIODS:]  # Valores reais que o modelo tentará prever

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
        df_rmse.to_excel("modelos.xlsx")
        exit(0)
    elif MODEL_TO_RUN == "Random_Forest":
        model_name = "Random Forest"
        prediction, model = random_forest(train_series, n_periods=12)
    elif MODEL_TO_RUN == "sliding_rmse_chart":
        model_name = "Auto ARIMA"
        fig = sliding_rmse_chart(series, auto_arima_forecast, model_name, test_periods=TEST_PERIODS, n_periods=FORECAST_PERIODS)
        fig.write_html(OUTPUT_HTML_PATH, auto_open=True)
        exit(0)
    else:
        raise ValueError(f"Modelo desconhecido: {MODEL_TO_RUN}")

    if len(test_series) == len(prediction):
        prediction.index = test_series.index

    rmse = calculate_rmse(test_series, prediction)

    print(f"\n--- VALIDAÇÃO DO MODELO {model_name} ---")
    print(f"Série Analisada: {UF} do produto {SHEET_NAME}")
    print(f"Períodos de Teste (Validação): {TEST_PERIODS}")
    print(f"RMSE (Root Mean Square Error): {rmse:.2f}")

    fig = generate_forecast_chart(series,prediction, SHEET_NAME,  uf=UF,value_name="Preço", model_name="ARIMA", rmse=rmse)
    fig.write_html(OUTPUT_HTML_PATH, auto_open=True)

