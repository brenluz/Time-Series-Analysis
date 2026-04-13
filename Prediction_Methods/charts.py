import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def generate_forecast_chart(serie_historica: pd.Series,
                             serie_previsoes: pd.Series,
                             nome_produto: str,
                             uf: str,
                             model_name: str,
                             rmse: float) -> go.Figure:
    """
    Line chart comparing historical series with model forecast.
    Historical series is solid blue; forecast is dashed red.
    """
    df_hist = serie_historica.reset_index()
    df_hist.columns = ['Date', 'Value']
    df_hist['Type'] = 'Historical'

    df_prev = serie_previsoes.reset_index()
    df_prev.columns = ['Date', 'Value']
    df_prev['Type'] = 'Forecast'

    df_long = pd.concat([df_hist, df_prev], ignore_index=True)
    df_long['Date'] = df_long['Date'].astype(str)

    fig = px.line(
        df_long, x='Date', y='Value', color='Type',
        color_discrete_map={'Historical': 'blue', 'Forecast': 'red'},
        title=f"Forecast — {uf}: {nome_produto} | {model_name} (RMSE: {rmse:.2f})",
        labels={'Date': 'Date', 'Value': 'Price', 'Type': 'Series'},
    )
    fig.update_traces(line=dict(dash='dash', width=3), selector=dict(name='Forecast'))
    fig.update_xaxes(rangeslider=dict(visible=True), type='date')
    fig.update_layout(title_x=0.5, title_font_size=20, legend_title='Series Type')
    return fig


def generate_sliding_chart(rmse_series: pd.Series,
                            model_name: str,
                            product_name: str,
                            uf: str) -> go.Figure:
    """
    Line chart of mean RMSE per forecast horizon step (sliding window validation).
    """
    if rmse_series.empty:
        return px.scatter(title="Insufficient data.")

    df_plot = rmse_series.reset_index()
    df_plot.columns = ['Horizon', 'RMSE']

    fig = px.line(
        df_plot, x='Horizon', y='RMSE', markers=True,
        title=f"RMSE by Step — {model_name} | {product_name} | {uf}",
        labels={'Horizon': 'Forecast Horizon', 'RMSE': 'Mean RMSE'},
    )
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        xaxis=dict(tickmode='linear'),
    )
    return fig