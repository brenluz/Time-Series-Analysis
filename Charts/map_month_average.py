import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import traceback
from datetime import datetime

# --- Configuração de Nomes e GeoJSON ---
# Nomes dos meses em Português para os títulos
MONTH_NAMES_PT = {
    1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
    5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
    9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
}

# Link GeoJSON mais estável
BRAZIL_GEOJSON = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'


def generate_monthly_grid_map(df_wide, product_name, value_name):
    """
    Agrega a série temporal por MÊS e gera uma grade estática (3x4) de 12 mapas
    choropleth, um para o preço médio de cada mês.

    Parameters:
    - df_wide (pd.DataFrame): DataFrame com Datas como índice e Estados (UFs) como colunas.
    - product_name (str): Nome do produto para o título.
    - value_name (str): Rótulo para a legenda de cores (e.g., 'Preço Médio (R$)').

    Returns:
    - fig: Objeto Plotly figure contendo a grade de mapas.
    """

    # 1. PREPARAÇÃO E AGREGAÇÃO DE DADOS (Reutilizando a lógica anterior)
    df_numeric = df_wide.select_dtypes(include=['number'])

    if df_numeric.empty:
        raise ValueError("Nenhuma coluna numérica encontrada após a leitura da planilha.")

    df_long = df_numeric.reset_index().melt(
        id_vars='Date', var_name='UF', value_name='Value'
    )

    df_long['Month'] = df_long['Date'].dt.month

    # Calcula a média mensal dos preços
    df_monthly = df_long.groupby(['Month', 'UF'])['Value'].mean().reset_index()

    # 2. ESCALA GLOBAL DE CORES
    # Encontra o valor mínimo (cmin) e máximo (cmax) global para todos os 12 meses.
    # Isso é CRUCIAL para garantir que todos os 12 mapas usem a mesma legenda de cores.
    cmin = df_monthly['Value'].min()
    cmax = df_monthly['Value'].max()

    # 3. CRIAÇÃO DA GRADE DE SUBPLOTS
    # 3 linhas por 4 colunas (3x4 = 12 meses)
    fig = make_subplots(
        rows=3,
        cols=4,
        specs=[[{'type': 'choropleth'}] * 4] * 3,
        subplot_titles=[MONTH_NAMES_PT[i] for i in range(1, 13)],
        vertical_spacing=0.01,
        horizontal_spacing=0.01
    )

    # 4. ITERAÇÃO E DESENHO DOS MAPAS
    for i in range(1, 13):
        # Mapeia o número do mês (i) para a posição da linha (row) e coluna (col) na grade
        row = (i - 1) // 4 + 1
        col = (i - 1) % 4 + 1

        # Filtra os dados apenas para o mês atual
        df_month = df_monthly[df_monthly['Month'] == i]

        # Cria o mapa choropleth
        trace = go.Choropleth(
            geojson=BRAZIL_GEOJSON,
            locations=df_month['UF'],
            z=df_month['Value'],
            featureidkey='properties.sigla',
            colorscale='Reds',  # Gradiente Vermelho
            autocolorscale=False,
            showscale=False,  # Não mostrar a legenda em cada subplot (será adicionada globalmente)
            # Define a escala de cor (cmin/cmax) para o valor GLOBAL
            zmin=cmin,
            zmax=cmax,
            # Configuração de cor para valores ausentes (NaNs)
            marker_line_color='gray',
            marker_line_width=0.5,
            # Força a cor branca para NaN
            showlegend=False,
            # A cor de NaN é definida no layout.coloraxis, mas este trace força os limites.
        )

        # Adiciona o mapa ao subplot correto
        fig.add_trace(trace, row=row, col=col)

        # Ajusta a configuração geográfica de cada subplot para focar no Brasil
        fig.update_geos(
            row=row, col=col,
            fitbounds="locations",
            visible=False,
            scope="south america",
            subunitcolor="black"
        )

    # 5. CONFIGURAÇÃO FINAL DA FIGURA
    # Adiciona uma única barra de cores global no centro da figura
    fig.update_layout(
        title_text=f"Média Mensal de Preços: {product_name} (Análise Sazonal)",
        title_x=0.5,
        height=1000,  # Altura ajustada para 3 linhas de mapas
        width=1200,  # Largura para 4 colunas
        coloraxis=dict(
            colorscale='Reds',
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                title=value_name,
                len=0.9,  # Ajusta o comprimento da barra de cores
                x=1.05,  # Posição x
                y=0.5,  # Posição y
            )
        )
    )

    # Remove os rótulos de eixo de todos os subplots para limpar a visualização
    fig.update_annotations(font_size=16)  # Título de cada mês

    return fig


# --- Execução Principal ---
if __name__ == "__main__":
    EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"  # Use o seu caminho real
    SHEET_NAME = "ARROZ"  # Exemplo, troque pelo seu sheet
    OUTPUT_HTML_PATH = '../arroz_map_grid_mensal.html'

    try:
        # 1. Carrega os dados e corrige o índice de data
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'

        print(f"Iniciando a geração da grade de mapas mensais para o produto: {SHEET_NAME}")

        # 2. Gera a grade de mapas
        fig = generate_monthly_grid_map(
            df_wide=df,
            product_name=SHEET_NAME,
            value_name="Preço Médio (R$)"
        )

        # 3. Salva a figura interativa
        fig.write_html(OUTPUT_HTML_PATH, auto_open=True)
        print(f"\nGrade de mapas mensais salva em: {os.path.abspath(OUTPUT_HTML_PATH)}")

    except FileNotFoundError:
        print(f"\nError: O arquivo '{EXCEL_FILE_PATH}' não foi encontrado. Por favor, verifique o caminho do arquivo.")
    except Exception as e:
        print("\n--- OCORREU UM ERRO INESPERADO ---")
        print(f"Tipo de Erro: {type(e).__name__}")
        print(f"Mensagem de Erro: {e}")
        print("\nTraceback (use para localizar a linha da falha):")
        traceback.print_exc()
