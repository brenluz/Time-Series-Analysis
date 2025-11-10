import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import traceback
from datetime import datetime
import math # Importar math para calcular a disposição da grade

# --- Configuração de Nomes e GeoJSON ---
# Nomes dos meses em Português (mantido, mas não usado na nova função)
MONTH_NAMES_PT = {
    1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
    5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
    9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
}

# Link GeoJSON mais estável
BRAZIL_GEOJSON = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'


def generate_yearly_grid_map(df_wide, product_name, value_name):
    """
    Agrega a série temporal por ANO e gera uma grade estática de mapas choropleth,
    um para o preço médio de cada ano.

    Parameters:
    - df_wide (pd.DataFrame): DataFrame com Datas como índice e Estados (UFs) como colunas.
    - product_name (str): Nome do produto para o título.
    - value_name (str): Rótulo para a legenda de cores (e.g., 'Preço Médio (R$)').

    Returns:
    - fig: Objeto Plotly figure contendo a grade de mapas anuais.
    """

    # 1. PREPARAÇÃO E AGREGAÇÃO DE DADOS
    df_numeric = df_wide.select_dtypes(include=['number'])

    if df_numeric.empty:
        raise ValueError("Nenhuma coluna numérica encontrada após a leitura da planilha.")

    df_long = df_numeric.reset_index().melt(
        id_vars='Date', var_name='UF', value_name='Value'
    )

    # *** MUDANÇA CHAVE: Usar o Ano em vez do Mês ***
    df_long['Year'] = df_long['Date'].dt.year

    # Calcula a média anual dos preços
    df_yearly = df_long.groupby(['Year', 'UF'])['Value'].mean().reset_index()

    # Obtém a lista de anos e o número total de anos
    unique_years = sorted(df_yearly['Year'].unique())
    num_years = len(unique_years)

    if num_years == 0:
        raise ValueError("Nenhum dado com ano válido encontrado.")

    # 2. ESCALA GLOBAL DE CORES
    cmin = df_yearly['Value'].min()
    cmax = df_yearly['Value'].max()

    # 3. CRIAÇÃO DA GRADE DE SUBPLOTS
    # Define a melhor disposição (ex: 4 colunas, número de linhas calculado)
    cols = 4
    rows = math.ceil(num_years / cols)

    print(f"Gerando grade de {rows} linhas x {cols} colunas para {num_years} anos.")

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{'type': 'choropleth'}] * cols] * rows,
        # *** MUDANÇA CHAVE: Usar o Ano como título do subplot ***
        subplot_titles=[str(year) for year in unique_years],
        vertical_spacing=0.01,
        horizontal_spacing=0.01
    )

    # 4. ITERAÇÃO E DESENHO DOS MAPAS
    for index, year in enumerate(unique_years):
        # Mapeia o índice (0, 1, 2...) para a posição da linha (row) e coluna (col)
        row = (index // cols) + 1
        col = (index % cols) + 1

        # Filtra os dados apenas para o ano atual
        df_year = df_yearly[df_yearly['Year'] == year]

        # Cria o mapa choropleth
        trace = go.Choropleth(
            geojson=BRAZIL_GEOJSON,
            locations=df_year['UF'],
            z=df_year['Value'],
            featureidkey='properties.sigla',
            colorscale='Reds',
            autocolorscale=False,
            showscale=False,
            # Define a escala de cor (cmin/cmax) para o valor GLOBAL
            zmin=cmin,
            zmax=cmax,
            marker_line_color='gray',
            marker_line_width=0.5,
            showlegend=False,
        )

        # Adiciona o mapa ao subplot correto
        fig.add_trace(trace, row=row, col=col)

        # Ajusta a configuração geográfica de cada subplot
        fig.update_geos(
            row=row, col=col,
            fitbounds="locations",
            visible=False,
            scope="south america",
            subunitcolor="black"
        )

    # 5. CONFIGURAÇÃO FINAL DA FIGURA
    fig.update_layout(
        # *** MUDANÇA CHAVE: Atualizar o título principal ***
        title_text=f"Média Anual de Preços: {product_name}",
        title_x=0.5,
        height=rows * 330,  # Altura ajustada dinamicamente
        width=1200,
        coloraxis=dict(
            colorscale='Reds',
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(
                title=value_name,
                len=0.9,
                x=1.05,
                y=0.5,
            )
        )
    )

    fig.update_annotations(font_size=16)

    return fig


if __name__ == "__main__":
    print("Hello world")
    EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"
    SHEET_NAME = "ARROZ"
    # *** MUDANÇA CHAVE: Nome do arquivo de saída ***
    OUTPUT_HTML_PATH = '../arroz_map_grid_anual.html'

    try:
        # 1. Carrega os dados e corrige o índice de data
        # O código original está correto para carregar os dados
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'

        print(f"Iniciando a geração da grade de mapas anuais para o produto: {SHEET_NAME}")

        # 2. Gera a grade de mapas usando a nova função
        fig = generate_yearly_grid_map(
            df_wide=df,
            product_name=SHEET_NAME,
            value_name="Preço Médio (R$)"
        )

        # 3. Salva a figura interativa
        fig.write_html(OUTPUT_HTML_PATH, auto_open=True)
        print(f"\nGrade de mapas anuais salva em: {os.path.abspath(OUTPUT_HTML_PATH)}")

    except FileNotFoundError:
        print(f"\nError: O arquivo '{EXCEL_FILE_PATH}' não foi encontrado. Por favor, verifique o caminho do arquivo.")
    except Exception as e:
        print("\n--- OCORREU UM ERRO INESPERADO ---")
        print(f"Tipo de Erro: {type(e).__name__}")
        print(f"Mensagem de Erro: {e}")
        print("\nTraceback (use para localizar a linha da falha):")
        traceback.print_exc()