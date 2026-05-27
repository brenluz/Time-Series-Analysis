import os
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import traceback
import math

# --- Configuração ---
BRAZIL_GEOJSON = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'

ALL_UFS = [
    'AC', 'AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO',
    'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR',
    'RJ', 'RN', 'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO'
]

BASE_Z = [0] * len(ALL_UFS)


def generate_yearly_grid_map(df_wide, product_name, value_name):
    """
    Agrega a série temporal por ANO e gera uma grade estática de mapas choropleth,
    um para o preço médio de cada ano. Estados sem dados aparecem em branco
    com o contorno claramente desenhado.

    Parameters:
    - df_wide (pd.DataFrame): DataFrame com Datas como índice e Estados (UFs) como colunas.
    - product_name (str): Nome do produto para o título.
    - value_name (str): Rótulo para a legenda de cores (e.g., 'Preço Médio (R$)').

    Returns:
    - fig: Objeto Plotly figure contendo a grade de mapas anuais.
    """

    df_numeric = df_wide.select_dtypes(include=['number'])
    if df_numeric.empty:
        raise ValueError("Nenhuma coluna numérica encontrada.")

    df_long = df_numeric.reset_index().melt(id_vars='Date', var_name='UF', value_name='Value')
    df_long['Year'] = df_long['Date'].dt.year
    df_yearly = df_long.groupby(['Year', 'UF'])['Value'].mean().reset_index()

    unique_years = sorted(df_yearly['Year'].unique())
    num_years = len(unique_years)
    if num_years == 0:
        raise ValueError("Nenhum dado com ano válido encontrado.")

    cmin = df_yearly['Value'].min()
    cmax = df_yearly['Value'].max()

    cols = 4
    rows = math.ceil(num_years / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{'type': 'choropleth'}] * cols] * rows,
        subplot_titles=[str(y) for y in unique_years],
        vertical_spacing=0.01,
        horizontal_spacing=0.01
    )

    for index, year in enumerate(unique_years):
        row = (index // cols) + 1
        col = (index % cols) + 1

        df_year = df_yearly[df_yearly['Year'] == year]

        # Camada base: todos os 27 estados em branco (garante contorno sempre visível)
        fig.add_trace(go.Choropleth(
            geojson=BRAZIL_GEOJSON,
            locations=ALL_UFS,
            z=BASE_Z,
            featureidkey='properties.sigla',
            colorscale=[[0, 'white'], [1, 'white']],
            autocolorscale=False,
            showscale=False,
            zmin=0, zmax=1,
            marker_line_color='gray',
            marker_line_width=0.5,
            showlegend=False,
            hoverinfo='skip',
        ), row=row, col=col)

        # Camada de dados: apenas estados com valor real
        df_year_valid = df_year.dropna(subset=['Value'])
        fig.add_trace(go.Choropleth(
            geojson=BRAZIL_GEOJSON,
            locations=df_year_valid['UF'],
            z=df_year_valid['Value'],
            featureidkey='properties.sigla',
            colorscale='Reds',
            autocolorscale=False,
            showscale=False,
            zmin=cmin, zmax=cmax,
            marker_line_color='gray',
            marker_line_width=0.5,
            showlegend=False,
        ), row=row, col=col)

        fig.update_geos(
            row=row, col=col,
            fitbounds="locations",
            visible=False,
            scope="south america",
            subunitcolor="black"
        )

    fig.update_layout(
        height=rows * 330,
        width=1200,
        coloraxis=dict(
            colorscale='Reds',
            cmin=cmin,
            cmax=cmax,
            colorbar=dict(title=value_name, len=0.9, x=1.05, y=0.5)
        )
    )
    fig.update_annotations(font_size=16)

    return fig


def safe_filename(name):
    """Converte o nome da aba em um nome de arquivo seguro."""
    return re.sub(r'[^\w\-]', '_', name).strip('_')


if __name__ == "__main__":
    EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"
    OUTPUT_DIR = '../maps_output'
    VALUE_LABEL = "Preço Médio (R$)"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lê os nomes de todas as abas do arquivo
    try:
        xl = pd.ExcelFile(EXCEL_FILE_PATH)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{EXCEL_FILE_PATH}' não encontrado.")
        raise SystemExit(1)

    sheet_names = xl.sheet_names
    print(f"Encontradas {len(sheet_names)} abas: {sheet_names}\n")

    failed = []

    for sheet in sheet_names:
        print(f"[{sheet}] Processando...")
        try:
            df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet, index_col=0)
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[df.index.notna()]  # Remove linhas com datas inválidas
            df.index.name = 'Date'

            if df.empty:
                print(f"  [SKIP] Aba '{sheet}' sem dados válidos.\n")
                continue

            fig = generate_yearly_grid_map(
                df_wide=df,
                product_name=sheet,
                value_name=VALUE_LABEL
            )

            out_path = os.path.join(OUTPUT_DIR, f"{safe_filename(sheet)}.html")
            fig.write_html(out_path)
            print(f"  [OK] Salvo em: {os.path.abspath(out_path)}\n")

        except Exception as e:
            print(f"  [ERRO] Aba '{sheet}' falhou: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed.append(sheet)
            print()

    print("=" * 50)
    print(f"Concluído. {len(sheet_names) - len(failed)}/{len(sheet_names)} produtos gerados.")
    if failed:
        print(f"Falhas: {failed}")
    print(f"HTMLs salvos em: {os.path.abspath(OUTPUT_DIR)}")