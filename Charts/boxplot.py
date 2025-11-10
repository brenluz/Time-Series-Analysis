import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import traceback

MONTH_NAMES_PT = {
    1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
    5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
    9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
}


def generate_monthly_boxplot_grid(df_wide, product_name, value_name):
    """
    Agrega a série temporal por MÊS e gera uma grade estática (3x4) de 12 boxplots,
    um para cada mês, mostrando a distribuição de preços por Estado (UF).

    Parameters:
    - df_wide (pd.DataFrame): DataFrame com Datas como índice e Estados (UFs) como colunas.
    - product_name (str): Nome do produto para o título.
    - value_name (str): Rótulo para o eixo Y (e.g., 'Preço Médio (R$)').

    Returns:
    - fig: Objeto Plotly figure contendo a grade de boxplots.
    """

    # 1. PREPARAÇÃO E AGREGAÇÃO DE DADOS
    # Seleciona apenas colunas numéricas (que representam os preços)
    df_numeric = df_wide.select_dtypes(include=['number'])

    if df_numeric.empty:
        raise ValueError("Nenhuma coluna numérica encontrada após a leitura da planilha.")

    # Converte o formato wide para long (empilha os estados)
    df_long = df_numeric.reset_index().melt(
        id_vars='Date', var_name='UF', value_name='Value'
    )

    df_long['Month'] = df_long['Date'].dt.month

    # 2. ESCALA GLOBAL DE CORES E Y-AXIS
    # Encontra os valores mínimo (ymin) e máximo (ymax) globais para garantir que
    # todos os 12 boxplots usem a mesma escala Y para fácil comparação.
    global_min = df_long['Value'].min() * 0.95
    global_max = df_long['Value'].max() * 1.05

    # 3. CRIAÇÃO DA GRADE DE SUBPLOTS
    # 3 linhas por 4 colunas (3x4 = 12 meses)
    fig = make_subplots(
        rows=3,
        cols=4,
        # Define o tipo de subplot como 'xy' (padrão para gráficos de dispersão/caixa)
        specs=[[{'type': 'xy'}] * 4] * 3,
        subplot_titles=[MONTH_NAMES_PT[i] for i in range(1, 13)],
        vertical_spacing=0.15,
        horizontal_spacing=0.03
    )

    # 4. ITERAÇÃO E DESENHO DOS BOXPLOTS
    for i in range(1, 13):
        # Mapeia o número do mês (i) para a posição da linha (row) e coluna (col) na grade
        row = (i - 1) // 4 + 1
        col = (i - 1) % 4 + 1

        # Filtra os dados apenas para o mês atual
        df_month = df_long[df_long['Month'] == i]

        # O Boxplot será agrupado por 'UF' (x) e terá 'Value' (y) como a série de dados
        trace = go.Box(
            x=df_month['UF'],  # Estados no eixo X
            y=df_month['Value'],  # Preços no eixo Y
            name=MONTH_NAMES_PT[i],
            boxpoints='outliers',  # Mostra apenas outliers
            jitter=0.3,  # Espalhamento para pontos
            pointpos=-1.8,  # Posição dos pontos
            showlegend=False,
            # Define o nome do traço para a legenda, embora showlegend seja False,
            # é bom para identificação interna.
            hovertemplate='<b>UF:</b> %{x}<br><b>Preço:</b> R$%{y:.2f}<extra></extra>',
        )

        # Adiciona o boxplot ao subplot correto
        fig.add_trace(trace, row=row, col=col)

        # Ajusta o layout do eixo Y (vertical) para garantir a escala global consistente
        fig.update_yaxes(
            title_text=value_name if col == 1 else None,  # Apenas o primeiro de cada linha tem label
            row=row, col=col,
            range=[global_min, global_max],  # Escala Y consistente
            automargin=True
        )

        # Ajusta o layout do eixo X (horizontal)
        fig.update_xaxes(
            title_text='Estado (UF)' if row == 3 else None,  # Apenas o último de cada coluna tem label
            row=row, col=col,
            tickangle=45,  # Rotação dos rótulos para melhor leitura
            automargin=True
        )

    # 5. CONFIGURAÇÃO FINAL DA FIGURA
    fig.update_layout(
        title_text=f"Análise Sazonal da Distribuição de Preços: {product_name} por Mês e UF",
        title_x=0.5,
        height=1200,  # Altura ajustada
        width=1400,  # Largura ajustada
        template="plotly_white",  # Fundo branco para melhor contraste
        # Opções globais de layout para eixos (apenas para fallback, pois update_yaxes/update_xaxes já foram usados)
        # uniformtext_minsize=8,
        # uniformtext_mode='hide'
    )

    # Aumenta o tamanho da fonte dos títulos dos subplots (nomes dos meses)
    fig.update_annotations(font_size=14, yshift=5)

    return fig


# --- Execução Principal (Exemplo) ---
if __name__ == "__main__":
    # O seu ambiente de execução exige que você forneça o caminho correto.
    # Exemplo de caminho de arquivo (AJUSTE CONFORME NECESSÁRIO)
    EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"
    SHEET_NAME = "ARROZ"
    OUTPUT_HTML_PATH = 'arroz_boxplot_grid_mensal.html'
    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'

    try:
        print(f"Iniciando a geração da grade de boxplots mensais para o produto: {SHEET_NAME}")

        # 2. Gera a grade de boxplots
        fig = generate_monthly_boxplot_grid(
            df_wide=df,
            product_name=SHEET_NAME,
            value_name="Preço Médio (R$)"
        )

        # 3. Salva a figura interativa
        # Atenção: O parâmetro auto_open só funciona em ambientes que suportam a abertura automática.
        fig.write_html(OUTPUT_HTML_PATH, auto_open=True)
        print(f"\nGrade de boxplots mensais salva em: {os.path.abspath(OUTPUT_HTML_PATH)}")


    except Exception as e:
        print("\n--- OCORREU UM ERRO INESPERADO ---")
        print(f"Tipo de Erro: {type(e).__name__}")
        print(f"Mensagem de Erro: {e}")
        print("\nTraceback (use para localizar a linha da falha):")
        traceback.print_exc()
