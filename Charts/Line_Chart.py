import os
import pandas as pd
import plotly.express as px
import traceback


def generate_line_chart(df_wide, sheet_name, value_name):
    """
    Transforma o DataFrame de série temporal (estados em colunas) para o formato longo
    e gera um gráfico de linha interativo Plotly que compara todos os estados.

    Parameters:
    - df_wide (pd.DataFrame): DataFrame com Datas como índice e Estados (UFs) como colunas.
    - sheet_name (str): Nome do produto (para o título).
    - value_name (str): Rótulo dos valores (e.g., 'Preço do Produto (R$)').

    Returns:
    - fig: Objeto Plotly figure representando o gráfico de linha.
    """

    # 1. FILTRAR E PREPARAR DADOS
    df_numeric = df_wide.select_dtypes(include=['number'])

    if df_numeric.empty:
        print("\nErro: Nenhuma coluna numérica encontrada para plotar.")
        raise ValueError("No numeric data columns (state values) were found.")

    # 2. TRANSFORMAR: Converter para o formato longo para Plotly Express
    # O Plotly Express funciona melhor com dados no formato longo (colunas: Date, UF, Value)
    df_long = df_numeric.reset_index().melt(
        id_vars="Date",
        var_name='UF',
        value_name='Value'
    )

    # Garantir que a data seja reconhecida como o eixo X (o formato string está ok)
    df_long['Date'] = df_long['Date'].astype(str)

    # Remover NaNs para o gráfico de linha, se necessário.
    # O Plotly trata NaNs interrompendo a linha, o que é útil para visualização de missing values.
    # Vamos manter todos os pontos para que as linhas sejam contínuas onde houver dados.

    print("\n--- Gerando Gráfico de Linha com os 5 Primeiros Pontos ---")
    print(df_long.head())

    # 3. CRIAR O GRÁFICO DE LINHA INTERATIVO
    fig = px.line(
        df_long,
        x='Date',
        y='Value',
        color='UF',  # Cor diferente para cada estado (UF)
        title=f"Série Temporal de Preços do Produto: {sheet_name}",
        labels={'Date': 'Data', 'Value': value_name, 'UF': 'Estado'}
    )

    # Adicionar um slider de intervalo (range slider) para facilitar o zoom no tempo
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1a", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    )

    fig.update_layout(
        title_x=0.5,
        title_font_size=24,
        legend_title="Estados"
    )

    return fig


# --- Execução Principal ---
if __name__ == "__main__":
    EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"
    SHEET_NAME = "ARROZ"
    OUTPUT_HTML_PATH = 'line_chart.html'

    try:
        # 1. Carrega os dados (Feito em uma única chamada para consistência)
        xls = pd.ExcelFile(EXCEL_FILE_PATH)
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, index_col=0)

        # 2. Fixa o nome do índice (data)
        df.index.name = 'Date'

        print(f"Iniciando a geração do gráfico de linha para o produto: {SHEET_NAME}")

        # 3. Gera o gráfico de linha
        fig = generate_line_chart(
            df_wide=df,
            sheet_name=SHEET_NAME,
            value_name="Preço do Produto (R$)"
        )

        # 4. Salva o gráfico interativo
        fig.write_html(OUTPUT_HTML_PATH, auto_open=True)
        print(f"\nGráfico de linha interativo salvo em: {os.path.abspath(OUTPUT_HTML_PATH)}")

    except FileNotFoundError:
        print(f"\nErro: O arquivo '{EXCEL_FILE_PATH}' não foi encontrado. Por favor, verifique o caminho do arquivo.")
    except Exception as e:
        print("\n--- OCORREU UM ERRO INESPERADO ---")
        print(f"Tipo de Erro: {type(e).__name__}")
        print(f"Mensagem de Erro: {e}")
        print("\nTraceback (use para localizar a linha da falha):")
        traceback.print_exc()
