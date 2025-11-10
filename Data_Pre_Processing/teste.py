import pandas as pd
import numpy as np


def create_worksheets_by_comercialization_level(file_path):
    """
    Reads an Excel file, creates a pivot table for each unique 'Comercialization Level',
    and saves each pivot table to a separate worksheet in a single Excel file.

    Args:
        file_path (str): The path to your original Excel file.
    """
    try:
        # 1. Carregar os dados e preencher os valores
        df = pd.read_excel(file_path)

        # Certifique-se de que os nomes das colunas estão corretos
        product_col = 'Produto'
        date_col = 'Período'
        price_col = 'Preço medio'
        comercialization_col = 'Nível de comercialização'
        uf_col = 'UF'

        # Preenche os valores ausentes com o 'ffill'
        columns_to_fill = [product_col, comercialization_col, uf_col]
        df[columns_to_fill] = df[columns_to_fill].ffill()

        # Limpa e prepara as colunas de data e preço
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df.dropna(subset=[price_col], inplace=True)
        df[date_col] = pd.to_datetime(df[date_col])
        df['YearMonth'] = df[date_col].dt.to_period('M')

        # 2. Encontrar todos os níveis de comercialização únicos
        unique_levels = df[comercialization_col].unique()
        if len(unique_levels) > 3:
            print(
                f"Atenção: Existem mais de 3 níveis de comercialização ({len(unique_levels)}). Todas serão incluídas.")

        output_file_name = 'teste.xlsx'

        # 3. Usar ExcelWriter para escrever em várias planilhas
        print(f"Criando o arquivo '{output_file_name}' com múltiplas planilhas...")
        with pd.ExcelWriter(output_file_name, engine='xlsxwriter') as writer:
            for level in unique_levels:
                print(f"  > Criando planilha para: {level}")

                # Filtra o DataFrame para o nível de comercialização atual
                df_filtered = df[df[comercialization_col] == level]

                # Cria a tabela dinâmica para o nível filtrado
                pivot_df = pd.pivot_table(
                    df_filtered,
                    index=[uf_col, 'YearMonth'],
                    columns=product_col,
                    values=price_col,
                    aggfunc='sum'
                )

                # Formata o nome da planilha para evitar caracteres inválidos
                safe_sheet_name = level.replace('/', '-')

                # Salva a tabela dinâmica na planilha específica dentro do arquivo
                pivot_df.to_excel(writer, sheet_name=safe_sheet_name)

        print("\nProcesso concluído. O arquivo foi salvo com sucesso.")

    except FileNotFoundError:
        print(f"Erro: O arquivo em '{file_path}' não foi encontrado.")
    except KeyError as e:
        print(f"Erro: Uma coluna não foi encontrada. Verifique os nomes das colunas. Chave ausente: {e}")

file = '../Databases/Database.xlsx'

# Execute a função
create_worksheets_by_comercialization_level(file)