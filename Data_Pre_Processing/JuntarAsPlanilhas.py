import pandas as pd

# Lista para armazenar os DataFrames de cada planilha
todos_os_dfs = []

# Nomes dos seus arquivos Excel
nomes_dos_arquivos = ['Consulta-precos-mensal.xlsx', 'Consulta-precos-mensal (1).xlsx', 'Consulta-precos-mensal (2).xlsx']

cabecalho = 10

for arquivo in nomes_dos_arquivos:
    try:
        # Ler cada planilha para um DataFrame
        df = pd.read_excel(arquivo, header=cabecalho + 1)
        todos_os_dfs.append(df)
        print(f"Planilha '{arquivo}' lida com sucesso.")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{arquivo}' não foi encontrado.")
    except Exception as e:
        print(f"Erro ao ler o arquivo '{arquivo}': {e}")

# Concatenar todos os DataFrames em um único DataFrame
# axis=0 significa que você está empilhando as linhas
df_final = pd.concat(todos_os_dfs, ignore_index=True)

# Salvar o DataFrame combinado em um novo arquivo Excel
df_final.to_excel('Database.xlsx', index=False)

print("\nPlanilhas combinadas com sucesso!")
print("As primeiras 5 linhas do DataFrame combinado:")
print(df_final.head())
print("\nO DataFrame combinado foi salvo em 'precos_combinados.xlsx'")

