import pandas as pd

def create_detailed_pivot_table_from_excel(file_path):
    """
    Reads an Excel file, forward-fills key columns, and creates a pivot table
    that is organized first by UF, then by Comercialization Level, and finally by Date.

    Args:
        file_path (str): The path to your Excel file.

    Returns:
        pd.DataFrame: A new DataFrame with the detailed pivoted data.
    """
    try:
        # 1. Load the data from the Excel file
        df = pd.read_excel(file_path)

        # Make sure column names are correct for your spreadsheet
        product_col = 'Produto'
        date_col = 'Período'
        price_col = 'Preço medio'
        comercialization_col = 'Nível de comercialização'
        uf_col = 'UF'

        # 2. Forward-fill the data for 'Product', 'Comercialization Level', and 'UF'
        columns_to_fill = [product_col, comercialization_col, uf_col]
        df[columns_to_fill] = df[columns_to_fill].ffill()

        # 3. Clean and prepare the data
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df.dropna(subset=[price_col], inplace=True)

        # Corrigindo o aviso de formato de data
        # Se as suas datas estiverem em outro formato, ajuste a string de formato.
        df[date_col] = pd.to_datetime(df[date_col], format='%m/%Y')
        df['YearMonth'] = df[date_col].dt.to_period('M')

        # 4. Create the detailed PivotTable with the new order
        # A ordem do index agora é: UF, Comercialization Level, YearMonth
        pivot_df = pd.pivot_table(
            df,
            index=[uf_col, comercialization_col, 'YearMonth'],
            columns=product_col,
            values=price_col,
            aggfunc='sum'
        )

        # Ordena o index para uma visualização mais limpa
        pivot_df = pivot_df.sort_index()

        return pivot_df

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return None
    except KeyError as e:
        print(f"Error: A column was not found. Please check your column names. Missing key: {e}")
        return None
    except ValueError as e:
        print(
            f"Error: Failed to parse dates. Please check the date format in your spreadsheet and the `format` parameter in the script. Original error: {e}")
        return None


file = '../Databases/Database.xlsx'

# Run the function and store the result
pivoted_data = create_detailed_pivot_table_from_excel(file)

# Display the final pivoted DataFrame
if pivoted_data is not None:
    print(pivoted_data)

# Save the new pivot table to a new Excel file
pivoted_data.to_excel('DatabaseOpt2.xlsx')