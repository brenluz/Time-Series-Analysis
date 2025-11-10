import pandas as pd

def forward_fill_data(file_path):
    """
    Loads an Excel file and forward-fills data for 'Product', 'Comercialization Level',
    and 'UF' columns, assuming these values are only present in the first row of each block.

    Args:
        file_path (str): The path to your Excel file.

    Returns:
        pd.DataFrame: A new DataFrame with the filled data.
    """
    try:
        # 1. Read the Excel file into a DataFrame
        # Assumes the first row contains headers.
        df = pd.read_excel(file_path)

        # 2. Identify the columns to be filled.
        # Ensure these column names match your spreadsheet exactly.
        columns_to_fill = ['Produto', 'Nível de comercialização', 'UF']

        # 3. Apply the forward fill (ffill) method to the specified columns.
        # This will copy the value from the last non-empty cell downwards.
        df[columns_to_fill] = df[columns_to_fill].ffill()

        return df

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return None
    except KeyError as e:
        print(f"Error: A column was not found. Please check your column names. Missing key: {e}")
        return None


# --- Example Usage ---

# Replace 'your_spreadsheet.xlsx' with the actual path to your file
file = '../Databases/Database.xlsx'

# Run the function to get the filled DataFrame
filled_df = forward_fill_data(file)

# Display the resulting DataFrame
if filled_df is not None:
    print("Original DataFrame (first 10 rows):")
    print("-----------------------------------")
    print(pd.read_excel(file).head(10))
    print("\nDataFrame after forward filling (first 10 rows):")
    print("------------------------------------------------")
    print(filled_df.head(10))

    # Optional: Save the new, filled DataFrame to a new Excel file
filled_df.to_excel('filled_data.xlsx', index=False)