import pandas as pd

def remove_zeros_from_excel(file_path):
    """
    Loads an Excel file and removes rows where the 'Preço medio' column has a value of zero.

    Args:
        file_path (str): The path to your Excel file.

    Returns:
        pd.DataFrame: A new DataFrame with rows containing zero in 'Preço medio' removed.
    """
    try:
        output_file_name = '../Databases/DatabaseConabv5.xlsx'
        excel_data = pd.read_excel(file_path, dtype=str, sheet_name=None)

        with pd.ExcelWriter(output_file_name, engine='openpyxl') as writer:
            for sheet_name, df in excel_data.items():
                if not isinstance(sheet_name, str):
                    continue
                print(f"Processing sheet: {sheet_name}")
                # Replace '0' and '0,00' with empty strings
                df.replace(['0', '0,00'], "", inplace=True)
                # Save the modified DataFrame to the output file
                df.to_excel(writer, sheet_name=sheet_name, index=False, float_format="%.2f".replace('.', ','))


    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return None
    except KeyError as e:
        print(f"Error: A column was not found. Please check your column names. Missing key: {e}")
        return None

remove_zeros_from_excel('../Databases/DatabaseConabv4.xlsx')