import pandas as pd

def linear_interpolation(file_path):
    """
    Loads an Excel file and performs linear interpolation to fill missing values
    in the 'Preço medio' column.

    Args:
        file_path (str): The path to your Excel file.

    Returns:
        pd.DataFrame: A new DataFrame with missing values in 'Preço medio' filled using linear interpolation.
    """
    try:
        # 1. Read the Excel file into a DataFrame
        df_sheet = pd.read_excel(file_path, sheet_name=None, index_col=0)
    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None

    # 2. Perform linear interpolation on each sheet
    interpolated_sheets = {}

    for sheet_name, wdf in df_sheet.items():
        print(f"\n--- Processing Sheet: {sheet_name} ---")

        print("Missing values (NaNs) before interpolation:")
        # Display the count of NaNs for each column (state)
        print(wdf.isna().sum())

        # --- NEW LOGIC: Identify and filter columns ---
        # Identify columns that are NOT entirely empty (i.e., have at least one non-NaN value).

        # We are replacing the concise pandas masking with a clearer, procedural loop
        # to improve understanding of which columns should be interpolated.
        cols_to_interpolate = []
        cols_to_skip = []

        # Iterate over every column (state)
        for col in wdf.columns:
            # Check if the column has at least one non-missing value
            # .notna() returns True for non-missing values. .any() checks if any True exists.
            if wdf[col].notna().any():
                cols_to_interpolate.append(col)
            else:
                # The entire column is missing (all NaNs)
                cols_to_skip.append(col)

        if cols_to_skip:
            print(f"\n[INFO] Skipping interpolation for entirely empty columns in {sheet_name}: {cols_to_skip}")

        # Create a copy to ensure interpolation is applied to the correct subset
        wdf_interpolated = wdf.copy()

        if cols_to_interpolate:
            # Perform linear interpolation ONLY on columns that have data.
            wdf_interpolated[cols_to_interpolate] = wdf[cols_to_interpolate].interpolate(
                method='linear',
                limit_direction='both'
            )
        else:
            print(f"\n[WARNING] Sheet {sheet_name} contains no columns with data. Skipping interpolation entirely.")

        interpolated_sheets[sheet_name] = wdf_interpolated

        print(f"\nMissing values (NaNs) after interpolation for {sheet_name}:")
        print(wdf_interpolated.isna().sum())

    return interpolated_sheets

# --- Main execution ---

if __name__ == "__main__":
    input_excel_path = '../Databases/DatabaseConabv4.xlsx'
    output_excel_path = '../Databases/DatabaseConabv5.xlsx'

    print(f"Starting linear interpolation process for file: {input_excel_path}")
    result_df = linear_interpolation(input_excel_path)

    if result_df is not None:
        try:
            with pd.ExcelWriter(output_excel_path) as writer:
                for sheet_name, df in result_df.items():
                    # Write each DataFrame to a sheet named after the original sheet
                    df.to_excel(writer, sheet_name=sheet_name)

        except Exception as e:
            print(f"\nAn error occurred while writing the Excel file: {e}")