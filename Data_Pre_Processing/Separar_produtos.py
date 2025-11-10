import pandas as pd
import numpy as np
from datetime import date, timedelta


def process_avg_prices_data(input_excel_path, output_excel_path):
    """
    Reads the AVGPRICES worksheet from the input Excel file, reshapes the data,
    and saves it to a new Excel file with a separate sheet for each product.

    The new sheets will have UFs as columns and a complete date range (2014-01 to 2024-12) as rows.

    Args:
        input_excel_path (str): Path to the input Excel file.
        output_excel_path (str): Path where the output Excel file will be saved.
    """
    try:
        # --- 1. Load the data from the AVGPRICES worksheet ---
        # The first two lines of the worksheet are part of a complex header.
        # We need to read it in and then clean it up to get the correct column names.
        df = pd.read_excel(input_excel_path, sheet_name='AVGPRICES', header=[0, 1])

        # Drop the second row of headers, as it's redundant.
        df.columns = df.columns.droplevel(1)

        # Explicitly rename the first two columns to handle cases where they might be unnamed
        # or have inconsistent names in the raw data.
        df.columns.values[0] = 'UF'
        df.columns.values[1] = 'YearMonth'

        # --- 2. Clean and fill the UF column ---
        # The UF column only has a value for the first row of each state.
        # We need to forward-fill these values to the subsequent rows.
        df['UF'] = df['UF'].ffill()

        # Drop any rows where 'YearMonth' is NaN. This is a more reliable way
        # to remove header rows or invalid entries while keeping all valid UF data.
        df.dropna(subset=['YearMonth'], inplace=True)

        # --- 3. Melt the DataFrame into a long format ---
        # Unpivot the table to have a single column for products and a single column for prices.
        # 'UF' and 'YearMonth' are the identifier variables.
        product_columns = df.columns.drop(['UF', 'YearMonth'])
        df_melted = df.melt(id_vars=['UF', 'YearMonth'], value_vars=product_columns,
                            var_name='Product', value_name='Price')

        # --- 4. Create a complete date range for the new sheets ---
        start_date = '2014-01-01'
        end_date = '2024-12-31'
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        full_date_range_str = full_date_range.strftime('%Y-%m')

        # --- 5. Get unique products and UFs ---
        products = df_melted['Product'].unique()
        ufs = df_melted['UF'].unique()

        # Create a Pandas ExcelWriter object to save multiple sheets
        with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
            print(f"Generating new Excel file at {output_excel_path}...")

            # --- 6. Iterate through each product and create a new sheet ---
            for product in products:
                # Filter the melted data for the current product
                df_product = df_melted[df_melted['Product'] == product].copy()

                # Pivot the data to get UFs as columns and dates as the index
                df_pivoted = df_product.pivot(index='YearMonth', columns='UF', values='Price')

                # Reindex the pivoted table to the full date range.
                # This ensures every sheet has the complete date range, with NaN for missing values.
                df_final = df_pivoted.reindex(full_date_range_str)

                # Write the final DataFrame to a new sheet in the Excel file.
                # The sheet name is the product's name.
                df_final.to_excel(writer, sheet_name=str(product))
                print(f"  - Sheet '{product}' created successfully.")

        print("Data processing complete. The new Excel file has been saved.")

    except FileNotFoundError:
        print(f"Error: The file at {input_excel_path} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- Main execution ---
if __name__ == "__main__":
    # Define the input and output file paths.
    input_file = "../Databases/DatabaseConabv2.xlsx"
    output_file = "../Databases/DatabaseConabv4.xlsx"

    # Call the function to perform the data transformation.
    process_avg_prices_data(input_file, output_file)
