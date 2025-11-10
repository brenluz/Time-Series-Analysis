import os
import pandas as pd
import plotly.express as px
import traceback


def generate_brazil_map(df_wide, title, value_name):
    """
    Transforms the time series DataFrame (states as columns) into a long format
    and generates an ANIMATED choropleth map of Brazil with a time slider.

    Parameters:
    - df_wide (pd.DataFrame): DataFrame with Dates as index and States (UFs) as columns.
    - title (str): The title of the map.
    - value_name (str): The label for the values (e.g., 'Product Price ($)').

    Returns:
    - fig: A Plotly figure object representing the animated map.
    """

    # 1. FILTER: Select only numeric columns (the price data) to prevent errors during melt.
    df_numeric = df_wide.select_dtypes(include=['number'])

    if df_numeric.empty:
        print("\nError: After filtering, no numeric columns were found for plotting.")
        raise ValueError("No numeric data columns (state values) were found after reading the sheet.")

    # 2. TRANSFORM: Convert the DataFrame from wide format to long format.
    # The 'Date' index is converted into a column named 'Date' for melting.
    df_long = df_numeric.reset_index().melt(
        id_vars='Date',
        var_name='UF',
        value_name='Value'
    )

    # Ensure Date column is a string type for proper animation framing
    df_long['Date'] = df_long['Date'].astype(str)

    # Remove any rows where the original value was NaN (as Plotly cannot plot them)
    df_long.dropna(subset=['Value'], inplace=True)

    print("\n--- Diagnostic Check: Final Animated Data Structure (df_long.head()) ---")
    print("If this looks wrong, the data transformation failed.")
    print(df_long.head())

    # Load GeoJSON data for Brazilian states (Crucial for drawing boundaries)
    # FIX: Updated to a reliable, publicly accessible GeoJSON link.
    brazil_geojson = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson'

    # --- 3. Create the animated CHOROPLETH map (using GeoJSON for boundaries) ---
    fig = px.choropleth(
        df_long,
        geojson=brazil_geojson,
        locations='UF',  # Column in df with state abbreviations
        featureidkey='properties.sigla',  # Key in GeoJSON for state abbreviations
        color='Value',
        color_continuous_scale="Reds",
        hover_name='UF',
        hover_data={'Value': True},
        scope="south america",
        title=title,
        labels={'Value': value_name},

        # --- KEY FOR ANIMATION ---
        animation_frame='Date',
        animation_group='UF'  # Ensures the state features persist across frames
    )

    # Adjust map properties to fit Brazil and show state lines
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showsubunits=True,
        subunitcolor="black"
    )

    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        title_x=0.5,
        title_font_size=24
    )

    # Optional: Adjust the animation speed
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 750
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 250

    return fig


# --- Main execution ---
if __name__ == "__main__":
    # Define file paths and parameters
    EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"
    SHEET_NAME = "ARROZ"
    OUTPUT_HTML_PATH = '../arroz_map_animated.html'

    try:
        # Diagnostic Check: Read the file to see what sheets are actually present
        xls = pd.ExcelFile(EXCEL_FILE_PATH)
        sheet_names = xls.sheet_names

        # 1. Load the data (Dates as index, UFs as columns)
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, index_col=0)

        # --- ROBUST INDEX NAME FIX ---
        # Explicitly assign the index name to 'Date' for consistent melting.
        df.index.name = 'Date'

        # --- NEW DIAGNOSTIC PRINTS (for troubleshooting data structure) ---
        print("\n--- Diagnostic Check: DataFrame after reading Excel ---")
        print(f"Index Name: {df.index.name}")
        print(f"Columns (should be UF codes): {df.columns.tolist()}")
        print("First 5 rows (Index should be dates, columns should be states):")
        print(df.head())
        print("---------------------------------------------------------")

        # 2. Generate the animated map
        fig = generate_brazil_map(
            df_wide=df,
            title=f"Animated Product Price for {SHEET_NAME} Across Brazilian States",
            value_name="Product Price ($)"
        )

        # 3. Save the interactive map
        fig.write_html(OUTPUT_HTML_PATH, auto_open=True)
        print(f"\nInteractive animated map successfully saved to: {os.path.abspath(OUTPUT_HTML_PATH)}")

    except FileNotFoundError:
        print(f"\nError: The file '{EXCEL_FILE_PATH}' was not found. Please check the file path.")
    except Exception as e:
        # Print the full error and traceback to identify the exact cause
        print("\n--- AN UNEXPECTED ERROR OCCURRED ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {e}")
        print("\nTraceback (use this to locate the line where it failed):")
        traceback.print_exc()
