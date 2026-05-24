"""
map_brazil_rmse.py
------------------
For a chosen commodity, draws a filled-polygon choropleth map of Brazil
where each state is coloured by the best-performing model (lowest mean
RMSE across h=1..12).

Adapted directly from the existing generate_brazil_map pipeline.
Run locally — requires internet access for the GeoJSON boundary file.

Dependencies
------------
    pip install pandas plotly openpyxl kaleido
"""

import os
import traceback
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Configuration ─────────────────────────────────────────────────────────────
EXCEL_FILE_PATH = "sliding_rmse_all_products.xlsx"
COMMODITY       = "ACUCAR"          # ← change to any sheet name
OUTPUT_HTML     = f"map_best_model_{COMMODITY.lower()}.html"
OUTPUT_PNG      = f"map_best_model_{COMMODITY.lower()}.png"

MODELS = ['ARIMA', 'ETS', 'Prophet', 'Random Forest',
          'LSTM', 'GRU', 'Transformer', 'Informer']

# One distinct colour per model — colourblind-friendly palette
MODEL_COLORS = {
    'ARIMA':          '#4472C4',   # blue
    'ETS':            '#ED7D31',   # orange
    'Prophet':        '#70AD47',   # green
    'Random Forest':  '#FFC000',   # amber
    'LSTM':           '#FF6666',   # salmon
    'GRU':            '#00B0F0',   # cyan
    'Transformer':    '#7030A0',   # purple
    'Informer':       '#C00000',   # dark red
}

# Same GeoJSON source as your animated price map
BRAZIL_GEOJSON = (
    'https://raw.githubusercontent.com/codeforamerica/'
    'click_that_hood/master/public/data/brazil-states.geojson'
)

# ── 1. Load RMSE file and compute best model per state ────────────────────────
def load_best_model_per_state(filepath: str, sheet: str) -> pd.DataFrame:
    """
    Reads one sheet from the RMSE Excel file.
    Returns a DataFrame with one row per state and columns:
        UF, Best_Model, Min_RMSE, RMSE_ARIMA, RMSE_ETS, ...
    """
    df        = pd.read_excel(filepath, sheet_name=sheet, header=None)
    model_row = df.iloc[0].tolist()
    data      = df.iloc[2:].copy()
    data.columns = range(data.shape[1])
    data      = data[data[0].notna()].reset_index(drop=True)

    records = []
    for _, row in data.iterrows():
        state = row[0]
        rmse_per_model = {}
        for model in MODELS:
            col_indices = [i for i, v in enumerate(model_row) if v == model]
            vals = pd.to_numeric(row[col_indices], errors='coerce').values
            rmse_per_model[model] = round(float(np.nanmean(vals)), 4)

        best_model = min(rmse_per_model, key=rmse_per_model.get)
        best_rmse  = round(min(rmse_per_model.values()), 4)

        records.append({
            'UF':         state,
            'Best_Model': best_model,
            'Min_RMSE':   best_rmse,
            **{f'RMSE_{m}': v for m, v in rmse_per_model.items()}
        })

    return pd.DataFrame(records)


# ── 2. Build the choropleth map ───────────────────────────────────────────────
def generate_best_model_map(
    df_results: pd.DataFrame,
    commodity:  str,
    geojson_url: str,
) -> go.Figure:
    """
    Generates a filled-polygon choropleth of Brazil where each state is
    coloured by its best-performing model.

    Follows the same px.choropleth structure as generate_brazil_map()
    but uses a discrete colour map instead of a continuous colour scale.
    """

    # Build a rich hover string showing all model RMSEs for each state
    def build_hover(row):
        lines = [
            f"<b>{row['UF']}</b>",
            f"Best model: <b>{row['Best_Model']}</b>",
            f"Min mean RMSE: {row['Min_RMSE']:.4f} BRL/kg",
            "",
            "<i>All models (mean RMSE across h=1..12):</i>",
        ]
        for m in MODELS:
            marker = " ◀" if m == row['Best_Model'] else ""
            lines.append(f"  {m}: {row[f'RMSE_{m}']:.4f}{marker}")
        return "<br>".join(lines)

    df_results = df_results.copy()
    df_results['Hover'] = df_results.apply(build_hover, axis=1)

    # Add ALL 27 Brazilian states so every polygon is drawn by Plotly.
    # States missing from the RMSE data get Best_Model = "No Data" and a
    # neutral grey fill — without this, Plotly silently drops their polygons.
    ALL_STATES = [
        'AC','AL','AM','AP','BA','CE','DF','ES','GO','MA','MG','MS','MT',
        'PA','PB','PE','PI','PR','RJ','RN','RO','RR','RS','SC','SE','SP','TO'
    ]
    present_states = set(df_results['UF'].tolist())
    missing = [s for s in ALL_STATES if s not in present_states]

    if missing:
        rmse_cols = {f'RMSE_{m}': np.nan for m in MODELS}
        missing_rows = pd.DataFrame([
            {'UF': s, 'Best_Model': 'No Data',
             'Min_RMSE': np.nan, 'Hover': f'<b>{s}</b><br>No data available',
             **rmse_cols}
            for s in missing
        ])
        df_results = pd.concat([df_results, missing_rows], ignore_index=True)

    # Grey for states with no data
    MODEL_COLORS['No Data'] = '#D3D3D3'

    model_order = sorted(
        [m for m in df_results['Best_Model'].unique() if m != 'No Data']
    ) + (['No Data'] if 'No Data' in df_results['Best_Model'].values else [])

    fig = px.choropleth(
        df_results,
        geojson=geojson_url,
        locations='UF',                          # state abbreviation column
        featureidkey='properties.sigla',         # key in the GeoJSON
        color='Best_Model',
        color_discrete_map=MODEL_COLORS,
        category_orders={'Best_Model': model_order},
        hover_name='UF',
        custom_data=['Hover'],
        scope='south america',
        title=(
            f'Best-Performing Forecasting Model by State<br>'
            f'<sup>Commodity: {commodity.title()} — '
            f'colour = model with lowest mean RMSE (BRL/kg) across h=1..12</sup>'
        ),
    )

    # Replace the default hover with our custom tooltip
    fig.update_traces(
        hovertemplate='%{customdata[0]}<extra></extra>'
    )

    # Fit tightly to Brazil, show state borders in white
    fig.update_geos(
        fitbounds='locations',
        visible=False,
        showsubunits=True,
        subunitcolor='white',
        subunitwidth=1.0,
    )

    fig.update_layout(
        margin={'r': 0, 't': 90, 'l': 0, 'b': 0},
        title_x=0.5,
        title_font_size=15,
        title_font_family='Arial',
        paper_bgcolor='white',
        legend=dict(
            title=dict(text='Best Model', font=dict(size=12)),
            font=dict(size=11, family='Arial'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cccccc',
            borderwidth=1,
        ),
    )

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        print(f"Loading RMSE data — commodity: {COMMODITY}")
        df_results = load_best_model_per_state(EXCEL_FILE_PATH, COMMODITY)

        print("\n--- State results ---")
        print(df_results[['UF', 'Best_Model', 'Min_RMSE']].to_string(index=False))
        print(f"\nModel win counts:\n"
              f"{df_results['Best_Model'].value_counts().to_string()}")

        fig = generate_best_model_map(df_results, COMMODITY, BRAZIL_GEOJSON)

        fig.write_html(OUTPUT_HTML, auto_open=True)
        print(f"\nSaved: {os.path.abspath(OUTPUT_HTML)}")

        # PNG export — requires kaleido: pip install kaleido
        try:
            fig.write_image(OUTPUT_PNG, width=900, height=900, scale=2)
            print(f"Saved: {os.path.abspath(OUTPUT_PNG)}")
        except Exception as img_err:
            print(f"PNG skipped ({img_err}). "
                  "Install kaleido and run with internet access.")

    except FileNotFoundError:
        print(f"Error: '{EXCEL_FILE_PATH}' not found. "
              "Place it in the same folder as this script.")
    except Exception as e:
        print(f"\n{type(e).__name__}: {e}")
        traceback.print_exc()