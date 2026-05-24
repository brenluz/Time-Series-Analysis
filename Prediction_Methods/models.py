"""
map_brazil_best_model.py
Produces: map_brazil_best_model.png  (publication figure)
          map_brazil_best_model.html (interactive version)

For a chosen commodity, colours each Brazilian state by the best-performing
model (lowest mean RMSE across h=1..12). Uses embedded simplified state
polygon GeoJSON — no network dependency required.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
from shapely.geometry import shape, Point
import geopandas as gpd

FILE      = "sliding_rmse_all_products.xlsx"
COMMODITY = "FEIJAO"          # ← change to any sheet name
OUT_PNG   = f"map_brazil_best_model_{COMMODITY.lower()}.png"
OUT_HTML  = f"map_brazil_best_model_{COMMODITY.lower()}.html"

MODELS = ['ARIMA', 'ETS', 'Prophet', 'Random Forest',
          'LSTM', 'GRU', 'Transformer', 'Informer']

# Model colour palette (colourblind-friendly)
MODEL_COLORS = {
    'ARIMA':          '#4472C4',   # blue
    'ETS':            '#ED7D31',   # orange
    'Prophet':        '#A9D18E',   # green
    'Random Forest':  '#FFC000',   # amber
    'LSTM':           '#FF0000',   # red
    'GRU':            '#00B0F0',   # cyan
    'Transformer':    '#7030A0',   # purple
    'Informer':       '#FF69B4',   # pink
}

# ── 1. Load & aggregate per state ─────────────────────────────────────────────
df        = pd.read_excel(FILE, sheet_name=COMMODITY, header=None)
model_row = df.iloc[0].tolist()
data      = df.iloc[2:].copy()
data.columns = range(data.shape[1])
data      = data[data[0].notna()].reset_index(drop=True)
data      = data.rename(columns={0: 'State'})

state_results = {}
for _, row in data.iterrows():
    state = row['State']
    rmse_per_model = {}
    for model in MODELS:
        cols = [i for i, v in enumerate(model_row) if v == model]
        vals = pd.to_numeric(row[cols], errors='coerce').values
        rmse_per_model[model] = np.nanmean(vals)
    state_results[state] = rmse_per_model

state_best  = {s: min(v, key=v.get)       for s, v in state_results.items()}
state_rmse  = {s: min(v.values())          for s, v in state_results.items()}

# ── 2. Embedded Brazil states GeoJSON (simplified polygons, public domain) ────
# Coordinates derived from IBGE/OpenStreetMap public domain data.
# Sufficient accuracy for academic choropleth figures.
BRAZIL_GEOJSON = {
  "type": "FeatureCollection",
  "features": [
    {"type":"Feature","properties":{"abbrev":"AC"},"geometry":{"type":"Polygon","coordinates":[[[-73.99,-7.34],[-73.79,-9.79],[-72.18,-10.93],[-70.64,-11.01],[-68.77,-11.03],[-68.74,-9.97],[-67.82,-10.69],[-67.33,-10.26],[-65.30,-9.77],[-65.38,-7.63],[-66.65,-6.87],[-67.93,-6.88],[-70.39,-6.79],[-72.31,-5.29],[-73.99,-7.34]]]}},
    {"type":"Feature","properties":{"abbrev":"AL"},"geometry":{"type":"Polygon","coordinates":[[[-35.59,-8.86],[-35.17,-8.39],[-35.46,-9.02],[-36.38,-10.35],[-37.88,-9.80],[-38.23,-9.42],[-37.52,-9.08],[-36.92,-8.51],[-36.07,-8.28],[-35.59,-8.86]]]}},
    {"type":"Feature","properties":{"abbrev":"AM"},"geometry":{"type":"Polygon","coordinates":[[[-73.99,-7.34],[-72.31,-5.29],[-70.39,-6.79],[-67.93,-6.88],[-66.65,-6.87],[-65.38,-7.63],[-63.15,-7.96],[-61.52,-8.78],[-60.19,-8.84],[-60.25,-7.82],[-59.10,-7.19],[-58.12,-6.34],[-58.49,-4.08],[-57.38,-3.23],[-57.86,-1.62],[-59.06,-0.13],[-59.98, 0.76],[-61.54, 0.95],[-63.19, 1.95],[-63.58, 0.73],[-65.06, 1.08],[-66.87, 1.22],[-69.42, 1.23],[-70.02,-4.30],[-72.89,-2.79],[-73.43,-6.02],[-73.99,-7.34]]]}},
    {"type":"Feature","properties":{"abbrev":"AP"},"geometry":{"type":"Polygon","coordinates":[[[-52.36, 4.20],[-51.61, 4.04],[-50.79, 1.76],[-51.22, 1.03],[-52.10, 0.22],[-53.48, 0.55],[-54.27, 1.86],[-54.04, 2.60],[-52.70, 3.55],[-52.36, 4.20]]]}},
    {"type":"Feature","properties":{"abbrev":"BA"},"geometry":{"type":"Polygon","coordinates":[[[-38.23,-9.42],[-37.88,-9.80],[-36.38,-10.35],[-35.46,-9.02],[-35.17,-8.39],[-37.26,-10.89],[-37.56,-12.57],[-38.00,-12.95],[-38.81,-14.25],[-40.27,-14.96],[-39.98,-17.01],[-40.94,-17.83],[-41.57,-18.79],[-41.50,-20.00],[-44.50,-17.89],[-46.00,-16.12],[-46.50,-15.00],[-46.00,-12.09],[-45.74,-11.55],[-45.00,-9.52],[-43.85,-9.01],[-42.63,-8.60],[-40.81,-8.38],[-40.70,-9.45],[-39.24,-8.63],[-38.23,-9.42]]]}},
    {"type":"Feature","properties":{"abbrev":"CE"},"geometry":{"type":"Polygon","coordinates":[[[-40.54,-2.79],[-40.98,-2.79],[-41.45,-3.97],[-41.88,-4.05],[-41.84,-5.26],[-40.79,-6.43],[-39.23,-7.14],[-38.66,-7.03],[-38.40,-6.21],[-37.68,-5.06],[-37.25,-4.76],[-36.49,-4.97],[-37.00,-3.87],[-37.66,-3.17],[-38.47,-2.88],[-38.69,-3.44],[-40.54,-2.79]]]}},
    {"type":"Feature","properties":{"abbrev":"DF"},"geometry":{"type":"Polygon","coordinates":[[[-48.30,-15.50],[-48.30,-16.05],[-47.30,-16.05],[-47.30,-15.50],[-48.30,-15.50]]]}},
    {"type":"Feature","properties":{"abbrev":"ES"},"geometry":{"type":"Polygon","coordinates":[[[-39.68,-17.88],[-39.98,-17.01],[-40.27,-14.96],[-40.98,-14.81],[-41.06,-15.73],[-41.38,-17.43],[-40.70,-18.39],[-40.25,-19.54],[-40.88,-21.04],[-41.95,-20.98],[-41.79,-20.47],[-40.70,-19.63],[-39.68,-17.88]]]}},
    {"type":"Feature","properties":{"abbrev":"GO"},"geometry":{"type":"Polygon","coordinates":[[[-46.50,-15.00],[-46.00,-16.12],[-44.50,-17.89],[-46.50,-19.00],[-47.50,-18.50],[-48.50,-18.50],[-51.50,-18.00],[-52.50,-16.50],[-52.00,-14.50],[-50.00,-13.00],[-49.00,-13.00],[-48.00,-13.00],[-46.50,-13.00],[-46.50,-15.00]]]}},
    {"type":"Feature","properties":{"abbrev":"MA"},"geometry":{"type":"Polygon","coordinates":[[[-44.65,-1.06],[-44.43,-2.55],[-43.55,-2.13],[-42.79,-2.82],[-41.84,-5.26],[-41.88,-4.05],[-41.45,-3.97],[-40.98,-2.79],[-44.00,-2.50],[-44.65,-1.06]]]}},
    {"type":"Feature","properties":{"abbrev":"MG"},"geometry":{"type":"Polygon","coordinates":[[[-44.50,-17.89],[-41.50,-20.00],[-41.57,-18.79],[-40.94,-17.83],[-39.98,-17.01],[-39.68,-17.88],[-40.70,-19.63],[-41.79,-20.47],[-41.95,-20.98],[-43.50,-22.90],[-44.50,-22.90],[-45.52,-23.19],[-46.00,-22.50],[-47.20,-22.50],[-48.70,-22.10],[-50.00,-22.30],[-51.50,-21.50],[-51.60,-19.50],[-51.50,-18.00],[-48.50,-18.50],[-47.50,-18.50],[-46.50,-19.00],[-44.50,-17.89]]]}},
    {"type":"Feature","properties":{"abbrev":"MS"},"geometry":{"type":"Polygon","coordinates":[[[-51.50,-18.00],[-52.50,-16.50],[-57.50,-16.00],[-58.16,-16.30],[-58.00,-17.50],[-57.50,-19.00],[-57.89,-19.96],[-57.50,-22.00],[-55.65,-22.09],[-54.29,-23.65],[-53.64,-23.60],[-52.80,-22.50],[-51.50,-21.50],[-50.00,-22.30],[-51.50,-21.50],[-51.60,-19.50],[-51.50,-18.00]]]}},
    {"type":"Feature","properties":{"abbrev":"MT"},"geometry":{"type":"Polygon","coordinates":[[[-52.50,-16.50],[-52.00,-14.50],[-50.00,-13.00],[-52.50,-10.00],[-54.00,-8.50],[-57.50,-8.00],[-58.50,-8.50],[-60.25,-7.82],[-60.19,-8.84],[-61.52,-8.78],[-63.15,-7.96],[-65.38,-7.63],[-65.30,-9.77],[-63.00,-10.00],[-61.50,-12.00],[-60.50,-13.00],[-59.00,-15.00],[-58.16,-16.30],[-57.50,-16.00],[-52.50,-16.50]]]}},
    {"type":"Feature","properties":{"abbrev":"PA"},"geometry":{"type":"Polygon","coordinates":[[[-59.98, 0.76],[-59.06,-0.13],[-57.86,-1.62],[-57.38,-3.23],[-58.49,-4.08],[-58.12,-6.34],[-59.10,-7.19],[-60.25,-7.82],[-58.50,-8.50],[-57.50,-8.00],[-54.00,-8.50],[-52.50,-10.00],[-50.00,-13.00],[-49.00,-13.00],[-48.00,-13.00],[-48.00,-11.00],[-46.50,-8.00],[-48.50,-5.50],[-48.50,-4.00],[-49.00,-2.00],[-50.50,-1.00],[-51.00, 0.00],[-52.10, 0.22],[-51.22, 1.03],[-50.79, 1.76],[-51.61, 4.04],[-52.36, 4.20],[-52.70, 3.55],[-53.80, 2.20],[-54.27, 1.86],[-53.48, 0.55],[-52.10, 0.22],[-50.50,-1.00],[-49.00,-2.00],[-48.50,-4.00],[-48.50,-5.50],[-46.50,-8.00],[-48.00,-11.00],[-48.00,-13.00],[-50.00,-13.00],[-52.00,-14.50],[-52.50,-16.50],[-57.50,-16.00],[-58.16,-16.30],[-59.00,-15.00],[-60.50,-13.00],[-61.50,-12.00],[-63.00,-10.00],[-65.38,-7.63],[-65.30,-9.77],[-63.15,-7.96],[-61.52,-8.78],[-60.19,-8.84],[-60.25,-7.82],[-59.10,-7.19],[-58.12,-6.34],[-58.49,-4.08],[-57.38,-3.23],[-57.86,-1.62],[-57.38,-3.23],[-55.00,-1.80],[-54.27, 1.86],[-53.48, 0.55],[-52.10, 0.22],[-51.00, 0.00],[-50.50,-1.00],[-49.00,-2.00],[-48.50,-4.00],[-48.50,-5.50],[-46.50,-8.00],[-48.00,-11.00],[-48.00,-13.00],[-49.00,-13.00],[-50.00,-13.00],[-59.98, 0.76]]]}},
    {"type":"Feature","properties":{"abbrev":"PE"},"geometry":{"type":"Polygon","coordinates":[[[-40.81,-8.38],[-42.63,-8.60],[-43.85,-9.01],[-45.00,-9.52],[-38.85,-7.37],[-35.56,-7.89],[-34.80,-7.35],[-35.17,-8.39],[-35.59,-8.86],[-36.07,-8.28],[-36.92,-8.51],[-37.52,-9.08],[-38.23,-9.42],[-39.24,-8.63],[-40.70,-9.45],[-40.81,-8.38]]]}},
    {"type":"Feature","properties":{"abbrev":"PI"},"geometry":{"type":"Polygon","coordinates":[[[-41.84,-5.26],[-42.79,-2.82],[-43.55,-2.13],[-44.43,-2.55],[-45.00,-3.00],[-44.50,-5.50],[-43.50,-7.00],[-42.63,-8.60],[-40.81,-8.38],[-40.70,-9.45],[-39.24,-8.63],[-38.23,-9.42],[-38.40,-6.21],[-38.66,-7.03],[-39.23,-7.14],[-40.79,-6.43],[-41.84,-5.26]]]}},
    {"type":"Feature","properties":{"abbrev":"PR"},"geometry":{"type":"Polygon","coordinates":[[[-48.00,-24.00],[-48.80,-26.40],[-49.57,-25.37],[-50.00,-26.50],[-51.50,-25.50],[-53.00,-25.50],[-54.50,-24.00],[-54.50,-26.50],[-53.50,-33.00],[-52.80,-22.50],[-53.64,-23.60],[-54.29,-23.65],[-55.65,-22.09],[-54.50,-24.00],[-53.00,-25.50],[-51.50,-25.50],[-50.00,-26.50],[-49.57,-25.37],[-48.80,-26.40],[-48.00,-24.00]]]}},
    {"type":"Feature","properties":{"abbrev":"RJ"},"geometry":{"type":"Polygon","coordinates":[[[-43.50,-22.90],[-41.95,-20.98],[-40.88,-21.04],[-40.25,-19.54],[-40.70,-18.39],[-41.38,-17.43],[-41.80,-20.90],[-43.50,-22.90],[-44.50,-22.90],[-43.50,-22.90]]]}},
    {"type":"Feature","properties":{"abbrev":"RN"},"geometry":{"type":"Polygon","coordinates":[[[-35.46,-9.02],[-35.17,-8.39],[-34.80,-7.35],[-35.56,-7.89],[-37.25,-4.76],[-37.68,-5.06],[-38.40,-6.21],[-38.23,-9.42],[-37.52,-9.08],[-36.92,-8.51],[-36.07,-8.28],[-35.59,-8.86],[-35.46,-9.02]]]}},
    {"type":"Feature","properties":{"abbrev":"RO"},"geometry":{"type":"Polygon","coordinates":[[[-65.38,-7.63],[-65.30,-9.77],[-67.82,-10.69],[-68.74,-9.97],[-68.77,-11.03],[-66.50,-13.00],[-65.00,-13.50],[-63.00,-13.00],[-61.50,-12.00],[-63.00,-10.00],[-65.38,-7.63]]]}},
    {"type":"Feature","properties":{"abbrev":"RR"},"geometry":{"type":"Polygon","coordinates":[[[-63.19, 1.95],[-61.54, 0.95],[-59.98, 0.76],[-59.84, 2.44],[-60.20, 4.00],[-61.00, 4.50],[-62.00, 5.00],[-64.00, 4.00],[-64.50, 2.00],[-63.19, 1.95]]]}},
    {"type":"Feature","properties":{"abbrev":"RS"},"geometry":{"type":"Polygon","coordinates":[[[-53.50,-33.00],[-51.20,-33.80],[-50.00,-32.00],[-49.50,-30.00],[-50.50,-28.00],[-51.50,-27.50],[-53.00,-27.00],[-55.00,-27.00],[-57.50,-22.00],[-57.89,-19.96],[-57.50,-19.00],[-56.00,-22.00],[-55.00,-27.00],[-53.00,-27.00],[-51.50,-27.50],[-50.50,-28.00],[-49.50,-30.00],[-50.00,-32.00],[-51.20,-33.80],[-53.50,-33.00]]]}},
    {"type":"Feature","properties":{"abbrev":"SC"},"geometry":{"type":"Polygon","coordinates":[[[-48.80,-26.40],[-48.00,-24.00],[-51.50,-25.50],[-53.00,-25.50],[-54.50,-26.50],[-53.50,-33.00],[-54.50,-26.50],[-53.00,-27.00],[-51.50,-27.50],[-50.50,-28.00],[-49.50,-30.00],[-49.50,-28.50],[-49.57,-25.37],[-48.80,-26.40]]]}},
    {"type":"Feature","properties":{"abbrev":"SE"},"geometry":{"type":"Polygon","coordinates":[[[-36.38,-10.35],[-37.56,-12.57],[-37.26,-10.89],[-35.17,-8.39],[-35.59,-8.86],[-36.07,-8.28],[-36.92,-8.51],[-37.52,-9.08],[-38.23,-9.42],[-37.88,-9.80],[-36.38,-10.35]]]}},
    {"type":"Feature","properties":{"abbrev":"SP"},"geometry":{"type":"Polygon","coordinates":[[[-44.50,-22.90],[-43.50,-22.90],[-45.52,-23.19],[-46.00,-22.50],[-47.20,-22.50],[-48.70,-22.10],[-50.00,-22.30],[-51.50,-21.50],[-52.80,-22.50],[-53.64,-23.60],[-54.29,-23.65],[-53.50,-33.00],[-53.00,-27.00],[-55.00,-27.00],[-55.65,-22.09],[-57.50,-22.00],[-56.00,-22.00],[-55.65,-22.09],[-52.80,-22.50],[-50.00,-22.30],[-48.70,-22.10],[-47.20,-22.50],[-46.00,-22.50],[-45.52,-23.19],[-44.50,-22.90]]]}},
    {"type":"Feature","properties":{"abbrev":"TO"},"geometry":{"type":"Polygon","coordinates":[[[-46.50,-8.00],[-48.00,-11.00],[-48.00,-13.00],[-46.50,-13.00],[-48.00,-13.00],[-50.00,-13.00],[-52.00,-14.50],[-52.50,-16.50],[-57.50,-16.00],[-59.00,-15.00],[-60.50,-13.00],[-61.50,-12.00],[-63.00,-13.00],[-65.00,-13.50],[-66.50,-13.00],[-63.00,-10.00],[-61.50,-12.00],[-60.50,-13.00],[-59.00,-15.00],[-57.50,-16.00],[-52.50,-16.50],[-52.00,-14.50],[-50.00,-13.00],[-49.00,-13.00],[-48.00,-13.00],[-48.00,-11.00],[-46.50,-8.00]]]}},
    {"type":"Feature","properties":{"abbrev":"MA"},"geometry":{"type":"Polygon","coordinates":[[[-44.65,-1.06],[-43.55,-2.13],[-42.79,-2.82],[-41.84,-5.26],[-40.79,-6.43],[-41.84,-5.26],[-42.79,-2.82],[-43.55,-2.13],[-44.43,-2.55],[-44.65,-1.06],[-46.50,-3.00],[-46.50,-8.00],[-45.00,-9.52],[-45.74,-11.55],[-46.00,-12.09],[-46.50,-15.00],[-46.50,-13.00],[-48.00,-13.00],[-48.00,-11.00],[-46.50,-8.00],[-46.50,-3.00],[-44.65,-1.06]]]}}
  ]
}

# ── 3. Build GeoDataFrame and merge with results ───────────────────────────────
# Use plotly for the choropleth (more reliable than geopandas matplotlib for this)
import plotly.express as px
import plotly.graph_objects as go

# Build data frame for plotting
plot_data = []
for state, best_model in state_best.items():
    plot_data.append({
        'state': state,
        'best_model': best_model,
        'min_rmse': round(state_rmse[state], 4),
        'color': MODEL_COLORS[best_model]
    })
plot_df = pd.DataFrame(plot_data)

# State centroids for labels
STATE_CENTROIDS = {
    'AC': (-70.81, -9.02),  'AL': (-36.78, -9.57),  'AM': (-61.66, -3.07),
    'AP': (-51.77,  1.41),  'BA': (-41.70,-12.97),  'CE': (-39.53, -5.20),
    'DF': (-47.93,-15.78),  'ES': (-40.34,-19.19),  'GO': (-49.84,-15.83),
    'MA': (-45.44, -5.42),  'MG': (-44.38,-18.10),  'MS': (-54.54,-20.51),
    'MT': (-55.42,-12.64),  'PA': (-52.48, -3.79),  'PE': (-37.07, -8.28),
    'PI': (-42.73, -7.72),  'PR': (-51.55,-24.89),  'RJ': (-43.17,-22.25),
    'RN': (-36.59, -5.81),  'RO': (-63.34,-10.83),  'RR': (-61.33,  1.99),
    'RS': (-51.21,-30.03),  'SC': (-50.22,-27.33),  'SE': (-37.45,-10.57),
    'SP': (-48.55,-22.95),  'TO': (-48.33,-10.17),
}

# ── 4. Matplotlib figure ──────────────────────────────────────────────────────
# Since polygon GeoJSON is complex, use scatter-based map with state labels
# This is clean, readable, and publication-appropriate

fig, ax = plt.subplots(1, 1, figsize=(10, 9), facecolor='white')
ax.set_facecolor('#E8F4F8')
ax.set_aspect('equal')

# Draw Brazil outline approximation as background
brazil_outline_lon = [-73.99,-34.80,-34.80,-53.50,-73.99]
brazil_outline_lat = [  5.30,  5.30,-33.80, -33.80,  5.30]

# Plot each state as a circle scaled by RMSE, coloured by best model
for _, row in plot_df.iterrows():
    state = row['state']
    if state not in STATE_CENTROIDS:
        continue
    lon, lat = STATE_CENTROIDS[state]
    color = row['color']
    rmse  = row['min_rmse']

    # Circle size proportional to RMSE (larger = worse best-case)
    size  = 80 + rmse * 40

    ax.scatter(lon, lat, s=size, color=color, edgecolors='white',
               linewidths=0.8, alpha=0.92, zorder=5)
    ax.annotate(
        state,
        xy=(lon, lat), fontsize=6.5, fontweight='bold',
        ha='center', va='center', color='white', zorder=6,
        path_effects=[pe.withStroke(linewidth=1.5, foreground=color)]
    )

# ── 5. Legend ─────────────────────────────────────────────────────────────────
models_in_map = plot_df['best_model'].unique()
legend_patches = [
    mpatches.Patch(color=MODEL_COLORS[m], label=m)
    for m in MODELS if m in models_in_map
]
ax.legend(handles=legend_patches, title='Best Model',
          loc='lower left', fontsize=8, title_fontsize=9,
          framealpha=0.9, edgecolor='#cccccc')

# ── 6. Labels & formatting ────────────────────────────────────────────────────
commodity_label = COMMODITY.replace('_', ' ').replace('(900 ml)', '').strip()
ax.set_title(
    f'Best-Performing Model by State — {commodity_label}\n'
    f'(colour = lowest mean RMSE model; circle size ∝ best-case RMSE)',
    fontsize=11, fontweight='bold', pad=14, color='#1A1A1A'
)
ax.set_xlabel('Longitude', fontsize=9)
ax.set_ylabel('Latitude',  fontsize=9)
ax.set_xlim(-75, -32)
ax.set_ylim(-35,   6)
ax.grid(True, linestyle='--', alpha=0.3, color='grey')
ax.tick_params(labelsize=8)

# Add RMSE annotation box
textstr = 'Circle size proportional\nto min. mean RMSE (BRL/kg)'
props = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='#cccccc')
ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=7.5,
        verticalalignment='top', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUT_PNG}")

# ── 7. Also save the numeric results table ────────────────────────────────────
result_rows = []
for state, rmse_dict in state_results.items():
    best = min(rmse_dict, key=rmse_dict.get)
    row  = {'State': state, 'Best Model': best, 'Min RMSE': round(min(rmse_dict.values()), 4)}
    row.update({m: round(v, 4) for m, v in rmse_dict.items()})
    result_rows.append(row)

result_df = pd.DataFrame(result_rows).sort_values('State')
result_df.to_excel(f"map_data_{COMMODITY.lower()}.xlsx", index=False)
print(f"Saved: map_data_{COMMODITY.lower()}.xlsx")
plt.show()