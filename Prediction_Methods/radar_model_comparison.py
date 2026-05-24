"""
radar_model_comparison.py
Produces: radar_model_comparison.png  (publication figure)
          radar_model_comparison.html (interactive version)

Radar chart with one axis per commodity. Each polygon represents a model.
Values are NORMALISED mean RMSE (0 = best possible, 1 = worst), so all
commodities are on comparable scales despite different price magnitudes.
Models to display are configurable via MODELS_TO_PLOT.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

FILE   = "sliding_rmse_all_products.xlsx"
OUTPUT_PNG  = "radar_model_comparison.png"
OUTPUT_HTML = "radar_model_comparison.html"

MODELS = ['ARIMA', 'ETS', 'Prophet', 'Random Forest',
          'LSTM', 'GRU', 'Transformer', 'Informer']

# Select which models to overlay — keep ≤5 for readability
# Recommended: the story is RF vs classical vs attention
MODELS_TO_PLOT = ['Random Forest', 'ETS', 'ARIMA', 'Transformer', 'Informer']

MODEL_STYLES = {
    'ARIMA':          {'color': '#4472C4', 'lw': 1.8, 'ls': '--',  'alpha': 0.75},
    'ETS':            {'color': '#ED7D31', 'lw': 1.8, 'ls': '-.',  'alpha': 0.75},
    'Prophet':        {'color': '#70AD47', 'lw': 1.5, 'ls': ':',   'alpha': 0.70},
    'Random Forest':  {'color': '#FFC000', 'lw': 2.5, 'ls': '-',   'alpha': 0.90},
    'LSTM':           {'color': '#FF0000', 'lw': 1.5, 'ls': ':',   'alpha': 0.65},
    'GRU':            {'color': '#00B0F0', 'lw': 1.8, 'ls': '--',  'alpha': 0.75},
    'Transformer':    {'color': '#7030A0', 'lw': 2.0, 'ls': '-',   'alpha': 0.80},
    'Informer':       {'color': '#C00000', 'lw': 2.0, 'ls': '-.',  'alpha': 0.80},
}

# Commodity display labels (shorter for radar axes)
COMMODITY_LABELS = {
    'ARROZ':               'Rice',
    'ACUCAR':              'Sugar',
    'BATATA':              'Potato',
    'CAFÉ':                'Coffee',
    'CARNE BOVINA':        'Beef',
    'CARNE FRANGO':        'Chicken',
    'CARNE SUINA':         'Pork',
    'CEBOLA':              'Onion',
    'EXTRATO DE TOMATE':   'Tom. Paste',
    'FARINHA MANDIOCA':    'Cass. Flour',
    'FARINHA TRIGO':       'Wh. Flour',
    'FEIJAO':              'Beans',
    'FLOCOS DE MILHO':     'Corn Flakes',
    'LEITE':               'Milk',
    'MACARRAO':            'Pasta',
    'OLEO DE SOJA (900 ml)': 'Soy Oil',
    'PÃO FRANCÊS   (kg)':  'Fr. Bread',
    'SAL    (kg)':         'Salt',
    'TOMATE':              'Tomato',
}

# ── 1. Load & aggregate mean RMSE per model per commodity ─────────────────────
xl      = pd.ExcelFile(FILE)
sheets  = xl.sheet_names
summary = {}

for sheet in sheets:
    df        = pd.read_excel(FILE, sheet_name=sheet, header=None)
    model_row = df.iloc[0].tolist()
    data      = df.iloc[2:].copy()
    data.columns = range(data.shape[1])
    data      = data[data[0].notna()].reset_index(drop=True)

    result = {}
    for model in MODELS:
        cols = [i for i, v in enumerate(model_row) if v == model]
        vals = data[cols].apply(pd.to_numeric, errors='coerce').values.flatten()
        result[model] = np.nanmean(vals)
    summary[sheet] = result

summary_df = pd.DataFrame(summary).T   # index=commodity, columns=model

# ── 2. Normalise per commodity (0 = best, 1 = worst across all 8 models) ──────
norm_df = summary_df.copy()
for idx in norm_df.index:
    row_min = norm_df.loc[idx].min()
    row_max = norm_df.loc[idx].max()
    norm_df.loc[idx] = (norm_df.loc[idx] - row_min) / (row_max - row_min + 1e-12)

# ── 3. Radar chart setup ──────────────────────────────────────────────────────
categories  = [COMMODITY_LABELS.get(s, s) for s in sheets]
N           = len(categories)
angles      = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles     += angles[:1]   # close the polygon

fig, ax = plt.subplots(figsize=(10, 10),
                       subplot_kw=dict(polar=True),
                       facecolor='white')
ax.set_facecolor('#FAFBFF')

# ── 4. Draw gridlines and axis labels ─────────────────────────────────────────
gridlevels = [0.25, 0.50, 0.75, 1.00]
for level in gridlevels:
    ax.plot(angles, [level] * (N + 1),
            color='#CCCCCC', linewidth=0.6, linestyle='-', zorder=1)
    ax.text(angles[0], level + 0.03, f'{level:.2f}',
            ha='center', va='bottom', fontsize=7, color='#999999')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=8.5, fontweight='bold', color='#333333')
ax.set_yticks([])
ax.set_ylim(0, 1.15)
ax.spines['polar'].set_color('#CCCCCC')
ax.spines['polar'].set_linewidth(0.8)

# Offset axis labels outward
ax.tick_params(axis='x', pad=12)

# ── 5. Plot each model ────────────────────────────────────────────────────────
for model in MODELS_TO_PLOT:
    values  = norm_df[model].values.tolist()
    values += values[:1]
    style   = MODEL_STYLES[model]

    ax.plot(angles, values,
            color=style['color'], linewidth=style['lw'],
            linestyle=style['ls'], alpha=style['alpha'],
            zorder=3, label=model)
    ax.fill(angles, values,
            color=style['color'], alpha=0.07, zorder=2)

    # Mark each vertex
    ax.scatter(angles[:-1], values[:-1],
               color=style['color'], s=28, zorder=4,
               edgecolors='white', linewidths=0.5, alpha=style['alpha'])

# ── 6. Legend & title ─────────────────────────────────────────────────────────
ax.legend(
    loc='upper right',
    bbox_to_anchor=(1.28, 1.12),
    fontsize=9,
    title='Model',
    title_fontsize=9.5,
    framealpha=0.92,
    edgecolor='#CCCCCC',
    handlelength=2.5,
)

ax.set_title(
    'Normalised Mean RMSE by Model and Commodity\n'
    '(0 = best performance, 1 = worst performance within each commodity)',
    fontsize=11, fontweight='bold', pad=28, color='#1A1A1A',
    y=1.06
)

# Add note
fig.text(0.5, 0.01,
         'Note: Values are normalised within each commodity axis to enable '
         'cross-commodity comparison.\nRaw scale differences are removed; '
         'only relative model ranking is preserved.',
         ha='center', fontsize=7.5, color='#666666',
         style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUTPUT_PNG}")

# ── 7. Interactive HTML via plotly ────────────────────────────────────────────
try:
    import plotly.graph_objects as go

    fig_p = go.Figure()

    for model in MODELS_TO_PLOT:
        values = norm_df[model].values.tolist()
        values += values[:1]
        style  = MODEL_STYLES[model]

        fig_p.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=style['color'],
            opacity=0.10,
            line=dict(color=style['color'], width=style['lw'] * 1.2),
            name=model,
            hovertemplate='<b>%{theta}</b><br>Normalised RMSE: %{r:.3f}<extra>' + model + '</extra>'
        ))

    fig_p.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.1],
                            tickfont=dict(size=8), gridcolor='#DDDDDD'),
            angularaxis=dict(tickfont=dict(size=9))
        ),
        showlegend=True,
        title=dict(
            text='Normalised Mean RMSE by Model and Commodity<br>'
                 '<sup>(0 = best, 1 = worst within each commodity)</sup>',
            font=dict(size=13)
        ),
        font=dict(family='Arial'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(font=dict(size=10))
    )

    fig_p.write_html(OUTPUT_HTML)
    print(f"Saved: {OUTPUT_HTML}")

except ImportError:
    print("Plotly not available — skipping HTML output.")

plt.show()