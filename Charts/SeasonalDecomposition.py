"""
Seasonal Decomposition Plot (STL)
-----------------------------------
For a selected set of representative commodities, decomposes the national
average price series into Trend, Seasonality, and Residual components
using STL (Seasonal-Trend decomposition using LOESS).

One PNG is saved per commodity.

Output:
    - stats_output/decomposition/Tomato_decomposition.png
    - stats_output/decomposition/Rice_decomposition.png
    - stats_output/decomposition/Beef_decomposition.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.seasonal import STL

# ── Configuration ──────────────────────────────────────────────────────────────
EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"
OUTPUT_DIR      = "../stats_output/decomposition"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Commodities to decompose: {sheet_name: display_name}
# Add or remove entries here to change which commodities are plotted
TARGET_COMMODITIES = {
    "TOMATE":       "Tomato",
    "ARROZ":        "Rice",
    "CARNE BOVINA": "Beef",
}

COLOR_OBSERVED   = "#1a3a5c"
COLOR_TREND      = "#c0392b"
COLOR_SEASONAL   = "#16a085"
COLOR_RESIDUAL   = "#7f8c8d"

# ── Load one sheet and compute national average ────────────────────────────────
def load_national_avg(xl, sheet):
    df = pd.read_excel(xl, sheet_name=sheet, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].select_dtypes(include="number")
    df.index.name = "Date"
    series = df.mean(axis=1).sort_index()
    # STL requires a regular frequency — resample to month-start to be safe
    series = series.resample("MS").mean()
    return series


# ── Plot decomposition for one commodity ──────────────────────────────────────
def plot_decomposition(name, series, path):
    # STL decomposition — period=12 for monthly data
    stl = STL(series, period=12, robust=True)
    result = stl.fit()

    components = {
        "Observed":   series,
        "Trend":      result.trend,
        "Seasonality": result.seasonal,
        "Residual":   result.resid,
    }
    colors = [COLOR_OBSERVED, COLOR_TREND, COLOR_SEASONAL, COLOR_RESIDUAL]
    fills  = [True, False, True, False]   # fill under observed and seasonal

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    for ax, (label, data), color, do_fill in zip(axes, components.items(), colors, fills):
        ax.plot(data.index, data.values, color=color, linewidth=1.3, zorder=2)
        if do_fill:
            ax.fill_between(data.index, data.values,
                            alpha=0.07, color=color, zorder=1)

        # Zero line for seasonal and residual
        if label in ("Seasonality", "Residual"):
            ax.axhline(0, color="#aaaaaa", linewidth=0.6, linestyle="--", zorder=0)

        ax.set_ylabel(label, fontsize=9, fontweight="bold", color="#333333")
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.set_facecolor("#fafafa")
        ax.grid(axis="y", linewidth=0.4, color="#e0e0e0", zorder=0)

    # X-axis formatting on the bottom panel only
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].tick_params(axis="x", labelsize=8, rotation=45)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"  [OK]   '{name}' → {os.path.basename(path)}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Reading: {EXCEL_FILE_PATH}\n" + "─" * 50)

    try:
        xl = pd.ExcelFile(EXCEL_FILE_PATH)
    except FileNotFoundError:
        print(f"[ERROR] File not found: '{EXCEL_FILE_PATH}'")
        raise SystemExit(1)

    available = {s.strip().upper(): s for s in xl.sheet_names}

    for sheet_key, display_name in TARGET_COMMODITIES.items():
        sheet_key_upper = sheet_key.strip().upper()
        if sheet_key_upper not in available:
            print(f"  [SKIP] '{sheet_key}' not found in workbook. "
                  f"Available sheets: {list(available.keys())}")
            continue
        try:
            series = load_national_avg(xl, available[sheet_key_upper])
            filename = f"{display_name.replace(' ', '_')}_decomposition.png"
            plot_decomposition(display_name, series,
                               os.path.join(OUTPUT_DIR, filename))
        except Exception as e:
            print(f"  [ERR]  '{display_name}': {e}")

    print(f"\nDone. Plots saved to: {os.path.abspath(OUTPUT_DIR)}")