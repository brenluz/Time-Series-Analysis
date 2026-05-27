"""
Individual Time Series Panel Plots
------------------------------------
Reads ALL sheets from the CONAB database and saves one PNG per commodity,
showing the national average price series (mean across all states).

Output:
    - stats_output/panels/Rice.png
    - stats_output/panels/Beans.png
    - ... one file per commodity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


# ── Configuration ──────────────────────────────────────────────────────────────
EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"
OUTPUT_DIR      = "../stats_output/panels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMMODITY_LABELS = {
    "ARROZ":            "Rice",
    "FEIJAO":           "Beans",
    "ACUCAR":           "Sugar",
    "CAFÉ":             "Coffee",
    "MACARRAO":         "Pasta",
    "TOMATE":           "Tomato",
    "CEBOLA":           "Onion",
    "BATATA":           "Potato",
    "CARNE BOVINA":     "Beef",
    "FRANGO":           "Chicken",
    "FARINHA TRIGO":    "Wheat Flour",
    "FLOCOS MILHO":     "Corn Flakes",
    "OLEO SOJA":        "Soybean Oil",
    "EXTRATO TOMATE":   "Tomato Paste",
    "SAL":              "Salt",
    "LEITE":            "Milk",
    "PAO FRANCES":      "French Bread",
    "FARINHA MANDIOCA": "Cassava Flour",
}

COLOR = "#1a3a5c"

# ── Plot one commodity ─────────────────────────────────────────────────────────
def plot_single(name, series, path):
    fig, ax = plt.subplots(figsize=(8, 3.2))

    # Price line + fill
    ax.plot(series.index, series.values, color=COLOR, linewidth=1.4, zorder=2)
    ax.fill_between(series.index, series.values, alpha=0.08, color=COLOR, zorder=1)

    # Axes formatting
    ax.set_ylabel("R$/kg", fontsize=9, color="#555555")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.set_facecolor("#fafafa")
    ax.grid(axis="y", linewidth=0.4, color="#dddddd", zorder=0)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Reading: {EXCEL_FILE_PATH}\n" + "─" * 50)

    try:
        xl = pd.ExcelFile(EXCEL_FILE_PATH)
    except FileNotFoundError:
        print(f"[ERROR] File not found: '{EXCEL_FILE_PATH}'")
        raise SystemExit(1)

    saved, failed = [], []

    for sheet in xl.sheet_names:
        try:
            df = pd.read_excel(xl, sheet_name=sheet, index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
            df.index.name = "Date"
            df = df.select_dtypes(include="number")

            if df.empty:
                print(f"  [SKIP] '{sheet}' — no numeric data")
                continue

            national_avg = df.mean(axis=1).sort_index()
            display_name = COMMODITY_LABELS.get(sheet.strip().upper(), sheet)

            # Use display name as filename, replacing spaces with underscores
            filename = display_name.replace(" ", "_") + ".png"
            out_path = os.path.join(OUTPUT_DIR, filename)

            plot_single(display_name, national_avg, out_path)
            saved.append(display_name)
            print(f"  [OK]   '{display_name}' → {filename}")

        except Exception as e:
            print(f"  [ERR]  '{sheet}': {e}")
            failed.append(sheet)

    print(f"\n{'='*50}")
    print(f"Done. {len(saved)} plots saved to: {os.path.abspath(OUTPUT_DIR)}")
    if failed:
        print(f"Failed: {failed}")