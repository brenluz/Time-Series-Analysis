"""
Descriptive Statistics Table Generator
---------------------------------------
Reads ALL sheets from the CONAB food price database and produces a
publication-ready summary statistics table aggregated across all states.

Outputs:
    - descriptive_stats.csv
    - descriptive_stats.xlsx
    - descriptive_stats.png
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
EXCEL_FILE_PATH = "../Databases/DatabaseConabv5.xlsx"
OUTPUT_DIR      = "../stats_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load and compute ───────────────────────────────────────────────────────────
def load_sheet(xl, sheet):
    df = pd.read_excel(xl, sheet_name=sheet, index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df.index.name = "Date"
    return df.select_dtypes(include="number")


def compute_stats(df):
    values = df.values.flatten()
    values = values[~np.isnan(values)]
    mean_ = np.mean(values)
    std_  = np.std(values, ddof=1)
    return {
        "Mean (BRL/kg)": round(mean_, 2),
        "Std Dev":        round(std_,  2),
        "CV (%)":         round((std_ / mean_) * 100, 1) if mean_ else np.nan,
        "Min":            round(np.min(values), 2),
        "Max":            round(np.max(values), 2),
        "Skewness":       round(stats.skew(values), 3),
        "Kurtosis":       round(stats.kurtosis(values), 3),
        "N":              len(values),
    }


def build_stats_table(excel_path):
    xl = pd.ExcelFile(excel_path)
    records = []

    for sheet in xl.sheet_names:
        try:
            df = load_sheet(xl, sheet)
            if df.empty:
                print(f"  [SKIP] '{sheet}' — no numeric data")
                continue
            row = compute_stats(df)
            row["Commodity"] = sheet
            records.append(row)
            print(f"  [OK]   '{sheet}'  (N={row['N']:,})")
        except Exception as e:
            print(f"  [ERR]  '{sheet}': {e}")

    df_stats = pd.DataFrame(records).set_index("Commodity")
    df_stats = df_stats[["Mean (BRL/kg)", "Std Dev", "CV (%)", "Min", "Max", "Skewness", "Kurtosis", "N"]]
    return df_stats.sort_values("Mean (BRL/kg)", ascending=False)


# ── Export CSV ─────────────────────────────────────────────────────────────────
def export_csv(df, path):
    df.to_csv(path)
    print(f"[CSV]  → {os.path.abspath(path)}")


# ── Export Excel ───────────────────────────────────────────────────────────────
def export_excel(df, path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Descriptive Stats")
        ws = writer.sheets["Descriptive Stats"]
        for col in ws.columns:
            w = max(len(str(c.value)) if c.value else 0 for c in col)
            ws.column_dimensions[col[0].column_letter].width = w + 4
    print(f"[XLSX] → {os.path.abspath(path)}")


# ── Export PNG figure ──────────────────────────────────────────────────────────
def export_figure(df, path):
    STAT_COLS = ["Mean (BRL/kg)", "Std Dev", "CV (%)", "Min", "Max", "Skewness", "Kurtosis"]
    df_display = df[STAT_COLS]
    commodities = df_display.index.tolist()
    n_rows = len(commodities)
    n_cols = len(STAT_COLS)

    # Build cell text
    fmt = ["{:.2f}", "{:.2f}", "{:.1f}", "{:.2f}", "{:.2f}", "{:.3f}", "{:.3f}"]
    cell_data = [
        [fmt[j].format(row[col]) for j, col in enumerate(STAT_COLS)]
        for _, row in df_display.iterrows()
    ]

    # Layout — auto-scales to however many rows there are
    fig_w   = 14.0
    row_h   = 0.36
    head_h  = 0.52
    fig_h   = head_h + n_rows * row_h + 0.6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    HEADER_BG = "#1a3a5c"
    INDEX_BG  = "#2e5f8a"
    ALT_BG    = "#eef2f7"
    WHITE     = "#ffffff"
    BORDER    = "#c0cfe0"
    FG_LIGHT  = "#ffffff"
    FG_DARK   = "#1a1a2e"

    # Column widths (relative): commodity + 7 stat columns
    rel_w = [0.20] + [0.114] * n_cols
    total = sum(rel_w)
    xs = [0.0]
    for w in rel_w:
        xs.append(xs[-1] + w / total)

    h_norm = head_h / fig_h
    r_norm = row_h  / fig_h
    y_top  = 1.0 - 0.01

    def cell(x0, y0, w, h, text, bg=WHITE, fg=FG_DARK, fs=8, bold=False, align="center"):
        ax.add_patch(plt.Rectangle((x0, y0), w, h,
            transform=ax.transAxes, facecolor=bg,
            edgecolor=BORDER, linewidth=0.4, clip_on=False))
        ax.text(x0 + w * (0.5 if align == "center" else 0.05),
                y0 + h * 0.5, text,
                transform=ax.transAxes, ha=align, va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=fg, clip_on=False)

    # Header
    y0 = y_top - h_norm
    col_headers = ["Mean\n(BRL/kg)", "Std Dev", "CV (%)", "Min", "Max", "Skewness", "Kurtosis"]
    cell(xs[0], y0, xs[1]-xs[0], h_norm, "Commodity", bg=HEADER_BG, fg=FG_LIGHT, fs=8.5, bold=True)
    for j, label in enumerate(col_headers):
        cell(xs[j+1], y0, xs[j+2]-xs[j+1], h_norm, label, bg=HEADER_BG, fg=FG_LIGHT, fs=8.5, bold=True)

    # Data rows
    for i, (name, vals) in enumerate(zip(commodities, cell_data)):
        y0 = y_top - h_norm - (i + 1) * r_norm
        row_bg = ALT_BG if i % 2 == 1 else WHITE
        cell(xs[0], y0, xs[1]-xs[0], r_norm, name, bg=INDEX_BG, fg=FG_LIGHT, fs=8, bold=True, align="left")
        for j, v in enumerate(vals):
            cell(xs[j+1], y0, xs[j+2]-xs[j+1], r_norm, v, bg=row_bg, fg=FG_DARK, fs=8)

    # Bottom rule
    y_bottom = y_top - h_norm - n_rows * r_norm
    ax.plot([xs[0], xs[-1]], [y_bottom, y_bottom],
            transform=ax.transAxes, color=HEADER_BG, lw=1.0, clip_on=False)

    # Caption
    ax.text(xs[0], y_bottom - 0.03,
            "Table 1. Descriptive statistics of monthly retail food prices (BRL/kg or BRL/L) "
            "across all 27 Brazilian federative units, January 2016–December 2024. "
            "CV = Coefficient of Variation. Kurtosis = excess kurtosis (Fisher; Normal = 0).",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="#555555", style="italic", clip_on=False)

    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[PNG]  → {os.path.abspath(path)}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Reading: {EXCEL_FILE_PATH}\n" + "─" * 50)

    try:
        df_stats = build_stats_table(EXCEL_FILE_PATH)
    except FileNotFoundError:
        print(f"[ERROR] File not found: '{EXCEL_FILE_PATH}'")
        print("  → Update EXCEL_FILE_PATH at the top of the script.")
        raise SystemExit(1)

    print("\n" + df_stats.to_string())

    export_csv(df_stats,   os.path.join(OUTPUT_DIR, "descriptive_stats.csv"))
    export_excel(df_stats, os.path.join(OUTPUT_DIR, "descriptive_stats.xlsx"))
    export_figure(df_stats, os.path.join(OUTPUT_DIR, "descriptive_stats.png"))

    print(f"\nDone. {len(df_stats)} commodities → {os.path.abspath(OUTPUT_DIR)}")