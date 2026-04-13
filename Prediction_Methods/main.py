

"""
main.py
-------
Entry point. Set MODEL_TO_RUN and other constants below, then run:

    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import logging
import pandas as pd

from forecast_methods import MODEL_FUNCS
from metrics import calculate_rmse
from analysis import (
    sliding_rmse_boxplots,
    sliding_rmse_excel,
    error_comparison_table,
    sliding_error_chart,
)
from charts import generate_forecast_chart, generate_sliding_chart

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------------------------------------------------------------------------
# Configuration — edit these
# ---------------------------------------------------------------------------

EXCEL_FILE_PATH  = "../Databases/DatabaseConabv5.xlsx"
SHEET_NAME       = "ARROZ"
OUTPUT_HTML      = "sliding_rmse_boxplots.html"
CACHE_DIR        = ".boxplot_cache"

TEST_PERIODS     = 12
FORECAST_PERIODS = 12
UF               = "MS"

# Columns to exclude (e.g. test series accidentally left in the database)
EXCLUDE_STATES   = []   # e.g. ["RANDOM"]

# Choose what to run:
#   "sliding_rmse_boxplots"  — publication-quality boxplot HTML/PDF/PNG
#   "sliding_rmse_excel"     — RMSE table exported to Excel
#   "error_comparison_table" — single-window RMSE/MAPE for all states
#   "sliding_error_chart"    — mean RMSE by horizon for one model
#   "Auto_Arima" | "ETS" | "Prophet" | "Random_Forest"
#   "LSTM"       | "GRU" | "Transformer"
MODEL_TO_RUN = "sliding_rmse_boxplots"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME, index_col=0)
df.index.name = 'Date'

if EXCLUDE_STATES:
    df = df.drop(columns=EXCLUDE_STATES, errors="ignore")

series       = df[UF]
train_series = series[:-TEST_PERIODS]
test_series  = series[-TEST_PERIODS:]

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

if MODEL_TO_RUN == "sliding_rmse_boxplots":
    fig = sliding_rmse_boxplots(
        df,
        product_name=SHEET_NAME,
        test_periods=TEST_PERIODS,
        min_train_size=36,
        max_windows=12,
        output_html=OUTPUT_HTML,
        cache_dir=CACHE_DIR,
        n_jobs=-1,
    )
    print(f"Done. Open {OUTPUT_HTML} in your browser.")

elif MODEL_TO_RUN == "sliding_rmse_excel":
    summary = sliding_rmse_excel(
        df,
        product_name=SHEET_NAME,
        test_periods=TEST_PERIODS,
        min_train_size=36,
        max_windows=12,
        output_xlsx="sliding_rmse.xlsx",
        cache_dir=CACHE_DIR,
        n_jobs=-1,
    )
    print(summary)

elif MODEL_TO_RUN == "error_comparison_table":
    df_rmse = error_comparison_table(
        df, test_periods=TEST_PERIODS, n_periods=FORECAST_PERIODS
    )
    df_rmse.to_excel(f"{MODEL_TO_RUN}.xlsx")
    print(df_rmse)

elif MODEL_TO_RUN == "sliding_error_chart":
    from forecast_methods import random_forest
    errors = sliding_error_chart(
        series, random_forest, "Random Forest",
        test_periods=TEST_PERIODS, n_periods=FORECAST_PERIODS
    )
    fig = generate_sliding_chart(errors, "Random Forest", SHEET_NAME, UF)
    fig.write_html("sliding_error_chart.html", auto_open=True)

else:
    # Single-model forecast
    fn_map = {
        "Auto_Arima":    lambda: MODEL_FUNCS["ARIMA"](train_series, FORECAST_PERIODS),
        "ETS":           lambda: MODEL_FUNCS["ETS"](train_series, FORECAST_PERIODS),
        "Prophet":       lambda: MODEL_FUNCS["Prophet"](train_series, FORECAST_PERIODS),
        "Random_Forest": lambda: MODEL_FUNCS["Random Forest"](train_series, FORECAST_PERIODS),
        "LSTM":          lambda: MODEL_FUNCS["LSTM"](train_series, FORECAST_PERIODS),
        "GRU":           lambda: MODEL_FUNCS["GRU"](train_series, FORECAST_PERIODS),
        "Transformer":   lambda: MODEL_FUNCS["Transformer"](train_series, FORECAST_PERIODS),
    }

    if MODEL_TO_RUN not in fn_map:
        raise ValueError(f"Unknown MODEL_TO_RUN: {MODEL_TO_RUN!r}")

    prediction, model = fn_map[MODEL_TO_RUN]()
    if len(test_series) == len(prediction):
        prediction.index = test_series.index

    rmse = calculate_rmse(test_series, prediction)
    print(f"RMSE ({MODEL_TO_RUN}): {rmse:.2f}")

    fig = generate_forecast_chart(
        series, prediction, SHEET_NAME, UF, MODEL_TO_RUN, rmse
    )
    fig.write_html("forecast.html", auto_open=True)