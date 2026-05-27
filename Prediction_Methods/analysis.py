"""
analysis.py
-----------
Walk-forward validation functions:
  - sliding_rmse_boxplots : publication-quality boxplot figure (HTML + PDF + PNG)
  - sliding_rmse_excel    : per-state RMSE table exported to Excel
  - error_comparison_table: single-window RMSE/MAPE comparison across all states
  - sliding_error_chart   : mean RMSE by horizon for a single model
"""

import logging
import math
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from metrics import calculate_rmse, calculate_mape
from forecast_methods import MODEL_FUNCS
from cache import _worker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers shared across analysis functions
# ---------------------------------------------------------------------------

def _build_jobs(df: pd.DataFrame, test_periods: int,
                min_train_size: int, max_windows: int):
    """
    Build the list of (state, window) computation jobs and their metadata.

    Returns
    -------
    jobs     : list of argument tuples for _worker
    job_meta : list of state labels, same order as jobs
    """
    jobs, job_meta = [], []
    for state in df.columns:
        series = df[state].dropna()
        if len(series) < min_train_size + test_periods:
            logger.warning(f"Skipping {state}: only {len(series)} observations.")
            continue

        start_end = len(series)
        stop_end  = max(min_train_size + test_periods - 1, start_end - max_windows)

        for end_idx in range(start_end, stop_end, -1):
            if end_idx - test_periods < min_train_size:
                break
            train_s = series.iloc[:end_idx - test_periods]
            test_s  = series.iloc[end_idx - test_periods:end_idx]
            jobs.append((
                state,
                train_s.values, train_s.index,
                test_s.values,  test_s.index,
                test_periods,
            ))
            job_meta.append(state)

    return jobs, job_meta


def _resolve_max_workers(n_jobs: int) -> int:
    """
    Resolve an effective positive worker count from user request/env cap.

    Rules:
    - n_jobs=-1 means "use all available".
    - ML_MAX_CPU_CORES, when valid, caps the final worker count.
    - Any invalid or non-positive value falls back to 1.
    """
    cpu_total = os.cpu_count() or 1
    requested = cpu_total if n_jobs == -1 else n_jobs
    if requested is None:
        requested = cpu_total
    try:
        requested = int(requested)
    except (TypeError, ValueError):
        requested = cpu_total
    if requested < 1:
        requested = 1

    cap_raw = os.getenv("ML_MAX_CPU_CORES")
    cap = None
    if cap_raw is not None:
        try:
            cap_int = int(cap_raw)
            if cap_int > 0:
                cap = cap_int
        except (TypeError, ValueError):
            cap = None

    if cap is not None:
        requested = min(requested, cap)
    return min(requested, cpu_total)


def _run_parallel(jobs: list, job_meta: list, cache_dir: str, n_jobs: int,
                  desc: str = "Computing windows"):
    """
    Execute jobs in parallel, injecting cache_dir into each job tuple.

    Returns
    -------
    results : list of (state, result_dict) in completion order
    """
    full_jobs = [(*job, cache_dir) for job in jobs]
    results   = []

    max_workers = _resolve_max_workers(n_jobs)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker, job): idx
            for idx, job in enumerate(full_jobs)
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc=desc, unit="window"):
            idx = futures[future]
            try:
                results.append((job_meta[idx], future.result()))
            except Exception as exc:
                logger.error(f"Window failed: {exc}")

    return results


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------

def sliding_rmse_boxplots(
    df: pd.DataFrame,
    product_name: str = "",
    test_periods: int = 12,
    min_train_size: int = 36,
    max_windows: int = 12,
    output_html: str = "sliding_rmse_boxplots.html",
    cache_dir: str = ".boxplot_cache",
    n_jobs: int = -1,
    steps_per_row: int = 6,
) -> go.Figure:
    """
    Walk-forward RMSE boxplots — one box per (model, step) pair.

    Layout
    ------
    ceil(test_periods / steps_per_row) subplots stacked vertically.
    Each subplot has steps_per_row step groups on the x-axis; within each
    step, 7 side-by-side boxes (one per model) are color-coded.

    What each box represents
    ------------------------
    Each of the ~26 data points in a box is:
        RMSE_state = sqrt( mean( (actual_k - predicted_k)^2 ) across windows )
    for forecast step k, for one state. The box shows the distribution of
    these per-state RMSEs across all states.

    Exports
    -------
    .html : interactive browser preview
    .pdf  : vector format, preferred for LaTeX (requires kaleido)
    .png  : raster at scale=3, fallback for LaTeX
    """
    os.makedirs(cache_dir, exist_ok=True)
    n_jobs = _resolve_max_workers(n_jobs)

    logger.info(
        f"sliding_rmse_boxplots | product={product_name!r} "
        f"| states={len(df.columns)} | max_windows={max_windows} | n_jobs={n_jobs}"
    )

    jobs, job_meta = _build_jobs(df, test_periods, min_train_size, max_windows)
    logger.info(f"Total jobs: {len(jobs)}")

    # --- Accumulate squared errors per (state, model, step) ---
    sq_errors_by_state: dict = {}
    for state, result in _run_parallel(jobs, job_meta, cache_dir, n_jobs):
        if state not in sq_errors_by_state:
            sq_errors_by_state[state] = {
                m: [[] for _ in range(test_periods)] for m in MODEL_FUNCS
            }
        for model_name, abs_errors in result.items():
            for step, ae in enumerate(abs_errors):
                if step < test_periods:
                    sq_errors_by_state[state][model_name][step].append(ae ** 2)

    # --- One RMSE scalar per (state, model, step) ---
    rmse_by_step = {step: {m: [] for m in MODEL_FUNCS} for step in range(test_periods)}
    for state, models in sq_errors_by_state.items():
        for model_name, steps_sq in models.items():
            for step, sq_list in enumerate(steps_sq):
                if sq_list:
                    rmse_by_step[step][model_name].append(
                        float(np.sqrt(np.mean(sq_list)))
                    )

    # --- Global y-axis range ---
    all_vals = [v for step in range(test_periods)
                  for m in MODEL_FUNCS for v in rmse_by_step[step][m]]
    y_max = float(np.max(all_vals)) * 1.05 if all_vals else 1.0
    y_min = 0.0

    # --- Figure layout ---
    n_rows     = math.ceil(test_periods / steps_per_row)
    n_models   = len(MODEL_FUNCS)
    model_names = list(MODEL_FUNCS.keys())

    palette   = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA",
                 "#FFA15A", "#19D3F3", "#FF6692"]
    color_map = {m: palette[i % len(palette)] for i, m in enumerate(model_names)}

    row_titles = []
    for r in range(n_rows):
        s_start = r * steps_per_row + 1
        s_end   = min((r + 1) * steps_per_row, test_periods)

    fig = make_subplots(
        rows=n_rows, cols=1,
        vertical_spacing=0.03,
        shared_yaxes=True,
    )

    # Manual x-positioning so boxes sit exactly side-by-side with no overlap
    box_width  = 0.10
    step_gap   = 0.25
    step_span  = n_models * box_width

    x_centers = {}
    for r in range(n_rows):
        x_centers[r] = {}
        for s in range(steps_per_row):
            step_idx = r * steps_per_row + s
            if step_idx >= test_periods:
                break
            x_centers[r][step_idx] = s * (step_span + step_gap) + step_span / 2

    model_offsets = {
        m: (i - (n_models - 1) / 2) * box_width
        for i, m in enumerate(model_names)
    }

    row_xticks = {}
    for r in range(n_rows):
        positions = [cx for cx in x_centers[r].values()]
        labels    = [f"Step {si + 1}" for si in x_centers[r]]
        row_xticks[r] = (positions, labels)

    for model_name in model_names:
        show_legend = True
        for step in range(test_periods):
            row = step // steps_per_row
            if step not in x_centers[row]:
                continue
            vals = rmse_by_step[step][model_name]
            if not vals:
                continue

            x_pos = x_centers[row][step] + model_offsets[model_name]

            fig.add_trace(
                go.Box(
                    x=[x_pos] * len(vals),
                    y=vals,
                    name=model_name,
                    legendgroup=model_name,
                    showlegend=show_legend,
                    marker_color=color_map[model_name],
                    fillcolor=color_map[model_name],
                    width=box_width,
                    boxmean=False,
                    whiskerwidth=1.0,
                    boxpoints="outliers",
                    jitter=0.15,
                    marker=dict(symbol="circle", size=4, opacity=0.6,
                                line=dict(width=0)),
                    line=dict(width=1.5, color="rgba(0,0,0,0.5)"),
                    opacity=0.85,
                ),
                row=row + 1, col=1,
            )
            show_legend = False

    for r in range(n_rows):
        positions, labels = row_xticks[r]
        fig.update_yaxes(range=[y_min, y_max], title_text="RMSE",
                         title_font=dict(size=20), tickfont=dict(size=18),
                         row=r + 1, col=1)
        fig.update_xaxes(tickmode="array", tickvals=positions, ticktext=labels,
                         tickfont=dict(size=18), row=r + 1, col=1)

    title_suffix = f" - {product_name}" if product_name else ""
    fig.update_layout(
        title=dict(
            text=f"RMSE Distribution by Forecast Step and Model{title_suffix}",
            x=0.5, font=dict(size=22),
        ),
        width=1400,
        height=600 * n_rows,
        template="plotly_white",
        boxmode="overlay",
        font=dict(size=18),
        legend=dict(
            title=dict(text="Model", font=dict(size=18)),
            orientation="h",
            yanchor="top", y=-0.12,
            xanchor="center", x=0.5,
            font=dict(size=18), tracegroupgap=4,
        ),
        margin=dict(t=60, b=120, l=60, r=20),
    )

    fig.write_html(output_html, auto_open=True)
    logger.info(f"Saved HTML -> {output_html}")

    for fmt, kwargs in [
        ("pdf", dict(format="pdf", scale=1)),
        ("png", dict(format="png", scale=3)),
    ]:
        path = output_html.replace(".html", f".{fmt}")
        try:
            fig.write_image(path, width=1400, height=600 * n_rows, **kwargs)
            logger.info(f"Saved {fmt.upper()}  -> {path}")
        except Exception as e:
            logger.warning(f"{fmt.upper()} export failed (pip install kaleido): {e}")

    return fig


def sliding_rmse_excel(
    df: pd.DataFrame,
    product_name: str = "",
    test_periods: int = 12,
    min_train_size: int = 36,
    max_windows: int = 12,
    output_xlsx: str = "sliding_rmse.xlsx",
    cache_dir: str = ".boxplot_cache",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Walk-forward RMSE export to Excel.

    Excel structure
    ---------------
    Sheet "Summary" : mean RMSE per (state, model, step) across all windows.
    Sheet per state : per-window per-step absolute errors, MultiIndex columns.
    """
    os.makedirs(cache_dir, exist_ok=True)
    n_jobs = _resolve_max_workers(n_jobs)

    logger.info(
        f"sliding_rmse_excel | product={product_name!r} "
        f"| states={len(df.columns)} | max_windows={max_windows} | n_jobs={n_jobs}"
    )

    # Build jobs with window labels for row indices
    jobs_raw, job_meta_raw = [], []
    for state in df.columns:
        series = df[state].dropna()
        if len(series) < min_train_size + test_periods:
            logger.warning(f"Skipping {state}: only {len(series)} observations.")
            continue
        start_end = len(series)
        stop_end  = max(min_train_size + test_periods - 1, start_end - max_windows)
        for end_idx in range(start_end, stop_end, -1):
            if end_idx - test_periods < min_train_size:
                break
            train_s = series.iloc[:end_idx - test_periods]
            test_s  = series.iloc[end_idx - test_periods:end_idx]
            try:
                window_label = str(pd.Timestamp(train_s.index[-1]).date())
            except Exception:
                window_label = str(train_s.index[-1])
            jobs_raw.append((
                state,
                train_s.values, train_s.index,
                test_s.values,  test_s.index,
                test_periods,
            ))
            job_meta_raw.append((state, window_label))

    logger.info(f"Total jobs: {len(jobs_raw)}")

    full_jobs = [(*job, cache_dir) for job in jobs_raw]
    raw_results: dict = {}

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_worker, job): idx
                   for idx, job in enumerate(full_jobs)}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Computing windows", unit="window"):
            idx = futures[future]
            state, window_label = job_meta_raw[idx]
            try:
                raw_results[(state, window_label)] = future.result()
            except Exception as exc:
                logger.error(f"Window ({state}, {window_label}) failed: {exc}")

    model_names = list(MODEL_FUNCS.keys())
    step_labels = [f"Step {k + 1}" for k in range(test_periods)]

    state_groups: dict = {}
    for (state, window_label), result in raw_results.items():
        state_groups.setdefault(state, {})[window_label] = result

    per_state_dfs: dict = {}
    for state, windows in state_groups.items():
        rows = {}
        for window_label, model_errors in windows.items():
            row = {}
            for model in model_names:
                errs = model_errors.get(model, [np.nan] * test_periods)
                for k, step_label in enumerate(step_labels):
                    row[(model, step_label)] = errs[k] if k < len(errs) else np.nan
            rows[window_label] = row
        per_state_dfs[state] = pd.DataFrame(rows).T
        per_state_dfs[state].index.name = "Window (last train date)"
        per_state_dfs[state].columns = pd.MultiIndex.from_tuples(
            per_state_dfs[state].columns, names=["Model", "Step"]
        )

    summary_df = pd.DataFrame(
        {state: df_s.mean(axis=0) for state, df_s in per_state_dfs.items()}
    ).T
    summary_df.index.name = "State"

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary")
        for state, df_s in per_state_dfs.items():
            df_s.to_excel(writer, sheet_name=str(state)[:31])

    logger.info(f"Saved Excel -> {output_xlsx}  ({len(per_state_dfs)} state sheets + Summary)")
    return summary_df


def error_comparison_table(df: pd.DataFrame,
                           test_periods: int = 12,
                           n_periods: int = 12) -> pd.DataFrame:
    """
    Single-window RMSE and MAPE comparison for all models and states.
    Trains on all data except the last test_periods observations.
    """
    results = {}
    for state in df.columns:
        series = df[state]
        if len(series.dropna()) <= test_periods:
            continue
        train, test = series[:-test_periods], series[-test_periods:]
        results[state] = {}
        for label, fn in [
            ('ARIMA',         lambda s: MODEL_FUNCS['ARIMA'](s, n_periods)),
            ('ETS',           lambda s: MODEL_FUNCS['ETS'](s, n_periods)),
            ('PROPHET',       lambda s: MODEL_FUNCS['Prophet'](s, n_periods)),
            ('RANDOM_FOREST', lambda s: MODEL_FUNCS['Random Forest'](s, n_periods)),
            ('LSTM',          lambda s: MODEL_FUNCS['LSTM'](s, n_periods)),
            ('GRU',           lambda s: MODEL_FUNCS['GRU'](s, n_periods)),
            ('TRANSFORMER',   lambda s: MODEL_FUNCS['Transformer'](s, n_periods)),
        ]:
            try:
                fc, _ = fn(train)
                fc.index = test.index
                results[state][f'RMSE_{label}'] = calculate_rmse(test, fc)
                results[state][f'MAPE_{label}'] = calculate_mape(test, fc)
            except Exception:
                pass

    df_out = pd.DataFrame(results).T
    df_out.index.name = "UF"
    return df_out


def sliding_error_chart(series: pd.Series, model_func,
                         model_name: str,
                         test_periods: int = 12,
                         n_periods: int = 12) -> pd.Series:
    """
    Compute mean RMSE per forecast horizon step using sliding-window validation
    on a single series with a single model.
    """
    all_sq_errors = []
    start_index   = len(series) + 1
    stop_index    = start_index - 12

    for end_idx in range(start_index, stop_index, -1):
        test_s  = series[end_idx - test_periods: end_idx]
        train_s = series[:end_idx - test_periods]
        try:
            fc, _ = model_func(train_s, n_periods=test_periods)
            if len(test_s) != len(fc):
                continue
            aligned = pd.Series(fc.values, index=test_s.index)
            all_sq_errors.append(((test_s - aligned) ** 2).values)
        except Exception:
            continue

    if not all_sq_errors:
        return pd.Series(dtype='float64')

    rmse_by_horizon = np.sqrt(np.mean(np.array(all_sq_errors), axis=0))
    labels = [f"Step {i + 1}" for i in range(test_periods)]
    return pd.Series(rmse_by_horizon, index=labels, name=f'Mean RMSE - {model_name}')