"""
Standalone Web Server for Sales Forecasting Dashboard
Features dynamic Dark and Light Mode toggle, persistent state,
file upload support (CSV/Excel) with sheet selection, raw data preview,
and dynamic auto-column detection without altering existing design or tabs.
"""

import os
import sys
import io
import json
import re
import cgi
import http.server
import socketserver
import urllib.parse
from typing import Dict, Any, Optional, Tuple, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from utils.data_prep import load_raw_dataset, process_sales_data
from utils.evaluation import rolling_origin_validation, fit_best_future_forecast
from utils.insights import generate_business_insights

DATA_DIR = os.path.join(BASE_DIR, "data")
PORT = 8000

# Server-side cache for uploaded file & metadata
UPLOADED_DATA_CACHE: Dict[str, Any] = {
    "filename": None,
    "file_bytes": None,
    "df": None,
    "sheet_names": [],
    "selected_sheet": None,
    "date_col": None,
    "sales_col": None,
    "group_col": "None",
    "is_auto_detected": False,
}


def parse_uploaded_file(file_bytes: bytes, filename: str, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, List[str], Optional[str]]:
    """
    Parses CSV/Excel file bytes with auto-encoding and delimiter detection.
    Supports .csv, .xlsx, .xls and multi-sheet Excel files.
    """
    lower = filename.lower()
    if lower.endswith((".xlsx", ".xls")):
        try:
            engine = "xlrd" if lower.endswith(".xls") else "openpyxl"
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
            sheet_names = excel_file.sheet_names
            if not sheet_names:
                raise ValueError("Excel file contains no sheets.")
            chosen_sheet = sheet_name if sheet_name in sheet_names else sheet_names[0]
            df = pd.read_excel(excel_file, sheet_name=chosen_sheet)
            return df, sheet_names, chosen_sheet
        except Exception as e:
            raise ValueError(f"Corrupt or unreadable Excel file: {e}")
    elif lower.endswith(".csv"):
        # Auto-delimiter (comma, semicolon, tab) & encoding fallback
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            for sep in [None, ",", ";", "\t"]:
                try:
                    df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, sep=sep, engine="python")
                    if df is not None and not df.empty and df.shape[1] >= 2:
                        return df, [], None
                except Exception:
                    continue
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1", on_bad_lines="skip")
            return df, [], None
        except Exception as e:
            raise ValueError(f"Corrupt or unreadable CSV file: {e}")
    else:
        raise ValueError("Unsupported format. Please upload a .csv, .xlsx, or .xls file.")


def auto_detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Auto-detects Date, Sales, and Group columns from dataframe statistics and heuristics.
    """
    date_col = None
    sales_col = None
    group_col = None

    # 1. Date column: non-numeric columns that parse with >= 60% valid dates
    best_date_score = 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            valid_ratio = parsed.notna().sum() / max(1, len(df))
            if valid_ratio > best_date_score and valid_ratio >= 0.6:
                best_date_score = valid_ratio
                date_col = col
        except Exception:
            pass

    if not date_col:
        for col in df.columns:
            lower_name = str(col).lower()
            if any(k in lower_name for k in ["date", "time", "day", "period", "timestamp"]):
                date_col = col
                break

    # 2. Sales column: numeric column preferring sales/revenue/amount/qty keywords
    keywords = ["sales", "revenue", "amount", "qty", "demand", "units", "total", "volume"]
    num_candidates = []
    for col in df.columns:
        if col == date_col:
            continue
        try:
            s_clean = (
                df[col]
                .astype(str)
                .str.replace("₹", "", regex=False)
                .str.replace("Rs.", "", regex=False)
                .str.replace("Rs", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace("£", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            num_parsed = pd.to_numeric(s_clean, errors="coerce")
            if num_parsed.notna().sum() / max(1, len(df)) >= 0.6:
                name_lower = str(col).lower()
                kw_priority = 0
                for idx, kw in enumerate(keywords):
                    if kw in name_lower:
                        kw_priority = len(keywords) - idx
                        break
                num_candidates.append((col, kw_priority, num_parsed.std()))
        except Exception:
            pass

    if num_candidates:
        num_candidates.sort(key=lambda x: (x[1], x[2] if pd.notna(x[2]) else 0), reverse=True)
        sales_col = num_candidates[0][0]

    # 3. Group column (optional): low-cardinality text column
    group_candidates = []
    group_keywords = ["store", "product", "category", "region", "dept", "department", "channel", "segment"]
    for col in df.columns:
        if col in [date_col, sales_col]:
            continue
        nunique = df[col].nunique(dropna=True)
        if 1 < nunique <= 50:
            name_lower = str(col).lower()
            kw_match = any(gk in name_lower for gk in group_keywords)
            group_candidates.append((col, kw_match))

    if group_candidates:
        group_candidates.sort(key=lambda x: x[1], reverse=True)
        group_col = group_candidates[0][0]
    else:
        group_col = "None"

    return date_col, sales_col, group_col


def init_default_data():
    if UPLOADED_DATA_CACHE["df"] is None or UPLOADED_DATA_CACHE["df"].empty:
        file_path = os.path.join(DATA_DIR, "walmart_weekly_sales.csv")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                b = f.read()
            df = pd.read_csv(io.BytesIO(b))
            d, s, g = auto_detect_columns(df)
            UPLOADED_DATA_CACHE["filename"] = "walmart_weekly_sales.csv (Sample)"
            UPLOADED_DATA_CACHE["file_bytes"] = b
            UPLOADED_DATA_CACHE["df"] = df
            UPLOADED_DATA_CACHE["sheet_names"] = []
            UPLOADED_DATA_CACHE["selected_sheet"] = None
            UPLOADED_DATA_CACHE["date_col"] = d
            UPLOADED_DATA_CACHE["sales_col"] = s
            UPLOADED_DATA_CACHE["group_col"] = g
            UPLOADED_DATA_CACHE["is_auto_detected"] = True


def build_dashboard_html(
    dataset_choice=None,
    selected_group="Total (all groups)",
    horizon=None,
    custom_date_col=None,
    custom_sales_col=None,
    custom_group_col=None,
    selected_sheet=None,
    upload_error=None,
):
    init_default_data()

    sheet_options_html = ""
    raw_preview_html = ""

    # Multi-sheet Excel workbook selection check
    if selected_sheet and selected_sheet in UPLOADED_DATA_CACHE["sheet_names"]:
        if selected_sheet != UPLOADED_DATA_CACHE["selected_sheet"]:
            try:
                df, _, _ = parse_uploaded_file(
                    UPLOADED_DATA_CACHE["file_bytes"],
                    UPLOADED_DATA_CACHE["filename"],
                    sheet_name=selected_sheet,
                )
                UPLOADED_DATA_CACHE["df"] = df
                UPLOADED_DATA_CACHE["selected_sheet"] = selected_sheet
                d, s, g = auto_detect_columns(df)
                UPLOADED_DATA_CACHE["date_col"] = d
                UPLOADED_DATA_CACHE["sales_col"] = s
                UPLOADED_DATA_CACHE["group_col"] = g
            except Exception as e:
                upload_error = str(e)
                df = UPLOADED_DATA_CACHE["df"]
        else:
            df = UPLOADED_DATA_CACHE["df"]
    else:
        df = UPLOADED_DATA_CACHE["df"]

    source_name = UPLOADED_DATA_CACHE["filename"] or "Uploaded Dataset"
    if UPLOADED_DATA_CACHE.get("selected_sheet"):
        source_name += f" ({UPLOADED_DATA_CACHE['selected_sheet']})"

    # Column selection from auto-detection or user overrides
    date_col = custom_date_col or UPLOADED_DATA_CACHE["date_col"]
    sales_col = custom_sales_col or UPLOADED_DATA_CACHE["sales_col"]
    group_col = custom_group_col or UPLOADED_DATA_CACHE["group_col"]

    # Handle empty/unuploaded state cleanly
    if df is None or df.empty:
        error_banner = f"<div class='alert alert-danger mb-4'>{upload_error}</div>" if upload_error else ""
        return f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Sales Forecasting Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <style>
        body {{ background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        .sidebar-card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.35rem; }}
    </style>
</head>
<body class="p-4">
    <div class="container-fluid">
        <h2 class="fw-bold mb-3">📈 Enterprise Sales Forecasting Dashboard</h2>
        {error_banner}
        <div class="row g-4">
            <div class="col-lg-4">
                <div class="sidebar-card">
                    <h5 class="fw-bold mb-3"><i class="bi bi-cloud-arrow-up-fill me-2 text-primary"></i>Upload Dataset</h5>
                    <div class="mb-3">
                        <label class="form-label fw-semibold">Select File (.csv, .xlsx, .xls)</label>
                        <input type="file" id="emptyFileInput" class="form-control" accept=".csv,.xlsx,.xls">
                        <div id="emptyUploadStatusMsg" class="d-none mt-2"></div>
                    </div>
                    <hr>
                    <a href="/?dataset=walmart" class="btn btn-outline-secondary w-100">Back to Walmart Demo</a>
                </div>
            </div>
            <div class="col-lg-8">
                <div class="alert alert-info py-4 text-center">
                    <h5>👈 Please upload a CSV or Excel file to begin forecasting</h5>
                    <p class="text-secondary mb-0">Supports automatic date parsing, delimiter detection, and multi-sheet workbooks.</p>
                </div>
            </div>
        </div>
    </div>
    <script>
        document.getElementById("emptyFileInput")?.addEventListener("change", async function(e) {{
            e.preventDefault();
            const file = e.target.files && e.target.files[0];
            if (!file) return;
            const status = document.getElementById("emptyUploadStatusMsg");
            if (status) {{
                status.className = "alert alert-info py-2 px-2 small mb-2 d-flex align-items-center gap-2";
                status.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> Reading and processing ' + file.name + '...';
            }}
            try {{
                const formData = new FormData();
                formData.append("dataset_file", file);
                const uploadUrl = (window.location.protocol === "file:") ? "http://localhost:8000/upload" : "/upload";
                const res = await fetch(uploadUrl, {{ method: "POST", body: formData }});
                if (!res.ok) {{
                    const err = await res.json().catch(() => null);
                    throw new Error((err && err.error) ? err.error : ("Server returned HTTP " + res.status));
                }}
                const data = await res.json();
                if (!data.success) throw new Error(data.error || "Upload failed");
                if (data.html) {{
                    document.open();
                    document.write(data.html);
                    document.close();
                }}
            }} catch (err) {{
                console.error("Upload error:", err);
                if (status) {{
                    status.className = "alert alert-danger py-2 px-2 small mb-2";
                    status.innerHTML = '<i class="bi bi-exclamation-octagon-fill me-1"></i> <b>Upload failed:</b> ' + (err.message || err);
                }}
            }}
        }});
    </script>
</body>
</html>"""

    # 1. Natural Frequency Detection from date gaps
    detected_freq_label = "Weekly"
    if date_col and date_col in df.columns:
        try:
            from utils.data_prep import detect_frequency_and_seasonality
            sample_dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values().drop_duplicates()
            detected_freq_label, _, _, _ = detect_frequency_and_seasonality(pd.DatetimeIndex(sample_dates))
        except Exception:
            detected_freq_label = "Weekly"

    valid_freqs = ["Weekly", "Daily", "Monthly", "Yearly"]
    frequency_choice = dataset_choice if (dataset_choice and dataset_choice in valid_freqs) else detected_freq_label

    freq_config = {
        "Weekly": {"code": "W", "default_h": 13, "label": "Weekly"},
        "Daily": {"code": "D", "default_h": 14, "label": "Daily"},
        "Monthly": {"code": "MS", "default_h": 3, "label": "Monthly"},
        "Yearly": {"code": "YE", "default_h": 2, "label": "Yearly"},
    }
    freq_info = freq_config.get(frequency_choice, freq_config.get(detected_freq_label, freq_config["Weekly"]))
    resample_freq = freq_info["code"]
    default_h = freq_info["default_h"]

    if horizon is None or horizon <= 0:
        horizon = default_h

    # Frequency mismatch warning if user selection diverges from detected data cadence
    if frequency_choice != detected_freq_label:
        freq_warning_html = f"""
        <div class="alert alert-warning py-2 px-3 small my-2 d-flex align-items-center gap-2">
            <i class="bi bi-exclamation-triangle-fill text-warning"></i>
            <div><b>Frequency Notice:</b> Selected '<b>{frequency_choice}</b>' differs from data cadence (<b>Detected: {detected_freq_label}</b>). Resampling has been applied.</div>
        </div>
        """
    else:
        freq_warning_html = ""

    # Group options for filtering
    available_groups = ["Total (all groups)"]
    if group_col and group_col != "None" and group_col in df.columns:
        available_groups += sorted(df[group_col].dropna().astype(str).unique().tolist())

    if selected_group not in available_groups:
        selected_group = "Total (all groups)"

    # Process time series data pipeline
    prep = process_sales_data(
        df=df,
        date_col=date_col,
        sales_col=sales_col,
        group_col=group_col if group_col != "None" else None,
        selected_group=selected_group,
        resample_freq=resample_freq,
    )

    ts_df = prep["ts_df"]
    sales_series = ts_df["Sales"]
    n_obs = prep["n_observations"]
    seasonal_period = prep["seasonal_period"]
    pandas_freq = prep["pandas_freq"]
    dq = prep["data_quality_report"]

    # RAW DATA PREVIEW (Rows, Columns, Data Quality Summary, First 10 rows, Column summary)
    n_rows, n_cols = df.shape
    preview_rows = df.head(10)
    
    thead_html = "".join([f"<th>{col}</th>" for col in df.columns])
    tbody_10_html = "".join(
        [
            "<tr>" + "".join([f"<td>{str(val)}</td>" for val in row]) + "</tr>"
            for row in preview_rows.values
        ]
    )

    col_summary_rows = ""
    for col in df.columns:
        c_type = str(df[col].dtype)
        missing_count = int(df[col].isna().sum())
        missing_pct = round((missing_count / n_rows) * 100, 1)
        samples = ", ".join([str(x) for x in df[col].dropna().unique()[:3]])
        col_summary_rows += f"""
        <tr>
            <td class="fw-bold">{col}</td>
            <td><code>{c_type}</code></td>
            <td>{missing_count} ({missing_pct}%)</td>
            <td class="text-secondary small">{samples}</td>
        </tr>
        """

    # Data Quality Report Table (Before and After Cleaning)
    data_quality_table_html = f"""
    <div class="card border mb-3">
        <div class="card-header bg-body-tertiary py-2 d-flex justify-content-between align-items-center">
            <span class="fw-bold small"><i class="bi bi-shield-check text-success me-1"></i>Data Quality & Preprocessing Report</span>
            <span class="badge bg-success-subtle text-success-emphasis border border-success-subtle">Cleaned & Ready</span>
        </div>
        <div class="card-body p-2">
            <table class="table table-sm table-bordered text-center small mb-0 align-middle">
                <thead class="table-light">
                    <tr>
                        <th class="text-start">Data Quality Metric</th>
                        <th>Before Cleaning</th>
                        <th>After Cleaning</th>
                        <th class="text-start">Resolution Action</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="text-start">Missing Values</td>
                        <td><span class="badge bg-warning-subtle text-warning-emphasis">{dq['before']['missing_values']}</span></td>
                        <td><span class="badge bg-success-subtle text-success-emphasis">{dq['after']['missing_values']}</span></td>
                        <td class="text-start text-secondary">Filled via linear interpolation</td>
                    </tr>
                    <tr>
                        <td class="text-start">Duplicate Rows / Dates</td>
                        <td><span class="badge bg-warning-subtle text-warning-emphasis">{dq['before']['duplicate_records']}</span></td>
                        <td><span class="badge bg-success-subtle text-success-emphasis">{dq['after']['duplicate_records']}</span></td>
                        <td class="text-start text-secondary">Aggregated into continuous date totals</td>
                    </tr>
                    <tr>
                        <td class="text-start">Invalid Records (Unparseable / Non-numeric)</td>
                        <td><span class="badge bg-danger-subtle text-danger-emphasis">{dq['before']['invalid_records']}</span></td>
                        <td><span class="badge bg-success-subtle text-success-emphasis">{dq['after']['invalid_records']}</span></td>
                        <td class="text-start text-secondary">Filtered and dropped safely</td>
                    </tr>
                    <tr class="table-light fw-bold">
                        <td class="text-start">Total Records Count</td>
                        <td>{dq['before']['total_records']}</td>
                        <td>{dq['after']['total_records']}</td>
                        <td class="text-start text-success">Continuous Regular Series</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    """

    raw_preview_html = f"""
    <div class="card-panel mt-3">
        <div class="d-flex justify-content-between align-items-center mb-2">
            <h6 class="fw-bold mb-0"><i class="bi bi-table me-2 text-primary"></i>Raw Data Preview ({n_rows:,} rows, {n_cols} columns)</h6>
            <button class="btn btn-sm btn-outline-secondary" type="button" data-bs-toggle="collapse" data-bs-target="#collapseDataPreview">
                Toggle Table
            </button>
        </div>
        <div class="collapse show" id="collapseDataPreview">
            {data_quality_table_html}
            <div class="table-responsive my-2" style="max-height: 240px; overflow-y: auto;">
                <table class="table table-sm table-striped table-hover small mb-0">
                    <thead class="table-light sticky-top"><tr>{thead_html}</tr></thead>
                    <tbody>{tbody_10_html}</tbody>
                </table>
            </div>
            <div class="accordion accordion-flush mt-2" id="accSummary">
                <div class="accordion-item bg-transparent">
                    <h2 class="accordion-header">
                        <button class="accordion-button collapsed py-2 px-0 bg-transparent small fw-semibold" type="button" data-bs-toggle="collapse" data-bs-target="#colSummaryPanel">
                            Column Summary & Diagnostics ({n_cols} columns)
                        </button>
                    </h2>
                    <div id="colSummaryPanel" class="accordion-collapse collapse">
                        <div class="table-responsive my-2">
                            <table class="table table-sm table-bordered small">
                                <thead class="table-light">
                                    <tr><th>Column Name</th><th>Detected Type</th><th>Missing Count</th><th>Sample Values</th></tr>
                                </thead>
                                <tbody>{col_summary_rows}</tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """

    # Sheet selection dropdown HTML if Excel has multiple sheets
    if len(UPLOADED_DATA_CACHE["sheet_names"]) > 1:
        options = "".join(
            [
                f"<option value='{s}' {'selected' if s == UPLOADED_DATA_CACHE['selected_sheet'] else ''}>{s}</option>"
                for s in UPLOADED_DATA_CACHE["sheet_names"]
            ]
        )
        sheet_options_html = f"""
        <div class="mb-2 mt-2">
            <label class="form-label fw-semibold small text-secondary">Excel Sheet</label>
            <select name="selected_sheet" class="form-select form-select-sm" onchange="switchExcelSheet(this.value, '{frequency_choice}')">
                {options}
            </select>
        </div>
        """

    # Run backtest evaluation and future forecast
    eval_res = rolling_origin_validation(
        ts_series=sales_series,
        horizon=int(horizon),
        seasonal_period=int(seasonal_period),
        freq_label=prep["detected_label"],
        pandas_freq=pandas_freq,
    )

    future_res = fit_best_future_forecast(
        ts_series=sales_series,
        best_model_name=eval_res["best_model_name"],
        horizon=int(horizon),
        seasonal_period=int(seasonal_period),
        pandas_freq=pandas_freq,
        freq_label=prep["detected_label"],
    )

    insights = generate_business_insights(
        ts_df=ts_df,
        eval_results=eval_res,
        prep_data=prep,
        future_forecast_df=future_res["forecast_df"],
        horizon=int(horizon),
    )

    stats = insights["stats"]
    best_model_name = eval_res["best_model_name"]
    best_rmse = eval_res["best_rmse"]
    improvement_pct = eval_res["improvement_pct"]
    baseline_used = eval_res["baseline_used"]

    # 1. Overview Chart
    fig_overview = go.Figure()
    fig_overview.add_trace(
        go.Scatter(
            x=ts_df.index.strftime("%Y-%m-%d"),
            y=ts_df["Sales"],
            mode="lines",
            name="Actual Sales",
            line=dict(color="#3b82f6", width=2.4),
            hovertemplate="<b>Date</b>: %{x}<br><b>Actual Sales</b>: ₹%{y:,.2f}<extra></extra>",
        )
    )
    rolling_window = max(3, min(12, len(ts_df) // 10))
    ts_df["Rolling_Avg"] = ts_df["Sales"].rolling(rolling_window, min_periods=1).mean()
    fig_overview.add_trace(
        go.Scatter(
            x=ts_df.index.strftime("%Y-%m-%d"),
            y=ts_df["Rolling_Avg"],
            mode="lines",
            name=f"Moving Avg ({rolling_window}p)",
            line=dict(color="#f59e0b", width=1.8, dash="dot"),
            hovertemplate=f"<b>Date</b>: %{{x}}<br><b>Moving Avg ({rolling_window}p)</b>: ₹%{{y:,.2f}}<extra></extra>",
        )
    )
    outliers_df = ts_df[ts_df["Is_Outlier"]]
    if not outliers_df.empty:
        fig_overview.add_trace(
            go.Scatter(
                x=outliers_df.index.strftime("%Y-%m-%d"),
                y=outliers_df["Sales"],
                mode="markers",
                name=f"Outliers ({len(outliers_df)})",
                marker=dict(color="#ef4444", size=9, symbol="diamond"),
                hovertemplate="<b>Date</b>: %{x}<br><b>Outlier Sales</b>: ₹%{y:,.2f}<extra></extra>",
            )
        )
    fig_overview.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=90, r=30, t=55, b=65, pad=6),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        xaxis=dict(
            showgrid=True,
            automargin=True,
            title=dict(
                text="Date",
                font=dict(size=13, weight="bold"),
                standoff=15,
            ),
            tickfont=dict(size=11),
            showline=True,
            linecolor="#cbd5e1",
            linewidth=1,
            tickangle=-25 if len(ts_df) > 50 else 0,
        ),
        yaxis=dict(
            showgrid=True,
            automargin=True,
            title=dict(
                text="Sales (₹)",
                font=dict(size=13, weight="bold"),
                standoff=15,
            ),
            tickprefix="₹",
            tickformat=",.0f",
            tickfont=dict(size=11),
            showline=True,
            linecolor="#cbd5e1",
            linewidth=1,
        ),
    )
    chart_overview_html = pio.to_html(
        fig_overview,
        full_html=False,
        include_plotlyjs="cdn",
        div_id="chart_overview",
        config={"responsive": True, "displayModeBar": True, "displaylogo": False},
    )

    # Group bar chart
    if prep["has_groups"] and prep["group_summary"] is not None:
        grp_df = prep["group_summary"].head(10)
        fig_grp = px.bar(
            grp_df,
            x=group_col,
            y="Total_Sales",
            color="Total_Sales",
            color_continuous_scale="Blues",
            labels={"Total_Sales": "Total Sales (₹)", group_col: group_col},
        )
        fig_grp.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        chart_grp_html = pio.to_html(fig_grp, full_html=False, include_plotlyjs=False, div_id="chart_group")
    else:
        fig_hist = px.histogram(ts_df, x="Sales", nbins=25, color_discrete_sequence=["#3b82f6"])
        fig_hist.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=10, b=10))
        chart_grp_html = pio.to_html(fig_hist, full_html=False, include_plotlyjs=False, div_id="chart_group")

    # 2. Decomposition Chart
    from statsmodels.tsa.seasonal import seasonal_decompose
    if seasonal_period > 1 and len(ts_df) >= 2 * seasonal_period:
        try:
            decomp = seasonal_decompose(ts_df["Sales"], model="additive", period=int(seasonal_period), extrapolate_trend="freq")
            fig_decomp = make_subplots(
                rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                subplot_titles=("Observed Sales (₹)", "Underlying Trend (₹)", f"Seasonal Component (Period = {seasonal_period}) (₹)", "Residual / Noise")
            )
            fig_decomp.add_trace(go.Scatter(x=ts_df.index.strftime("%Y-%m-%d"), y=decomp.observed, mode="lines", line=dict(color="#3b82f6", width=1.5)), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(x=ts_df.index.strftime("%Y-%m-%d"), y=decomp.trend, mode="lines", line=dict(color="#10b981", width=2)), row=2, col=1)
            fig_decomp.add_trace(go.Scatter(x=ts_df.index.strftime("%Y-%m-%d"), y=decomp.seasonal, mode="lines", line=dict(color="#f59e0b", width=1.5)), row=3, col=1)
            fig_decomp.add_trace(go.Scatter(x=ts_df.index.strftime("%Y-%m-%d"), y=decomp.resid, mode="markers", marker=dict(color="#64748b", size=3.5)), row=4, col=1)
            fig_decomp.update_layout(template="plotly_white", height=600, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
            chart_decomp_html = pio.to_html(fig_decomp, full_html=False, include_plotlyjs=False, div_id="chart_decomp")
        except Exception:
            chart_decomp_html = "<div class='alert alert-warning'>Decomposition requires at least 2 full seasonal cycles.</div>"
    else:
        chart_decomp_html = f"<div class='alert alert-info py-3'><i class='bi bi-info-circle me-1'></i><b>Annual Data Rhythm:</b> The series is aggregated at the <b>{prep['detected_label']}</b> cadence (seasonal period = {seasonal_period}). Annual aggregation consolidates cyclical sub-periods into yearly totals; view overall historical trajectory and trend in the Overview and Forecast tabs.</div>"

    # Monthly Seasonality / Annual Comparison
    if prep["detected_label"] != "Yearly" and len(ts_df.index.month.unique()) > 1:
        month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        monthly_df = ts_df.groupby(ts_df.index.month_name())["Sales"].mean().reindex(month_order).dropna().reset_index().rename(columns={"Date": "Month", "Sales": "Avg_Sales"})
        fig_month = px.bar(monthly_df, x="Month", y="Avg_Sales", color="Avg_Sales", color_continuous_scale="Purples", labels={"Avg_Sales": "Avg Sales (₹)"})
        fig_month.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        chart_month_html = pio.to_html(fig_month, full_html=False, include_plotlyjs=False, div_id="chart_month")
    else:
        annual_df = ts_df.reset_index()
        annual_df["Year"] = annual_df["Date"].dt.year.astype(str)
        fig_year = px.bar(annual_df, x="Year", y="Sales", color="Sales", color_continuous_scale="Teal", labels={"Sales": "Annual Sales (₹)"})
        fig_year.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        chart_month_html = pio.to_html(fig_year, full_html=False, include_plotlyjs=False, div_id="chart_month")

    # 3. Forecast Chart
    future_df = future_res["forecast_df"]
    history_tail_len = min(len(ts_df), max(int(horizon) * 3, 26))
    hist_tail = ts_df.iloc[-history_tail_len:]

    fig_future = go.Figure()
    fig_future.add_trace(go.Scatter(x=hist_tail.index.strftime("%Y-%m-%d"), y=hist_tail["Sales"], mode="lines", name="Recent Actuals", line=dict(color="#64748b", width=2), hovertemplate="<b>Date</b>: %{x}<br><b>Actual Sales</b>: ₹%{y:,.2f}<extra></extra>"))
    fig_future.add_trace(go.Scatter(x=future_df.index.strftime("%Y-%m-%d"), y=future_df["Upper_CI_95"], mode="lines", line=dict(width=0), showlegend=False, hovertemplate="<b>Date</b>: %{x}<br><b>Upper 95% CI</b>: ₹%{y:,.2f}<extra></extra>"))
    fig_future.add_trace(go.Scatter(x=future_df.index.strftime("%Y-%m-%d"), y=future_df["Lower_CI_95"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(59, 130, 246, 0.22)", name="95% Confidence Band", hovertemplate="<b>Date</b>: %{x}<br><b>Lower 95% CI</b>: ₹%{y:,.2f}<extra></extra>"))
    fig_future.add_trace(go.Scatter(x=future_df.index.strftime("%Y-%m-%d"), y=future_df["Forecast"], mode="lines+markers", name=f"Forecast ({best_model_name})", line=dict(color="#2563eb", width=2.8), marker=dict(size=6, color="#1d4ed8"), hovertemplate=f"<b>Date</b>: %{{x}}<br><b>Forecast ({best_model_name})</b>: ₹%{{y:,.2f}}<extra></extra>"))
    fig_future.update_layout(template="plotly_white", height=450, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(showgrid=True, title="Date"), yaxis=dict(showgrid=True, title="Sales (₹)"))
    chart_future_html = pio.to_html(fig_future, full_html=False, include_plotlyjs=False, div_id="chart_future")

    # 4. Model Comparison Chart
    metrics_df = eval_res["metrics_df"]
    plot_m = metrics_df[metrics_df["Status"] == "Success"].copy()
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=plot_m["Model"], y=plot_m["RMSE"], name="RMSE", marker_color="#2563eb", hovertemplate="<b>%{x}</b><br>RMSE: ₹%{y:,.2f}<extra></extra>"))
    fig_bar.add_trace(go.Bar(x=plot_m["Model"], y=plot_m["MAE"], name="MAE", marker_color="#93c5fd", hovertemplate="<b>%{x}</b><br>MAE: ₹%{y:,.2f}<extra></extra>"))
    fig_bar.update_layout(barmode="group", template="plotly_white", height=340, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), yaxis=dict(showgrid=True, title="Error (₹)"))
    chart_bar_html = pio.to_html(fig_bar, full_html=False, include_plotlyjs=False, div_id="chart_bar")

    # Horizon degradation
    h_data = eval_res["horizon_error_data"]
    fig_horizon = go.Figure()
    h_steps = list(range(1, int(horizon) + 1))
    colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]
    for idx, (m_name, step_errs) in enumerate(h_data.items()):
        if len(step_errs) == len(h_steps):
            fig_horizon.add_trace(go.Scatter(x=h_steps, y=step_errs, mode="lines+markers", name=m_name, line=dict(color=colors[idx % len(colors)], width=2), marker=dict(size=4), hovertemplate=f"<b>{m_name} (h=%{{x}})</b><br>MAE: ₹%{{y:,.2f}}<extra></extra>"))
    fig_horizon.update_layout(template="plotly_white", height=340, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), xaxis=dict(title="Forecast Step (h)", tickmode="linear"), yaxis=dict(showgrid=True, title="MAE (₹)"))
    chart_horizon_html = pio.to_html(fig_horizon, full_html=False, include_plotlyjs=False, div_id="chart_horizon")

    metrics_rows = ""
    for _, row in metrics_df.iterrows():
        is_champion = (row["Model"] == best_model_name and row["Status"] == "Success")
        if is_champion:
            badge = "<span class='badge bg-success ms-1'>⭐ Best Model</span>"
            row_cls = 'class="table-primary fw-bold"'
        elif row["Status"] == "Skipped":
            badge = "<span class='badge bg-warning-subtle text-warning-emphasis border border-warning-subtle ms-1'>Skipped</span>"
            row_cls = 'class="text-secondary"'
        elif row["Status"] == "Failed":
            badge = "<span class='badge bg-danger-subtle text-danger-emphasis border border-danger-subtle ms-1'>Failed</span>"
            row_cls = 'class="text-secondary"'
        else:
            badge = "<span class='badge bg-secondary-subtle text-secondary-emphasis ms-1'>Evaluated</span>"
            row_cls = ""

        mae_str = f"₹{row['MAE']:,.2f}" if pd.notna(row['MAE']) else "—"
        rmse_str = f"₹{row['RMSE']:,.2f}" if pd.notna(row['RMSE']) else "—"
        smape_str = f"{row['sMAPE (%)']:.2f}%" if pd.notna(row['sMAPE (%)']) else "—"

        status_detail = ""
        if row["Status"] != "Success":
            status_detail = f"<div class='small text-muted fst-italic mt-1'><i class='bi bi-info-circle me-1'></i>{row['Details']}</div>"

        metrics_rows += f"""
        <tr {row_cls}>
            <td><b>{row['Model']}</b> {badge}{status_detail}</td>
            <td>{mae_str}</td>
            <td>{rmse_str}</td>
            <td>{smape_str}</td>
        </tr>
        """

    # Key Insights Box HTML
    ki = insights.get("key_insights", {})
    demand_icon = "📈" if ki.get("demand_direction_type") == "increasing" else "📉"
    demand_color = "success" if ki.get("demand_direction_type") == "increasing" else "warning"

    key_insights_box_html = f"""
    <div class="card-panel border-primary mb-4" style="border-left: 4px solid #2563eb; background: rgba(37, 99, 235, 0.04);">
        <div class="d-flex align-items-center mb-3">
            <span class="fs-4 me-2">💡</span>
            <h5 class="fw-bold mb-0 text-primary">Key Insights</h5>
            <span class="badge bg-primary ms-auto">Auto-Generated</span>
        </div>
        
        <div class="row g-3 mb-3">
            <div class="col-md-3 col-sm-6">
                <div class="p-3 rounded bg-body border h-100">
                    <span class="text-secondary small fw-semibold">Demand Direction</span>
                    <div class="fs-5 fw-bold mt-1 text-{demand_color}">
                        {demand_icon} {ki.get('demand_direction', 'N/A')}
                    </div>
                </div>
            </div>
            <div class="col-md-3 col-sm-6">
                <div class="p-3 rounded bg-body border h-100">
                    <span class="text-secondary small fw-semibold">Peak & Low Season</span>
                    <div class="small fw-semibold mt-1">
                        <div>🌟 <b>Peak:</b> {ki.get('peak_season', 'N/A')}</div>
                        <div class="text-muted">❄️ <b>Low:</b> {ki.get('low_season', 'N/A')}</div>
                    </div>
                </div>
            </div>
            <div class="col-md-3 col-sm-6">
                <div class="p-3 rounded bg-body border h-100">
                    <span class="text-secondary small fw-semibold">Anomalies Count</span>
                    <div class="fs-5 fw-bold mt-1 text-danger">
                        ⚠️ {ki.get('anomalies_count', 0)} Outlier(s)
                    </div>
                </div>
            </div>
            <div class="col-md-3 col-sm-6">
                <div class="p-3 rounded bg-body border h-100">
                    <span class="text-secondary small fw-semibold">Best Model & Gain</span>
                    <div class="small fw-bold mt-1 text-primary">
                        🏆 {ki.get('best_model_name', best_model_name)}
                        <div class="text-muted small">+{ki.get('improvement_pct', 0.0)}% vs {ki.get('baseline_used', 'Baseline')}</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="p-3 rounded bg-body border">
            <h6 class="fw-bold mb-2"><i class="bi bi-clipboard-check-fill text-primary me-2"></i>Recommended Actions</h6>
            <ul class="mb-0 small ps-3">
                <li class="mb-2"><b>📦 Increase Inventory:</b> {ki.get('recommended_actions', {}).get('increase_inventory', '')}</li>
                <li class="mb-2"><b>👥 Plan Staffing:</b> {ki.get('recommended_actions', {}).get('plan_staffing', '')}</li>
                <li><b>📅 Plan for Seasonal Demand:</b> {ki.get('recommended_actions', {}).get('plan_for_seasonal_demand', '')}</li>
            </ul>
        </div>
    </div>
    """

    def format_insight_text(text: str) -> str:
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
        return text.replace("\n", "<br>")

    peak_html = format_insight_text(insights["peak_analysis"])
    stock_html = format_insight_text(insights["safety_stock_text"])
    staff_html = format_insight_text(insights["staffing_text"])
    retrain_html = format_insight_text(insights["retraining_advice"])
    exec_summary_html = format_insight_text(insights["exec_summary"])

    group_options_html = ""
    for g in available_groups:
        sel = "selected" if g == selected_group else ""
        group_options_html += f"<option value='{g}' {sel}>{g}</option>"

    # Column dropdowns HTML with Auto-detected badge
    col_date_options = "".join(
        [f"<option value='{c}' {'selected' if c == date_col else ''}>{c}</option>" for c in df.columns]
    )
    col_sales_options = "".join(
        [f"<option value='{c}' {'selected' if c == sales_col else ''}>{c}</option>" for c in df.columns]
    )
    col_group_options = "<option value='None' " + ("selected" if group_col == "None" else "") + ">None</option>"
    col_group_options += "".join(
        [f"<option value='{c}' {'selected' if c == group_col else ''}>{c}</option>" for c in df.columns]
    )

    badge_html = '<span class="badge bg-info-subtle text-info-emphasis ms-1">Auto-detected</span>'

    error_banner = f"<div class='alert alert-danger mb-4'>{upload_error}</div>" if upload_error else ""

    # Upload file input widget in sidebar
    upload_input_html = f"""
    <div class="mb-3 p-2 border rounded bg-body-tertiary">
        <label class="form-label fw-semibold small mb-1"><i class="bi bi-cloud-arrow-up me-1"></i>Select File (.csv, .xlsx, .xls)</label>
        <input type="file" name="file" class="form-control form-control-sm mb-2" accept=".csv,.xlsx,.xls">
        <button type="submit" formaction="/upload" formmethod="POST" formenctype="multipart/form-data" class="btn btn-sm btn-outline-primary w-100 fw-semibold">
            Upload & Preview Data
        </button>
    </div>
    """

    # Build full responsive HTML with Light/Dark Mode CSS variables & switcher
    html = f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enterprise Sales Forecasting Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    <style>
        :root {{
            --bg-body: #f8fafc;
            --card-bg: #ffffff;
            --card-border: #e2e8f0;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --kpi-subtext-color: #64748b;
            --highlight-color: #2563eb;
        }}

        [data-bs-theme="dark"] {{
            --bg-body: #0b1120;
            --card-bg: #1e293b;
            --card-border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --kpi-subtext-color: #94a3b8;
            --highlight-color: #60a5fa;
        }}

        body {{
            background-color: var(--bg-body);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            transition: background-color 0.25s ease, color 0.25s ease;
        }}

        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 1.2rem 1.35rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            height: 100%;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            border-color: var(--highlight-color);
        }}
        .kpi-label {{
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 0.35rem;
        }}
        .kpi-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-main);
            line-height: 1.2;
        }}
        .kpi-subtext {{
            font-size: 0.82rem;
            color: #10b981;
            font-weight: 600;
            margin-top: 0.35rem;
        }}

        .sidebar-card, .card-panel {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            padding: 1.35rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
            margin-bottom: 1.5rem;
            transition: background-color 0.25s ease, border-color 0.25s ease;
        }}

        .nav-tabs {{
            border-bottom: 2px solid var(--card-border);
        }}
        .nav-tabs .nav-link {{
            color: var(--text-muted);
            font-weight: 600;
            border: none;
            border-bottom: 3px solid transparent;
            padding: 0.75rem 1.25rem;
            transition: all 0.2s ease;
        }}
        .nav-tabs .nav-link:hover {{
            color: var(--highlight-color);
        }}
        .nav-tabs .nav-link.active {{
            color: var(--highlight-color);
            background: transparent;
            border-bottom: 3px solid var(--highlight-color);
        }}

        .theme-switch-btn {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.95rem;
            font-weight: 600;
            border-radius: 9999px;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
    </style>
</head>
<body class="p-3 p-md-4">
    <div class="container-fluid">
        <!-- Top Nav / Header -->
        <div class="d-flex flex-wrap justify-content-between align-items-center mb-4 gap-3">
            <div>
                <h2 class="fw-bold mb-1 d-flex align-items-center gap-2">
                    <span>📈</span> Enterprise Sales Forecasting Dashboard
                </h2>
                <div class="text-secondary small">
                    Dataset: <b>{source_name}</b> • Cadence: <b>{prep['detected_label']}</b> • Observations: <b>{n_obs}</b> • Seasonal Cycle: <b>{seasonal_period}</b>
                </div>
            </div>
            <div class="d-flex align-items-center gap-2">
                <!-- Theme Toggle Button -->
                <button type="button" id="themeToggleBtn" class="btn btn-outline-secondary theme-switch-btn" onclick="toggleDashboardTheme()">
                    <i class="bi bi-moon-stars-fill" id="themeIcon"></i>
                    <span id="themeText">Dark Mode</span>
                </button>
            </div>
        </div>

        {error_banner}

        <!-- Top 5 KPI Cards -->
        <div class="row g-3 mb-4">
            <div class="col-12 col-sm-6 col-xl">
                <div class="kpi-card">
                    <div class="kpi-label">Total Historical Sales</div>
                    <div class="kpi-value">₹{stats['total_sales']:,.0f}</div>
                    <div class="kpi-subtext text-secondary">Min: ₹{stats['min_sales']:,.0f}</div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-xl">
                <div class="kpi-card">
                    <div class="kpi-label">Average Sales / Period</div>
                    <div class="kpi-value">₹{stats['mean_sales']:,.1f}</div>
                    <div class="kpi-subtext text-secondary">Median: ₹{stats['median_sales']:,.1f}</div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-xl">
                <div class="kpi-card">
                    <div class="kpi-label">All-Time Peak Sales</div>
                    <div class="kpi-value">₹{stats['max_sales']:,.0f}</div>
                    <div class="kpi-subtext text-primary">{stats['peak_date']}</div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-xl">
                <div class="kpi-card">
                    <div class="kpi-label">Optimal Model</div>
                    <div class="kpi-value text-primary fs-3">{best_model_name}</div>
                    <div class="kpi-subtext"><span class="badge bg-success">Rank #1 by RMSE</span></div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-xl">
                <div class="kpi-card">
                    <div class="kpi-label">Best RMSE</div>
                    <div class="kpi-value">₹{best_rmse:,.1f}</div>
                    <div class="kpi-subtext">{improvement_pct:+.1f}% vs {baseline_used}</div>
                </div>
            </div>
        </div>

        <div class="row g-4">
            <!-- Sidebar Controls -->
            <div class="col-lg-3">
                <!-- 1. SEPARATE FILE UPLOAD CARD (ABOVE Controls & Parameters) -->
                <div class="sidebar-card mb-3">
                    <h5 class="fw-bold mb-2 d-flex align-items-center gap-2">
                        <i class="bi bi-cloud-arrow-up-fill text-primary"></i> Upload Dataset
                    </h5>
                    <p class="small text-secondary mb-2">Upload CSV or Excel file (.csv, .xlsx, .xls) directly from your computer:</p>
                    <div class="mb-2">
                        <input type="file" id="datasetFileInput" class="form-control form-control-sm mb-2" accept=".csv,.xlsx,.xls">
                        <div id="uploadStatusMsg" class="d-none"></div>
                    </div>
                    {sheet_options_html}
                    <div class="mt-3 pt-2 border-top small d-flex flex-column gap-1">
                        <span class="text-secondary fw-semibold">📁 Uploaded File:</span>
                        <span class="badge bg-secondary-subtle text-secondary-emphasis text-start py-2 px-2 text-wrap" style="word-break: break-all; font-size: 0.8rem;">
                            {source_name}
                        </span>
                        <span class="text-muted small">
                            <b>{n_rows:,}</b> rows • <b>{n_cols}</b> columns
                        </span>
                    </div>
                </div>

                <!-- 2. CONTROLS & PARAMETERS WITH UPDATED DATASET DROPDOWN -->
                <div class="sidebar-card">
                    <h5 class="fw-bold mb-3 d-flex align-items-center gap-2">
                        <i class="bi bi-sliders"></i> Controls & Parameters
                    </h5>
                    <form method="GET" action="/" id="forecastForm">
                        <div class="mb-3">
                            <label class="form-label fw-semibold d-flex justify-content-between align-items-center">
                                <span>Dataset</span>
                                <span class="badge bg-primary-subtle text-primary border border-primary-subtle">Detected: {detected_freq_label}</span>
                            </label>
                            <select name="dataset" class="form-select" onchange="this.form.submit()">
                                <option value="Weekly" {'selected' if frequency_choice=='Weekly' else ''}>Weekly</option>
                                <option value="Daily" {'selected' if frequency_choice=='Daily' else ''}>Daily</option>
                                <option value="Monthly" {'selected' if frequency_choice=='Monthly' else ''}>Monthly</option>
                                <option value="Yearly" {'selected' if frequency_choice=='Yearly' else ''}>Yearly</option>
                            </select>
                            {freq_warning_html}
                        </div>

                        <!-- Dynamic Auto-detected Column Dropdowns with badge -->
                        <div class="mb-3">
                            <label class="form-label fw-semibold small d-flex align-items-center">
                                Date Column {badge_html}
                            </label>
                            <select name="date_col" class="form-select form-select-sm" onchange="this.form.submit()">
                                {col_date_options}
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label fw-semibold small d-flex align-items-center">
                                Sales Column {badge_html}
                            </label>
                            <select name="sales_col" class="form-select form-select-sm" onchange="this.form.submit()">
                                {col_sales_options}
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label fw-semibold small d-flex align-items-center">
                                Group Column (Optional) {badge_html}
                            </label>
                            <select name="group_col" class="form-select form-select-sm" onchange="this.form.submit()">
                                {col_group_options}
                            </select>
                        </div>

                        <div class="mb-3">
                            <label class="form-label fw-semibold">Group Subset</label>
                            <select name="group" class="form-select" onchange="this.form.submit()">
                                {group_options_html}
                            </select>
                        </div>
                        <div class="mb-3">
                            <label class="form-label fw-semibold">Forecast Horizon (steps)</label>
                            <input type="number" name="horizon" class="form-control" value="{horizon}" min="1" max="52">
                        </div>
                        <button type="submit" class="btn btn-primary w-100 fw-bold py-2">
                            <i class="bi bi-lightning-charge-fill me-1"></i> Run Forecast & Backtest
                        </button>
                    </form>
                    
                    <hr class="my-4">
                    <h6 class="fw-bold">📁 Dynamic Column Mappings</h6>
                    <ul class="list-unstyled small text-secondary mb-0">
                        <li class="mb-1"><b>Date Column:</b> <span class="badge bg-secondary-subtle text-secondary-emphasis">{date_col}</span></li>
                        <li class="mb-1"><b>Sales Column:</b> <span class="badge bg-secondary-subtle text-secondary-emphasis">{sales_col}</span></li>
                        <li><b>Group Column:</b> <span class="badge bg-secondary-subtle text-secondary-emphasis">{group_col}</span></li>
                    </ul>

                    <!-- Raw Data Preview Accordion Panel in Sidebar / Bottom -->
                    {raw_preview_html}
                </div>
            </div>

            <!-- Tab Panels -->
            <div class="col-lg-9">
                <ul class="nav nav-tabs mb-3" id="dashTabs" role="tablist">
                    <li class="nav-item"><button class="nav-link active" id="overview-tab" data-bs-toggle="tab" data-bs-target="#overview" type="button">📊 Overview</button></li>
                    <li class="nav-item"><button class="nav-link" id="decomp-tab" data-bs-toggle="tab" data-bs-target="#decomp" type="button">🌊 Trend & Seasonality</button></li>
                    <li class="nav-item"><button class="nav-link" id="forecast-tab" data-bs-toggle="tab" data-bs-target="#forecast" type="button">🎯 Forecast</button></li>
                    <li class="nav-item"><button class="nav-link" id="compare-tab" data-bs-toggle="tab" data-bs-target="#compare" type="button">🏆 Model Comparison</button></li>
                    <li class="nav-item"><button class="nav-link" id="insights-tab" data-bs-toggle="tab" data-bs-target="#insights" type="button">💡 Recommendations</button></li>
                </ul>

                <div class="tab-content">
                    <!-- Tab 1: Overview -->
                    <div class="tab-pane fade show active" id="overview">
                        <div class="card-panel">
                            <h5 class="fw-bold mb-1">Sales Trajectory & Outlier Analysis</h5>
                            <p class="text-secondary small mb-3">Interactive historical timeline with IQR anomalies and 10-period moving average.</p>
                            {chart_overview_html}
                        </div>
                        <div class="card-panel">
                            <h5 class="fw-bold mb-1">Sales Distribution & Volume Breakdown</h5>
                            {chart_grp_html}
                        </div>
                    </div>

                    <!-- Tab 2: Trend & Seasonality -->
                    <div class="tab-pane fade" id="decomp">
                        <div class="card-panel">
                            <h5 class="fw-bold mb-1">Classical Time Series Decomposition</h5>
                            <p class="text-secondary small mb-3">Additive components: Observed = Trend + Seasonal + Residual.</p>
                            {chart_decomp_html}
                        </div>
                        <div class="card-panel">
                            <h5 class="fw-bold mb-1">Average Sales by Month of Year</h5>
                            {chart_month_html}
                        </div>
                    </div>

                    <!-- Tab 3: Forecast -->
                    <div class="tab-pane fade" id="forecast">
                        <div class="card-panel">
                            <h5 class="fw-bold mb-1">Future Out-of-Sample Projections ({horizon} Periods Ahead)</h5>
                            <p class="text-secondary small mb-3">Champion Model: <b>{best_model_name}</b> with shaded 95% Confidence Intervals.</p>
                            {chart_future_html}
                        </div>
                    </div>

                    <!-- Tab 4: Model Comparison -->
                    <div class="tab-pane fade" id="compare">
                        <div class="card-panel">
                            <h5 class="fw-bold mb-3">Cross-Validation Leaderboard (Rolling-Origin Backtest)</h5>
                            <div class="table-responsive">
                                <table class="table table-hover align-middle">
                                    <thead>
                                        <tr>
                                            <th>Model</th>
                                            <th>MAE (₹)</th>
                                            <th>RMSE (₹)</th>
                                            <th>sMAPE (%)</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {metrics_rows}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <div class="row g-3">
                            <div class="col-md-6">
                                <div class="card-panel">
                                    <h6 class="fw-bold mb-2">Error Comparison (RMSE vs MAE)</h6>
                                    {chart_bar_html}
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card-panel">
                                    <h6 class="fw-bold mb-2">Horizon Error Degradation (h = 1 to H)</h6>
                                    {chart_horizon_html}
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Tab 5: Recommendations -->
                    <div class="tab-pane fade" id="insights">
                        {key_insights_box_html}
                        <div class="alert alert-info py-3 mb-4">
                            <b>Executive Summary:</b><br>{exec_summary_html}
                        </div>
                        <div class="row g-4">
                            <div class="col-md-6">
                                <div class="card-panel">
                                    <h5 class="fw-bold">🛒 Demand Patterns</h5>
                                    <div class="small">{peak_html}</div>
                                    <hr>
                                    <h5 class="fw-bold">📦 Safety Stock & Inventory</h5>
                                    <div class="small">{stock_html}</div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card-panel">
                                    <h5 class="fw-bold">👥 Staffing & Capacity</h5>
                                    <div class="small">{staff_html}</div>
                                    <hr>
                                    <h5 class="fw-bold">🔄 Retraining Protocols</h5>
                                    <div class="small">{retrain_html}</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function updatePlotlyTheme(isDark) {{
            const templateName = isDark ? "plotly_dark" : "plotly_white";
            const bgColor = isDark ? "#1e293b" : "#ffffff";
            const fontColor = isDark ? "#f8fafc" : "#0f172a";
            const gridColor = isDark ? "#334155" : "#e2e8f0";

            const chartIds = ["chart_overview", "chart_group", "chart_decomp", "chart_month", "chart_future", "chart_bar", "chart_horizon"];
            chartIds.forEach(id => {{
                const el = document.getElementById(id);
                if (el && typeof Plotly !== "undefined" && el.layout) {{
                    Plotly.relayout(el, {{
                        "template": templateName,
                        "paper_bgcolor": bgColor,
                        "plot_bgcolor": bgColor,
                        "font.color": fontColor,
                        "xaxis.gridcolor": gridColor,
                        "yaxis.gridcolor": gridColor
                    }});
                }}
            }});
        }}

        function toggleDashboardTheme() {{
            const htmlTag = document.documentElement;
            const currentTheme = htmlTag.getAttribute("data-bs-theme");
            const newTheme = currentTheme === "dark" ? "light" : "dark";
            applyTheme(newTheme);
            localStorage.setItem("dashboard_theme", newTheme);
        }}

        function applyTheme(theme) {{
            const htmlTag = document.documentElement;
            const themeIcon = document.getElementById("themeIcon");
            const themeText = document.getElementById("themeText");

            htmlTag.setAttribute("data-bs-theme", theme);
            if (theme === "dark") {{
                themeIcon.className = "bi bi-sun-fill text-warning";
                themeText.innerText = "Light Mode";
                updatePlotlyTheme(true);
            }} else {{
                themeIcon.className = "bi bi-moon-stars-fill";
                themeText.innerText = "Dark Mode";
                updatePlotlyTheme(false);
            }}
        }}

        async function switchExcelSheet(sheetName, datasetChoice) {{
            try {{
                const url = '/?selected_sheet=' + encodeURIComponent(sheetName) + '&dataset=' + encodeURIComponent(datasetChoice);
                const resp = await fetch(url);
                if (resp.ok) {{
                    const html = await resp.text();
                    document.open();
                    document.write(html);
                    document.close();
                }}
            }} catch (err) {{
                console.error("[Sheet Switch Error]", err);
            }}
        }}

        function initFileUploadHandler() {{
            const fileInput = document.getElementById("datasetFileInput");
            if (!fileInput) return;

            fileInput.addEventListener("change", async function(event) {{
                event.preventDefault();
                const file = event.target.files && event.target.files[0];
                if (!file) return;

                const statusDiv = document.getElementById("uploadStatusMsg");
                if (statusDiv) {{
                    statusDiv.className = "alert alert-info py-2 px-2 small mb-2 d-flex align-items-center gap-2";
                    statusDiv.innerHTML = '<span class="spinner-border spinner-border-sm" role="status"></span> Reading and processing ' + file.name + '...';
                }}

                try {{
                    const formData = new FormData();
                    formData.append("dataset_file", file);

                    const uploadUrl = (window.location.protocol === "file:") ? "http://localhost:8000/upload" : "/upload";
                    console.log("[Upload] Submitting file to " + uploadUrl + " (" + file.name + ", " + file.size + " bytes)");

                    const response = await fetch(uploadUrl, {{
                        method: "POST",
                        body: formData,
                    }});

                    console.log("[Upload] Response status: " + response.status);

                    if (!response.ok) {{
                        const errData = await response.json().catch(() => null);
                        const msg = (errData && errData.error) ? errData.error : ("Server responded with HTTP " + response.status);
                        throw new Error(msg);
                    }}

                    const data = await response.json();
                    if (!data.success) {{
                        throw new Error(data.error || "Upload failed");
                    }}

                    console.log("[Upload] Success: " + data.filename + ", " + data.rows + " rows, " + data.cols + " cols. Detected: Date=" + data.date_col + ", Sales=" + data.sales_col + ", Group=" + data.group_col);

                    if (data.html) {{
                        document.open();
                        document.write(data.html);
                        document.close();
                    }}
                }} catch (err) {{
                    console.error("[Upload Error]", err);
                    if (statusDiv) {{
                        statusDiv.className = "alert alert-danger py-2 px-2 small mb-2";
                        statusDiv.innerHTML = '<i class="bi bi-exclamation-octagon-fill me-1"></i> <b>Upload failed:</b> ' + (err.message || err);
                    }}
                }}
            }});
        }}

        document.addEventListener("DOMContentLoaded", function() {{
            const savedTheme = localStorage.getItem("dashboard_theme") || "light";
            applyTheme(savedTheme);
            setTimeout(() => {{ applyTheme(savedTheme); }}, 400);
            initFileUploadHandler();
        }});
        initFileUploadHandler();
    </script>
</body>
</html>
"""
    return html


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        """Enable CORS pre-flight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_POST(self):
        """Handles multipart file uploads for CSV / Excel files without page reload."""
        parsed = urllib.parse.urlparse(self.path)
        clean_path = parsed.path.rstrip("/")

        if clean_path == "/upload":
            content_type = self.headers.get("Content-Type", "")
            try:
                content_length = int(self.headers.get("Content-Length", 0))
            except Exception:
                content_length = 0

            print(f"[POST /upload] Content-Type: {content_type}, Content-Length: {content_length}")

            if not content_type.startswith("multipart/form-data"):
                error_msg = f"Invalid Content-Type '{content_type}'. Expected multipart/form-data."
                print(f"[Upload Error] {error_msg}", file=sys.stderr)
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": error_msg}).encode("utf-8"))
                return

            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                        "CONTENT_LENGTH": str(content_length),
                    },
                )
            except Exception as e:
                error_msg = f"Failed to parse multipart payload: {e}"
                print(f"[Upload Error] {error_msg}", file=sys.stderr)
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": error_msg}).encode("utf-8"))
                return

            file_item = None
            for key in ["dataset_file", "file"]:
                if key in form:
                    val = form[key]
                    if not isinstance(val, list) and getattr(val, "filename", None):
                        file_item = val
                        break

            if file_item is not None and getattr(file_item, "filename", None):
                filename = os.path.basename(file_item.filename)
                file_bytes = file_item.file.read()

                if not file_bytes or len(file_bytes) == 0:
                    error_msg = f"The uploaded file '{filename}' is empty (0 bytes)."
                    print(f"[Upload Error] {error_msg}", file=sys.stderr)
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": error_msg}).encode("utf-8"))
                    return

                try:
                    df, sheet_names, chosen_sheet = parse_uploaded_file(file_bytes, filename)
                    if df is None or df.empty:
                        raise ValueError(f"File '{filename}' contains 0 rows.")

                    # Auto-detect Date, Sales, and Group columns
                    d, s, g = auto_detect_columns(df)

                    # Update persistent memory cache
                    UPLOADED_DATA_CACHE["filename"] = filename
                    UPLOADED_DATA_CACHE["file_bytes"] = file_bytes
                    UPLOADED_DATA_CACHE["df"] = df
                    UPLOADED_DATA_CACHE["sheet_names"] = sheet_names
                    UPLOADED_DATA_CACHE["selected_sheet"] = chosen_sheet
                    UPLOADED_DATA_CACHE["date_col"] = d
                    UPLOADED_DATA_CACHE["sales_col"] = s
                    UPLOADED_DATA_CACHE["group_col"] = g
                    UPLOADED_DATA_CACHE["is_auto_detected"] = True

                    # Generate new dashboard HTML with updated state
                    new_html = build_dashboard_html()

                    print(f"[Upload Success] '{filename}' loaded: {len(df):,} rows, {len(df.columns)} cols. Detected: Date={d}, Sales={s}, Group={g}")

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    response_payload = {
                        "success": True,
                        "filename": filename,
                        "rows": len(df),
                        "cols": len(df.columns),
                        "date_col": d,
                        "sales_col": s,
                        "group_col": g,
                        "html": new_html,
                    }
                    self.wfile.write(json.dumps(response_payload).encode("utf-8"))
                    return
                except Exception as e:
                    error_msg = f"Upload failed: {e}"
                    print(f"[Upload Error] {error_msg}", file=sys.stderr)
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": error_msg}).encode("utf-8"))
                    return
            else:
                error_msg = "Upload failed: No file was selected or received in the request."
                print(f"[Upload Error] {error_msg}", file=sys.stderr)
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": error_msg}).encode("utf-8"))
                return

        error_msg = f"POST path '{self.path}' not found."
        print(f"[404 Error] {error_msg}", file=sys.stderr)
        self.send_response(404)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"success": False, "error": error_msg}).encode("utf-8"))

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        dataset = params.get("dataset", [None])[0]
        if dataset and dataset not in ["Weekly", "Daily", "Monthly", "Yearly"]:
            dataset = None

        group = params.get("group", ["Total (all groups)"])[0]
        try:
            horizon_param = params.get("horizon", [None])[0]
            horizon = int(horizon_param) if horizon_param is not None and int(horizon_param) > 0 else None
        except Exception:
            horizon = None

        date_col = params.get("date_col", [None])[0]
        sales_col = params.get("sales_col", [None])[0]
        group_col = params.get("group_col", [None])[0]
        selected_sheet = params.get("selected_sheet", [None])[0]

        try:
            html = build_dashboard_html(
                dataset_choice=dataset,
                selected_group=group,
                horizon=horizon,
                custom_date_col=date_col,
                custom_sales_col=sales_col,
                custom_group_col=group_col,
                selected_sheet=selected_sheet,
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Error rendering dashboard: {e}".encode("utf-8"))


def run_server():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"Standalone Sales Dashboard server listening on http://localhost:{PORT}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server()
