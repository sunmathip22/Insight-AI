"""
Universal Sales Forecasting Dashboard
A generic, enterprise-grade sales forecasting web application built with Streamlit.
Works with ANY sales dataset (CSV/Excel) with dynamic column mapping,
automated data cleaning, rolling-origin validation, multiple models, and business insights.
"""

import os
import sys
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure base directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.data_prep import (
    load_raw_dataset,
    process_sales_data,
    get_excel_sheets,
    auto_detect_columns,
    get_column_summary,
)
from utils.evaluation import rolling_origin_validation, fit_best_future_forecast
from utils.insights import generate_business_insights


DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


def ensure_sample_datasets():
    """Guarantees that demo CSV files exist on disk."""
    walmart_file = os.path.join(DATA_DIR, "walmart_weekly_sales.csv")
    retail_file = os.path.join(DATA_DIR, "retail_monthly_sales.csv")
    daily_file = os.path.join(DATA_DIR, "daily_ecommerce_sales.csv")

    if not os.path.exists(walmart_file) or not os.path.exists(retail_file) or not os.path.exists(daily_file):
        try:
            import generate_samples
        except Exception:
            pass


ensure_sample_datasets()


# ---------------------------------------------------------
# Page Configuration & Professional Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Clean layout adjustments */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    
    /* Header subtitle */
    .header-subtitle {
        font-size: 0.95rem;
        opacity: 0.8;
        margin-bottom: 1.2rem;
    }

    /* Metric card border enhancement */
    [data-testid="stMetric"] {
        background: rgba(125, 125, 125, 0.06);
        border: 1px solid rgba(125, 125, 125, 0.18);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
    }
    
    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-green {
        background-color: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-blue {
        background-color: rgba(59, 130, 246, 0.15);
        color: #2563eb;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Cached Computation Functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def cached_load_data(file_key_or_path: str, _uploaded_file=None):
    if _uploaded_file is not None:
        return load_raw_dataset(_uploaded_file)
    return load_raw_dataset(file_key_or_path)


@st.cache_data(show_spinner=False)
def cached_run_pipeline(
    df: pd.DataFrame,
    date_col: str,
    sales_col: str,
    group_col: Optional[str],
    selected_group: Optional[str],
    resample_freq: Optional[str],
):
    return process_sales_data(
        df=df,
        date_col=date_col,
        sales_col=sales_col,
        group_col=group_col,
        selected_group=selected_group,
        resample_freq=resample_freq,
    )


@st.cache_data(show_spinner=False)
def cached_evaluate_and_forecast(
    ts_series: pd.Series,
    horizon: int,
    seasonal_period: int,
    pandas_freq: str,
    freq_label: str = None,
):
    eval_results = rolling_origin_validation(
        ts_series=ts_series,
        horizon=horizon,
        seasonal_period=seasonal_period,
        freq_label=freq_label,
        pandas_freq=pandas_freq,
    )

    future_res = fit_best_future_forecast(
        ts_series=ts_series,
        best_model_name=eval_results["best_model_name"],
        horizon=horizon,
        seasonal_period=seasonal_period,
        pandas_freq=pandas_freq,
        freq_label=freq_label,
    )

    return eval_results, future_res


# ---------------------------------------------------------
# Sidebar Configuration & Dynamic Dataset Input
# ---------------------------------------------------------
# ---------------------------------------------------------
# 1. SEPARATE FILE UPLOAD CARD (ABOVE Controls & Parameters)
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="background: rgba(125, 125, 125, 0.08); border: 1px solid rgba(125, 125, 125, 0.22); border-radius: 12px; padding: 1rem; margin-bottom: 1.25rem;">
        <div style="font-weight: 700; font-size: 1rem; margin-bottom: 0.25rem; display: flex; align-items: center; gap: 0.4rem;">
            <span>📁</span> Upload Dataset
        </div>
        <div style="font-size: 0.8rem; opacity: 0.75; margin-bottom: 0.65rem;">
            Upload CSV or Excel file (.csv, .xlsx, .xls) directly from your computer:
        </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.sidebar.file_uploader(
    "Choose file",
    type=["csv", "xlsx", "xls"],
    label_visibility="collapsed",
    help="Upload a sales dataset (.csv, .xlsx, .xls).",
)

raw_df: Optional[pd.DataFrame] = None
source_name = "walmart_weekly_sales.csv (Sample)"
is_uploaded = False

if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    # Reset cached selections when a new file is uploaded
    if st.session_state.get("active_file_id") != file_id:
        st.session_state["active_file_id"] = file_id
        st.session_state.pop("selected_excel_sheet", None)

    # Multi-sheet Excel support
    selected_sheet = None
    lower_name = uploaded_file.name.lower()
    if lower_name.endswith((".xlsx", ".xls")):
        try:
            sheet_names = get_excel_sheets(uploaded_file)
            if len(sheet_names) > 1:
                selected_sheet = st.sidebar.selectbox(
                    "Select Excel Sheet",
                    options=sheet_names,
                    index=0,
                    key="selected_excel_sheet",
                    help="Workbook has multiple sheets. Choose sheet to analyze.",
                )
        except Exception as e:
            st.sidebar.error(f"⚠️ Corrupt Excel workbook: {e}")

    # Read and validate the uploaded dataset
    try:
        raw_df = load_raw_dataset(uploaded_file, sheet_name=selected_sheet)
        source_name = uploaded_file.name
        is_uploaded = True
        st.sidebar.markdown(
            f"""
            <div style="margin-top: 0.5rem; font-size: 0.82rem; word-break: break-all;">
                <b>📁 Active File:</b> <span class="badge-pill badge-green">{source_name}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.sidebar.error(f"⚠️ Upload Error: {e}")
        raw_df = None
else:
    raw_df = None
    source_name = None
    is_uploaded = False
    st.sidebar.markdown(
        """
        <div style="margin-top: 0.5rem; font-size: 0.82rem; word-break: break-all;">
            <b>📁 Active File:</b> <span class="badge-pill badge-blue">No file chosen</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. CONTROLS & PARAMETERS WITH UPDATED DATASET DROPDOWN
# ---------------------------------------------------------
st.sidebar.title("📈 Controls & Parameters")

# Auto-detect data frequency from date gaps
auto_date_init, _, _ = auto_detect_columns(raw_df) if raw_df is not None else (None, None, None)
detected_freq_label = "Weekly"
if auto_date_init and auto_date_init in raw_df.columns:
    try:
        from utils.data_prep import detect_frequency_and_seasonality
        sample_dates = pd.to_datetime(raw_df[auto_date_init], errors="coerce").dropna().sort_values().drop_duplicates()
        detected_freq_label, _, _, _ = detect_frequency_and_seasonality(pd.DatetimeIndex(sample_dates))
    except Exception:
        detected_freq_label = "Weekly"

freq_options = ["Weekly", "Daily", "Monthly", "Yearly"]
default_idx = freq_options.index(detected_freq_label) if detected_freq_label in freq_options else 0

st.sidebar.markdown(f"**Dataset** <span class='badge-pill badge-green'>Detected: {detected_freq_label}</span>", unsafe_allow_html=True)
dataset_frequency = st.sidebar.selectbox(
    "Dataset Cadence",
    options=freq_options,
    index=default_idx,
    label_visibility="collapsed",
    help="Select sales analysis cadence: Weekly, Daily, Monthly, or Yearly.",
)

if dataset_frequency != detected_freq_label:
    st.sidebar.warning(f"⚠️ Notice: Selected '{dataset_frequency}' differs from detected cadence '{detected_freq_label}'. Resampling applied.")

# Frequency mapping: pandas code and default forecast horizons
freq_config = {
    "Weekly": {"code": "W", "default_h": 13, "label": "Weekly"},
    "Daily": {"code": "D", "default_h": 14, "label": "Daily"},
    "Monthly": {"code": "MS", "default_h": 3, "label": "Monthly"},
    "Yearly": {"code": "YE", "default_h": 2, "label": "Yearly"},
}
selected_freq_info = freq_config.get(dataset_frequency, freq_config["Weekly"])
resample_freq = selected_freq_info["code"]
default_h = selected_freq_info["default_h"]

# Main page title (Always visible)
st.title("📈 Enterprise Sales Forecasting Dashboard")

# If no data is available
if raw_df is None or raw_df.empty:
    st.info("Please upload a CSV or Excel dataset to view sales analysis and forecasting.")
    st.stop()


# ---------------------------------------------------------
# RAW DATA PREVIEW & COLUMN DIAGNOSTICS
# ---------------------------------------------------------
with st.expander("📋 Raw Data Preview & Data Quality Report", expanded=is_uploaded):
    st.markdown(f"**Dataset Dimensions:** `{raw_df.shape[0]:,}` rows × `{raw_df.shape[1]}` columns")
    tab_prev1, tab_prev2, tab_prev3 = st.tabs(["Data Quality Summary", "First 10 Rows (Preview)", "Column Diagnostics"])
    with tab_prev1:
        # Pre-audit data quality summary before/after cleaning
        raw_missing = int(raw_df.isna().sum().sum())
        raw_dupes = int(raw_df.duplicated().sum())
        dq_summary_df = pd.DataFrame([
            {"Metric": "Missing Values", "Before Cleaning": f"{raw_missing} values", "After Cleaning": "0 (Interpolated)", "Action": "Linear interpolation"},
            {"Metric": "Duplicate Records", "Before Cleaning": f"{raw_dupes} rows", "After Cleaning": "0 (Deduplicated)", "Action": "Aggregated by date sum"},
            {"Metric": "Invalid Records", "Before Cleaning": "Audited on load", "After Cleaning": "0 (Filtered)", "Action": "Dropped non-parseable entries"},
        ])
        st.dataframe(dq_summary_df, use_container_width=True, hide_index=True)
    with tab_prev2:
        st.dataframe(raw_df.head(10), use_container_width=True)
        if st.checkbox("Expand to view all rows", value=False):
            st.dataframe(raw_df, use_container_width=True)
    with tab_prev3:
        st.dataframe(get_column_summary(raw_df), use_container_width=True)


# ---------------------------------------------------------
# 3. AUTO COLUMN DETECTION & DYNAMIC COLUMN MAPPINGS
# ---------------------------------------------------------
all_cols = list(raw_df.columns)
auto_date, auto_sales, auto_group = auto_detect_columns(raw_df)

# Date column dropdown with Auto-detected badge
date_idx = all_cols.index(auto_date) if auto_date in all_cols else 0
st.sidebar.markdown(
    "**Date Column** <span class='badge-pill badge-green'>✨ Auto-detected</span>",
    unsafe_allow_html=True,
)
date_col = st.sidebar.selectbox("Select Date Column", options=all_cols, index=date_idx, label_visibility="collapsed")

# Sales column dropdown with Auto-detected badge
remaining_cols = [c for c in all_cols if c != date_col]
sales_idx = remaining_cols.index(auto_sales) if auto_sales in remaining_cols else 0
st.sidebar.markdown(
    "**Sales Column** <span class='badge-pill badge-green'>✨ Auto-detected</span>",
    unsafe_allow_html=True,
)
sales_col = st.sidebar.selectbox("Select Sales Column", options=remaining_cols, index=sales_idx, label_visibility="collapsed")

# Group column dropdown with Auto-detected badge
group_options = ["None (Single series)"] + [c for c in all_cols if c not in [date_col, sales_col]]
group_idx = group_options.index(auto_group) if (auto_group and auto_group in group_options) else 0
st.sidebar.markdown(
    "**Group Column (Optional)** <span class='badge-pill badge-green'>✨ Auto-detected</span>",
    unsafe_allow_html=True,
)
chosen_group = st.sidebar.selectbox("Select Group Column", options=group_options, index=group_idx, label_visibility="collapsed")

group_col: Optional[str] = None
selected_group: Optional[str] = None

if chosen_group != "None (Single series)":
    group_col = chosen_group
    unique_groups = sorted(raw_df[group_col].dropna().astype(str).unique().tolist())
    group_selection_options = ["Total (all groups)"] + unique_groups
    selected_group = st.sidebar.selectbox(
        "Group Subset",
        options=group_selection_options,
        index=0,
        help="Forecast total aggregated sales across all groups or isolate a specific entity.",
    )

# Visual Dynamic Column Mappings Badge Panel
st.sidebar.markdown(
    f"""
    <div style="background: rgba(125,125,125,0.08); border-radius: 8px; padding: 10px; margin-top: 10px; font-size: 0.82rem;">
        <b>Active Mappings:</b><br>
        📅 Date: <code>{date_col}</code><br>
        💰 Sales: <code>{sales_col}</code><br>
        🏷️ Group: <code>{group_col or 'None'}</code><br>
        ⏱️ Cadence: <b>{dataset_frequency}</b>
    </div>
    """,
    unsafe_allow_html=True,
)

# Forecast Horizon & Run Button
forecast_horizon = st.sidebar.number_input(
    "Forecast Horizon (periods ahead)",
    min_value=1,
    max_value=104,
    value=default_h,
    step=1,
    help=f"Periods ahead to project. Default for {dataset_frequency}: {default_h}.",
)

run_button = st.sidebar.button("🚀 Run Forecast & Backtest", type="primary", use_container_width=True)


# ---------------------------------------------------------
# Execute Pipeline & Forecast
# ---------------------------------------------------------
with st.spinner("Processing data pipeline and temporal structures..."):
    try:
        prep_data = cached_run_pipeline(
            df=raw_df,
            date_col=date_col,
            sales_col=sales_col,
            group_col=group_col,
            selected_group=selected_group,
            resample_freq=resample_freq,
        )
    except Exception as e:
        st.error(f"Data Pipeline Error: {e}")
        st.stop()

ts_df = prep_data["ts_df"]
sales_series = ts_df["Sales"]
n_obs = prep_data["n_observations"]
seasonal_period = prep_data["seasonal_period"]
pandas_freq = prep_data["pandas_freq"]

with st.spinner("Backtesting models with rolling-origin validation & generating future forecasts..."):
    try:
        eval_results, future_res = cached_evaluate_and_forecast(
            ts_series=sales_series,
            horizon=int(forecast_horizon),
            seasonal_period=int(seasonal_period),
            pandas_freq=pandas_freq,
            freq_label=prep_data["detected_label"],
        )
    except Exception as e:
        st.error(f"Modeling Engine Error: {e}")
        st.stop()

insights = generate_business_insights(
    ts_df=ts_df,
    eval_results=eval_results,
    prep_data=prep_data,
    future_forecast_df=future_res["forecast_df"],
    horizon=int(forecast_horizon),
)


# ---------------------------------------------------------
# Header Subtitle & Top KPI Metrics (Theme Adaptive)
# ---------------------------------------------------------
group_context = f" • Entity: {selected_group}" if selected_group else ""
st.markdown(
    f"<div class='header-subtitle'>Dataset: <b>{source_name}</b>{group_context} • "
    f"Frequency: <b>{prep_data['detected_label']}</b> • "
    f"Observations: <b>{n_obs}</b> • Seasonal Period: <b>{seasonal_period}</b></div>",
    unsafe_allow_html=True,
)

stats = insights["stats"]
best_model_name = eval_results["best_model_name"]
best_rmse = eval_results["best_rmse"]
improvement_pct = eval_results["improvement_pct"]
baseline_used = eval_results["baseline_used"]

# 5 Top KPI Cards using Native Streamlit Metrics
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)

with kpi_col1:
    st.metric(
        label="Total Historical Sales",
        value=f"₹{stats['total_sales']:,.0f}",
        help=f"Minimum period sales: ₹{stats['min_sales']:,.0f}",
    )

with kpi_col2:
    st.metric(
        label="Average Sales / Period",
        value=f"₹{stats['mean_sales']:,.1f}",
        help=f"Median period sales: ₹{stats['median_sales']:,.1f}",
    )

with kpi_col3:
    st.metric(
        label="All-Time Peak Sales",
        value=f"₹{stats['max_sales']:,.0f}",
        help=f"Peak Date: {stats['peak_date']}",
    )

with kpi_col4:
    st.metric(
        label="Optimal Model",
        value=best_model_name,
        delta="Rank #1 (Lowest RMSE)",
        delta_color="normal",
    )

with kpi_col5:
    delta_str = f"{improvement_pct:+.1f}% vs {baseline_used}"
    st.metric(
        label="Best RMSE",
        value=f"₹{best_rmse:,.1f}",
        delta=delta_str,
        delta_color="normal",
    )

st.markdown("<br>", unsafe_allow_html=True)

# Pipeline Validation Notices
if prep_data["warnings"]:
    with st.expander("⚠️ Pipeline Validation Notices", expanded=not prep_data["has_sufficient_cycles"]):
        for w in prep_data["warnings"]:
            st.warning(w)


# ---------------------------------------------------------
# Dashboard Navigation Tabs
# ---------------------------------------------------------
tab_overview, tab_decomposition, tab_forecast, tab_comparison, tab_insights = st.tabs(
    [
        "📊 Overview",
        "🌊 Trend & Seasonality",
        "🎯 Forecast",
        "🏆 Model Comparison",
        "💡 Business Recommendations",
    ]
)


# =========================================================
# TAB 1: OVERVIEW
# =========================================================
with tab_overview:
    st.subheader("Sales Trajectory & Outlier Analysis")
    st.caption("Interactive historical time series with IQR-flagged anomalies and volume patterns.")

    # Line chart with Outliers highlighted
    fig_overview = go.Figure()

    # Main Sales Line
    fig_overview.add_trace(
        go.Scatter(
            x=ts_df.index,
            y=ts_df["Sales"],
            mode="lines",
            name="Actual Sales",
            line=dict(color="#2563eb", width=2.2),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Sales</b>: ₹%{y:,.2f}<extra></extra>",
        )
    )

    # Rolling average
    rolling_window = max(3, min(12, len(ts_df) // 10))
    ts_df["Rolling_Avg"] = ts_df["Sales"].rolling(rolling_window, min_periods=1).mean()
    fig_overview.add_trace(
        go.Scatter(
            x=ts_df.index,
            y=ts_df["Rolling_Avg"],
            mode="lines",
            name=f"Moving Avg ({rolling_window}p)",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
            opacity=0.85,
            hovertemplate=f"<b>Date</b>: %{{x|%Y-%m-%d}}<br><b>Moving Avg ({rolling_window}p)</b>: ₹%{{y:,.2f}}<extra></extra>",
        )
    )

    # Highlight statistical outliers
    outliers_df = ts_df[ts_df["Is_Outlier"]]
    if not outliers_df.empty:
        fig_overview.add_trace(
            go.Scatter(
                x=outliers_df.index,
                y=outliers_df["Sales"],
                mode="markers",
                name=f"Statistical Outliers ({len(outliers_df)})",
                marker=dict(color="#ef4444", size=9, symbol="diamond", line=dict(color="#991b1b", width=1.5)),
                hovertemplate="<b>⚠️ Outlier</b><br><b>Date</b>: %{x|%Y-%m-%d}<br><b>Sales</b>: ₹%{y:,.2f}<extra></extra>",
            )
        )

    fig_overview.update_layout(
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
            title=dict(text="Date", font=dict(size=13, weight="bold"), standoff=15),
            tickfont=dict(size=11),
            showline=True,
            linecolor="#cbd5e1",
            linewidth=1,
            tickangle=-25 if len(ts_df) > 50 else 0,
        ),
        yaxis=dict(
            showgrid=True,
            automargin=True,
            title=dict(text="Sales (₹)", font=dict(size=13, weight="bold"), standoff=15),
            tickprefix="₹",
            tickformat=",.0f",
            tickfont=dict(size=11),
            showline=True,
            linecolor="#cbd5e1",
            linewidth=1,
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig_overview, use_container_width=True, config={"responsive": True, "displayModeBar": True, "displaylogo": False})

    # Group Distribution & Descriptive Stats
    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        if prep_data["has_groups"] and prep_data["group_summary"] is not None:
            st.markdown(f"**Sales Breakdown by {group_col}**")
            grp_df = prep_data["group_summary"].head(10)
            fig_grp = px.bar(
                grp_df,
                x=group_col,
                y="Total_Sales",
                color="Total_Sales",
                color_continuous_scale="Blues",
                labels={"Total_Sales": "Total Sales (₹)", group_col: group_col},
            )
            fig_grp.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_grp, use_container_width=True)
        else:
            st.markdown("**Sales Distribution Histogram**")
            fig_hist = px.histogram(
                ts_df,
                x="Sales",
                nbins=25,
                color_discrete_sequence=["#3b82f6"],
                marginal="box",
            )
            fig_hist.update_layout(
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Sales (₹)",
                yaxis_title="Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with col_g2:
        st.markdown("**Statistical Distribution Summary**")
        desc_df = pd.DataFrame(
            {
                "Metric": [
                    "Count",
                    "Mean",
                    "Std Dev",
                    "Min",
                    "25th Percentile",
                    "Median (50%)",
                    "75th Percentile",
                    "Max",
                    "IQR Outliers Detected",
                ],
                "Value": [
                    f"{len(ts_df):,}",
                    f"₹{ts_df['Sales'].mean():,.2f}",
                    f"₹{ts_df['Sales'].std():,.2f}",
                    f"₹{ts_df['Sales'].min():,.2f}",
                    f"₹{ts_df['Sales'].quantile(0.25):,.2f}",
                    f"₹{ts_df['Sales'].median():,.2f}",
                    f"₹{ts_df['Sales'].quantile(0.75):,.2f}",
                    f"₹{ts_df['Sales'].max():,.2f}",
                    f"{prep_data['outlier_count']}",
                ],
            }
        )
        st.dataframe(desc_df, use_container_width=True, hide_index=True)


# =========================================================
# TAB 2: TREND & SEASONALITY
# =========================================================
with tab_decomposition:
    st.subheader("Time Series Decomposition & Cyclical Seasonality")
    st.caption("Isolating long-term trend, recurring seasonal cycles, and irregular residuals.")

    can_decompose = prep_data["has_sufficient_cycles"] and (seasonal_period > 1)

    if can_decompose:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            decomp = seasonal_decompose(
                ts_df["Sales"],
                model="additive",
                period=int(seasonal_period),
                extrapolate_trend="freq",
            )

            fig_decomp = make_subplots(
                rows=4,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=(
                    "Observed Sales",
                    "Underlying Trend",
                    f"Seasonal Component (Period = {seasonal_period})",
                    "Residual / Noise",
                ),
            )

            fig_decomp.add_trace(
                go.Scatter(x=ts_df.index, y=decomp.observed, mode="lines", line=dict(color="#2563eb", width=1.5)),
                row=1,
                col=1,
            )
            fig_decomp.add_trace(
                go.Scatter(x=ts_df.index, y=decomp.trend, mode="lines", line=dict(color="#10b981", width=2)),
                row=2,
                col=1,
            )
            fig_decomp.add_trace(
                go.Scatter(x=ts_df.index, y=decomp.seasonal, mode="lines", line=dict(color="#f59e0b", width=1.5)),
                row=3,
                col=1,
            )
            fig_decomp.add_trace(
                go.Scatter(x=ts_df.index, y=decomp.resid, mode="markers", marker=dict(color="#64748b", size=3.5)),
                row=4,
                col=1,
            )

            fig_decomp.update_layout(
                height=650,
                showlegend=False,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_decomp, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not compute classical decomposition: {e}")
    else:
        st.warning(
            f"Classical decomposition requires at least 2 full seasonal cycles ({2 * seasonal_period} observations). "
            f"Currently have {len(ts_df)} observations. Showing cyclical aggregates below."
        )

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**Average Sales by Month of Year**")
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        monthly_df = (
            ts_df.groupby(ts_df.index.month_name())["Sales"]
            .mean()
            .reindex(month_order)
            .dropna()
            .reset_index()
            .rename(columns={"Date": "Month", "Sales": "Avg_Sales"})
        )

        fig_month = px.bar(
            monthly_df,
            x="Month",
            y="Avg_Sales",
            color="Avg_Sales",
            color_continuous_scale="Purples",
            labels={"Avg_Sales": "Average Sales (₹)", "Month": "Month"},
        )
        fig_month.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_month, use_container_width=True)

    with col_s2:
        if prep_data["detected_label"] in ["Daily", "Weekly"]:
            st.markdown("**Average Sales by Day of Week**")
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_df = (
                ts_df.groupby(ts_df.index.day_name())["Sales"]
                .mean()
                .reindex(dow_order)
                .dropna()
                .reset_index()
                .rename(columns={"Date": "Day_of_Week", "Sales": "Avg_Sales"})
            )
            fig_dow = px.bar(
                dow_df,
                x="Day_of_Week",
                y="Avg_Sales",
                color="Avg_Sales",
                color_continuous_scale="Teal",
                labels={"Avg_Sales": "Average Sales (₹)", "Day_of_Week": "Day of Week"},
            )
            fig_dow.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_dow, use_container_width=True)
        else:
            st.markdown("**Quarterly Seasonality Patterns**")
            ts_df["Quarter"] = "Q" + ts_df.index.quarter.astype(str)
            q_df = ts_df.groupby("Quarter")["Sales"].mean().reset_index()
            fig_q = px.bar(
                q_df,
                x="Quarter",
                y="Sales",
                color="Sales",
                color_continuous_scale="Emrld",
                labels={"Sales": "Average Sales (₹)"},
            )
            fig_q.update_layout(
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_q, use_container_width=True)


# =========================================================
# TAB 3: FORECAST
# =========================================================
with tab_forecast:
    st.subheader(f"Future Projections ({forecast_horizon} Periods Ahead)")
    st.caption(f"Champion Model: **{best_model_name}** • 95% Confidence Bounds")

    future_df = future_res["forecast_df"]

    history_tail_len = min(len(ts_df), max(forecast_horizon * 3, 26))
    hist_tail = ts_df.iloc[-history_tail_len:]

    fig_future = go.Figure()

    # Historical line
    fig_future.add_trace(
        go.Scatter(
            x=hist_tail.index,
            y=hist_tail["Sales"],
            mode="lines",
            name="Recent Actuals",
            line=dict(color="#64748b", width=2),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Actual</b>: ₹%{y:,.2f}<extra></extra>",
        )
    )

    # 95% CI Ribbon Upper
    fig_future.add_trace(
        go.Scatter(
            x=future_df.index,
            y=future_df["Upper_CI_95"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # 95% CI Ribbon Lower + Fill
    fig_future.add_trace(
        go.Scatter(
            x=future_df.index,
            y=future_df["Lower_CI_95"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)",
            name="95% Confidence Band",
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Lower CI</b>: ₹%{y:,.2f}<extra></extra>",
        )
    )

    # Point Forecast Line
    fig_future.add_trace(
        go.Scatter(
            x=future_df.index,
            y=future_df["Forecast"],
            mode="lines+markers",
            name=f"Forecast ({best_model_name})",
            line=dict(color="#2563eb", width=2.8),
            marker=dict(size=6, color="#1d4ed8"),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Forecast</b>: ₹%{y:,.2f}<extra></extra>",
        )
    )

    fig_future.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True, title="Sales (₹)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_future, use_container_width=True)

    # Backtest Validation Window
    test_slice = eval_results.get("last_test_slice")
    if test_slice is not None:
        st.markdown("**Backtest Validation Window: Actual vs Models**")
        st.caption("Out-of-sample comparison on the holdout test window without look-ahead bias.")

        fig_backtest = go.Figure()
        act_series = test_slice["actual"]
        fig_backtest.add_trace(
            go.Scatter(
                x=act_series.index,
                y=act_series.values,
                mode="lines+markers",
                name="Ground Truth Actuals",
                line=dict(color="#0f172a", width=3),
                marker=dict(size=6),
            )
        )

        colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
        for idx, (m_name, pred_series) in enumerate(test_slice["predictions"].items()):
            is_best = m_name == best_model_name
            fig_backtest.add_trace(
                go.Scatter(
                    x=pred_series.index,
                    y=pred_series.values,
                    mode="lines",
                    name=f"{m_name} {'⭐' if is_best else ''}",
                    line=dict(
                        color=colors[idx % len(colors)],
                        width=2.5 if is_best else 1.5,
                        dash="solid" if is_best else "dash",
                    ),
                )
            )

        fig_backtest.update_layout(
            height=360,
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(showgrid=True, title="Sales (₹)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_backtest, use_container_width=True)

    # Future Forecast Table and Export
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.markdown("**Forecast Projections Table**")
        display_forecast = future_df.copy()
        display_forecast["Forecast"] = display_forecast["Forecast"].round(2)
        display_forecast["Lower_CI_95"] = display_forecast["Lower_CI_95"].round(2)
        display_forecast["Upper_CI_95"] = display_forecast["Upper_CI_95"].round(2)
        display_forecast.reset_index(inplace=True)
        display_forecast["Date"] = display_forecast["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_forecast, use_container_width=True, height=220)

    with col_t2:
        st.markdown("**Export Options**")
        st.caption("Download forecast points and 95% confidence intervals.")
        csv_forecast = display_forecast.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv_forecast,
            file_name=f"sales_forecast_{best_model_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# TAB 4: MODEL COMPARISON
# =========================================================
with tab_comparison:
    st.subheader("Model Evaluation Leaderboard")
    st.caption("Cross-validation ranking candidate algorithms across rolling origin splits. Best model highlighted by RMSE.")

    metrics_df = eval_results["metrics_df"]
    # Model Comparison table: Model | MAE | RMSE | sMAPE, with the best model highlighted
    display_cols = ["Model", "MAE", "RMSE", "sMAPE (%)", "Status", "Details"]
    styled_df = metrics_df[[c for c in display_cols if c in metrics_df.columns]].copy()

    def highlight_best(row):
        is_best = (row["Model"] == best_model_name and row["Status"] == "Success")
        return ["background-color: rgba(37, 99, 235, 0.18); font-weight: bold;" if is_best else "" for _ in row]

    st.dataframe(styled_df.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("**Error Comparison by Model (RMSE & MAE)**")
        plot_m = metrics_df[metrics_df["Status"] == "Success"].copy()
        if not plot_m.empty:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=plot_m["Model"], y=plot_m["RMSE"], name="RMSE", marker_color="#2563eb"))
            fig_bar.add_trace(go.Bar(x=plot_m["Model"], y=plot_m["MAE"], name="MAE", marker_color="#93c5fd"))
            fig_bar.update_layout(
                barmode="group",
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(showgrid=True, title="Error (₹)"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_c2:
        st.markdown("**Horizon Error Degradation (Step 1 to H)**")
        h_data = eval_results["horizon_error_data"]
        fig_horizon = go.Figure()
        h_steps = list(range(1, int(forecast_horizon) + 1))

        colors = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]
        for idx, (m_name, step_errs) in enumerate(h_data.items()):
            if len(step_errs) == len(h_steps):
                fig_horizon.add_trace(
                    go.Scatter(
                        x=h_steps,
                        y=step_errs,
                        mode="lines+markers",
                        name=m_name,
                        line=dict(color=colors[idx % len(colors)], width=2),
                        marker=dict(size=4),
                    )
                )

        fig_horizon.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Forecast Step (h)", tickmode="linear", dtick=max(1, forecast_horizon // 10)),
            yaxis=dict(showgrid=True, title="Mean Absolute Error (₹)"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_horizon, use_container_width=True)

    metrics_csv = metrics_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Model Evaluation Metrics CSV",
        data=metrics_csv,
        file_name="forecast_model_metrics.csv",
        mime="text/csv",
    )


# =========================================================
# TAB 5: RECOMMENDATIONS & INSIGHTS
# =========================================================
with tab_insights:
    st.subheader("Data-Driven Business Recommendations")
    st.caption("Translating statistical probabilities into operational supply chain actions.")

    # KEY INSIGHTS BOX
    ki = insights.get("key_insights", {})
    demand_icon = "📈" if ki.get("demand_direction_type") == "increasing" else "📉"
    
    with st.container():
        st.markdown(
            f"""
            <div style="background: rgba(37, 99, 235, 0.08); border-left: 4px solid #2563eb; border-radius: 10px; padding: 1.2rem; margin-bottom: 1.5rem;">
                <div style="font-weight: 700; font-size: 1.15rem; color: #2563eb; margin-bottom: 0.75rem;">
                    💡 Key Insights (Auto-Generated)
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 1rem;">
                    <div style="background: rgba(125,125,125,0.06); padding: 0.75rem; border-radius: 8px;">
                        <span style="font-size: 0.78rem; opacity: 0.75;">Demand Direction</span><br>
                        <b>{demand_icon} {ki.get('demand_direction', 'N/A')}</b>
                    </div>
                    <div style="background: rgba(125,125,125,0.06); padding: 0.75rem; border-radius: 8px;">
                        <span style="font-size: 0.78rem; opacity: 0.75;">Peak vs Low Season</span><br>
                        <b>🌟 Peak:</b> {ki.get('peak_season', 'N/A')}<br>
                        <span style="font-size: 0.8rem; opacity: 0.75;">❄️ Low: {ki.get('low_season', 'N/A')}</span>
                    </div>
                    <div style="background: rgba(125,125,125,0.06); padding: 0.75rem; border-radius: 8px;">
                        <span style="font-size: 0.78rem; opacity: 0.75;">Anomalies Count</span><br>
                        <b style="color: #ef4444;">⚠️ {ki.get('anomalies_count', 0)} Outlier(s)</b>
                    </div>
                    <div style="background: rgba(125,125,125,0.06); padding: 0.75rem; border-radius: 8px;">
                        <span style="font-size: 0.78rem; opacity: 0.75;">Best Model & Baseline Gain</span><br>
                        <b style="color: #2563eb;">🏆 {ki.get('best_model_name', best_model_name)}</b><br>
                        <span style="font-size: 0.8rem; opacity: 0.75;">+{ki.get('improvement_pct', 0.0)}% vs {ki.get('baseline_used', 'Baseline')}</span>
                    </div>
                </div>
                <div style="border-top: 1px solid rgba(125,125,125,0.2); padding-top: 0.75rem; font-size: 0.85rem;">
                    <b>📋 Recommended Actions:</b>
                    <ul style="margin-top: 0.4rem; padding-left: 1.2rem;">
                        <li><b>Increase Inventory:</b> {ki.get('recommended_actions', {}).get('increase_inventory', '')}</li>
                        <li><b>Plan Staffing:</b> {ki.get('recommended_actions', {}).get('plan_staffing', '')}</li>
                        <li><b>Plan for Seasonal Demand:</b> {ki.get('recommended_actions', {}).get('plan_for_seasonal_demand', '')}</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.info(insights["exec_summary"])

    col_rec1, col_rec2 = st.columns(2)

    with col_rec1:
        st.markdown("### 🛒 Demand Patterns & Seasonality")
        st.markdown(insights["peak_analysis"])

        st.markdown("### 📦 Safety Stock & Buffer Inventory")
        st.markdown(insights["safety_stock_text"])

    with col_rec2:
        st.markdown("### 👥 Staffing & Capacity Allocation")
        st.markdown(insights["staffing_text"])

        st.markdown("### 🔄 Retraining & Drift Monitoring")
        st.markdown(insights["retraining_advice"])

    st.markdown("---")
    st.markdown("### ⚠️ Model Governance & Operational Limitations")
    for lim in insights["limitations_text"]:
        st.markdown(lim)
