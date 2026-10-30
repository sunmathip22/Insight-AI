"""
Automated Business Insights & Recommendations Engine
Generates data-driven executive narratives, seasonality summaries,
safety stock estimations, staffing advice, retraining protocols, and model limitations.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def generate_business_insights(
    ts_df: pd.DataFrame,
    eval_results: Dict[str, Any],
    prep_data: Dict[str, Any],
    future_forecast_df: pd.DataFrame,
    horizon: int,
) -> Dict[str, Any]:
    """
    Synthesizes historical time series patterns, backtest performance,
    and future projections into actionable supply chain & operational advice.
    """
    sales = ts_df["Sales"]
    total_sales = float(sales.sum())
    mean_sales = float(sales.mean())
    median_sales = float(sales.median())
    std_sales = float(sales.std())
    max_sales = float(sales.max())
    min_sales = float(sales.min())

    # Peak & low dates
    peak_date = sales.idxmax().strftime("%Y-%m-%d")
    min_date = sales.idxmin().strftime("%Y-%m-%d")

    # Time-based patterns
    freq_label = prep_data["detected_label"]
    seasonal_period = prep_data["seasonal_period"]
    best_model = eval_results["best_model_name"]
    best_rmse = eval_results["best_rmse"]
    best_mae = eval_results["best_mae"]
    best_mape = eval_results["best_mape"]
    metric_type = "sMAPE" if eval_results.get("has_zeros") else "MAPE"
    improvement_pct = eval_results["improvement_pct"]
    baseline_used = eval_results["baseline_used"]

    # Month-of-year analysis
    sales_by_month = ts_df.groupby(ts_df.index.month_name())["Sales"].mean()
    if not sales_by_month.empty:
        top_month = sales_by_month.idxmax()
        top_month_val = sales_by_month.max()
        bottom_month = sales_by_month.idxmin()
        bottom_month_val = sales_by_month.min()
    else:
        top_month, top_month_val = "N/A", 0
        bottom_month, bottom_month_val = "N/A", 0

    # Day-of-week analysis (for daily/weekly)
    sales_by_dow = ts_df.groupby(ts_df.index.day_name())["Sales"].mean()
    if not sales_by_dow.empty and freq_label in ["Daily", "Weekly"]:
        top_day = sales_by_dow.idxmax()
        top_day_val = sales_by_dow.max()
        bottom_day = sales_by_dow.idxmin()
        bottom_day_val = sales_by_dow.min()
    else:
        top_day, top_day_val = "N/A", 0
        bottom_day, bottom_day_val = "N/A", 0

    # Future forecast trajectory
    if not future_forecast_df.empty:
        future_total = float(future_forecast_df["Forecast"].sum())
        future_mean = float(future_forecast_df["Forecast"].mean())
        growth_vs_hist = ((future_mean - mean_sales) / mean_sales) * 100.0 if mean_sales > 0 else 0.0
        peak_forecast_date = future_forecast_df["Forecast"].idxmax().strftime("%Y-%m-%d")
        peak_forecast_val = float(future_forecast_df["Forecast"].max())
    else:
        future_total = 0.0
        future_mean = 0.0
        growth_vs_hist = 0.0
        peak_forecast_date = "N/A"
        peak_forecast_val = 0.0

    # Operational Calculations: Safety Stock & Buffer
    # Safety Stock formula: Z * RMSE * sqrt(Lead Time in periods)
    # 95% service level -> Z = 1.645; 99% service level -> Z = 2.326
    lead_time_periods = 1
    safety_stock_95 = 1.645 * (best_rmse if not np.isnan(best_rmse) else std_sales) * np.sqrt(lead_time_periods)
    safety_stock_99 = 2.326 * (best_rmse if not np.isnan(best_rmse) else std_sales) * np.sqrt(lead_time_periods)

    # Narrative generation (small, concise, min 2-3 actionable points per area)
    exec_summary = (
        f"• **Project Baseline**: Across {len(sales)} periods, sales averaged **₹{mean_sales:,.2f}** (all-time peak: **₹{max_sales:,.2f}** on **{peak_date}**).\n"
        f"• **Optimal Model**: Selected **{best_model}** achieving an RMSE of **₹{best_rmse:,.2f}** and {metric_type} of **{best_mape:.1f}%** (**{improvement_pct:+.1f}%** vs {baseline_used} baseline)."
    )

    dow_peak_str = f" ({top_day})" if top_day != "N/A" else ""
    dow_low_str = f" ({bottom_day})" if bottom_day != "N/A" else ""

    peak_analysis = (
        f"• **Peak Demand**: Highest sales in **{top_month}**{dow_peak_str} averaging **₹{top_month_val:,.2f}** per period.\n"
        f"• **Low Demand**: Slowest period in **{bottom_month}**{dow_low_str} averaging **₹{bottom_month_val:,.2f}** per period.\n"
        f"• **Forecast Peak**: Next projected peak on **{peak_forecast_date}** at **₹{peak_forecast_val:,.2f}**."
    )

    safety_stock_text = (
        f"• **Standard Buffer (95% SL)**: Maintain **₹{safety_stock_95:,.2f}** buffer above projected demand to protect against standard variability.\n"
        f"• **Critical Buffer (99% SL)**: Maintain **₹{safety_stock_99:,.2f}** buffer for mission-critical or high-margin items.\n"
        f"• **Reorder Trigger**: Place replenishment orders 1-2 periods before projected peak on **{peak_forecast_date}** (**₹{peak_forecast_val:,.2f}**)."
    )

    top_month_surge = ((top_month_val - mean_sales) / mean_sales * 100) if mean_sales > 0 else 0.0
    staff_peak_day = f" and {top_day}s" if top_day != "N/A" else ""

    staffing_text = (
        f"• **Peak Scheduling**: Increase shift coverage during **{top_month}**{staff_peak_day} where demand is **{top_month_surge:+.1f}%** above baseline.\n"
        f"• **Downtime & Training**: Schedule maintenance, stock counts, and team training during slower **{bottom_month}**.\n"
        f"• **Projected Growth**: Align capacity for future average demand of **₹{future_mean:,.2f}** ({growth_vs_hist:+.1f}% vs historical mean)."
    )

    cadence = 'weekly' if freq_label == 'Daily' else 'monthly' if freq_label == 'Weekly' else 'quarterly'

    retraining_advice = (
        f"• **Retraining Schedule**: Re-fit and validate models **{cadence}** or when new actual sales batches arrive.\n"
        f"• **Drift Monitoring**: Trigger immediate re-run if Tracking Signal (`CFE / MAD`) exceeds ±4.0.\n"
        f"• **Anomaly Review**: Review {prep_data['outlier_count']} detected anomaly periods before updating model parameters."
    )

    limitations_text = [
        f"**1. Historical Data Span**: Dataset contains {prep_data['cycle_count']} seasonal cycles. "
        + (f"Since this is under 2 full cycles, long-term seasonal estimates carry higher uncertainty."
           if not prep_data["has_sufficient_cycles"]
           else f"Having {prep_data['cycle_count']} cycles provides a solid statistical baseline."),
        f"**2. Forecast Horizon Decay**: Model confidence intervals naturally widen as the forecast horizon extends "
        f"from step 1 to step {horizon}. Decisions beyond the near-term horizon should be reviewed cautiously.",
        "**3. Exogenous Factors**: Univariate statistical models assume future patterns mirror historical dynamics. "
        "They do not account for external shocks such as marketing campaign shifts, competitor pricing actions, "
        "supply chain black swans, or macroeconomic disruptions.",
        "**4. Zero & Intermittent Demand**: If sales data contains long zero stretches, continuous time-series models "
        "can occasionally produce fractional positive predictions rather than discrete batch arrivals."
    ]

    # Key Insights auto-generation
    half_idx = max(1, len(sales) // 2)
    first_half_mean = float(sales.iloc[:half_idx].mean())
    second_half_mean = float(sales.iloc[half_idx:].mean())
    if first_half_mean > 0:
        demand_pct = ((second_half_mean - first_half_mean) / first_half_mean) * 100.0
    else:
        demand_pct = 0.0

    if demand_pct >= 0:
        demand_direction_label = f"Increasing (+{demand_pct:.1f}%)"
        demand_direction_type = "increasing"
    else:
        demand_direction_label = f"Decreasing ({demand_pct:.1f}%)"
        demand_direction_type = "decreasing"

    peak_season_label = f"{top_month} (avg ₹{top_month_val:,.2f})" if top_month != "N/A" else "N/A"
    low_season_label = f"{bottom_month} (avg ₹{bottom_month_val:,.2f})" if bottom_month != "N/A" else "N/A"
    anomalies_count = int(prep_data.get("outlier_count", 0))

    best_model_label = f"{best_model} (RMSE: ₹{best_rmse:,.2f}, +{improvement_pct:.1f}% improvement over {baseline_used})" if not np.isnan(best_rmse) else f"{best_model} (baseline)"

    key_insights = {
        "demand_direction": demand_direction_label,
        "demand_direction_type": demand_direction_type,
        "demand_pct": round(demand_pct, 1),
        "peak_season": peak_season_label,
        "low_season": low_season_label,
        "anomalies_count": anomalies_count,
        "best_model_label": best_model_label,
        "best_model_name": best_model,
        "improvement_pct": improvement_pct,
        "baseline_used": baseline_used,
        "recommended_actions": {
            "increase_inventory": f"Maintain ₹{safety_stock_95:,.2f} safety stock ahead of {top_month} peak and {peak_forecast_date} surge.",
            "plan_staffing": f"Scale shift labor during {top_month} ({top_month_surge:+.1f}% vs baseline) and plan downtime during {bottom_month}.",
            "plan_for_seasonal_demand": f"Trigger purchase orders 1-2 periods before {peak_forecast_date} peak (₹{peak_forecast_val:,.2f}).",
        },
    }

    return {
        "exec_summary": exec_summary,
        "peak_analysis": peak_analysis,
        "safety_stock_text": safety_stock_text,
        "staffing_text": staffing_text,
        "retraining_advice": retraining_advice,
        "limitations_text": limitations_text,
        "key_insights": key_insights,
        "stats": {
            "total_sales": total_sales,
            "mean_sales": mean_sales,
            "median_sales": median_sales,
            "max_sales": max_sales,
            "min_sales": min_sales,
            "peak_date": peak_date,
            "future_total": future_total,
            "future_mean": future_mean,
            "growth_vs_hist": growth_vs_hist,
            "safety_stock_95": safety_stock_95,
        },
    }

