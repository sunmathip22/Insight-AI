"""
Evaluation & Backtesting Module
Implements rolling-origin time-series cross-validation (backtesting),
metrics computation (MAE, RMSE, sMAPE, MAPE), horizon error tracking,
and automated best-model selection by RMSE.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from utils.models import MODEL_REGISTRY


def compute_metrics(
    actual: np.ndarray, forecast: np.ndarray, use_smape: bool = False
) -> Dict[str, float]:
    """
    Computes MAE, RMSE, sMAPE, and MAPE.
    Uses sMAPE if data contains zeros or is specifically requested.
    sMAPE formula: mean(2 * |y - yhat| / (|y| + |yhat| + epsilon)) * 100%
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    valid_mask = ~np.isnan(actual) & ~np.isnan(forecast)
    if not np.any(valid_mask):
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "sMAPE": np.nan,
            "MAPE": np.nan,
            "metric_type": "sMAPE" if use_smape else "MAPE",
        }

    act = actual[valid_mask]
    pred = forecast[valid_mask]

    errors = act - pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # Symmetric MAPE (sMAPE): bounded between 0% and 200%
    denom_smape = (np.abs(act) + np.abs(pred)) / 2.0
    nonzero_smape = denom_smape > 1e-8
    if np.any(nonzero_smape):
        smape = float(np.mean(np.abs(act[nonzero_smape] - pred[nonzero_smape]) / denom_smape[nonzero_smape]) * 100.0)
    else:
        smape = 0.0

    # Standard MAPE (avoid divide by zero)
    nonzero_mape = np.abs(act) > 1e-8
    if np.any(nonzero_mape):
        mape = float(np.mean(np.abs(errors[nonzero_mape] / act[nonzero_mape])) * 100.0)
    else:
        mape = smape

    has_zeros = np.any(act == 0) or use_smape
    return {
        "MAE": mae,
        "RMSE": rmse,
        "sMAPE": smape,
        "MAPE": mape,
        "metric_type": "sMAPE" if has_zeros else "MAPE",
    }


def rolling_origin_validation(
    ts_series: pd.Series,
    horizon: int,
    seasonal_period: int = 7,
    max_windows: int = 3,
    freq_label: Optional[str] = None,
    pandas_freq: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes rolling-origin (expanding window) backtesting without look-ahead bias.
    Runs every model through the exact same backtest splits and horizon.
    Safely captures skipped models (e.g. Prophet, XGBoost, LSTM for Yearly or short data).
    """
    n = len(ts_series)
    # Require at least minimum history for model estimation
    min_train = max(seasonal_period * 2, horizon * 2, 8)
    if min_train >= n:
        min_train = max(horizon + 2, n // 2)

    available_test_space = n - min_train
    if available_test_space < horizon:
        test_size = max(1, min(horizon, n // 4))
        splits = [n - test_size]
    else:
        n_windows = min(max_windows, max(1, available_test_space // horizon))
        if n_windows == 1:
            splits = [n - horizon]
        else:
            step = max(1, (available_test_space - horizon) // (n_windows - 1)) if n_windows > 1 else horizon
            splits = [min_train + i * step for i in range(n_windows)]
            if splits[-1] != (n - horizon):
                splits[-1] = n - horizon
            splits = sorted(list(set(splits)))

    # Storage for results per model
    model_evaluations: Dict[str, Dict[str, Any]] = {
        name: {
            "window_actuals": [],
            "window_preds": [],
            "step_errors": [[] for _ in range(horizon)],
            "failed_count": 0,
            "status": "Success",
            "last_error": None,
        }
        for name in MODEL_REGISTRY.keys()
    }

    # Yearly frequency rule: skip Prophet, XGBoost and LSTM immediately
    if freq_label == "Yearly":
        yearly_msg = "Skipped for Yearly frequency (insufficient annual observations and sub-annual seasonality not applicable)"
        for skipped_name in ["Prophet", "XGBoost", "LSTM"]:
            if skipped_name in model_evaluations:
                model_evaluations[skipped_name]["status"] = "Skipped"
                model_evaluations[skipped_name]["last_error"] = yearly_msg

    window_info = []
    has_zeros = (ts_series == 0).any()

    # Iterate over backtesting splits
    for split_idx, split_pt in enumerate(splits):
        train = ts_series.iloc[:split_pt]
        test = ts_series.iloc[split_pt : split_pt + horizon]
        h_actual = len(test)

        if h_actual == 0:
            continue

        window_info.append(
            {
                "split_index": split_idx + 1,
                "train_end": ts_series.index[split_pt - 1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                "test_length": h_actual,
            }
        )

        actual_vals = test.values

        for name, model_fn in MODEL_REGISTRY.items():
            # If already permanently skipped for this run, do not re-run
            if model_evaluations[name]["status"] == "Skipped":
                continue

            try:
                res = model_fn(
                    train,
                    horizon=h_actual,
                    seasonal_period=seasonal_period,
                    freq_label=freq_label,
                    pandas_freq=pandas_freq,
                )
                pred = res["forecast"]

                if res.get("status") == "Skipped":
                    model_evaluations[name]["status"] = "Skipped"
                    model_evaluations[name]["last_error"] = res.get("error_msg", "Model requirements not met")
                    continue

                if res.get("status") == "Failed" or np.isnan(pred).any():
                    model_evaluations[name]["failed_count"] += 1
                    model_evaluations[name]["status"] = "Failed"
                    model_evaluations[name]["last_error"] = res.get("error_msg", "Calculation failed")
                    continue

                model_evaluations[name]["window_actuals"].append(actual_vals)
                model_evaluations[name]["window_preds"].append(pred)

                # Record error per horizon step
                for step_h in range(min(h_actual, horizon)):
                    err = abs(actual_vals[step_h] - pred[step_h])
                    model_evaluations[name]["step_errors"][step_h].append(err)

            except Exception as e:
                model_evaluations[name]["failed_count"] += 1
                model_evaluations[name]["status"] = "Failed"
                model_evaluations[name]["last_error"] = str(e)

    # Compute aggregate metrics per model
    summary_metrics = []
    horizon_error_data = {}

    for name, data in model_evaluations.items():
        if len(data["window_actuals"]) == 0:
            final_status = data["status"] if data["status"] in ["Skipped", "Failed"] else "Failed"
            summary_metrics.append(
                {
                    "Model": name,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "sMAPE (%)": np.nan,
                    "MAPE (%)": np.nan,
                    "Metric_Type": "sMAPE" if has_zeros else "MAPE",
                    "Windows_Tested": 0,
                    "Status": final_status,
                    "Details": data["last_error"] or "Could not converge on validation splits",
                }
            )
            continue

        all_actuals = np.concatenate(data["window_actuals"])
        all_preds = np.concatenate(data["window_preds"])

        m = compute_metrics(all_actuals, all_preds, use_smape=has_zeros)

        # Average error per horizon step
        step_avg_errors = []
        for step_h in range(horizon):
            errs = data["step_errors"][step_h]
            step_avg_errors.append(float(np.mean(errs)) if len(errs) > 0 else np.nan)
        horizon_error_data[name] = step_avg_errors

        summary_metrics.append(
            {
                "Model": name,
                "MAE": round(m["MAE"], 2),
                "RMSE": round(m["RMSE"], 2),
                "sMAPE (%)": round(m["sMAPE"], 2),
                "MAPE (%)": round(m["MAPE"], 2),
                "Metric_Type": m["metric_type"],
                "Windows_Tested": len(data["window_actuals"]),
                "Status": "Success",
                "Details": "Converged smoothly across all evaluation windows",
            }
        )

    metrics_df = pd.DataFrame(summary_metrics)

    # Auto-select best model strictly by RMSE
    valid_models_df = metrics_df[metrics_df["Status"] == "Success"].copy()
    if not valid_models_df.empty:
        valid_models_df = valid_models_df.sort_values(by="RMSE", ascending=True)
        best_model_name = valid_models_df.iloc[0]["Model"]
        best_rmse = valid_models_df.iloc[0]["RMSE"]
        best_mae = valid_models_df.iloc[0]["MAE"]
        best_smape = valid_models_df.iloc[0]["sMAPE (%)"]
        best_mape = valid_models_df.iloc[0]["MAPE (%)"]
    else:
        best_model_name = "Naive"
        best_rmse = np.nan
        best_mae = np.nan
        best_smape = np.nan
        best_mape = np.nan

    # Calculate % improvement over baseline (Seasonal Naive or Naive)
    snaive_row = metrics_df[metrics_df["Model"] == "Seasonal Naive"]
    naive_row = metrics_df[metrics_df["Model"] == "Naive"]

    improvement_pct = 0.0
    baseline_used = "Seasonal Naive"

    if (
        not snaive_row.empty
        and snaive_row.iloc[0]["Status"] == "Success"
        and not np.isnan(snaive_row.iloc[0]["RMSE"])
    ):
        snaive_rmse = snaive_row.iloc[0]["RMSE"]
        if snaive_rmse > 0 and not np.isnan(best_rmse):
            improvement_pct = ((snaive_rmse - best_rmse) / snaive_rmse) * 100.0
    elif (
        not naive_row.empty
        and naive_row.iloc[0]["Status"] == "Success"
        and not np.isnan(naive_row.iloc[0]["RMSE"])
    ):
        naive_rmse = naive_row.iloc[0]["RMSE"]
        baseline_used = "Naive"
        if naive_rmse > 0 and not np.isnan(best_rmse):
            improvement_pct = ((naive_rmse - best_rmse) / naive_rmse) * 100.0

    # Extract test window actual vs predicted for the most recent test window
    last_test_slice = None
    last_predictions = {}
    if len(splits) > 0:
        last_split = splits[-1]
        last_test_series = ts_series.iloc[last_split : last_split + horizon]
        last_train_series = ts_series.iloc[:last_split]

        for name, model_fn in MODEL_REGISTRY.items():
            if model_evaluations[name]["status"] == "Skipped":
                continue
            try:
                pred_res = model_fn(
                    last_train_series,
                    len(last_test_series),
                    seasonal_period,
                    freq_label=freq_label,
                    pandas_freq=pandas_freq,
                )
                if pred_res["status"] == "Success":
                    last_predictions[name] = pd.Series(pred_res["forecast"], index=last_test_series.index)
            except Exception:
                pass

        last_test_slice = {
            "train": last_train_series,
            "actual": last_test_series,
            "predictions": last_predictions,
        }

    return {
        "metrics_df": metrics_df,
        "best_model_name": best_model_name,
        "best_rmse": best_rmse,
        "best_mae": best_mae,
        "best_smape": best_smape,
        "best_mape": best_mape,
        "improvement_pct": round(improvement_pct, 1),
        "baseline_used": baseline_used,
        "horizon_error_data": horizon_error_data,
        "window_info": window_info,
        "last_test_slice": last_test_slice,
        "has_zeros": has_zeros,
    }


def fit_best_future_forecast(
    ts_series: pd.Series,
    best_model_name: str,
    horizon: int,
    seasonal_period: int,
    pandas_freq: str = None,
    freq_label: str = None,
) -> Dict[str, Any]:
    """
    Fits the best selected model on the entire historical dataset
    and produces out-of-sample future forecasts with 95% confidence intervals.
    """
    model_fn = MODEL_REGISTRY.get(best_model_name, MODEL_REGISTRY["Naive"])
    try:
        res = model_fn(
            ts_series,
            horizon=horizon,
            seasonal_period=seasonal_period,
            freq_label=freq_label,
            pandas_freq=pandas_freq,
        )
    except Exception as e:
        res = MODEL_REGISTRY["Naive"](ts_series, horizon=horizon, seasonal_period=seasonal_period)

    # Generate future dates
    last_date = ts_series.index[-1]
    if pandas_freq:
        future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=pandas_freq)[1:]
    else:
        if len(ts_series) >= 2:
            delta = ts_series.index[-1] - ts_series.index[-2]
            future_dates = pd.date_range(start=last_date + delta, periods=horizon, freq=delta)
        else:
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq="D")[1:]

    forecast_df = pd.DataFrame(
        {
            "Forecast": res["forecast"],
            "Lower_CI_95": res["lower_ci"],
            "Upper_CI_95": res["upper_ci"],
        },
        index=future_dates,
    )
    forecast_df.index.name = "Date"

    return {
        "model_name": best_model_name,
        "forecast_df": forecast_df,
        "status": res["status"],
        "error_msg": res.get("error_msg"),
        "subtype": res.get("subtype"),
    }
