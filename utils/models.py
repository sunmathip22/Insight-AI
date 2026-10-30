"""
Forecasting Models Module
Contains time series models: Naive, Seasonal Naive, Moving Average,
Holt-Winters (Exponential Smoothing), and SARIMA.
Each model is robustly wrapped in try/except with confidence intervals.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_naive(
    train_series: pd.Series, horizon: int, seasonal_period: int = 1, **kwargs
) -> Dict[str, Any]:
    """
    Naive Model: Forecasts future periods using the most recent observation.
    """
    try:
        last_val = float(train_series.iloc[-1])
        forecast = np.full(horizon, last_val)

        # Residual std for confidence interval
        residuals = train_series.diff().dropna()
        residual_std = float(residuals.std()) if len(residuals) > 1 else max(1.0, last_val * 0.05)
        if np.isnan(residual_std) or residual_std == 0:
            residual_std = max(1.0, last_val * 0.05)

        # Standard error grows with sqrt of horizon step
        steps = np.sqrt(np.arange(1, horizon + 1))
        lower_ci = np.maximum(0, forecast - 1.96 * residual_std * steps)
        upper_ci = forecast + 1.96 * residual_std * steps

        return {
            "name": "Naive",
            "forecast": forecast,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "status": "Success",
            "error_msg": None,
        }
    except Exception as e:
        return {
            "name": "Naive",
            "forecast": np.full(horizon, np.nan),
            "lower_ci": np.full(horizon, np.nan),
            "upper_ci": np.full(horizon, np.nan),
            "status": "Failed",
            "error_msg": str(e),
        }


def forecast_seasonal_naive(
    train_series: pd.Series, horizon: int, seasonal_period: int = 7, **kwargs
) -> Dict[str, Any]:
    """
    Seasonal Naive Model: Repeats the observation from the previous seasonal cycle.
    y_{t+k} = y_{t+k - m}
    """
    try:
        m = seasonal_period
        n = len(train_series)
        if n < m:
            # Fall back to naive if not even 1 cycle
            return forecast_naive(train_series, horizon, seasonal_period)

        # Extract last seasonal cycle
        last_cycle = train_series.iloc[-m:].values
        # Tile across horizon
        repeats = int(np.ceil(horizon / m))
        forecast = np.tile(last_cycle, repeats)[:horizon]

        # Empirical seasonal residual std
        if n > m:
            seasonal_diff = train_series.values[m:] - train_series.values[:-m]
            residual_std = float(np.std(seasonal_diff)) if len(seasonal_diff) > 1 else 1.0
        else:
            residual_std = float(train_series.std()) if len(train_series) > 1 else 1.0

        if np.isnan(residual_std) or residual_std == 0:
            residual_std = max(1.0, float(np.mean(forecast)) * 0.05)

        # Confidence intervals
        cycle_idx = np.floor(np.arange(0, horizon) / m) + 1
        steps = np.sqrt(cycle_idx)
        lower_ci = np.maximum(0, forecast - 1.96 * residual_std * steps)
        upper_ci = forecast + 1.96 * residual_std * steps

        return {
            "name": "Seasonal Naive",
            "forecast": forecast,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "status": "Success",
            "error_msg": None,
        }
    except Exception as e:
        return {
            "name": "Seasonal Naive",
            "forecast": np.full(horizon, np.nan),
            "lower_ci": np.full(horizon, np.nan),
            "upper_ci": np.full(horizon, np.nan),
            "status": "Failed",
            "error_msg": str(e),
        }


def forecast_moving_average(
    train_series: pd.Series, horizon: int, seasonal_period: int = 7, **kwargs
) -> Dict[str, Any]:
    """
    Moving Average Model: Forecasts based on trailing rolling window average.
    Window size defaults to min(len(train)//2, max(3, seasonal_period)).
    """
    try:
        n = len(train_series)
        window = min(max(3, seasonal_period), max(2, n // 2))
        ma_val = float(train_series.iloc[-window:].mean())
        forecast = np.full(horizon, ma_val)

        # Residual std
        rolling_mean = train_series.rolling(window=window).mean().dropna()
        residuals = train_series.iloc[-len(rolling_mean):] - rolling_mean
        residual_std = float(residuals.std()) if len(residuals) > 1 else max(1.0, ma_val * 0.05)
        if np.isnan(residual_std) or residual_std == 0:
            residual_std = max(1.0, ma_val * 0.05)

        steps = np.sqrt(np.arange(1, horizon + 1))
        lower_ci = np.maximum(0, forecast - 1.96 * residual_std * steps)
        upper_ci = forecast + 1.96 * residual_std * steps

        return {
            "name": "Moving Average",
            "forecast": forecast,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "status": "Success",
            "error_msg": None,
        }
    except Exception as e:
        return {
            "name": "Moving Average",
            "forecast": np.full(horizon, np.nan),
            "lower_ci": np.full(horizon, np.nan),
            "upper_ci": np.full(horizon, np.nan),
            "status": "Failed",
            "error_msg": str(e),
        }


def forecast_holt_winters(
    train_series: pd.Series, horizon: int, seasonal_period: int = 7, **kwargs
) -> Dict[str, Any]:
    """
    Holt-Winters Exponential Smoothing.
    Attempts:
    1. Additive trend + additive seasonal (if n >= 2 * m)
    2. Additive trend + multiplicative seasonal (if strictly positive)
    3. Holt's linear (trend only) fallback
    4. Simple exponential smoothing fallback
    """
    n = len(train_series)
    m = seasonal_period
    can_use_seasonal = (n >= 2 * m) and (m > 1) and (m <= 52)
    strictly_positive = (train_series > 0).all()

    model_fit = None
    fitted_name = "Holt-Winters"

    # Strategy 1: Additive Trend & Seasonal
    if can_use_seasonal:
        try:
            hw = ExponentialSmoothing(
                train_series,
                trend="add",
                seasonal="add",
                seasonal_periods=m,
                initialization_method="estimated",
            )
            model_fit = hw.fit(optimized=True)
            fitted_name = "Holt-Winters (Add, Add)"
        except Exception:
            model_fit = None

    # Strategy 2: Multiplicative Seasonal if positive
    if model_fit is None and can_use_seasonal and strictly_positive:
        try:
            hw = ExponentialSmoothing(
                train_series,
                trend="add",
                seasonal="mul",
                seasonal_periods=m,
                initialization_method="estimated",
            )
            model_fit = hw.fit(optimized=True)
            fitted_name = "Holt-Winters (Add, Mul)"
        except Exception:
            model_fit = None

    # Strategy 3: Non-seasonal Holt's Linear (Trend only)
    if model_fit is None:
        try:
            hw = ExponentialSmoothing(
                train_series,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            )
            model_fit = hw.fit(optimized=True)
            fitted_name = "Holt-Winters (Trend Only)"
        except Exception:
            model_fit = None

    # Strategy 4: Simple Exponential Smoothing
    if model_fit is None:
        try:
            hw = ExponentialSmoothing(
                train_series,
                trend=None,
                seasonal=None,
                initialization_method="estimated",
            )
            model_fit = hw.fit(optimized=True)
            fitted_name = "Holt-Winters (Simple ES)"
        except Exception as e:
            return {
                "name": "Holt-Winters",
                "forecast": np.full(horizon, np.nan),
                "lower_ci": np.full(horizon, np.nan),
                "upper_ci": np.full(horizon, np.nan),
                "status": "Failed",
                "error_msg": str(e),
            }

    try:
        forecast_vals = model_fit.forecast(horizon).values
        # Ensure non-negative forecast for sales
        forecast_vals = np.maximum(0, forecast_vals)

        # Confidence intervals using residual variance
        residuals = model_fit.resid
        res_std = float(residuals.std()) if len(residuals) > 1 else max(1.0, float(np.mean(forecast_vals)) * 0.05)
        if np.isnan(res_std) or res_std == 0:
            res_std = max(1.0, float(np.mean(forecast_vals)) * 0.05)

        steps = np.sqrt(np.arange(1, horizon + 1))
        lower_ci = np.maximum(0, forecast_vals - 1.96 * res_std * steps)
        upper_ci = forecast_vals + 1.96 * res_std * steps

        return {
            "name": "Holt-Winters",
            "subtype": fitted_name,
            "forecast": forecast_vals,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "status": "Success",
            "error_msg": None,
        }
    except Exception as e:
        return {
            "name": "Holt-Winters",
            "forecast": np.full(horizon, np.nan),
            "lower_ci": np.full(horizon, np.nan),
            "upper_ci": np.full(horizon, np.nan),
            "status": "Failed",
            "error_msg": str(e),
        }


def forecast_sarima(
    train_series: pd.Series, horizon: int, seasonal_period: int = 7, **kwargs
) -> Dict[str, Any]:
    """
    SARIMA / ARIMA Model:
    Uses seasonal SARIMAX((1,1,1)x(0,1,1,m)) when seasonal period m <= 12 and n >= 2*m.
    For high-frequency or long cycles (e.g. m > 12) or short datasets, fits ARIMA(1,1,1)
    to guarantee blazing fast, robust convergence without numerical stall.
    """
    import warnings
    warnings.filterwarnings("ignore")

    n = len(train_series)
    m = seasonal_period
    can_use_seasonal = (n >= 2 * m) and (1 < m <= 12)

    model_fit = None
    fitted_name = "SARIMA"

    # Strategy 1: Seasonal SARIMAX for m <= 12 (Daily m=7, Monthly m=12)
    if can_use_seasonal:
        try:
            sarima = SARIMAX(
                train_series,
                order=(1, 1, 1),
                seasonal_order=(0, 1, 1, m),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            model_fit = sarima.fit(disp=False, maxiter=25)
            fitted_name = f"SARIMA(1,1,1)x(0,1,1,{m})"
        except Exception:
            model_fit = None

    # Strategy 2: Fast ARIMA(1, 1, 1)
    if model_fit is None:
        try:
            arima = SARIMAX(
                train_series,
                order=(1, 1, 1),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            model_fit = arima.fit(disp=False, maxiter=25)
            fitted_name = "ARIMA(1,1,1)"
        except Exception:
            model_fit = None

    # Strategy 3: Fast ARIMA(1, 0, 0)
    if model_fit is None:
        try:
            arima = SARIMAX(
                train_series,
                order=(1, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            model_fit = arima.fit(disp=False, maxiter=20)
            fitted_name = "ARIMA(1,0,0)"
        except Exception as e:
            return {
                "name": "SARIMA",
                "forecast": np.full(horizon, np.nan),
                "lower_ci": np.full(horizon, np.nan),
                "upper_ci": np.full(horizon, np.nan),
                "status": "Failed",
                "error_msg": str(e),
            }

    try:
        # Forecast with state-space prediction intervals
        forecast_res = model_fit.get_forecast(steps=horizon)
        forecast_vals = np.maximum(0, forecast_res.predicted_mean.values)
        conf_int = forecast_res.conf_int(alpha=0.05)

        lower_ci = np.maximum(0, conf_int.iloc[:, 0].values)
        upper_ci = conf_int.iloc[:, 1].values

        # If CI values were degenerate, fallback to empirical
        if np.isnan(lower_ci).any() or np.isnan(upper_ci).any():
            residuals = model_fit.resid
            res_std = float(residuals.std()) if len(residuals) > 1 else float(np.mean(forecast_vals)) * 0.05
            steps = np.sqrt(np.arange(1, horizon + 1))
            lower_ci = np.maximum(0, forecast_vals - 1.96 * res_std * steps)
            upper_ci = forecast_vals + 1.96 * res_std * steps

        return {
            "name": "SARIMA",
            "subtype": fitted_name,
            "forecast": forecast_vals,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "status": "Success",
            "error_msg": None,
        }
    except Exception as e:
        return {
            "name": "SARIMA",
            "forecast": np.full(horizon, np.nan),
            "lower_ci": np.full(horizon, np.nan),
            "upper_ci": np.full(horizon, np.nan),
            "status": "Failed",
            "error_msg": str(e),
        }


def forecast_arima(
    train_series: pd.Series,
    horizon: int,
    seasonal_period: int = 1,
    freq_label: str = None,
    pandas_freq: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    ARIMA Model: Used specifically for Yearly frequency or general non-seasonal trends.
    Uses ARIMA(1,1,0) or ARIMA(1,1,1) via statsmodels without seasonal orders.
    """
    import warnings
    warnings.filterwarnings("ignore")

    try:
        n = len(train_series)
        if n < 3:
            return forecast_naive(train_series, horizon, seasonal_period)

        model_fit = None
        # Strategy 1: ARIMA(1, 1, 0)
        try:
            arima = SARIMAX(
                train_series,
                order=(1, 1, 0) if n >= 5 else (1, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            model_fit = arima.fit(disp=False, maxiter=25)
        except Exception:
            model_fit = None

        # Strategy 2: ARIMA(0, 1, 1)
        if model_fit is None:
            try:
                arima = SARIMAX(
                    train_series,
                    order=(0, 1, 1),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                model_fit = arima.fit(disp=False, maxiter=25)
            except Exception:
                model_fit = None

        if model_fit is None:
            return forecast_naive(train_series, horizon, seasonal_period)

        forecast_res = model_fit.get_forecast(steps=horizon)
        forecast_vals = np.maximum(0, forecast_res.predicted_mean.values)
        conf_int = forecast_res.conf_int(alpha=0.05)

        lower_ci = np.maximum(0, conf_int.iloc[:, 0].values)
        upper_ci = conf_int.iloc[:, 1].values

        if np.isnan(lower_ci).any() or np.isnan(upper_ci).any():
            residuals = model_fit.resid
            res_std = float(residuals.std()) if len(residuals) > 1 else max(1.0, float(np.mean(forecast_vals)) * 0.05)
            steps = np.sqrt(np.arange(1, horizon + 1))
            lower_ci = np.maximum(0, forecast_vals - 1.96 * res_std * steps)
            upper_ci = forecast_vals + 1.96 * res_std * steps

        return {
            "name": "ARIMA",
            "forecast": forecast_vals,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "status": "Success",
            "error_msg": None,
        }
    except Exception as e:
        return {
            "name": "ARIMA",
            "forecast": np.full(horizon, np.nan),
            "lower_ci": np.full(horizon, np.nan),
            "upper_ci": np.full(horizon, np.nan),
            "status": "Failed",
            "error_msg": str(e),
        }


def forecast_prophet(
    train_series: pd.Series,
    horizon: int,
    seasonal_period: int = 7,
    freq_label: str = None,
    pandas_freq: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Prophet Model: Additive model where non-linear trends fit with yearly/weekly/daily seasonality.
    Requires at least 2 seasonal cycles. Skipped for Yearly frequency.
    """
    try:
        # Yearly frequency rule: skip Prophet
        if freq_label == "Yearly":
            return {
                "name": "Prophet",
                "forecast": np.full(horizon, np.nan),
                "lower_ci": np.full(horizon, np.nan),
                "upper_ci": np.full(horizon, np.nan),
                "status": "Skipped",
                "error_msg": "Skipped for Yearly frequency (insufficient annual observations and sub-annual seasonality not applicable)",
            }

        n = len(train_series)
        min_cycles = 2 * seasonal_period if seasonal_period > 1 else 14
        if n < min_cycles:
            return {
                "name": "Prophet",
                "forecast": np.full(horizon, np.nan),
                "lower_ci": np.full(horizon, np.nan),
                "upper_ci": np.full(horizon, np.nan),
                "status": "Skipped",
                "error_msg": f"Requires at least 2 seasonal cycles ({min_cycles} observations, has {n})",
            }

        import logging
        from prophet import Prophet

        # Mute logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

        # Build clean ds and y dataframe
        ds_vals = pd.to_datetime(train_series.index)
        if hasattr(ds_vals, "tz") and ds_vals.tz is not None:
            ds_vals = ds_vals.tz_localize(None)

        df = pd.DataFrame({"ds": ds_vals, "y": train_series.values.astype(float)})

        # Configure fast seasonalities
        has_weekly = seasonal_period == 7
        has_yearly = (seasonal_period == 52 and n >= 104) or (seasonal_period == 12 and n >= 24) or (n >= 730)

        m = Prophet(
            yearly_seasonality=has_yearly,
            weekly_seasonality=has_weekly,
            daily_seasonality=False,
            interval_width=0.95,
        )
        m.fit(df)

        p_freq = pandas_freq or ("D" if seasonal_period == 7 else "W" if seasonal_period == 52 else "MS" if seasonal_period == 12 else "D")
        future = m.make_future_dataframe(periods=horizon, freq=p_freq, include_history=False)
        forecast_res = m.predict(future)

        forecast_vals = np.maximum(0, forecast_res["yhat"].values[:horizon])
        lower_ci = np.maximum(0, forecast_res["yhat_lower"].values[:horizon])
        upper_ci = forecast_res["yhat_upper"].values[:horizon]

        return {
            "name": "Prophet",
            "forecast": forecast_vals,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "status": "Success",
            "error_msg": None,
        }
    except Exception as e:
        return {
            "name": "Prophet",
            "forecast": np.full(horizon, np.nan),
            "lower_ci": np.full(horizon, np.nan),
            "upper_ci": np.full(horizon, np.nan),
            "status": "Failed",
            "error_msg": str(e),
        }


def forecast_xgboost(
    train_series: pd.Series,
    horizon: int,
    seasonal_period: int = 7,
    freq_label: str = None,
    pandas_freq: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    XGBoost Model: Uses lag features (lag_1, lag_2, lag_3), rolling mean (window=3),
    and calendar features (month, week, day of week).
    Skipped for Yearly frequency or if observations < 20.
    """
    try:
        # Yearly frequency rule: skip XGBoost
        if freq_label == "Yearly":
            return {
                "name": "XGBoost",
                "forecast": np.full(horizon, np.nan),
                "lower_ci": np.full(horizon, np.nan),
                "upper_ci": np.full(horizon, np.nan),
                "status": "Skipped",
                "error_msg": "Skipped for Yearly frequency (insufficient annual observations and sub-annual seasonality not applicable)",
            }

        n = len(train_series)
        if n < 20:
            return {
                "name": "XGBoost",
                "forecast": np.full(horizon, np.nan),
                "lower_ci": np.full(horizon, np.nan),
                "upper_ci": np.full(horizon, np.nan),
                "status": "Skipped",
                "error_msg": f"Requires at least 20 observations for feature extraction (has {n})",
            }

        import xgboost as xgb

        dates = pd.to_datetime(train_series.index)
        y = train_series.values.astype(float)

        # Build feature matrix
        df_feat = pd.DataFrame(index=range(n))
        df_feat["y"] = y
        df_feat["lag_1"] = df_feat["y"].shift(1)
        df_feat["lag_2"] = df_feat["y"].shift(2)
        df_feat["lag_3"] = df_feat["y"].shift(3)
        df_feat["rolling_mean_3"] = df_feat["y"].shift(1).rolling(window=3).mean()
        df_feat["month"] = dates.month
        df_feat["week"] = dates.isocalendar().week.astype(int).values
        df_feat["dayofweek"] = dates.dayofweek

        train_data = df_feat.dropna().reset_index(drop=True)
        feature_cols = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "month", "week", "dayofweek"]
        X_train = train_data[feature_cols].values
        y_train = train_data["y"].values

        model = xgb.XGBRegressor(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X_train, y_train)

        # In-sample residuals for CI
        in_preds = model.predict(X_train)
        residuals = y_train - in_preds
        res_std = float(np.std(residuals)) if len(residuals) > 1 else max(1.0, float(np.mean(y)) * 0.05)
        if np.isnan(res_std) or res_std == 0:
            res_std = max(1.0, float(np.mean(y)) * 0.05)

        # Future calendar dates
        last_date = dates[-1]
        p_freq = pandas_freq or ("D" if seasonal_period == 7 else "W" if seasonal_period == 52 else "MS" if seasonal_period == 12 else "D")
        try:
            future_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=p_freq)[1:]
        except Exception:
            delta = dates[-1] - dates[-2] if len(dates) >= 2 else pd.Timedelta(days=1)
            future_dates = pd.date_range(start=last_date + delta, periods=horizon, freq=delta)

        # Multi-step recursive forecasting
        history = list(y)
        forecast_vals = []
        for h_idx in range(horizon):
            f_date = future_dates[h_idx]
            l1 = history[-1]
            l2 = history[-2]
            l3 = history[-3]
            rm3 = np.mean(history[-3:])
            x_step = np.array([[l1, l2, l3, rm3, f_date.month, int(f_date.isocalendar().week), f_date.dayofweek]])
            pred_val = max(0.0, float(model.predict(x_step)[0]))
            forecast_vals.append(pred_val)
            history.append(pred_val)

        forecast_arr = np.array(forecast_vals)
        steps = np.sqrt(np.arange(1, horizon + 1))
        lower_ci = np.maximum(0, forecast_arr - 1.96 * res_std * steps)
        upper_ci = forecast_arr + 1.96 * res_std * steps

        return {
            "name": "XGBoost",
            "forecast": forecast_arr,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "status": "Success",
            "error_msg": None,
        }
    except Exception as e:
        return {
            "name": "XGBoost",
            "forecast": np.full(horizon, np.nan),
            "lower_ci": np.full(horizon, np.nan),
            "upper_ci": np.full(horizon, np.nan),
            "status": "Failed",
            "error_msg": str(e),
        }


def forecast_lstm(
    train_series: pd.Series,
    horizon: int,
    seasonal_period: int = 7,
    freq_label: str = None,
    pandas_freq: str = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    LSTM Model: Lightweight sliding window LSTM implemented in PyTorch.
    Uses scaled data (MinMaxScaler) and a compact 1-layer architecture for fast training.
    Skipped for Yearly frequency or if observations < 50.
    """
    try:
        # Yearly frequency rule: skip LSTM
        if freq_label == "Yearly":
            return {
                "name": "LSTM",
                "forecast": np.full(horizon, np.nan),
                "lower_ci": np.full(horizon, np.nan),
                "upper_ci": np.full(horizon, np.nan),
                "status": "Skipped",
                "error_msg": "Skipped for Yearly frequency (insufficient annual observations and sub-annual seasonality not applicable)",
            }

        n = len(train_series)
        if n < 50:
            return {
                "name": "LSTM",
                "forecast": np.full(horizon, np.nan),
                "lower_ci": np.full(horizon, np.nan),
                "upper_ci": np.full(horizon, np.nan),
                "status": "Skipped",
                "error_msg": f"Requires at least 50 observations for sequence training (has {n})",
            }

        import torch
        import torch.nn as nn
        from sklearn.preprocessing import MinMaxScaler

        torch.manual_seed(42)
        np.random.seed(42)

        y_vals = train_series.values.astype(np.float32).reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0.05, 0.95))
        y_scaled = scaler.fit_transform(y_vals).flatten()

        window_size = min(12, max(6, seasonal_period if 1 < seasonal_period <= 12 else 8))
        if n - window_size < 10:
            window_size = max(4, n // 5)

        X_list, y_list = [], []
        for i in range(len(y_scaled) - window_size):
            X_list.append(y_scaled[i : i + window_size])
            y_list.append(y_scaled[i + window_size])

        X_train = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1)

        class SmallLSTM(nn.Module):
            def __init__(self, input_dim=1, hidden_dim=16, output_dim=1):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=1)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        model = SmallLSTM(input_dim=1, hidden_dim=16, output_dim=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.015)

        model.train()
        for epoch in range(25):
            optimizer.zero_grad()
            preds = model(X_train)
            loss = criterion(preds, y_train)
            loss.backward()
            optimizer.step()

        model.eval()
        curr_window = list(y_scaled[-window_size:])
        preds_scaled = []

        with torch.no_grad():
            for _ in range(horizon):
                inp = torch.tensor(curr_window[-window_size:], dtype=torch.float32).view(1, window_size, 1)
                next_pred = max(0.0, float(model(inp).item()))
                preds_scaled.append(next_pred)
                curr_window.append(next_pred)

        preds_unscaled = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        forecast_vals = np.maximum(0, preds_unscaled)

        # Residual std for CI
        with torch.no_grad():
            in_preds_scaled = model(X_train).numpy()
            in_preds = scaler.inverse_transform(in_preds_scaled).flatten()
            actuals_slice = y_vals[window_size:].flatten()
            residuals = actuals_slice - in_preds
            res_std = float(np.std(residuals)) if len(residuals) > 1 else max(1.0, float(np.mean(forecast_vals)) * 0.05)
            if np.isnan(res_std) or res_std == 0:
                res_std = max(1.0, float(np.mean(forecast_vals)) * 0.05)

        steps = np.sqrt(np.arange(1, horizon + 1))
        lower_ci = np.maximum(0, forecast_vals - 1.96 * res_std * steps)
        upper_ci = forecast_vals + 1.96 * res_std * steps

        return {
            "name": "LSTM",
            "forecast": forecast_vals,
            "lower_ci": lower_ci,
            "upper_ci": upper_ci,
            "status": "Success",
            "error_msg": None,
        }
    except Exception as e:
        return {
            "name": "LSTM",
            "forecast": np.full(horizon, np.nan),
            "lower_ci": np.full(horizon, np.nan),
            "upper_ci": np.full(horizon, np.nan),
            "status": "Failed",
            "error_msg": str(e),
        }


MODEL_REGISTRY = {
    "Naive": forecast_naive,
    "Seasonal Naive": forecast_seasonal_naive,
    "Moving Average": forecast_moving_average,
    "Holt-Winters": forecast_holt_winters,
    "SARIMA": forecast_sarima,
    "ARIMA": forecast_arima,
    "Prophet": forecast_prophet,
    "XGBoost": forecast_xgboost,
    "LSTM": forecast_lstm,
}

