"""
Data Preparation & Pipeline Module for Sales Forecasting Dashboard
Handles generic dataset loading, type coercion, cleaning, aggregation,
frequency detection, outlier flagging, and validation checks.
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np


import io


def get_excel_sheets(uploaded_file_or_path) -> List[str]:
    """
    Returns the list of sheet names from an Excel file or file-like object.
    """
    try:
        excel_file = pd.ExcelFile(uploaded_file_or_path)
        return excel_file.sheet_names
    except Exception as e:
        raise ValueError(f"Unable to read Excel workbook sheets: {e}")


def load_raw_dataset(uploaded_file_or_path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Loads dataset from an uploaded file or local filepath (CSV or Excel).
    Handles encoding variations (utf-8, latin-1), auto delimiters (comma, semicolon, tab),
    and multi-sheet Excel selections with clear error messages.
    """
    if uploaded_file_or_path is None:
        raise ValueError("No file provided for upload.")

    filename = getattr(uploaded_file_or_path, "name", str(uploaded_file_or_path)).lower()

    if filename.endswith((".xlsx", ".xls")):
        try:
            excel_file = pd.ExcelFile(uploaded_file_or_path)
            sheets = excel_file.sheet_names
            if not sheets:
                raise ValueError("Excel file contains no readable sheets.")
            target_sheet = sheet_name if sheet_name in sheets else sheets[0]
            df = pd.read_excel(excel_file, sheet_name=target_sheet)
            if df.empty:
                raise ValueError(f"The sheet '{target_sheet}' in the uploaded Excel file is empty.")
            return df
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Corrupt or unreadable Excel file: {e}")

    elif filename.endswith((".csv", ".tsv", ".txt")) or not filename.endswith((".xlsx", ".xls")):
        # Read file bytes to support rewind across multiple encodings and delimiter sniffers
        if hasattr(uploaded_file_or_path, "read"):
            data_bytes = uploaded_file_or_path.read()
            if hasattr(uploaded_file_or_path, "seek"):
                uploaded_file_or_path.seek(0)
        elif isinstance(uploaded_file_or_path, str):
            with open(uploaded_file_or_path, "rb") as f:
                data_bytes = f.read()
        elif isinstance(uploaded_file_or_path, bytes):
            data_bytes = uploaded_file_or_path
        else:
            raise ValueError("Unsupported file object type.")

        if not data_bytes or len(data_bytes.strip()) == 0:
            raise ValueError("Uploaded file is empty (0 bytes).")

        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        last_error = None

        for enc in encodings:
            try:
                # sep=None with engine='python' detects comma, semicolon, tab, pipe automatically
                df = pd.read_csv(io.BytesIO(data_bytes), encoding=enc, sep=None, engine="python")
                if df.empty or len(df.columns) == 0:
                    continue
                return df
            except Exception as e:
                last_error = e
                continue

        # Fallback to standard comma separated latin-1 with line skipping
        try:
            df = pd.read_csv(io.BytesIO(data_bytes), encoding="latin-1", on_bad_lines="skip")
            if not df.empty:
                return df
        except Exception as e:
            last_error = e

        raise ValueError(f"Corrupt or unsupported CSV file: {last_error or 'Unable to parse tabular data'}")

    else:
        raise ValueError("Unsupported file format. Please upload a .csv, .xlsx, or .xls file.")


def auto_detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Intelligently identifies Date, Sales, and Group columns:
    - Date: Column with highest parse ratio as valid dates (>= 60%), skipping purely numeric cols.
    - Sales: Numeric column prioritizing names like sales, revenue, amount, qty, demand, units, total.
    - Group: Low-cardinality categorical column (store, product, category, region), or 'None'.
    """
    date_col = None
    sales_col = None
    group_col = "None"

    # 1. Date column: check non-numeric columns for highest datetime parse ratio
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

    # Fallback date keyword scan
    if not date_col:
        for col in df.columns:
            lower = str(col).lower()
            if any(k in lower for k in ["date", "time", "day", "period", "timestamp"]):
                date_col = col
                break

    # 2. Sales column: numeric column with keyword preference
    sales_keywords = ["sales", "revenue", "amount", "qty", "demand", "units", "total", "volume"]
    candidates = []
    for col in df.columns:
        if col == date_col:
            continue
        try:
            s_clean = (
                df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace("£", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.strip()
            )
            parsed_num = pd.to_numeric(s_clean, errors="coerce")
            valid_ratio = parsed_num.notna().sum() / max(1, len(df))
            if valid_ratio >= 0.6:
                name_lower = str(col).lower()
                kw_priority = 0
                for idx, kw in enumerate(sales_keywords):
                    if kw in name_lower:
                        kw_priority = len(sales_keywords) - idx
                        break
                candidates.append((col, kw_priority, parsed_num.std()))
        except Exception:
            pass

    if candidates:
        candidates.sort(key=lambda x: (x[1], x[2] if pd.notna(x[2]) else 0), reverse=True)
        sales_col = candidates[0][0]

    # 3. Group column: low cardinality categorical/text column
    group_keywords = ["store", "product", "category", "region", "dept", "department", "channel", "segment", "brand", "item", "location"]
    group_candidates = []
    for col in df.columns:
        if col in [date_col, sales_col]:
            continue
        nunique = df[col].nunique(dropna=True)
        if 1 < nunique <= 50:
            name_lower = str(col).lower()
            kw_match = any(k in name_lower for k in group_keywords)
            is_text = not pd.api.types.is_numeric_dtype(df[col]) or kw_match
            if is_text:
                group_candidates.append((col, kw_match, nunique))

    if group_candidates:
        group_candidates.sort(key=lambda x: (1 if x[1] else 0, -x[2]), reverse=True)
        group_col = group_candidates[0][0]
    else:
        group_col = "None"

    return date_col, sales_col, group_col


def get_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates column diagnostic summary: column name, detected data type,
    missing count, missing percentage, and sample values.
    """
    records = []
    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_pct = round((missing_count / max(1, len(df))) * 100, 1)

        # Detect human-friendly data type
        if pd.api.types.is_datetime64_any_dtype(s):
            dtype_label = "Datetime"
        elif pd.api.types.is_numeric_dtype(s):
            dtype_label = "Numeric (Float/Int)"
        elif pd.api.types.is_bool_dtype(s):
            dtype_label = "Boolean"
        else:
            try:
                date_parsed = pd.to_datetime(s, errors="coerce")
                if date_parsed.notna().sum() / max(1, len(df)) >= 0.6:
                    dtype_label = "Date (String)"
                else:
                    num_parsed = pd.to_numeric(s.astype(str).str.replace(",", "").str.replace("$", ""), errors="coerce")
                    if num_parsed.notna().sum() / max(1, len(df)) >= 0.6:
                        dtype_label = "Numeric (Currency/String)"
                    else:
                        dtype_label = "Categorical / Text"
            except Exception:
                dtype_label = "Categorical / Text"

        sample_vals = [str(x) for x in s.dropna().unique()[:3]]
        sample_str = ", ".join(sample_vals) if sample_vals else "—"

        records.append({
            "Column Name": str(col),
            "Detected Type": dtype_label,
            "Missing Count": f"{missing_count} ({missing_pct}%)",
            "Sample Values": sample_str,
        })

    return pd.DataFrame(records)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    """
    Cleans a series by removing currency symbols, commas, and converting to float.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    cleaned = (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def detect_frequency_and_seasonality(dates: pd.DatetimeIndex) -> Tuple[str, str, int, int]:
    """
    Infers the series frequency from date diffs and assigns:
    - freq_label ('Daily', 'Weekly', 'Monthly', 'Yearly', 'Irregular')
    - pandas_freq ('D', 'W', 'MS', 'YE', None)
    - seasonal_period (7 for daily, 52 for weekly, 12 for monthly, 1 for yearly)
    - default_horizon (14 for daily, 13 for weekly, 3 for monthly, 2 for yearly)
    """
    if len(dates) < 3:
        return "Irregular", None, 1, 3

    diffs = dates.to_series().diff().dropna().dt.total_seconds() / (24 * 3600)
    median_days = diffs.median()

    if 0.5 <= median_days <= 1.5:
        return "Daily", "D", 7, 14
    elif 5.0 <= median_days <= 9.0:
        return "Weekly", "W", 52, 13
    elif 25.0 <= median_days <= 35.0:
        return "Monthly", "MS", 12, 3
    elif median_days >= 300.0:
        return "Yearly", "YE", 1, 2
    else:
        if median_days < 4.0:
            return "Daily", "D", 7, 14
        elif median_days < 20.0:
            return "Weekly", "W", 52, 13
        elif median_days < 180.0:
            return "Monthly", "MS", 12, 3
        else:
            return "Yearly", "YE", 1, 2


def flag_outliers(series: pd.Series, method: str = "iqr") -> pd.Series:
    """
    Flags outliers without removing them.
    Returns boolean Series where True indicates an outlier.
    """
    if len(series) < 5:
        return pd.Series(False, index=series.index)

    if method == "iqr":
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        return (series < lower_bound) | (series > upper_bound)
    else:
        mean = series.mean()
        std = series.std()
        if std == 0 or np.isnan(std):
            return pd.Series(False, index=series.index)
        z = (series - mean) / std
        return z.abs() > 3.0


def process_sales_data(
    df: pd.DataFrame,
    date_col: str,
    sales_col: str,
    group_col: Optional[str] = None,
    selected_group: Optional[str] = None,
    resample_freq: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes the complete generic data pipeline:
    1. Audits initial data quality (missing values, duplicates, invalid records)
    2. Filters group if requested
    3. Drops unparseable dates and invalid records
    4. Interpolates missing sales and flags outliers
    5. Aggregates duplicates and resamples to continuous regular cadence
    6. Returns clean time series along with Data Quality before & after report
    """
    warnings: List[str] = []
    info: List[str] = []

    if df.empty:
        raise ValueError("The uploaded dataset contains no rows.")

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in dataset.")
    if sales_col not in df.columns:
        raise ValueError(f"Sales column '{sales_col}' not found in dataset.")

    work_df = df.copy()

    # --- DATA QUALITY AUDIT (BEFORE CLEANING) ---
    raw_total_records = len(work_df)
    missing_dates_raw = int(work_df[date_col].isna().sum())
    missing_sales_raw = int(work_df[sales_col].isna().sum())
    missing_values_before = missing_dates_raw + missing_sales_raw

    duplicate_dates_count = int(work_df[date_col].duplicated().sum())
    duplicate_rows_count = int(work_df.duplicated().sum())
    duplicates_before = int(work_df.duplicated(subset=[date_col]).sum())

    parsed_dates_test = pd.to_datetime(work_df[date_col], errors="coerce")
    unparseable_dates_count = int(parsed_dates_test.isna().sum())
    cleaned_sales_test = clean_numeric_series(work_df[sales_col])
    non_numeric_sales_count = int(cleaned_sales_test.isna().sum())
    invalid_records_before = unparseable_dates_count + non_numeric_sales_count

    # Group filtering or total
    has_groups = False
    group_summary = None
    if group_col and group_col in work_df.columns:
        has_groups = True
        group_summary = (
            work_df.groupby(group_col)[sales_col]
            .apply(lambda s: clean_numeric_series(s).sum())
            .reset_index()
            .rename(columns={sales_col: "Total_Sales"})
            .sort_values(by="Total_Sales", ascending=False)
        )

        if selected_group and selected_group != "Total (all groups)":
            work_df = work_df[work_df[group_col].astype(str) == str(selected_group)]
            info.append(f"Filtered for group: {group_col} = '{selected_group}'")
        else:
            info.append(f"Aggregating across all groups in '{group_col}'")

    # Clean step 1: Date parsing & dropping unparseable dates
    work_df["__parsed_date__"] = pd.to_datetime(work_df[date_col], errors="coerce")
    invalid_dates = int(work_df["__parsed_date__"].isna().sum())
    if invalid_dates > 0:
        warnings.append(f"Dropped {invalid_dates} invalid record(s) with unparseable date values.")
        work_df = work_df.dropna(subset=["__parsed_date__"])

    if work_df.empty:
        raise ValueError("No valid date records remained after cleaning.")

    # Clean step 2: Sales numeric conversion & interpolating non-numeric / missing
    work_df["__clean_sales__"] = clean_numeric_series(work_df[sales_col])
    nan_sales = int(work_df["__clean_sales__"].isna().sum())
    if nan_sales > 0:
        warnings.append(f"Found {nan_sales} non-numeric/missing sales value(s); interpolated.")
        work_df["__clean_sales__"] = work_df["__clean_sales__"].interpolate(method="linear").bfill().ffill()

    # Clean step 3: Zero / negative handling
    zero_or_neg = int((work_df["__clean_sales__"] <= 0).sum())
    if zero_or_neg > 0:
        warnings.append(
            f"Flagged {zero_or_neg} zero or negative sales value(s). "
            f"Interpolation applied for negative values."
        )
        work_df.loc[work_df["__clean_sales__"] < 0, "__clean_sales__"] = np.nan
        work_df["__clean_sales__"] = work_df["__clean_sales__"].interpolate(method="linear").bfill().ffill()

    # Clean step 4: Aggregate duplicate rows & dates by summing sales
    ts_df = (
        work_df.groupby("__parsed_date__")["__clean_sales__"]
        .sum(min_count=1)
        .reset_index()
        .sort_values("__parsed_date__")
    )
    ts_df.rename(columns={"__parsed_date__": "Date", "__clean_sales__": "Sales"}, inplace=True)
    ts_df.set_index("Date", inplace=True)

    # Detect natural frequency from date gaps
    detected_label, pandas_freq, seasonal_period, default_horizon = detect_frequency_and_seasonality(
        ts_df.index
    )

    if resample_freq:
        rf = str(resample_freq).upper()
        if any(yr in rf for yr in ["YE", "YS", "Y", "A"]):
            detected_label = "Yearly"
            pandas_freq = "YE"
            seasonal_period = 1
            default_horizon = 2
        elif any(mo in rf for mo in ["MS", "M"]):
            detected_label = "Monthly"
            pandas_freq = "MS"
            seasonal_period = 12
            default_horizon = 3
        elif "W" in rf:
            detected_label = "Weekly"
            pandas_freq = "W"
            seasonal_period = 52
            default_horizon = 13
        elif "D" in rf:
            detected_label = "Daily"
            pandas_freq = "D"
            seasonal_period = 7
            default_horizon = 14
        chosen_freq = pandas_freq
    else:
        chosen_freq = pandas_freq

    # Resample to guarantee continuous regular cadence
    if chosen_freq:
        try:
            ts_resampled = ts_df["Sales"].resample(chosen_freq).sum()
            if ts_resampled.isna().any():
                ts_resampled = ts_resampled.interpolate(method="linear").bfill().ffill()
            ts_df = pd.DataFrame({"Sales": ts_resampled})
        except Exception:
            pass

    # Ensure no NaN remains in sales series
    ts_df["Sales"] = ts_df["Sales"].interpolate(method="linear").bfill().ffill()

    # Outlier detection
    ts_df["Is_Outlier"] = flag_outliers(ts_df["Sales"], method="iqr")
    outlier_count = int(ts_df["Is_Outlier"].sum())
    if outlier_count > 0:
        info.append(f"Flagged {outlier_count} statistical outlier(s) using IQR (preserved in dataset).")

    # Seasonal cycles check
    n_points = len(ts_df)
    if seasonal_period <= 1:
        has_sufficient_cycles = n_points >= 2
        cycle_count = n_points
    else:
        has_sufficient_cycles = n_points >= (2 * seasonal_period)
        cycle_count = round(n_points / seasonal_period, 2) if seasonal_period > 0 else 0

    if not has_sufficient_cycles and seasonal_period > 1:
        warnings.append(
            f"Dataset length ({n_points} observations) provides only {cycle_count} seasonal cycles "
            f"(seasonal period = {seasonal_period}). At least 2 full cycles ({2 * seasonal_period} points) "
            f"are recommended for seasonal decomposition and Holt-Winters/SARIMA seasonal components."
        )

    # --- DATA QUALITY REPORT (BEFORE vs AFTER) ---
    data_quality_report = {
        "before": {
            "total_records": raw_total_records,
            "missing_values": missing_values_before,
            "duplicate_records": duplicates_before,
            "invalid_records": invalid_records_before,
        },
        "after": {
            "total_records": n_points,
            "missing_values": 0,
            "duplicate_records": 0,
            "invalid_records": 0,
        },
        "summary": {
            "missing_filled_interpolation": missing_values_before,
            "duplicates_aggregated": duplicates_before,
            "invalid_dropped": invalid_records_before,
        },
    }

    return {
        "ts_df": ts_df,
        "n_observations": n_points,
        "detected_label": detected_label,
        "pandas_freq": pandas_freq,
        "seasonal_period": seasonal_period,
        "default_horizon": default_horizon,
        "has_sufficient_cycles": has_sufficient_cycles,
        "cycle_count": cycle_count,
        "outlier_count": outlier_count,
        "group_summary": group_summary,
        "has_groups": has_groups,
        "warnings": warnings,
        "info": info,
        "data_quality_report": data_quality_report,
    }

