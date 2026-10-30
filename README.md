# 📈 Enterprise Sales Forecasting Dashboard

A universal, production-grade sales forecasting web application built in Python using **Streamlit**, **Pandas**, **Statsmodels**, **Plotly**, and **Scikit-Learn**. 

Designed to operate seamlessly on **ANY sales dataset** (CSV or Excel) without hard-coded column dependencies, the dashboard automates the entire forecasting lifecycle: from data cleaning, frequency detection, and anomaly flagging to rolling-origin backtesting, automated model selection, future projections with 95% confidence intervals, and operational inventory/staffing recommendations.

---

## 🚀 Key Features

- **Generic & Agnostic Input**: Upload any CSV or Excel file and dynamically map Date, Sales, and optional Group columns (store, department, product line, region, channel).
- **Auto-Adaptive Data Pipeline**:
  - Handles mixed date formats, sorts chronologically, and aggregates duplicate timestamps.
  - Cleans currencies/symbols, interpolates missing values, and flags negative/zero observations.
  - Auto-detects frequency (Daily, Weekly, Monthly) from median timestamp gaps.
  - Automatically sets seasonal period ($m=7$ for daily, $m=52$ for weekly, $m=12$ for monthly).
  - Preserves statistical outliers using the Interquartile Range (IQR) rule and highlights them on interactive charts.
  - Verifies data length against minimum seasonal cycle requirements ($\ge 2m$).
- **5 Diverse Forecasting Engines**:
  1. **Naive**: Last observed sales value repeated with horizon variance scaling.
  2. **Seasonal Naive**: Repeats observations from $m$ periods ago with cyclical wrapping.
  3. **Moving Average**: Rolling trailing window average with empirical error bounds.
  4. **Holt-Winters Exponential Smoothing**: Additive/multiplicative trend and seasonal smoothing with automatic fallback.
  5. **SARIMA / ARIMA**: Seasonal and non-seasonal state-space modeling with parametric and empirical confidence intervals.
- **Leak-Free Rolling-Origin Validation**:
  - Expanding-window backtesting across multiple splits without look-ahead bias.
  - Adapts validation window count dynamically to historical dataset length.
  - Computes **MAE**, **RMSE**, and **MAPE** (automatically switches to **sMAPE** when zero values are present).
  - Automatically ranks and selects the best model by lowest RMSE.
  - Computes the percentage improvement of the champion model over the Seasonal Naive baseline.
- **5 Comprehensive Dashboard Tabs**:
  - **Overview**: Interactive sales timeline, rolling average, highlighted IQR outliers, group-wise sales breakdown, and descriptive statistics.
  - **Trend & Seasonality**: Classical additive decomposition (Observed, Trend, Seasonal, Residual) and cyclical aggregations by month of year and day of week.
  - **Forecast**: Out-of-sample future projections with shaded 95% confidence intervals plus backtest validation overlay.
  - **Model Comparison**: Full metrics leaderboard, error comparison bar chart, and Horizon vs Error degradation curves ($h=1$ to $h=H$).
  - **Business Recommendations**: Actionable executive summary, peak/trough demand analysis, safety stock calculations (95% & 99% service levels), staffing guidance, model retraining playbook, and operational limitations.
- **Exports**: One-click downloads for future forecast CSVs (with bounds) and model metrics CSVs.

---

## 📁 Repository Structure

```text
sales-dashboard/
├── app.py                      # Main Streamlit web application
├── utils/
│   ├── data_prep.py            # Dynamic loading, cleaning, frequency detection, IQR outliers
│   ├── models.py               # Naive, Seasonal Naive, MA, Holt-Winters, SARIMA engines
│   ├── evaluation.py           # Rolling-origin cross-validation, metrics, horizon tracking
│   └── insights.py             # Business intelligence, safety stock, staffing, limitations
├── data/
│   ├── walmart_weekly_sales.csv# Demo 1: Weekly department-level retail sales
│   ├── retail_monthly_sales.csv# Demo 2: Monthly category revenues
│   └── daily_ecommerce_sales.csv# Demo 3: Daily multi-channel ecommerce transactions
├── generate_samples.py         # Script to generate sample datasets
├── test_app.py                 # Automated unit and integration test suite
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation and guide
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Clone or Navigate to Directory
```bash
cd "c:/Users/santh/OneDrive/Desktop/sales dashboard"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-generate Demo Datasets
```bash
python generate_samples.py
```

### 5. Run Integration Tests
```bash
python test_app.py
```

---

## 💻 Running the Dashboard

### 1. Standalone Web Server (Recommended)
Launch the standalone server:
```bash
python server.py
```
Open your browser and navigate to the application URL:
👉 **`http://localhost:8000`**

> **Important Note on File Uploads**: Always access the dashboard through the server at `http://localhost:8000`. Double-clicking an HTML file locally (`file://`) can block asynchronous browser uploads due to browser security policies.

### 2. Streamlit Mode
Alternatively, launch the application with Streamlit:
```bash
streamlit run app.py
```
Once started, the dashboard will open at `http://localhost:8501`.

---

## 📖 Usage Guide

### Sidebar Controls
1. **Data Source**:
   - Choose one of the built-in demo datasets (**Walmart Weekly Sales**, **Retail Monthly Sales**, **Daily Ecommerce Sales**) or select **Upload Custom CSV / Excel**.
2. **Column Mapping**:
   - Select your dataset's **Date Column** (supports standard timestamps, ISO dates, slashes, or mixed strings).
   - Select your **Sales Column** (automatically cleans currency symbols like `$`, `€`, `£` and commas).
   - *(Optional)* Select a **Group Column** (e.g. `Department`, `Store`, `Category`, `Channel`). You can forecast for **Total (all groups)** or filter by an individual group entity.
3. **Forecast Horizon & Resampling**:
   - Default horizon adjusts based on frequency (Daily: 14, Weekly: 13, Monthly: 3). You can adjust this to any integer value.
   - Choose whether to let the app auto-detect frequency or enforce resampling (`Daily (D)`, `Weekly (W)`, `Monthly (MS)`).
4. **Run Forecast**:
   - Click **Run Forecast & Analysis** (or let the app automatically run when inputs change).

---

## 📊 Models & Evaluation Methodology

### Rolling-Origin Validation
Unlike standard train/test splits that introduce temporal leakage or test on an arbitrary single period, this dashboard uses **expanding-window backtesting**:
- The model trains on historical data $[1 \dots t]$ and forecasts the subsequent window $[t+1 \dots t+H]$.
- The origin rolls forward through time across multiple non-overlapping test folds.
- Metrics are calculated across all backtest periods:
  $$\text{MAE} = \frac{1}{N}\sum |y_t - \hat{y}_t|$$
  $$\text{RMSE} = \sqrt{\frac{1}{N}\sum (y_t - \hat{y}_t)^2}$$
  $$\text{MAPE} = \frac{100\%}{N}\sum \left|\frac{y_t - \hat{y}_t}{y_t}\right| \quad \text{or} \quad \text{sMAPE} = \frac{100\%}{N}\sum \frac{2|y_t - \hat{y}_t|}{|y_t| + |\hat{y}_t|}$$

### Automated Best Model Selection
The model with the lowest RMSE across all validation windows is selected as champion. The dashboard explicitly measures its percentage improvement over the Seasonal Naive benchmark:
$$\text{Improvement \%} = \frac{\text{RMSE}_{\text{Seasonal Naive}} - \text{RMSE}_{\text{Best}}}{\text{RMSE}_{\text{Seasonal Naive}}} \times 100\%$$

---

## 📦 Output Artifacts & Exports
- **Forecast CSV**: Download the future forecast table including `Date`, `Forecast`, `Lower_CI_95`, and `Upper_CI_95`.
- **Metrics Leaderboard CSV**: Download the backtest metrics table for auditing and governance.
