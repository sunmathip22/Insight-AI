"""
Script to generate sample datasets for the Sales Forecasting Dashboard:
1. Walmart Weekly Sales (Store, Dept, Date, Weekly_Sales, IsHoliday)
2. Retail Monthly Sales (Category, Date, Sales)
3. Daily Ecommerce Sales (Channel, Date, Sales)
"""
import os
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

# 1. Walmart Weekly Sales (~3 years of weekly data, multiple departments)
np.random.seed(42)
dates_weekly = pd.date_range(start="2021-01-08", end="2023-12-29", freq="W-FRI")
n_weeks = len(dates_weekly)

walmart_records = []
depts = ["Electronics", "Grocery", "Apparel", "Home & Garden"]
for dept in depts:
    base_level = {"Electronics": 25000, "Grocery": 50000, "Apparel": 18000, "Home & Garden": 15000}[dept]
    trend = np.linspace(0, base_level * 0.15, n_weeks)
    # Seasonal wave (peaks in Q4 holiday season: Nov-Dec)
    week_of_year = dates_weekly.isocalendar().week.values
    holiday_boost = np.where((week_of_year >= 46) & (week_of_year <= 52), base_level * 0.45, 0)
    summer_boost = np.where((week_of_year >= 24) & (week_of_year <= 32), base_level * 0.12, 0)
    noise = np.random.normal(0, base_level * 0.05, n_weeks)
    
    sales = base_level + trend + holiday_boost + summer_boost + noise
    sales = np.maximum(sales, 1000).round(2)
    
    for d, s, w in zip(dates_weekly, sales, week_of_year):
        is_holiday = bool(w in [6, 36, 47, 51])
        walmart_records.append({
            "Store": 1,
            "Department": dept,
            "Date": d.strftime("%Y-%m-%d"),
            "Weekly_Sales": s,
            "IsHoliday": is_holiday
        })

df_walmart = pd.DataFrame(walmart_records)
df_walmart.to_csv("data/walmart_weekly_sales.csv", index=False)
print(f"Generated Walmart Weekly Sales: {len(df_walmart)} rows -> data/walmart_weekly_sales.csv")

# 2. Retail Monthly Sales (4 years of monthly data across 3 regions)
dates_monthly = pd.date_range(start="2020-01-01", end="2023-12-01", freq="MS")
n_months = len(dates_monthly)
monthly_records = []
categories = ["Consumer Goods", "Industrial Supplies", "Software & Services"]

for cat in categories:
    base = {"Consumer Goods": 85000, "Industrial Supplies": 120000, "Software & Services": 60000}[cat]
    trend = np.linspace(0, base * 0.35, n_months)
    month_num = dates_monthly.month.values
    seasonality = base * 0.20 * np.sin(2 * np.pi * month_num / 12) + (month_num == 12) * (base * 0.25)
    noise = np.random.normal(0, base * 0.04, n_months)
    sales = (base + trend + seasonality + noise).round(2)
    sales = np.maximum(sales, 5000)
    
    for d, s in zip(dates_monthly, sales):
        monthly_records.append({
            "Category": cat,
            "Date": d.strftime("%Y-%m-%d"),
            "Revenue": s
        })

df_monthly = pd.DataFrame(monthly_records)
df_monthly.to_csv("data/retail_monthly_sales.csv", index=False)
print(f"Generated Retail Monthly Sales: {len(df_monthly)} rows -> data/retail_monthly_sales.csv")

# 3. Daily Ecommerce Sales (1.5 years daily data)
dates_daily = pd.date_range(start="2022-06-01", end="2023-12-31", freq="D")
n_days = len(dates_daily)
daily_records = []
channels = ["Organic Search", "Paid Social", "Direct"]

for ch in channels:
    base = {"Organic Search": 4500, "Paid Social": 3200, "Direct": 2100}[ch]
    trend = np.linspace(0, base * 0.25, n_days)
    dow = dates_daily.dayofweek.values  # 0=Mon, 6=Sun
    # Weekend surge
    dow_effect = np.where(dow >= 5, base * 0.3, -base * 0.05)
    # Month end effect
    day = dates_daily.day.values
    month_end_effect = np.where(day >= 27, base * 0.15, 0)
    noise = np.random.normal(0, base * 0.08, n_days)
    sales = (base + trend + dow_effect + month_end_effect + noise).round(2)
    sales = np.maximum(sales, 200)
    
    for d, s in zip(dates_daily, sales):
        daily_records.append({
            "Channel": ch,
            "Order_Date": d.strftime("%Y-%m-%d"),
            "Sales_Amount": s
        })

df_daily = pd.DataFrame(daily_records)
df_daily.to_csv("data/daily_ecommerce_sales.csv", index=False)
print(f"Generated Daily Ecommerce Sales: {len(df_daily)} rows -> data/daily_ecommerce_sales.csv")
