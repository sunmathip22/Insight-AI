"""
End-to-end unit and integration test for the sales dashboard pipeline,
evaluating Walmart Weekly Sales, Retail Monthly Sales, and Daily Ecommerce Sales.
"""
import pandas as pd
from utils.data_prep import load_raw_dataset, process_sales_data
from utils.evaluation import rolling_origin_validation, fit_best_future_forecast
from utils.insights import generate_business_insights

def test_dataset(filepath, date_col, sales_col, group_col=None, horizon=None):
    print(f"\n================ Testing {filepath} ================")
    df = load_raw_dataset(filepath)
    print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
    
    # Run data prep
    prep = process_sales_data(
        df=df,
        date_col=date_col,
        sales_col=sales_col,
        group_col=group_col,
        selected_group="Total (all groups)" if group_col else None,
    )
    print(f"Detected Label: {prep['detected_label']}, Seasonal Period: {prep['seasonal_period']}, Sufficient Cycles: {prep['has_sufficient_cycles']}")
    print(f"Clean TS Length: {len(prep['ts_df'])}, Outliers: {prep['outlier_count']}")
    
    h = horizon or prep['default_horizon']
    print(f"Using Horizon: {h}")
    
    # Run evaluation
    eval_res = rolling_origin_validation(
        ts_series=prep['ts_df']['Sales'],
        horizon=h,
        seasonal_period=prep['seasonal_period']
    )
    print("Metrics Table:")
    print(eval_res['metrics_df'][['Model', 'MAE', 'RMSE', 'MAPE (%)', 'Status']])
    print(f"Best Model: {eval_res['best_model_name']} with RMSE: {eval_res['best_rmse']} (% improvement: {eval_res['improvement_pct']}%)")
    
    # Run future forecast
    future = fit_best_future_forecast(
        ts_series=prep['ts_df']['Sales'],
        best_model_name=eval_res['best_model_name'],
        horizon=h,
        seasonal_period=prep['seasonal_period'],
        pandas_freq=prep['pandas_freq']
    )
    print("Future Forecast Preview:")
    print(future['forecast_df'].head(3))
    
    # Generate insights
    ins = generate_business_insights(
        ts_df=prep['ts_df'],
        eval_results=eval_res,
        prep_data=prep,
        future_forecast_df=future['forecast_df'],
        horizon=h
    )
    print("Insights Peak Summary:")
    print(ins['peak_analysis'][:150].encode('ascii', errors='replace').decode('ascii') + "...")
    print(">> TEST PASSED!")

if __name__ == "__main__":
    # Test 1: Walmart Weekly Sales
    test_dataset("data/walmart_weekly_sales.csv", date_col="Date", sales_col="Weekly_Sales", group_col="Department", horizon=13)
    
    # Test 2: Retail Monthly Sales
    test_dataset("data/retail_monthly_sales.csv", date_col="Date", sales_col="Revenue", group_col="Category", horizon=3)
    
    # Test 3: Daily Ecommerce Sales
    test_dataset("data/daily_ecommerce_sales.csv", date_col="Order_Date", sales_col="Sales_Amount", group_col="Channel", horizon=14)
    print("\n\n>>> ALL 3 DATASETS PASSED WITH 100% SUCCESS! <<<")
