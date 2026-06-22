import os
import json
import pandas as pd
import numpy as np
from prophet import Prophet
import warnings

# Suppress Prophet warnings and logs to keep output clean
warnings.filterwarnings("ignore")
import logging
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sales_history.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast_results.json")

def load_data() -> pd.DataFrame:
    """Loads and formats the sales history dataset."""
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Calculates MAE and MAPE (handles division by zero)."""
    mae = float(np.mean(np.abs(actual - predicted)))
    # Avoid dividing by zero if actual is 0
    actual_denom = np.where(actual == 0, 1.0, actual)
    mape = float(np.mean(np.abs(actual - predicted) / actual_denom) * 100)
    return mae, mape

def run_forecasts():
    print("Loading sales data...")
    df = load_data()
    
    # Identify unique SKUs and dates
    skus = df["sku"].unique()
    all_dates = sorted(df["date"].unique())
    num_weeks = len(all_dates)
    
    print(f"Dataset contains {len(skus)} SKUs and {num_weeks} weeks of sales data per SKU.")
    
    # The last 4 weeks are held out as the test set
    cutoff_date = all_dates[-4] # 2026-05-18
    train_dates = all_dates[:-4]
    test_dates = all_dates[-4:]
    
    print(f"Training set: {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')} ({len(train_dates)} weeks)")
    print(f"Test set:     {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')} ({len(test_dates)} weeks)")
    
    sku_results = {}
    
    overall_prophet_mae = []
    overall_prophet_mape = []
    overall_naive_mae = []
    overall_naive_mape = []
    overall_snaive_mae = []
    overall_snaive_mape = []
    
    for i, sku in enumerate(skus):
        # Extract SKU data
        sku_df = df[df["sku"] == sku].sort_values("date").copy()
        
        # Split train and test
        train_df = sku_df[sku_df["date"] < cutoff_date].copy()
        test_df = sku_df[sku_df["date"] >= cutoff_date].copy()
        
        # Prepare data for Prophet
        # Fields must be 'ds' and 'y'
        prophet_train = train_df[["date", "units_sold", "promo_flag"]].rename(
            columns={"date": "ds", "units_sold": "y"}
        )
        
        # --- 1. Prophet Model Training ---
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.95
        )
        # Add promo_flag as an additional regressor
        model.add_regressor("promo_flag")
        
        # Fit model
        model.fit(prophet_train)
        
        # Predict for future periods
        # Prophet make_future_dataframe doesn't copy regressors, so we merge them back
        future = model.make_future_dataframe(periods=4, freq="W-MON")
        
        # Merge promo_flag from the original dataframe to get future promo flags
        promo_lookup = sku_df[["date", "promo_flag"]].rename(columns={"date": "ds"})
        future = future.merge(promo_lookup, on="ds", how="left")
        
        forecast = model.predict(future)
        
        # Extract the predictions for the test dates
        test_forecast = forecast[forecast["ds"] >= cutoff_date]
        prophet_preds = test_forecast["yhat"].values
        # Ensure predictions are non-negative integers
        prophet_preds = np.clip(np.round(prophet_preds), 0, None)
        
        # --- 2. Baseline Model 1: Naive (Last value repeat) ---
        last_val = train_df["units_sold"].values[-1]
        naive_preds = np.full(4, last_val)
        
        # --- 3. Baseline Model 2: Seasonal Naive (Value from 52 weeks ago) ---
        snaive_preds = []
        for d in test_df["date"]:
            snaive_date = d - pd.DateOffset(weeks=52)
            past_val_df = sku_df[sku_df["date"] == snaive_date]
            if not past_val_df.empty:
                snaive_preds.append(past_val_df["units_sold"].values[0])
            else:
                # Fallback to naive if 52 weeks ago is not in dataset
                snaive_preds.append(last_val)
        snaive_preds = np.array(snaive_preds)
        
        # Actual values
        actuals = test_df["units_sold"].values
        
        # Calculate error metrics
        p_mae, p_mape = calculate_metrics(actuals, prophet_preds)
        n_mae, n_mape = calculate_metrics(actuals, naive_preds)
        s_mae, s_mape = calculate_metrics(actuals, snaive_preds)
        
        overall_prophet_mae.append(p_mae)
        overall_prophet_mape.append(p_mape)
        overall_naive_mae.append(n_mae)
        overall_naive_mape.append(n_mape)
        overall_snaive_mae.append(s_mae)
        overall_snaive_mape.append(s_mape)
        
        # Save historical vs forecast dates for plotting later in Streamlit
        historical_history = train_df.tail(12) # last 12 weeks of training data for context
        sku_results[sku] = {
            "historical_dates": [d.strftime("%Y-%m-%d") for d in historical_history["date"]],
            "historical_sales": [int(v) for v in historical_history["units_sold"]],
            "test_dates": [d.strftime("%Y-%m-%d") for d in test_df["date"]],
            "test_sales": [int(v) for v in actuals],
            "prophet_forecast": [int(v) for v in prophet_preds],
            "naive_forecast": [int(v) for v in naive_preds],
            "snaive_forecast": [int(v) for v in snaive_preds],
            "metrics": {
                "prophet": {"mae": p_mae, "mape": p_mape},
                "naive": {"mae": n_mae, "mape": n_mape},
                "snaive": {"mae": s_mae, "mape": s_mape}
            }
        }
        
        if (i + 1) % 10 == 0 or (i + 1) == len(skus):
            print(f"Processed forecasting for {i + 1}/{len(skus)} SKUs...")
            
    # Calculate overall metrics
    overall_metrics = {
        "prophet": {
            "mae": float(np.mean(overall_prophet_mae)),
            "mape": float(np.mean(overall_prophet_mape))
        },
        "naive": {
            "mae": float(np.mean(overall_naive_mae)),
            "mape": float(np.mean(overall_naive_mape))
        },
        "snaive": {
            "mae": float(np.mean(overall_snaive_mae)),
            "mape": float(np.mean(overall_snaive_mape))
        }
    }
    
    # Save results structure
    output_data = {
        "overall_metrics": overall_metrics,
        "sku_results": sku_results
    }
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
        
    print("\n=== Forecasting Run Complete! ===")
    print("\nOverall Performance Comparison (Averaged across 30 SKUs):")
    print(f"Model             | Mean Absolute Error (MAE) | Mean Absolute Percentage Error (MAPE)")
    print(f"------------------|---------------------------|--------------------------------------")
    print(f"Prophet (W/Promo) | {overall_metrics['prophet']['mae']:25.2f} | {overall_metrics['prophet']['mape']:34.2f}%")
    print(f"Naive (Last-Val)  | {overall_metrics['naive']['mae']:25.2f} | {overall_metrics['naive']['mape']:34.2f}%")
    print(f"Seasonal Naive    | {overall_metrics['snaive']['mae']:25.2f} | {overall_metrics['snaive']['mape']:34.2f}%")
    
    better_than_naive = overall_metrics['prophet']['mae'] < overall_metrics['naive']['mae']
    better_than_snaive = overall_metrics['prophet']['mae'] < overall_metrics['snaive']['mae']
    print(f"\nEvaluation Conclusion:")
    if better_than_naive and better_than_snaive:
        print("Success! The Prophet model beats both naive baselines.")
    else:
        print("Notice: The baselines performed very competitively compared to Prophet.")

if __name__ == "__main__":
    run_forecasts()
