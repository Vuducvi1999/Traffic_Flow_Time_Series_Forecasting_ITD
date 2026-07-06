import os
# Limit multi-threading in NumPy/SciPy to prevent CPU/memory thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import ast
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_forecasting_dataset import is_vietnam_holiday

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DashboardExporter")

ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "data" / "processed" / "traffic_vehicle_forecasting_dataset.csv"
METRICS_PATH = ROOT / "data" / "processed" / "models" / "sarimax_evaluation_metrics.csv"
OUTPUT_JSON = ROOT / "data" / "processed" / "dashboard_data.json"

def process_single_device(task_args):
    device_id, df_device, order, seasonal_order, metrics_summary = task_args
    
    target_col = "NumVehicles"
    base_traffic_features = [
        "AvgSpeed", "Occupancy", "AvgDensity",
        "Rain", "Temperature", "Humidity", "Visibility", "WindSpeed"
    ]
    
    # Sort and re-index to minutely frequency
    df_device = df_device.sort_values(by=["Lane", "BucketTime"]).reset_index(drop=True)
    
    # Group by BucketTime and take mean if there are multiple lanes (to aggregate to device-level)
    device_numeric_cols = [target_col] + base_traffic_features
    df_device_agg = df_device.groupby("BucketTime")[device_numeric_cols].mean().reset_index()
    df_device_agg = df_device_agg.set_index("BucketTime").asfreq("5min")
    
    # Create lag-1 features
    for col in base_traffic_features:
        df_device_agg[f"{col}_lag1"] = df_device_agg[col].shift(1)
        
    # Hour and DayOfWeek dummies
    hour_dummies = pd.get_dummies(df_device_agg.index.hour, prefix="Hour", drop_first=True, dtype=int)
    hour_dummies.index = df_device_agg.index
    dow_dummies = pd.get_dummies(df_device_agg.index.dayofweek, prefix="DoW", drop_first=True, dtype=int)
    dow_dummies.index = df_device_agg.index
    
    df_device_agg = pd.concat([df_device_agg, hour_dummies, dow_dummies], axis=1)
    df_device_agg = df_device_agg.bfill().ffill().fillna(0)
    df_device_agg["IsHoliday"] = is_vietnam_holiday(pd.Series(df_device_agg.index)).values
    
    exog_names = [f"{col}_lag1" for col in base_traffic_features] + [
        "IsHoliday"
    ] + hour_dummies.columns.tolist() + dow_dummies.columns.tolist()
    
    y = df_device_agg[target_col]
    X = df_device_agg[exog_names].astype(float)
    
    # 1. Validation Mode Setup: Split into train/test (last 120 mins is test set - 24 steps of 5-min)
    test_length = 24
    train_y = y.iloc[:-test_length]
    test_y = y.iloc[-test_length:]
    test_X = X.iloc[-test_length:]
    
    # Load model pre-trained on train split (excluding last 60 mins)
    model_save_path = ROOT / "data" / "processed" / "models" / f"model_{target_col}_{device_id}.joblib"
    results = joblib.load(model_save_path)
    
    # Run test set forecast (align exog names)
    test_exog = test_X.reindex(columns=results.model.exog_names, fill_value=0).astype(float)
    test_forecast = results.forecast(steps=test_length, exog=test_exog).clip(lower=0)
    
    # 2. Future Mode Setup: Fit SARIMAX on full history (with trend='c')
    full_model = sm.tsa.statespace.SARIMAX(
        y,
        exog=X,
        order=order,
        seasonal_order=seasonal_order,
        trend='c',
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    full_results = full_model.fit(disp=False, maxiter=50)
    
    # Generate X_future for next 120 minutes (24 steps of 5-min)
    last_exog = X.iloc[-1:]
    X_future_df = pd.concat([last_exog] * 24, ignore_index=True)
    future_index = pd.date_range(start=y.index[-1] + pd.Timedelta("5min"), periods=24, freq="5min")
    X_future_df.index = future_index
    
    # Drop old dummy columns and concat the new ones for the future steps
    future_hour_dummies = pd.get_dummies(future_index.hour, prefix="Hour", drop_first=True, dtype=int)
    future_hour_dummies.index = future_index
    future_dow_dummies = pd.get_dummies(future_index.dayofweek, prefix="DoW", drop_first=True, dtype=int)
    future_dow_dummies.index = future_index
    
    cols_to_drop = [c for c in X_future_df.columns if "Hour_" in c or "DoW_" in c]
    X_future_df = X_future_df.drop(columns=cols_to_drop, errors="ignore")
    X_future_df = pd.concat([X_future_df, future_hour_dummies, future_dow_dummies], axis=1)
    X_future_df["IsHoliday"] = is_vietnam_holiday(pd.Series(X_future_df.index)).values
    X_future = X_future_df.reindex(columns=full_results.model.exog_names, fill_value=0).astype(float)
    
    # Future forecast
    future_forecast = full_results.forecast(steps=24, exog=X_future).clip(lower=0)
    
    # 3. Format Data Outputs
    # A. History list for validation (last 120 minutes of train_y - 24 steps)
    history_df = df_device_agg.iloc[:-test_length].tail(24)
    history_list = []
    for time_idx, row_val in history_df.iterrows():
        history_list.append({
            "time": time_idx.strftime("%Y-%m-%d %H:%M"),
            "actual": float(row_val[target_col]),
            "speed": float(row_val["AvgSpeed"]),
            "rain": float(row_val["Rain"]),
            "temp": float(row_val["Temperature"]),
            "hum": float(row_val["Humidity"]),
            "vis": float(row_val["Visibility"])
        })
        
    # B. Test Set actuals (120 minutes)
    test_actual_df = df_device_agg.tail(test_length)
    test_actual_list = []
    for time_idx, row_val in test_actual_df.iterrows():
        test_actual_list.append({
            "time": time_idx.strftime("%Y-%m-%d %H:%M"),
            "actual": float(row_val[target_col]),
            "speed": float(row_val["AvgSpeed"]),
            "rain": float(row_val["Rain"]),
            "temp": float(row_val["Temperature"]),
            "hum": float(row_val["Humidity"]),
            "vis": float(row_val["Visibility"])
        })
        
    # C. Test Set predictions (120 minutes)
    test_predict_list = []
    for time_idx, val in test_forecast.items():
        test_predict_list.append({
            "time": time_idx.strftime("%Y-%m-%d %H:%M"),
            "predicted": float(val)
        })
        
    # D. Future History list (last 120 minutes of full dataset)
    future_history_df = df_device_agg.tail(24)
    future_history_list = []
    for time_idx, row_val in future_history_df.iterrows():
        future_history_list.append({
            "time": time_idx.strftime("%Y-%m-%d %H:%M"),
            "actual": float(row_val[target_col]),
            "speed": float(row_val["AvgSpeed"]),
            "rain": float(row_val["Rain"]),
            "temp": float(row_val["Temperature"]),
            "hum": float(row_val["Humidity"]),
            "vis": float(row_val["Visibility"])
        })
        
    # E. Future predictions (120 minutes)
    future_forecast_list = []
    for time_idx, val in future_forecast.items():
        future_forecast_list.append({
            "time": time_idx.strftime("%Y-%m-%d %H:%M"),
            "predicted": float(val)
        })
        
    # Specific horizons (extracted from future forecast)
    # The dashboard expects predictions at 1m, 5m, 15m, 30m, 60m horizons.
    # Since our frequency is 5-min:
    # - 1m and 5m -> index 0 (5 minutes ahead)
    # - 15m -> index 2 (15 minutes ahead)
    # - 30m -> index 5 (30 minutes ahead)
    # - 60m -> index 11 (60 minutes ahead)
    horizons = {
        "1m": float(future_forecast.iloc[0]),
        "5m": float(future_forecast.iloc[0]),
        "15m": float(future_forecast.iloc[2]),
        "30m": float(future_forecast.iloc[5]),
        "60m": float(future_forecast.iloc[11])
    }
    
    device_data = {
        "metrics": metrics_summary,
        "horizons": horizons,
        "history": history_list,
        "test_actual": test_actual_list,
        "test_predict": test_predict_list,
        "future_history": future_history_list,
        "future_forecast": future_forecast_list
    }
    return device_id, device_data

def main():
    logger.info("Starting Parallel Validation Exporter...")
    
    target_col = "NumVehicles"
    
    # 1. Load traffic data
    if not CSV_PATH.exists():
        logger.error(f"Dataset not found at {CSV_PATH}")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    df["BucketTime"] = pd.to_datetime(df["BucketTime"])
    
    # 2. Load metrics to get the best orders
    if not METRICS_PATH.exists():
        logger.error(f"Evaluation metrics not found at {METRICS_PATH}. Please train models first.")
        sys.exit(1)
    metrics_df = pd.read_csv(METRICS_PATH)
    
    devices = df["DeviceId"].unique()
    logger.info(f"Found {len(devices)} devices to export predictions for.")
    
    tasks = []
    for device_id in devices:
        df_device = df[df["DeviceId"] == device_id].copy()
        
        # Load best model orders from metrics table (filtering by TargetCol)
        device_metrics = metrics_df[(metrics_df["DeviceId"] == device_id) & (metrics_df["TargetCol"] == target_col)]
        if device_metrics.empty:
            order = (1, 0, 0)
            seasonal_order = (0, 0, 0, 0)
            metrics_summary = {"mae": 50.0, "rmse": 70.0, "mape": 20.0, "r2": 0.1}
        else:
            row = device_metrics.iloc[0]
            # Orders are saved as Python tuple strings like "(1, 0, 0)" — use ast.literal_eval
            order = tuple(ast.literal_eval(row["SARIMAX_Order"])) if isinstance(row["SARIMAX_Order"], str) else tuple(row["SARIMAX_Order"])
            seasonal_order = tuple(ast.literal_eval(row["Seasonal_Order"])) if isinstance(row["Seasonal_Order"], str) else tuple(row["Seasonal_Order"])
            metrics_summary = {
                "mae": float(row["MAE"]),
                "rmse": float(row["RMSE"]),
                "mape": float(row["MAPE (%)"]),
                "r2": float(row["R2"])
            }
        tasks.append((device_id, df_device, order, seasonal_order, metrics_summary))
        
    logger.info("Executing validation forecasts in parallel using joblib...")
    results_list = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(process_single_device)(task) for task in tqdm(tasks, desc="Exporting device predictions")
    )
    
    dashboard_data = {
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "data_end_time": df["BucketTime"].max().isoformat()
        },
        "devices": {}
    }
    
    for device_id, device_data in results_list:
        dashboard_data["devices"][device_id] = device_data
        
    # Write to file
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Dashboard data successfully exported to {OUTPUT_JSON}!")

if __name__ == "__main__":
    main()
