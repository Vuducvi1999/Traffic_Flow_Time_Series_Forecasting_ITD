import os
# Limit multi-threading in NumPy/SciPy to prevent CPU/memory thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import logging
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import joblib
from prepare_forecasting_dataset import is_vietnam_holiday

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("OnlineEngine")

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "processed" / "traffic_vehicle_forecasting_dataset.csv"
METRICS_PATH = ROOT / "data" / "processed" / "models" / "sarimax_evaluation_metrics.csv"
OUTPUT_JSON = ROOT / "data" / "processed" / "dashboard_data.json"

def main():
    logger.info("Initializing Online Forecasting Engine...")
    
    # 1. Load data
    if not CSV_PATH.exists():
        logger.error(f"Dataset not found at {CSV_PATH}")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    df["BucketTime"] = pd.to_datetime(df["BucketTime"])
    
    # 2. Load metrics
    if not METRICS_PATH.exists():
        logger.error(f"Metrics not found at {METRICS_PATH}")
        sys.exit(1)
    metrics_df = pd.read_csv(METRICS_PATH)
    
    devices = df["DeviceId"].unique()
    
    target_col = "NumVehicles"
    base_traffic_features = [
        "AvgSpeed", "Occupancy", "AvgDensity",
        "Rain", "Temperature", "Humidity", "Visibility", "WindSpeed"
    ]
    
    # Pre-process all devices and load models into memory
    active_models = {}
    device_data_grids = {}
    active_histories = {}
    metrics_summaries = {}
    
    logger.info("Loading pre-trained models into memory...")
    for device_id in devices:
        df_device = df[df["DeviceId"] == device_id].copy()
        df_device = df_device.sort_values(by=["Lane", "BucketTime"]).reset_index(drop=True)
        
        device_numeric_cols = [target_col] + base_traffic_features
        df_device_agg = df_device.groupby("BucketTime")[device_numeric_cols].mean().reset_index()
        df_device_agg = df_device_agg.set_index("BucketTime").asfreq("5min")
        
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
        
        device_data_grids[device_id] = df_device_agg
        
        # Load pre-trained model (trained on history, excluding last 60 minutes)
        model_path = ROOT / "data" / "processed" / "models" / f"model_{target_col}_{device_id}.joblib"
        if not model_path.exists():
            logger.error(f"Model file not found for device {device_id}")
            sys.exit(1)
        active_models[device_id] = joblib.load(model_path)
        
        # Load metrics summary (filtering by TargetCol)
        device_metrics = metrics_df[(metrics_df["DeviceId"] == device_id) & (metrics_df["TargetCol"] == target_col)]
        if device_metrics.empty:
            metrics_summaries[device_id] = {"mae": 50.0, "rmse": 70.0, "mape": 20.0, "r2": 0.1}
        else:
            row = device_metrics.iloc[0]
            metrics_summaries[device_id] = {
                "mae": float(row["MAE"]),
                "rmse": float(row["RMSE"]),
                "mape": float(row["MAPE (%)"]),
                "r2": float(row["R2"])
            }
            
        # Initial sliding history (last 120 minutes of the training set - 24 steps of 5-min)
        active_histories[device_id] = df_device_agg.iloc[:-12].tail(24)
        
    # Get the time index for the last 60 minutes (12 steps of 5-min)
    any_device = devices[0]
    stream_times = device_data_grids[any_device].index[-12:]
    
    logger.info(f"Stream simulation ready: 60 minutes of data from {stream_times[0]} to {stream_times[-1]}")
    logger.info("Starting simulation loop. Refresh your dashboard to watch it update live!")
    
    buffer = []
    
    for tick, current_time in enumerate(stream_times):
        logger.info(f"--- Step {tick+1}/12 | Timestamp: {current_time.strftime('%Y-%m-%d %H:%M')} ---")
        
        dashboard_data = {
            "metadata": {
                "generated_at": pd.Timestamp.now().isoformat(),
                "data_end_time": current_time.isoformat(),
                "simulation_tick": tick + 1
            },
            "devices": {}
        }
        
        # Process each VDS device
        for device_id in devices:
            grid = device_data_grids[device_id]
            row = grid.loc[current_time]
            model = active_models[device_id]
            exog_names = model.model.exog_names
            
            new_y = pd.Series([row[target_col]], index=[current_time])
            new_X_df = pd.DataFrame([row], index=[current_time])
            new_X = new_X_df.reindex(columns=exog_names, fill_value=0).astype(float)
            
            # 1. Minutely Step: Extend incrementally — append state, chỉ 1 dòng mới mỗi tick
            extended_results = model.extend(endog=new_y, exog=new_X)
            active_models[device_id] = extended_results
            
            future_index = pd.date_range(start=current_time + pd.Timedelta("5min"), periods=24, freq="5min")
            X_future_list = []
            for f_t in future_index:
                if f_t in grid.index:
                    X_future_list.append(grid.loc[f_t])
                else:
                    # If we go out of bounds of the dataset, persist the last row
                    X_future_list.append(grid.iloc[-1])
            X_future_df = pd.DataFrame(X_future_list, index=future_index)
            
            # Drop old dummy columns and concat the new ones for the future steps
            future_hour_dummies = pd.get_dummies(future_index.hour, prefix="Hour", drop_first=True, dtype=int)
            future_hour_dummies.index = future_index
            future_dow_dummies = pd.get_dummies(future_index.dayofweek, prefix="DoW", drop_first=True, dtype=int)
            future_dow_dummies.index = future_index
            
            cols_to_drop = [c for c in X_future_df.columns if "Hour_" in c or "DoW_" in c]
            X_future_df = X_future_df.drop(columns=cols_to_drop, errors="ignore")
            X_future_df = pd.concat([X_future_df, future_hour_dummies, future_dow_dummies], axis=1)
            
            X_future_df["IsHoliday"] = is_vietnam_holiday(pd.Series(X_future_df.index)).values
            X_future = X_future_df.reindex(columns=exog_names, fill_value=0).astype(float)
            
            # Forecast next 120 minutes (24 steps of 5-min)
            forecast = extended_results.forecast(steps=24, exog=X_future).clip(lower=0)
            
            # 2. Update sliding history (remove oldest step, add current step)
            hist_df = active_histories[device_id]
            hist_df = pd.concat([hist_df, pd.DataFrame([row], index=[current_time])])
            hist_df = hist_df.tail(24)  # Maintain last 120 minutes window (24 steps)
            active_histories[device_id] = hist_df
            
            # Format history list for charting
            history_list = []
            for time_idx, row_val in hist_df.iterrows():
                history_list.append({
                    "time": time_idx.strftime("%Y-%m-%d %H:%M"),
                    "actual": float(row_val[target_col]),
                    "speed": float(row_val["AvgSpeed"]),
                    "rain": float(row_val["Rain"]),
                    "temp": float(row_val["Temperature"]),
                    "hum": float(row_val["Humidity"]),
                    "vis": float(row_val["Visibility"])
                })
                
            forecast_list = []
            for f_t, val in zip(future_index, forecast):
                forecast_list.append({
                    "time": f_t.strftime("%Y-%m-%d %H:%M"),
                    "predicted": float(val)
                })
                
            # Specific horizons
            horizons = {
                "5m": float(forecast.iloc[0]),
                "15m": float(forecast.iloc[2]),
                "30m": float(forecast.iloc[5]),
                "60m": float(forecast.iloc[11])
            }
            
            # For the dashboard visual flow:
            # We map the sliding history to both 'history' and 'future_history'
            # We map the forecast to both 'test_predict' and 'future_forecast'
            # We map the forecast window actuals (from grid) to 'test_actual' to support dynamic validation mode
            test_actual_list = []
            for f_t in future_index:
                if f_t in grid.index:
                    row_val = grid.loc[f_t]
                    test_actual_list.append({
                        "time": f_t.strftime("%Y-%m-%d %H:%M"),
                        "actual": float(row_val[target_col]),
                        "speed": float(row_val["AvgSpeed"]),
                        "rain": float(row_val["Rain"]),
                        "temp": float(row_val["Temperature"]),
                        "hum": float(row_val["Humidity"]),
                        "vis": float(row_val["Visibility"])
                    })
                else:
                    # Pad if future extends beyond dataset
                    row_val = grid.iloc[-1]
                    test_actual_list.append({
                        "time": f_t.strftime("%Y-%m-%d %H:%M"),
                        "actual": float(row_val[target_col]),
                        "speed": float(row_val["AvgSpeed"]),
                        "rain": float(row_val["Rain"]),
                        "temp": float(row_val["Temperature"]),
                        "hum": float(row_val["Humidity"]),
                        "vis": float(row_val["Visibility"])
                    })
            
            dashboard_data["devices"][device_id] = {
                "metrics": metrics_summaries[device_id],
                "horizons": horizons,
                "history": history_list,
                "test_actual": test_actual_list,
                "test_predict": forecast_list,
                "future_history": history_list,
                "future_forecast": forecast_list
            }
            
        # Write to JSON file to update the dashboard live
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
            
        # Buffer the new records for the hourly sync simulation
        buffer.append(current_time)
        
        # 3. Hourly Step: Sync buffer to persistent database (simulated every 10 ticks for demo)
        if len(buffer) >= 10:
            logger.info(f"[HOURLY SYNC] Syncing last {len(buffer)} minutely records to historical CSV storage...")
            buffer.clear()
            
        # Sleep to simulate time progression (2 seconds = 1 minute)
        time.sleep(2.0)
        
    # 4. Daily Step: Retrain models (simulated at the end of the stream)
    logger.info("[DAILY RETRAIN] Daily trigger (2:00 AM) fired! Re-running train_sarimax.py to refit coefficients on the new data...")
    logger.info("Simulation completed successfully!")

if __name__ == "__main__":
    main()
