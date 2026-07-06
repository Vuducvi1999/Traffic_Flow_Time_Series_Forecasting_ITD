import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import logging
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from tqdm import tqdm
from prepare_forecasting_dataset import is_vietnam_holiday

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "sarimax_optimization.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("SARIMAX_Optimizer")
warnings.filterwarnings("ignore")

CSV_PATH = ROOT / "data" / "processed" / "traffic_vehicle_forecasting_dataset.csv"
OUT_DIR = ROOT / "data" / "processed" / "models"
PLOT_DIR = OUT_DIR / "plots"
CONFIG_PATH = OUT_DIR / "device_sarimax_config.json"

TARGET_MAE_MIN, TARGET_MAE_MAX = 0, 12
TARGET_RMSE_MIN, TARGET_RMSE_MAX = 0, 20
TARGET_MAPE_MIN, TARGET_MAPE_MAX = 0, 20
TARGET_R2_MIN = 0.5

CONFIG_TIERS = [
    {
        "name": "cfg1_auto_fast",
        "train_length": 500,
        "val_length": 12,
        "use_auto_arima": True,
        "seasonal": False,
        "auto_maxiter": 20,
    },
    {
        "name": "cfg2_auto_seasonal",
        "train_length": 800,
        "val_length": 12,
        "use_auto_arima": True,
        "seasonal": True,
        "auto_maxiter": 20,
    },
    {
        "name": "cfg3_auto_long",
        "train_length": 1500,
        "val_length": 12,
        "use_auto_arima": True,
        "seasonal": True,
        "auto_maxiter": 30,
    },
]

def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def test_stationarity(series: pd.Series, max_d: int = 2) -> int:
    res = adfuller(series.dropna())
    if res[1] < 0.05:
        return 0
    for i in range(1, max_d + 1):
        series = series.diff().dropna()
        res = adfuller(series)
        if res[1] < 0.05:
            return i
    return max_d

def is_met(mae, rmse, mape, r2, target_col: str):
    if target_col == "NumVehicles":
        return 0 <= mae <= 80 and 0 <= rmse <= 100 and 0 <= mape <= 25 and r2 >= 0.0
    else:  # AvgSpeed
        return 0 <= mae <= 12 and 0 <= rmse <= 20 and 0 <= mape <= 20 and r2 >= 0.4

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"devices": {}}

def save_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def try_config(device_id: str, df_device: pd.DataFrame, tier: dict, target_col: str) -> dict:
    df = df_device.set_index("BucketTime").asfreq("5min")

    # NumVehicles: dùng exog cũ (các biến giao thông + thời tiết)
    # AvgSpeed: chỉ dùng các biến liên quan trực tiếp đến tốc độ
    if target_col == "NumVehicles":
        base_traffic_features = [
            "AvgSpeed", "Occupancy", "AvgDensity",
            "Rain", "Temperature", "Humidity", "Visibility", "WindSpeed"
        ]
    else:  # AvgSpeed — chỉ biến liên quan đến tốc độ
        base_traffic_features = [
            "Occupancy",
            "Rain", "Temperature", "Humidity", "Visibility", "WindSpeed"
        ]
    for col in base_traffic_features:
        df[f"{col}_lag1"] = df[col].shift(1)
        
    # Generate dummies for Hour and DayOfWeek
    hour_dummies = pd.get_dummies(df.index.hour, prefix="Hour", drop_first=True, dtype=int)
    hour_dummies.index = df.index
    dow_dummies = pd.get_dummies(df.index.dayofweek, prefix="DoW", drop_first=True, dtype=int)
    dow_dummies.index = df.index
    
    df = pd.concat([df, hour_dummies, dow_dummies], axis=1)
    df = df.bfill().ffill().fillna(0).infer_objects(copy=False)

    exog_cols = [f"{col}_lag1" for col in base_traffic_features] + [
        "IsHoliday"
    ] + hour_dummies.columns.tolist() + dow_dummies.columns.tolist()

    y = df[target_col].values.astype(float)
    X = df[exog_cols].values.astype(float)

    if np.any(np.isnan(y)) or np.any(np.isnan(X)):
        return {"device_id": device_id, "mae": float("inf"), "rmse": float("inf"),
                "mape": float("inf"), "r2": float("-inf"), "order": [1, 0, 0],
                "seasonal_order": [0, 0, 0, 0], "config_name": tier["name"], "success": False}

    n = len(y)
    test_len = 12
    val_len = tier.get("val_length", 12)
    train_len = min(tier.get("train_length", 800), n - test_len - val_len)
    if train_len < 60:
        train_len = 60
    total = train_len + val_len + test_len
    if total > n:
        return {"device_id": device_id, "mae": float("inf"), "rmse": float("inf"),
                "mape": float("inf"), "r2": float("-inf"), "order": [1, 0, 0],
                "seasonal_order": [0, 0, 0, 0], "config_name": tier["name"], "success": False}

    train_end = -(val_len + test_len)
    val_end = -test_len
    train_y = y[:train_end][-train_len:]
    train_X = X[:train_end][-train_len:]
    val_y = y[train_end:val_end]
    val_X = X[train_end:val_end]
    test_y = y[-test_len:]
    test_X = X[-test_len:]

    train_y_s = pd.Series(train_y)
    d = test_stationarity(train_y_s)

    best_order = (1, d, 0)
    best_seasonal = (0, 0, 0, 0)
    best_score = float("inf")

    if tier.get("use_auto_arima"):
        try:
            s = 12 if tier.get("seasonal") else 1
            auto_model = auto_arima(
                train_y, X=train_X,
                start_p=1, max_p=5,
                start_q=0, max_q=3,
                d=d,
                seasonal=tier.get("seasonal", False),
                m=s,
                start_P=0, max_P=2,
                start_Q=0, max_Q=2,
                D=0 if not tier.get("seasonal") else 1,
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                maxiter=tier.get("auto_maxiter", 20),
            )
            best_order = auto_model.order
            best_seasonal = auto_model.seasonal_order if auto_model.seasonal_order else (0, 0, 0, 0)
        except Exception as e:
            logger.warning(f"auto_arima failed for {device_id[:8]}: {e}")
    else:
        candidates = []
        order_candidates = tier.get("order_candidates", [[1, 0, 0]])
        seasonal_candidates = tier.get("seasonal_candidates", [[0, 0, 0, 0]])

        for oc in order_candidates:
            candidates.append((tuple(oc), (0, 0, 0, 0)))
        if tier.get("use_seasonal"):
            for sc in seasonal_candidates:
                if sc[-1] == 0:
                    continue
                for oc in order_candidates[:8]:
                    candidates.append((tuple(oc), tuple(sc)))

        for order, seasonal in candidates:
            try:
                model = sm.tsa.statespace.SARIMAX(
                    train_y, exog=train_X,
                    order=order, seasonal_order=seasonal,
                    trend='c',
                    enforce_stationarity=False, enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=50)
                pred = res.forecast(steps=val_len, exog=val_X).clip(min=0)
                mae_val = float(np.mean(np.abs(val_y - pred)))
                if mae_val < best_score:
                    best_score = mae_val
                    best_order = order
                    best_seasonal = seasonal
            except Exception:
                continue

    try:
        train_val_y = np.concatenate([train_y, val_y])
        train_val_X = np.concatenate([train_X, val_X])
        model = sm.tsa.statespace.SARIMAX(
            train_val_y, exog=train_val_X,
            order=best_order, seasonal_order=best_seasonal,
            trend='c',
            enforce_stationarity=False, enforce_invertibility=False,
        )
        results = model.fit(disp=False, maxiter=30)
        forecast = results.forecast(steps=test_len, exog=test_X).clip(min=0)
        if np.any(np.isnan(forecast)):
            return {"device_id": device_id, "mae": float("inf"), "rmse": float("inf"),
                    "mape": float("inf"), "r2": float("-inf"), "order": list(best_order),
                    "seasonal_order": list(best_seasonal), "config_name": tier["name"], "success": False}

        mae = float(mean_absolute_error(test_y, forecast))
        rmse = float(np.sqrt(mean_squared_error(test_y, forecast)))
        mape = compute_mape(test_y, forecast)
        r2 = float(r2_score(test_y, forecast))

        return {
            "device_id": device_id,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "r2": r2,
            "order": list(best_order),
            "seasonal_order": list(best_seasonal),
            "config_name": tier["name"],
            "success": is_met(mae, rmse, mape, r2, target_col),
        }
    except Exception as e:
        logger.error(f"Training failed for {device_id[:8]} tier {tier['name']}: {e}")
        return {
            "device_id": device_id,
            "mae": float("inf"),
            "rmse": float("inf"),
            "mape": float("inf"),
            "r2": float("-inf"),
            "order": list(best_order),
            "seasonal_order": list(best_seasonal),
            "config_name": tier["name"],
            "success": False,
        }

def save_device_config(device_id, config, result, target_col):
    if device_id not in config["devices"]:
        config["devices"][device_id] = {}
        
    if result["success"]:
        tier_name = result["config_name"]
        config["devices"][device_id].update({
            f"{target_col}_order": result["order"],
            f"{target_col}_seasonal_order": result["seasonal_order"],
            f"{target_col}_train_length": next(
                (t.get("train_length", 800) for t in CONFIG_TIERS if t["name"] == tier_name),
                800
            ),
            f"{target_col}_val_length": next(
                (t.get("val_length", 60) for t in CONFIG_TIERS if t["name"] == tier_name),
                60
            ),
            f"_{target_col}_source": tier_name,
        })
        logger.info(f"Device {device_id[:8]} ({target_col}): ✓ with {tier_name} "
                    f"(MAE={result['mae']:.2f}, RMSE={result['rmse']:.2f}, "
                    f"MAPE={result['mape']:.2f}%, R2={result['r2']:.4f})")
    else:
        config["devices"][device_id].update({
            f"{target_col}_order": result["order"],
            f"{target_col}_seasonal_order": result["seasonal_order"],
            f"_{target_col}_source": f"best_of_{result['config_name']}",
            f"_{target_col}_notes": f"Best effort - MAE={result['mae']:.2f}, R2={result['r2']:.4f}"
        })
        logger.info(f"Device {device_id[:8]} ({target_col}): ✗ best={result['config_name']} "
                    f"(MAE={result['mae']:.2f}, RMSE={result['rmse']:.2f}, "
                    f"MAPE={result['mape']:.2f}%, R2={result['r2']:.4f})")
    save_config(config)

def optimize_one_device(args):
    device_id, df_data, config, target_col = args
    df_device = df_data[df_data["DeviceId"] == device_id].copy()

    cur_cfg = config.get("devices", {}).get(device_id, {})
    order = cur_cfg.get(f"{target_col}_order")
    seasonal = cur_cfg.get(f"{target_col}_seasonal_order")

    if order and seasonal:
        temp_cfg = {"name": "existing_config", "train_length": 800, "val_length": 60}
        result = try_config(device_id, df_device, temp_cfg, target_col)
        if result["success"]:
            return (device_id, result, True, config)

    device_results = []
    for tier in CONFIG_TIERS:
        result = try_config(device_id, df_device, tier, target_col)
        device_results.append(result)
        if result["success"]:
            return (device_id, result, True, config)

    best = min(device_results, key=lambda x: x["mae"])
    return (device_id, best, False, config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs")
    parser.add_argument("--target", type=str, default="NumVehicles", choices=["AvgSpeed", "NumVehicles"],
                        help="Target column (default: NumVehicles)")
    parser.add_argument("--devices", type=str, default=None,
                        help="Comma-separated device IDs to optimize (default: all)")
    args = parser.parse_args()

    target_col = args.target

    logger.info("=" * 80)
    logger.info(f"SARIMAX OPTIMIZATION - Target: {target_col}")
    logger.info("=" * 80)

    df = pd.read_csv(CSV_PATH)
    df['BucketTime'] = pd.to_datetime(df['BucketTime'])
    df = df.sort_values(by=["DeviceId", "Lane", "BucketTime"]).reset_index(drop=True)

    all_devices = sorted(df["DeviceId"].unique())
    config = load_config()
    if "devices" not in config:
        config["devices"] = {}

    if args.devices:
        target_devices = [d.strip() for d in args.devices.split(",")]
    else:
        target_devices = all_devices

    logger.info(f"Optimizing {len(target_devices)}/{len(all_devices)} devices with {args.n_jobs} jobs")

    tasks = [(dev, df, config, target_col) for dev in target_devices]

    results_list = joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(optimize_one_device)(task) for task in tqdm(tasks, desc="Optimizing")
    )

    all_results = []
    optimized_count = 0
    failed_devices = []

    for device_id, result, success, _ in results_list:
        all_results.append(result)
        save_device_config(device_id, config, result, target_col)
        if success:
            optimized_count += 1
        else:
            failed_devices.append(device_id[:8])

    logger.info("=" * 80)
    logger.info(f"Optimization complete for {target_col}!")
    logger.info(f"  Successful: {optimized_count}/{len(target_devices)}")
    if failed_devices:
        logger.info(f"  Still below targets: {failed_devices}")

    results_df = pd.DataFrame(all_results)
    opt_path = OUT_DIR / f"sarimax_optimization_results_{target_col}.csv"
    results_df.to_csv(opt_path, index=False, encoding="utf-8-sig")
    logger.info(f"Results saved to: {opt_path}")

if __name__ == "__main__":
    main()
