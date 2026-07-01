import os
# Limit multi-threading in NumPy/SciPy to prevent memory/CPU thrashing during parallel execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server/command-line execution
import matplotlib.pyplot as plt

# Time-series libraries
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.stats as stats
import joblib
from tqdm import tqdm
from prepare_forecasting_dataset import is_vietnam_holiday

# Define Paths
ROOT = Path(__file__).resolve().parent.parent

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "sarimax_training.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("SARIMAX_Pipeline")

# Suppress Convergence and User Warnings from statsmodels/pmdarima
warnings.filterwarnings("ignore")

CSV_PATH = ROOT / "data" / "processed" / "traffic_vehicle_forecasting_dataset.csv"
OUT_DIR = ROOT / "data" / "processed" / "models"
PLOT_DIR = OUT_DIR / "plots"
CONFIG_PATH = OUT_DIR / "device_sarimax_config.json"
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


def load_device_config() -> dict:
    """Load per-device SARIMAX config from JSON file."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"default": {}, "devices": {}}


def parse_args():
    parser = argparse.ArgumentParser(description="SARIMAX Traffic Volume Forecasting Pipeline")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode (fast execution: only 2 devices, last 300 rows each)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs to run. Default is -1 (all available CPUs)"
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Skip grid search; only use manual config from device_sarimax_config.json"
    )
    parser.add_argument(
        "--seasonal-search",
        action="store_true",
        help="Include seasonal (P,D,Q,s) candidates in grid search (slower but may improve accuracy)"
    )
    return parser.parse_args()


def load_and_preprocess_data(path: Path) -> pd.DataFrame:
    logger.info(f"Step 1 & 2: Loading dataset from {path}...")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    df['BucketTime'] = pd.to_datetime(df['BucketTime'])
    
    logger.info(f"Step 3: Sorting data by DeviceId, Lane, and BucketTime...")
    df = df.sort_values(by=["DeviceId", "Lane", "BucketTime"]).reset_index(drop=True)
    return df


def test_stationarity(series: pd.Series, max_d: int = 2) -> Tuple[int, float, float]:
    """
    Perform Augmented Dickey-Fuller (ADF) test.
    Returns:
        - d: order of differencing needed (0, 1, or 2)
        - p_before: p-value of the original series
        - p_after: p-value of the series after d-differencing
    """
    res_before = adfuller(series.dropna())
    p_before = res_before[1]
    
    if p_before < 0.05:
        return 0, p_before, p_before
    
    # Try differencing
    d = 0
    curr_series = series.copy()
    p_after = p_before
    
    for i in range(1, max_d + 1):
        curr_series = curr_series.diff().dropna()
        res_after = adfuller(curr_series)
        p_after = res_after[1]
        d = i
        if p_after < 0.05:
            break
            
    return d, p_before, p_after


def select_best_orders(
    y: pd.Series, exog: pd.DataFrame, s: int,
    config: dict = None,
    seasonal_search: bool = False
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], str]:
    """
    Perform grid search to find the optimal order (p,d,q) and seasonal order (P,D,Q,s)
    on a validation split of recent training data for each device.
    Returns (order, seasonal_order, config_source) where config_source is "auto" or "manual".
    """
    if config is None:
        config = {}

    cfg = config.get("default", {})
    val_len = cfg.get("val_length", 120)
    train_len = cfg.get("train_length", 1000)
    order_candidates = cfg.get("order_candidates", [
        [1, 0, 0], [2, 0, 0], [3, 0, 0],
        [1, 0, 1], [2, 0, 1],
        [0, 0, 1], [0, 0, 2],
        [1, 1, 0], [2, 1, 0], [1, 1, 1], [2, 1, 1], [3, 1, 0],
    ])
    seasonal_candidates = cfg.get("seasonal_candidates", [
        [0, 0, 0, 0],
        [1, 0, 0, 60],
        [1, 1, 0, 60],
        [2, 0, 0, 60],
        [1, 0, 1, 60],
    ]) if seasonal_search else [[0, 0, 0, 0]]

    try:
        available = len(y)
        total_needed = train_len + val_len
        if available < total_needed:
            train_len = max(60, available - val_len)
            total_needed = train_len + val_len

        y_sub = y.tail(total_needed)
        exog_sub = exog.tail(total_needed)

        g_train_y = y_sub.iloc[:-val_len]
        g_train_X = exog_sub.iloc[:-val_len]
        g_val_y = y_sub.iloc[-val_len:]
        g_val_X = exog_sub.iloc[-val_len:]

        # Test stationarity
        res_before = adfuller(g_train_y.dropna())
        p_before = res_before[1]
        rec_d = 0
        if p_before >= 0.05:
            diff_s = g_train_y.diff().dropna()
            res_after = adfuller(diff_s)
            rec_d = 1 if res_after[1] < 0.05 else 2

        # Build full candidate list: combine non-seasonal + seasonal
        candidates = []
        for oc in order_candidates:
            candidates.append((tuple(oc), (0, 0, 0, 0)))
        for sc in seasonal_candidates:
            if sc[-1] == 0:
                continue
            for oc in order_candidates[:5]:
                candidates.append((tuple(oc), tuple(sc)))

        best_order = (1, rec_d, 0)
        best_seasonal = (0, 0, 0, 0)
        best_val_mae = float("inf")

        for order, seasonal in candidates:
            try:
                model = sm.tsa.statespace.SARIMAX(
                    g_train_y,
                    exog=g_train_X,
                    order=order,
                    seasonal_order=seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = model.fit(disp=False, maxiter=100)
                pred = res.forecast(steps=val_len, exog=g_val_X).clip(lower=0)
                mae = float(np.mean(np.abs(g_val_y.values - pred.values)))
                if mae < best_val_mae:
                    best_val_mae = mae
                    best_order = order
                    best_seasonal = seasonal
            except Exception:
                continue

        return best_order, best_seasonal, "auto"
    except Exception as e:
        logger.warning(f"Grid search failed: {e}. Falling back to baseline ARIMAX(1, 0, 0)")
        return (1, 0, 0), (0, 0, 0, 0), "auto"


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero by setting a small threshold or excluding zero values
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def generate_evaluation_plots(
    device_id: str,
    train_y: pd.Series,
    test_y: pd.Series,
    forecast_y: pd.Series,
    save_path: Path
):
    """
    Step 11: Plot Train, Test, and Forecast on the same figure.
    """
    plt.figure(figsize=(12, 6))
    
    # We display only the last 3 hours of train data to make the plot readable
    recent_train = train_y.tail(180)
    
    plt.plot(recent_train.index, recent_train.values, label="Train Data (Recent)", color="blue", alpha=0.6)
    plt.plot(test_y.index, test_y.values, label="Test Data (Actual)", color="green", linewidth=1.5)
    plt.plot(test_y.index, forecast_y.values, label="Forecast (SARIMAX)", color="red", linestyle="--", linewidth=1.5)
    
    plt.title(f"Traffic Volume Forecast - Device {device_id[:8]}")
    plt.xlabel("BucketTime")
    plt.ylabel("NumVehicles")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_diagnostic_plots(
    device_id: str,
    residuals: pd.Series,
    save_path: Path
):
    """
    Step 12: Plot ACF, PACF, Histogram, and QQ-plot for residuals.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Residual Diagnostics - Device {device_id[:8]}", fontsize=16)
    
    # 1. Residuals Line Plot
    axes[0, 0].plot(residuals.index, residuals.values, color="purple", alpha=0.7)
    axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[0, 0].set_title("Residuals over Time")
    
    # 2. Histogram & KDE
    axes[0, 1].hist(residuals, bins=30, density=True, color="gray", alpha=0.6, edgecolor='white')
    # Fit normal distribution
    mu, std = stats.norm.fit(residuals.dropna())
    xmin, xmax = axes[0, 1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    axes[0, 1].plot(x, p, 'r-', linewidth=2, label=f"N({mu:.2f}, {std:.2f})")
    axes[0, 1].set_title("Residual Histogram & fitted Normal Curve")
    axes[0, 1].legend()
    
    # 3. ACF Plot
    sm.graphics.tsa.plot_acf(residuals.dropna(), ax=axes[1, 0], lags=40, title="Residual ACF")
    
    # 4. QQ Plot
    stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("QQ Plot")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def process_single_device(args_tuple: Tuple[str, pd.DataFrame, bool, dict, bool, bool]) -> Dict[str, Any]:
    device_id, df_device, is_test_mode, device_config_dict, no_search, seasonal_search = args_tuple
    logger.info(f"Processing device {device_id}...")
    
    # Step 5: Set BucketTime as DatetimeIndex with freq='1min'
    df_device = df_device.set_index("BucketTime").asfreq("1min")
    
    # Define targets and exogenous columns
    target_col = "NumVehicles"
    
    # To prevent mathematical leakage (where concurrent counts sum directly to NumVehicles),
    # we convert vehicle class counts and ratios into lag-1 features. We also convert weather
    # measurements (Rain, Temperature, Humidity, Visibility, WindSpeed) into lag-1 features.
    # This allows the model to learn from historical vehicle composition and recent weather changes
    # using only information available up to the previous minute.
    base_traffic_features = [
        "AvgSpeed", "Occupancy", "AvgDensity", "AvgHeadway", "FlowRate",
        "AvgTravelTime", "MedianSpeed", "SpeedStd", "MeanConfidence",
        "CarCount", "TruckCount", "BusCount", "MotorcycleCount", "OtherVehicleCount",
        "Rain", "Temperature", "Humidity", "Visibility", "WindSpeed"
    ]
    
    for col in base_traffic_features:
        df_device[f"{col}_lag1"] = df_device[col].shift(1)
        
    # Handle NaNs from shifting
    df_device = df_device.bfill().ffill().fillna(0)
    
    exog_cols = [f"{col}_lag1" for col in base_traffic_features] + [
        "Hour", "DayOfWeek", "IsWeekend", "IsHoliday"
    ]
    
    # If in test mode, subset the data to speed up execution
    if is_test_mode:
        df_device = df_device.tail(300)  # Use smaller history (300 mins)
        
    y = df_device[target_col]
    X = df_device[exog_cols]
    
    # Train-test split (last 60 minutes as test)
    test_length = 60
    train_y = y.iloc[:-test_length]
    train_X = X.iloc[:-test_length]
    test_y = y.iloc[-test_length:]
    test_X = X.iloc[-test_length:]
    
    # Step 6: Augmented Dickey-Fuller (ADF) test
    d, p_before, p_after = test_stationarity(train_y)
    logger.info(f"Device {device_id[:8]} ADF p-value before diff: {p_before:.4f}, after diff (d={d}): {p_after:.4f}")
    
    s = 60
    
    # Step 7: Select model orders — check manual config first, then grid search
    manual_order = device_config_dict.get("order")
    manual_seasonal = device_config_dict.get("seasonal_order")

    if manual_order and manual_seasonal:
        order = tuple(manual_order)
        seasonal_order = tuple(manual_seasonal)
        config_source = "manual"
        logger.info(f"Device {device_id[:8]} using MANUAL config: Order={order}, Seasonal={seasonal_order}")
    elif no_search:
        # Fallback default for --no-search without manual config
        order = (1, d, 0)
        seasonal_order = (0, 0, 0, 0)
        config_source = "default"
        logger.info(f"Device {device_id[:8]} --no-search mode, using fallback: Order={order}")
    else:
        # Auto grid search with full config
        order, seasonal_order, _ = select_best_orders(train_y, train_X, s, device_config_dict, seasonal_search)
        order = (order[0], d, order[2])
        config_source = "auto"
        logger.info(f"Device {device_id[:8]} auto-selected Order: {order}, Seasonal: {seasonal_order}")
    
    # Step 8: Train SARIMAX Model
    model = sm.tsa.statespace.SARIMAX(
        train_y,
        exog=train_X,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    try:
        results = model.fit(disp=False)
    except Exception as e:
        logger.error(f"Failed to fit SARIMAX for device {device_id}: {e}. Retrying without seasonal component.")
        # Fallback to standard ARIMAX
        model = sm.tsa.statespace.SARIMAX(
            train_y,
            exog=train_X,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        results = model.fit(disp=False)
    
    # Step 9: Forecast on Test set
    forecast = results.forecast(steps=test_length, exog=test_X)
    forecast = pd.Series(forecast, index=test_y.index).clip(lower=0) # Vehicle counts cannot be negative
    
    # Step 10: Evaluate metrics
    mae = mean_absolute_error(test_y, forecast)
    rmse = np.sqrt(mean_squared_error(test_y, forecast))
    mape = compute_mape(test_y, forecast)
    r2 = r2_score(test_y, forecast)
    
    logger.info(f"Device {device_id[:8]} Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R2: {r2:.2f}")
    
    # Step 11 & 12: Generate Plots
    plot_eval_path = PLOT_DIR / f"forecast_{device_id}.png"
    generate_evaluation_plots(device_id, train_y, test_y, forecast, plot_eval_path)
    
    residuals = test_y - forecast
    plot_diag_path = PLOT_DIR / f"diagnostics_{device_id}.png"
    generate_diagnostic_plots(device_id, residuals, plot_diag_path)
    
    # Ljung-Box Test on Residuals
    ljung_box_results = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
    ljung_p_value = float(ljung_box_results["lb_pvalue"].iloc[0])
    
    # Step 13: Analysis of Coefficients
    summary_coefs = {}
    for param in results.params.index:
        summary_coefs[param] = {
            "coef": float(results.params[param]),
            "pvalue": float(results.pvalues[param])
        }
        
    # Step 14: Multi-horizon Forecast (15, 30, 60 minutes into the absolute future)
    # We need exogenous variables for the future.
    # We construct X_future by persisting the last row of X test_length times.
    last_exog = X.iloc[-1:]
    X_future = pd.concat([last_exog] * 60, ignore_index=True)
    future_index = pd.date_range(start=y.index[-1] + pd.Timedelta("1min"), periods=60, freq="1min")
    X_future.index = future_index
    
    # Update time-dependent features dynamically for the future index
    X_future["Hour"] = X_future.index.hour
    X_future["DayOfWeek"] = X_future.index.dayofweek
    X_future["IsWeekend"] = X_future.index.dayofweek.isin([5, 6]).astype(int)
    X_future["IsHoliday"] = is_vietnam_holiday(pd.Series(X_future.index)).values
    
    # Train full model on all data before future forecast
    full_model = sm.tsa.statespace.SARIMAX(
        y,
        exog=X,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    full_results = full_model.fit(disp=False)
    
    future_forecast = full_results.forecast(steps=60, exog=X_future).clip(lower=0)
    future_forecast.index = future_index
    
    future_predictions = {
        "15min": float(future_forecast.iloc[14]),
        "30min": float(future_forecast.iloc[29]),
        "60min": float(future_forecast.iloc[59])
    }
    
    # Step 15: Save model to file
    model_save_path = OUT_DIR / f"model_{device_id}.joblib"
    joblib.dump(results, model_save_path)
    
    return {
        "device_id": device_id,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "adf_p_before": p_before,
        "adf_p_after": p_after,
        "order": order,
        "seasonal_order": seasonal_order,
        "config_source": config_source,
        "ljung_box_pvalue": ljung_p_value,
        "coefs": summary_coefs,
        "future_predictions": future_predictions,
        "model_path": str(model_save_path),
        "forecast_png": str(plot_eval_path),
        "diagnostics_png": str(plot_diag_path)
    }


def main():
    args = parse_args()
    logger.info("Initializing SARIMAX Traffic Volume Forecasting Pipeline...")
    
    # Load dataset
    try:
        df = load_and_preprocess_data(CSV_PATH)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)
        
    # Get distinct devices
    devices = df["DeviceId"].unique()
    logger.info(f"Found {len(devices)} unique device IDs in the dataset.")
    
    # In test mode, select only 2 devices to keep it very fast
    if args.test_mode:
        logger.info("TEST MODE ENABLED: Limiting pipeline to 2 devices with limited history.")
        devices = devices[:2]
        
    # Load per-device config
    full_config = load_device_config()
    default_cfg = full_config.get("default", {})
    device_cfgs = full_config.get("devices", {})

    # Build list of tasks for parallel execution
    tasks = []
    for device_id in devices:
        df_device = df[df["DeviceId"] == device_id].copy()
        dev_cfg = device_cfgs.get(device_id, {})
        # Merge default into device config (device overrides default)
        merged = {**default_cfg, **dev_cfg}
        tasks.append((device_id, df_device, args.test_mode, merged, args.no_search, args.seasonal_search))
        
    logger.info(f"Starting model training for {len(devices)} devices using {args.n_jobs} parallel jobs...")
    
    # Execute training in parallel
    results_list = joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(process_single_device)(task) for task in tqdm(tasks, desc="Training SARIMAX models")
    )
    
    # Step 10: Compile and display final results
    results_df = pd.DataFrame([
        {
            "DeviceId": r["device_id"],
            "MAE": r["mae"],
            "RMSE": r["rmse"],
            "MAPE (%)": r["mape"],
            "R2": r["r2"],
            "ADF_p_before": r["adf_p_before"],
            "ADF_p_after": r["adf_p_after"],
            "SARIMAX_Order": str(r["order"]),
            "Seasonal_Order": str(r["seasonal_order"]),
            "ConfigSource": r["config_source"],
            "LjungBox_pvalue": r["ljung_box_pvalue"],
            "Forecast_15min": r["future_predictions"]["15min"],
            "Forecast_30min": r["future_predictions"]["30min"],
            "Forecast_60min": r["future_predictions"]["60min"],
        }
        for r in results_list
    ])
    
    # Save forecast statistics to CSV
    metrics_path = OUT_DIR / "sarimax_evaluation_metrics.csv"
    results_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    
    logger.info("=== FINAL SARIMAX EVALUATION METRICS ===")
    print(results_df.to_string(index=False))
    logger.info(f"Evaluation metrics saved to: {metrics_path}")
    
    # Save coefficient breakdown to JSON for later analysis
    coef_report = {r["device_id"]: {"coefs": r["coefs"]} for r in results_list}
    coef_report_path = OUT_DIR / "sarimax_coefficients.json"
    import json
    coef_report_path.write_text(json.dumps(coef_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Coefficients details saved to: {coef_report_path}")
    
    logger.info("SARIMAX Forecasting Pipeline completed successfully!")


if __name__ == "__main__":
    main()
