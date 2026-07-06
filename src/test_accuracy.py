
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")

# Paths
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "processed" / "traffic_vehicle_forecasting_dataset.csv"
METRICS_PATH = ROOT / "data" / "processed" / "models" / "sarimax_evaluation_metrics.csv"
MODEL_DIR = ROOT / "data" / "processed" / "models"

BASE_TRAFFIC_FEATURES = [
    "NumVehicles", "Occupancy", "AvgDensity", "AvgHeadway", "FlowRate",
    "AvgTravelTime", "MedianSpeed", "SpeedStd", "MeanConfidence",
    "Rain", "Temperature", "Humidity", "Visibility", "WindSpeed",
    "NumVehicles_roll_mean_15m", "NumVehicles_roll_std_15m",
    "NumVehicles_roll_mean_30m", "NumVehicles_roll_std_30m",
    "AvgSpeed_roll_mean_15m", "AvgSpeed_roll_std_15m",
    "AvgSpeed_roll_mean_30m", "AvgSpeed_roll_std_30m",
]
TARGET_COL = "AvgSpeed"

def load_data():
    print("[LOG] Dang tai du lieu (CSV)...")
    df = pd.read_csv(CSV_PATH)
    df["BucketTime"] = pd.to_datetime(df["BucketTime"])
    metrics = pd.read_csv(METRICS_PATH)
    return df, metrics

def prepare_device_data(df, device_id):
    from prepare_forecasting_dataset import is_vietnam_holiday

    df_dev = df[df["DeviceId"] == device_id].copy()
    df_dev = df_dev.sort_values(by=["Lane", "BucketTime"]).reset_index(drop=True)

    if TARGET_COL == "NumVehicles":
        base_traffic_features = [
            "AvgSpeed", "Occupancy", "AvgDensity",
            "Rain", "Temperature", "Humidity", "Visibility", "WindSpeed"
        ]
    else:  # AvgSpeed
        base_traffic_features = [
            "Occupancy",
            "Rain", "Temperature", "Humidity", "Visibility", "WindSpeed"
        ]

    numeric_cols = [TARGET_COL] + base_traffic_features
    agg = df_dev.groupby("BucketTime")[numeric_cols].mean().reset_index()
    agg = agg.set_index("BucketTime").asfreq("5min")

    for col in base_traffic_features:
        agg[f"{col}_lag1"] = agg[col].shift(1)

    # Hour and DayOfWeek dummies
    hour_dummies = pd.get_dummies(agg.index.hour, prefix="Hour", drop_first=True, dtype=int)
    hour_dummies.index = agg.index
    dow_dummies = pd.get_dummies(agg.index.dayofweek, prefix="DoW", drop_first=True, dtype=int)
    dow_dummies.index = agg.index

    agg = pd.concat([agg, hour_dummies, dow_dummies], axis=1)
    agg = agg.bfill().ffill().fillna(0)

    agg["IsHoliday"] = is_vietnam_holiday(pd.Series(agg.index)).values

    exog_cols = [f"{col}_lag1" for col in base_traffic_features] + [
        "IsHoliday"
    ] + hour_dummies.columns.tolist() + dow_dummies.columns.tolist()

    y = agg[TARGET_COL]
    X = agg[exog_cols].astype(float)
    return y, X, agg

def plot_device_accuracy(y, X, agg, metrics_df, device_id, horizon=60, history_len=24):
    # Load model
    model_path = MODEL_DIR / f"model_{TARGET_COL}_{device_id}.joblib"
    if not model_path.exists():
        print(f"[LOI] Khong tim thay model cho tram: {device_id}")
        return
    results = joblib.load(model_path)

    # Metrics
    matching_metrics = metrics_df[(metrics_df["DeviceId"] == device_id) & (metrics_df["TargetCol"] == TARGET_COL)]
    if len(matching_metrics) > 0:
        dev_metrics = matching_metrics.iloc[0]
        mape = dev_metrics["MAPE (%)"]
        r2 = dev_metrics["R2"]
        mae = dev_metrics["MAE"]
    else:
        mape, r2, mae = 0.0, 0.0, 0.0

    # Calculate First & Second Difference as Percentage
    y_last = float(y.iloc[-1])
    y_prev1 = float(y.iloc[-2]) if len(y) > 1 else y_last
    v1_pct = ((y_last - y_prev1) / y_prev1 * 100) if y_prev1 > 0 else 0.0

    y_prev2 = float(y.iloc[-3]) if len(y) > 2 else y_prev1
    v1_prev_pct = ((y_prev1 - y_prev2) / y_prev2 * 100) if y_prev2 > 0 else 0.0
    v2_pct = v1_pct - v1_prev_pct

    # Friendly indicators
    v1_status = "Tang" if v1_pct >= 0.5 else ("Giam" if v1_pct <= -0.5 else "On dinh")
    v2_status = "Tang toc (De un tac)" if v2_pct >= 1.0 else ("Giam toc (Thoang dan)" if v2_pct <= -1.0 else "On dinh")

    unit = "km/h" if TARGET_COL == "AvgSpeed" else "xe"
    print("\n" + "="*60)
    print(f"THONG TIN KIEM TRA DO CHINH XAC - Tram {device_id[:8]}...")
    print("="*60)
    print(f"- MAPE (Sai so du bao): {mape:.2f}% ({'Tot' if mape < 25 else 'Trung binh/Kem'})")
    print(f"- R2 (Do tuong quan xu huong): {r2:.3f} ({'Tot' if r2 > 0.0 else 'Kem'})")
    print(f"- MAE (Lech trung binh): +/-{mae:.2f} {unit}")
    print(f"- Vi phan cap 1 (% Thay doi): {v1_pct:+.2f}% -> Trang thai: {v1_status}")
    print(f"- Vi phan cap 2 (% Gia toc): {v2_pct:+.2f}% -> Trang thai: {v2_status}")
    print("="*60)

    # Slices (5-minute steps)
    test_len = 12  # Matches exactly the 12 steps (60 minutes) test split from training
    train_y  = y.iloc[:-test_len]
    test_y   = y.iloc[-test_len:]
    test_X   = X.iloc[-test_len:]

    fitted_vals = results.fittedvalues

    # Forecast
    horizon_steps = min(horizon // 5, test_len)
    forecast_obj = results.get_forecast(steps=horizon_steps, exog=test_X.iloc[:horizon_steps])
    forecast_mean = forecast_obj.predicted_mean.clip(lower=0)
    ci = forecast_obj.conf_int(alpha=0.05) # 95% CI
    ci_lower = ci.iloc[:, 0].clip(lower=0)
    ci_upper = ci.iloc[:, 1]

    # Matplotlib Plot: Create 2 Subplots for direct comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
    
    # --- SUBPLOT 1: ZOOM-IN FORECAST PERIOD (Direct Comparison / Doi Chat) ---
    test_act = test_y.iloc[:horizon_steps]
    
    ax1.plot(test_act.index, test_act.values, label="Thuc te dien ra (Actual)", color="#4caf50", linewidth=3.0, marker="o", markersize=4)
    ax1.plot(forecast_mean.index, forecast_mean.values, label="Moi hinh Du bao (Forecast)", color="#9c27b0", linewidth=3.0, linestyle="--", marker="x", markersize=4)
    ax1.fill_between(forecast_mean.index, ci_lower.values, ci_upper.values, color="#9c27b0", alpha=0.15, label="Khoang tin cay 95%")
    
    # Calculate step-by-step errors for the hover or annotation
    mae_horizon = np.mean(np.abs(test_act.values - forecast_mean.values))
    mae_unit = "km/h" if TARGET_COL == "AvgSpeed" else "xe/phut"
    mae_label = "Lech toc do" if TARGET_COL == "AvgSpeed" else "Lech so xe"
    y_label = "Toc do (km/h)" if TARGET_COL == "AvgSpeed" else "Luu luong (xe / phut)"
    
    ax1.set_title(f"DOI CHAT TRUC TIEP: Thuc te vs Du bao trong {horizon_steps * 5} phut (MAE: {mae_horizon:.2f} {mae_unit})", fontsize=11, fontweight="bold", color="#1e88e5")
    ax1.set_xlabel("Thoi gian", fontsize=9)
    ax1.set_ylabel(y_label, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")
    
    # --- SUBPLOT 2: OVERALL TIMELINE CONTEXT ---
    hist_act = train_y.iloc[-history_len:]
    hist_fit = fitted_vals.iloc[-history_len:]
    
    ax2.plot(hist_act.index, hist_act.values, label="Thuc te (Qua khu)", color="#00bcd4", linewidth=2)
    ax2.plot(hist_fit.index, hist_fit.values, label="Mo hinh Khops (Fitted)", color="#ff5722", linewidth=1.5, linestyle="--")
    ax2.plot(forecast_mean.index, forecast_mean.values, label="Du bao (Forecast)", color="#9c27b0", linewidth=2, linestyle="-.")
    ax2.plot(test_act.index, test_act.values, label="Thuc te (Tuong lai)", color="#4caf50", linewidth=2)
    
    # Separate line
    split_time = forecast_mean.index[0]
    ax2.axvline(x=split_time, color="#757575", linestyle=":", linewidth=2)
    ax2.text(split_time, ax2.get_ylim()[1]*0.9, " Bat dau du bao", color="#616161", fontsize=9, fontweight="bold")
    
    ax2.set_title(f"TOAN CANH CHUOI THOI GIAN: Lich su & Giai doan Du doan", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Thoi gian", fontsize=9)
    ax2.set_ylabel("Luu luong (xe / phut)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")
    
    plt.suptitle(f"KIEM TRA CHAT LUONG DU BAO - TRAM VDS: {device_id}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.show()

def main():
    global TARGET_COL
    print("\n" + "="*60)
    print("CHUONG TRINH DO CHIEU VA KIEM TRA DO CHINH XAC FORECAST")
    print("="*60)
    print("[1] NumVehicles (Luu luong xe - TARGET CHINH)")
    print("[2] AvgSpeed (Toc do xe)")
    tgt_choice = input("Chon bien muc tieu muon kiem tra [Mac dinh: 1]: ")
    if tgt_choice.strip() == "2":
        TARGET_COL = "AvgSpeed"
    else:
        TARGET_COL = "NumVehicles"

    df, metrics = load_data()
    # Filter device list using only devices present for selected target
    target_metrics = metrics[metrics["TargetCol"] == TARGET_COL]
    device_ids = sorted(target_metrics["DeviceId"].tolist())
    if not device_ids:
        # Fallback to all devices in data
        device_ids = sorted(df["DeviceId"].unique().tolist())

    while True:
        print(f"\nDanh sach cac tram VDS (Bien: {TARGET_COL}):")
        for idx, dev in enumerate(device_ids):
            print(f"[{idx+1:2d}] {dev}")
            
        print("\n[0] Thoat")
        
        try:
            choice = input("\nChon so tram can xem (hoac 0 de thoat): ")
            if not choice.strip().isdigit():
                print("[!] Vui loi nhap so.")
                continue
            choice_num = int(choice)
            if choice_num == 0:
                print("Goodbye!")
                break
            if choice_num < 1 or choice_num > len(device_ids):
                print("[!] Lua chon khong hop le, hay thu lai.")
                continue
                
            selected_device = device_ids[choice_num-1]
            
            h_choice = input("Nhap so phut du bao can xem (mac dinh 60, toi da 120): ")
            h_val = int(h_choice) if h_choice.strip().isdigit() else 60
            
            y, X, agg = prepare_device_data(df, selected_device)
            plot_device_accuracy(y, X, agg, metrics, selected_device, horizon=h_val)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[!] Loi: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
