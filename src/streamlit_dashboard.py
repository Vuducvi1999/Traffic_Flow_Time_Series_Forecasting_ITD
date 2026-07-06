import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="SARIMAX Testing Dashboard",
    page_icon="📊",
    layout="wide"
)

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
# We will select target_col dynamically in the sidebar.

# Clean custom CSS for simple styling (no fancy colors, just nice layout)
st.markdown("""
<style>
    .reportview-container { background: #f3f4f6; }
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title { font-size: 0.85rem; color: #6b7280; text-transform: uppercase; font-weight: bold; }
    .metric-value { font-size: 1.6rem; font-weight: bold; margin: 5px 0; color: #1f2937; }
    .metric-desc { font-size: 0.75rem; color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="⏳ Đang tải dữ liệu...")
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["BucketTime"] = pd.to_datetime(df["BucketTime"])
    metrics = pd.read_csv(METRICS_PATH)
    return df, metrics

@st.cache_resource(show_spinner=False)
def load_model(device_id: str, target_col: str):
    path = MODEL_DIR / f"model_{target_col}_{device_id}.joblib"
    if path.exists():
        return joblib.load(path)
    return None

def compute_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)

def prepare_device_data(df, device_id, target_col):
    from prepare_forecasting_dataset import is_vietnam_holiday

    df_dev = df[df["DeviceId"] == device_id].copy()
    df_dev = df_dev.sort_values(by=["Lane", "BucketTime"]).reset_index(drop=True)

    if target_col == "NumVehicles":
        base_traffic_features = [
            "AvgSpeed", "Occupancy", "AvgDensity",
            "Rain", "Temperature", "Humidity", "Visibility", "WindSpeed"
        ]
    else:  # AvgSpeed
        base_traffic_features = [
            "NumVehicles", "Occupancy", "AvgDensity",
            "Rain", "Temperature", "Humidity", "Visibility", "WindSpeed"
        ]

    numeric_cols = [target_col] + base_traffic_features
    agg = df_dev.groupby("BucketTime")[numeric_cols].mean().reset_index()
    agg = agg.set_index("BucketTime").asfreq("5min")

    for col in base_traffic_features:
        agg[f"{col}_lag1"] = agg[col].shift(1)

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
    y = agg[target_col]
    X = agg[exog_cols].astype(float)
    return y, X, agg

def main():
    st.title("📊 SARIMAX Accuracy Test & Backtest Simulator")
    st.write("Giao diện kiểm chứng độ chính xác trực quan. Bạn có thể chọn bất kỳ thời điểm nào trong lịch sử để chạy mô phỏng đối chất song song Thực tế vs Dự báo.")

    df, metrics_df = load_data()

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("⚙️ Cấu hình bộ lọc")
    
    target_col = st.sidebar.selectbox(
        "Biến mục tiêu", 
        options=["NumVehicles", "AvgSpeed"], 
        format_func=lambda x: "NumVehicles (Lưu lượng xe)" if x == "NumVehicles" else "AvgSpeed (Tốc độ)"
    )
    
    # Filter device list using only devices present for selected target
    target_metrics = metrics_df[metrics_df["TargetCol"] == target_col]
    device_ids = sorted(target_metrics["DeviceId"].tolist())
    if not device_ids:
        device_ids = sorted(df["DeviceId"].unique().tolist())
        
    selected_device = st.sidebar.selectbox("Trạm VDS", options=device_ids, format_func=lambda x: f"Trạm {x[:8]}...")
    
    horizon = st.sidebar.slider("Số phút dự báo (Horizon)", min_value=5, max_value=120, value=60, step=5)
    history_window = st.sidebar.slider("Số phút lịch sử hiển thị", min_value=30, max_value=240, value=120, step=30)
    
    show_weather = st.sidebar.checkbox("Hiển thị cột lượng mưa", value=True)

    # --- PROCESS DATA & MODEL ---
    y, X, agg = prepare_device_data(df, selected_device, target_col)
    results = load_model(selected_device, target_col)

    if results is None:
        st.error(f"Không tìm thấy file model cho trạm: {selected_device}")
        return

    # --- CHẾ ĐỘ MÔ PHỎNG ĐỐI CHẤT ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏰ Chế độ mô phỏng đối chất")
    mode = st.sidebar.radio(
        "Chọn khoảng thời gian",
        options=["Cuối tập dữ liệu (Mặc định)", "Tự chọn thời điểm lịch sử (Chạy đối chất)"]
    )

    if mode == "Cuối tập dữ liệu (Mặc định)":
        forecast_start_time = agg.index[-1 - horizon]
    else:
        unique_dates = sorted(list(set(agg.index.date)))
        selected_date = st.sidebar.selectbox("Chọn Ngày", options=unique_dates, index=len(unique_dates)-2 if len(unique_dates) > 1 else 0)
        selected_hour = st.sidebar.slider("Chọn Giờ", min_value=0, max_value=23, value=17)
        selected_minute = st.sidebar.slider("Chọn Phút", min_value=0, max_value=55, value=0, step=5)
        
        forecast_start_time = pd.Timestamp(f"{selected_date} {selected_hour:02d}:{selected_minute:02d}:00")
        
        # Guard rails
        if forecast_start_time < agg.index[history_window]:
            forecast_start_time = agg.index[history_window]
        if forecast_start_time > agg.index[-1 - horizon]:
            forecast_start_time = agg.index[-1 - horizon]

    st.sidebar.info(f"Đang dự báo cho 60 phút tiếp theo tính từ: **{forecast_start_time.strftime('%Y-%m-%d %H:%M')}**")

    # Split slices dynamically based on forecast_start_time
    train_y = y.loc[:forecast_start_time]
    
    start_future = forecast_start_time + pd.Timedelta("5min")
    end_future = forecast_start_time + pd.Timedelta(minutes=horizon)
    
    test_act = y.loc[start_future:end_future]
    test_exog = X.loc[start_future:end_future]
    
    hist_act = train_y.iloc[-history_window:]
    fitted_vals = results.fittedvalues.reindex(hist_act.index).ffill().bfill()

    # Forecast using statsmodels predict/forecast starting at the selected time
    try:
        pred_obj = results.get_prediction(start=start_future, end=end_future, exog=test_exog)
        forecast_mean = pred_obj.predicted_mean.clip(lower=0)
        ci = pred_obj.conf_int(alpha=0.05)
        ci_lower = ci.iloc[:, 0].clip(lower=0)
        ci_upper = ci.iloc[:, 1]
    except Exception:
        # Fallback to standard forecast
        steps = horizon // 5
        forecast_mean = results.forecast(steps=steps, exog=test_exog).clip(lower=0)
        forecast_mean.index = pd.date_range(start=start_future, periods=steps, freq="5min")
        std_est = float(fitted_vals.std()) * 1.5 if fitted_vals.notna().any() else 10.0
        ci_lower = (forecast_mean - std_est * 1.96).clip(lower=0)
        ci_upper = forecast_mean + std_est * 1.96

    # Calculate LIVE accuracy metrics specifically for the selected visualization window
    common_idx = test_act.index.intersection(forecast_mean.index)
    if len(common_idx) > 0:
        y_true_eval = test_act.loc[common_idx]
        y_pred_eval = forecast_mean.loc[common_idx]
        mape = compute_mape(y_true_eval, y_pred_eval)
        mae = float(mean_absolute_error(y_true_eval, y_pred_eval))
        r2 = float(r2_score(y_true_eval, y_pred_eval)) if len(y_true_eval) > 1 and y_true_eval.std() > 0 else 0.0
    else:
        mape, r2, mae = 0.0, 0.0, 0.0

    # Calculate First & Second Difference as Percentage at the chosen forecast_start_time
    y_at_t = float(y.loc[forecast_start_time])
    t_prev1 = forecast_start_time - pd.Timedelta("5min")
    y_prev1 = float(y.loc[t_prev1]) if t_prev1 in y.index else y_at_t
    v1_pct = ((y_at_t - y_prev1) / y_prev1 * 100) if y_prev1 > 0 else 0.0

    t_prev2 = forecast_start_time - pd.Timedelta("10min")
    y_prev2 = float(y.loc[t_prev2]) if t_prev2 in y.index else y_prev1
    v1_prev_pct = ((y_prev1 - y_prev2) / y_prev2 * 100) if y_prev2 > 0 else 0.0
    v2_pct = v1_pct - v1_prev_pct

    # Format status text
    v1_status = "Tăng ↗" if v1_pct >= 0.5 else ("Giảm ↘" if v1_pct <= -0.5 else "Ổn định ➡️")
    v2_status = "Đang tăng nhanh ⚡" if v2_pct >= 1.0 else ("Giảm tốc 🍃" if v2_pct <= -1.0 else "Ổn định ➡️")

    # --- METRIC CARDS ROW ---
    st.markdown("### 📊 Các chỉ số đánh giá độ chính xác (Tính thực tế trên khung nhìn hiện tại)")
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-title">MAPE (SAI SỐ)</div>
            <div class="metric-value" style="color: {'#10b981' if mape < 20 else '#f59e0b'}">{mape:.2f}%</div>
            <div class="metric-desc">Chênh lệch trung bình với thực tế</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-title">R² (ĐỘ KHỚP XU HƯỚNG)</div>
            <div class="metric-value" style="color: {'#10b981' if r2 > 0.4 else '#ef4444'}">{r2:.3f}</div>
            <div class="metric-desc">Độ tương quan đường dự đoán</div>
        </div>""", unsafe_allow_html=True)
    mae_title = "MAE (LỆCH TỐC ĐỘ)" if target_col == "AvgSpeed" else "MAE (LỆCH SỐ XE)"
    mae_unit = "km/h" if target_col == "AvgSpeed" else "xe"
    with c3:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-title">{mae_title}</div>
            <div class="metric-value" style="color: #3b82f6">±{mae:.1f} {mae_unit}</div>
            <div class="metric-desc">Lệch trung bình mỗi phút</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-title">VI PHÂN 1 (% THAY ĐỔI)</div>
            <div class="metric-value" style="color: #6366f1">{v1_pct:+.2f}%</div>
            <div class="metric-desc">Tại mốc dự báo: {v1_status}</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-title">VI PHÂN 2 (% GIA TỐC)</div>
            <div class="metric-value" style="color: #8b5cf6">{v2_pct:+.2f}%</div>
            <div class="metric-desc">Gia tốc xu hướng: {v2_status}</div>
        </div>""", unsafe_allow_html=True)

    st.write("")

    # --- UNIFIED ACCURACY CHART ---
    st.markdown(f"### 📈 Biểu đồ đối chất song song Thực tế vs Dự báo (Mốc: {forecast_start_time.strftime('%Y-%m-%d %H:%M')})")
    
    fig = go.Figure()

    # 1. CI Band (Future)
    x_fc = list(forecast_mean.index)
    fig.add_trace(go.Scatter(
        x=x_fc + x_fc[::-1],
        y=list(ci_upper.values) + list(ci_lower.values[::-1]),
        fill="toself",
        fillcolor="rgba(156, 39, 176, 0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Khoảng tin cậy 95%",
        showlegend=True,
        hoverinfo="skip"
    ))

    # 2. Historical Actual
    fig.add_trace(go.Scatter(
        x=hist_act.index, y=hist_act.values,
        name="Thực tế (Quá khứ)",
        mode="lines",
        line=dict(color="#2196f3", width=2),
    ))

    # 3. Historical Fitted
    fig.add_trace(go.Scatter(
        x=hist_act.index, y=fitted_vals.values,
        name="Mô hình khớp (Fitted)",
        mode="lines",
        line=dict(color="#ff9800", width=1.5, dash="dot"),
    ))

    # 4. Forecasted Mean
    fig.add_trace(go.Scatter(
        x=forecast_mean.index, y=forecast_mean.values,
        name="Đường dự đoán (Forecast)",
        mode="lines+markers",
        line=dict(color="#9c27b0", width=3, dash="dash"),
        marker=dict(size=4),
    ))

    # 5. Actual Future
    fig.add_trace(go.Scatter(
        x=test_act.index, y=test_act.values,
        name="Đường thực tế (Actual Future)",
        mode="lines+markers",
        line=dict(color="#4caf50", width=3),
        marker=dict(size=4),
    ))

    # Vertical split line
    split_time = forecast_mean.index[0]
    fig.add_vline(x=split_time, line_color="#757575", line_width=2, line_dash="dot")

    # Optional weather overlay (bar chart on secondary y-axis)
    if show_weather and "Rain" in agg.columns:
        combined_idx = hist_act.index.union(forecast_mean.index)
        rain_series = agg["Rain"].reindex(combined_idx).fillna(0)
        fig.add_trace(go.Bar(
            x=rain_series.index, y=rain_series.values,
            name="Lượng mưa (mm)",
            yaxis="y2",
            marker_color="rgba(33, 150, 243, 0.15)",
            marker_line_width=0,
        ))

    # Layout config
    layout_extra = {}
    if show_weather:
        layout_extra = dict(
            yaxis2=dict(
                title=dict(text="Lượng mưa (mm)", font=dict(size=10, color="#2196f3")),
                overlaying="y",
                side="right",
                showgrid=False
            )
        )

    y_axis_title = "Tốc độ (km/h)" if target_col == "AvgSpeed" else "Lưu lượng xe (xe/5 phút)"
    fig.update_layout(
        **layout_extra,
        margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        hovermode="x unified",
        height=500
    )
    fig.update_xaxes(gridcolor="#e5e7eb", linecolor="#cccccc")
    fig.update_yaxes(title_text=y_axis_title, gridcolor="#e5e7eb", linecolor="#cccccc")

    st.plotly_chart(fig, width='stretch')

    # --- RAW DATA TABLE ---
    with st.expander("📋 Xem bảng số liệu đối chất chi tiết", expanded=False):
        fc_table = pd.DataFrame({
            "Thời gian": forecast_mean.index.strftime("%H:%M"),
            f"Dự báo ({mae_unit})": forecast_mean.values.round(1),
            f"Thực tế tương lai ({mae_unit})": test_act.reindex(forecast_mean.index).values.round(1),
            f"Lệch ({mae_unit})": np.abs(test_act.reindex(forecast_mean.index).values - forecast_mean.values).round(1),
            "Giới hạn dưới (CI Lower)": ci_lower.values.round(1),
            "Giới hạn trên (CI Upper)": ci_upper.values.round(1)
        })
        st.dataframe(fc_table, width='stretch', hide_index=True)

if __name__ == "__main__":
    main()
