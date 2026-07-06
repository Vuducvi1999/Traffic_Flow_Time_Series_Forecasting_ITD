import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from prepare_forecasting_dataset import is_vietnam_holiday

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("APIServer")

# ── Paths ────────────────────────────────────────────────────────────────────
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

# ── Global state (set during startup) ────────────────────────────────────────
df: pd.DataFrame = None
metrics_df: pd.DataFrame = None
models: Dict[str, object] = {}
device_metrics_map: Dict[str, dict] = {}
device_agg_data: Dict[str, pd.DataFrame] = {}


# ── Helpers (reused from existing codebase) ──────────────────────────────────
def compute_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)


def prepare_device_data(device_id: str) -> pd.DataFrame:
    """Aggregate device data to 1-min frequency, compute features & lags."""
    df_dev = df[df["DeviceId"] == device_id].copy()
    df_dev = df_dev.sort_values(by=["Lane", "BucketTime"]).reset_index(drop=True)

    numeric_cols = [TARGET_COL] + BASE_TRAFFIC_FEATURES
    agg = df_dev.groupby("BucketTime")[numeric_cols].mean().reset_index()
    agg = agg.set_index("BucketTime").asfreq("5min")

    agg["Hour"] = agg.index.hour
    agg["DayOfWeek"] = agg.index.dayofweek
    agg["IsWeekend"] = (agg.index.dayofweek >= 5).astype(int)
    agg["IsHoliday"] = is_vietnam_holiday(pd.Series(agg.index)).values

    for col in BASE_TRAFFIC_FEATURES:
        agg[f"{col}_lag1"] = agg[col].shift(1)

    return agg.bfill().ffill().fillna(0)


def compute_exog_row(row: pd.Series) -> pd.Series:
    """Build exogenous feature vector from a single data row."""
    features = {}
    for col in BASE_TRAFFIC_FEATURES:
        if col in row:
            features[f"{col}_lag1"] = float(row[col])
        else:
            features[f"{col}_lag1"] = 0.0
    features["Hour"] = row.get("Hour", 0)
    features["DayOfWeek"] = row.get("DayOfWeek", 0)
    features["IsWeekend"] = row.get("IsWeekend", 0)
    features["IsHoliday"] = row.get("IsHoliday", 0)
    return pd.Series(features)


# ── Pydantic models ──────────────────────────────────────────────────────────
class TrafficObservation(BaseModel):
    NumVehicles: float = 0
    AvgSpeed: float = 0
    Occupancy: float = 0
    AvgDensity: float = 0
    AvgHeadway: float = 0
    FlowRate: float = 0
    AvgTravelTime: float = 0
    MedianSpeed: float = 0
    SpeedStd: float = 0
    MeanConfidence: float = 0
    Rain: float = 0
    Temperature: float = 0
    Humidity: float = 0
    Visibility: float = 0
    WindSpeed: float = 0


class ForecastRequest(BaseModel):
    horizon: int = Field(default=60, ge=1, le=1440, description="Number of minutes to forecast")
    current_observation: Optional[TrafficObservation] = Field(default=None, description="Current traffic measurements. If omitted, the last row from the dataset is used.")
    observation_time: Optional[str] = Field(default=None, description="ISO timestamp of the observation. If omitted, uses the last dataset timestamp + 1min.")


class ForecastPoint(BaseModel):
    time: str
    predicted: float
    ci_lower: float
    ci_upper: float


class HistoryPoint(BaseModel):
    time: str
    actual: float


class ExtendRequest(BaseModel):
    observation: TrafficObservation
    timestamp: str = Field(description="ISO timestamp of the observation (e.g. '2026-06-29 09:18')")


class ExtendResponse(BaseModel):
    device_id: str
    status: str
    model_updated_at: str


class NodeTrafficObservation(BaseModel):
    NumVehicles: float = 0
    AvgSpeed: float = 0
    Occupancy: float = 0
    AvgDensity: float = 0
    AvgHeadway: float = 0
    FlowRate: float = 0
    Confidence: float = 0
    Rain: float = 0
    Temperature: float = 0
    Humidity: float = 0
    Visibility: float = 0
    WindSpeed: float = 0


class NodeForecastRequest(BaseModel):
    Horizon: int = Field(default=60, ge=1, le=1440, description="Number of minutes to forecast")
    CurrentObservation: Optional[NodeTrafficObservation] = Field(default=None, description="Current traffic measurements. If omitted, the last row from the dataset is used.")
    ObservationTime: Optional[str] = Field(default=None, description="ISO timestamp of the observation. If omitted, uses the last dataset timestamp + 1min.")


class NodeForecastPoint(BaseModel):
    Time: datetime
    NumVehicles: float
    AvgSpeed: float
    Occupancy: float
    AvgDensity: float
    AvgHeadway: float
    FlowRate: float
    Confidence: float


class NodeForecastResponse(BaseModel):
    Forecast: List[NodeForecastPoint]
    Metrics: Dict[str, float]



# ── FastAPI app ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, metrics_df, models, device_metrics_map, device_agg_data
    logger.info("Loading dataset...")
    if not CSV_PATH.exists():
        logger.error(f"Dataset not found at {CSV_PATH}")
        yield
        return
    df = pd.read_csv(CSV_PATH)
    df["BucketTime"] = pd.to_datetime(df["BucketTime"])

    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)
        for _, row in metrics_df.iterrows():
            did = row["DeviceId"]
            device_metrics_map[did] = {
                "mae": float(row["MAE"]),
                "rmse": float(row["RMSE"]),
                "mape": float(row["MAPE (%)"]),
                "r2": float(row["R2"]),
            }

    logger.info("Loading models and aggregating device data...")
    device_ids = df["DeviceId"].unique()
    for device_id in device_ids:
        model_path = MODEL_DIR / f"model_{TARGET_COL}_{device_id}.joblib"
        if model_path.exists():
            try:
                models[device_id] = joblib.load(model_path)
                device_agg_data[device_id] = prepare_device_data(device_id)
            except Exception as e:
                logger.warning(f"Failed to load model for {device_id}: {e}")
        else:
            logger.warning(f"No model file for device {device_id}")

    logger.info(f"API ready: {len(models)} devices loaded")
    yield


app = FastAPI(
    title="Traffic Forecasting API",
    description="SARIMAX-based traffic volume forecasting API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "devices_loaded": len(models),
        "dataset_period": (
            f"{df['BucketTime'].min()} → {df['BucketTime'].max()}" if df is not None else "N/A"
        ),
    }


@app.get("/api/status/{status_code}")
async def status_code_endpoint(status_code: int, response: Response):
    if status_code == 200:
        return {"status": 200, "message": "OK"}
    elif status_code == 400:
        response.status_code = 400
        return {"status": 400, "message": "Bad Request"}
    elif status_code == 404:
        response.status_code = 404
        return {"status": 404, "message": "Not Found"}
    elif status_code == 500:
        response.status_code = 500
        return {"status": 500, "message": "Internal Server Error"}
    else:
        response.status_code = status_code
        return {"status": status_code, "message": f"Custom status {status_code}"}


@app.get("/api/devices")
async def list_devices():
    return {
        "devices": sorted(models.keys()),
        "total": len(models),
    }


@app.get("/api/devices/{device_id}/metrics")
async def get_metrics(device_id: str):
    if device_id not in models:
        raise HTTPException(404, f"Device '{device_id}' not found")
    return {
        "device_id": device_id,
        "metrics": device_metrics_map.get(device_id, {"mae": 0, "rmse": 0, "mape": 0, "r2": 0}),
    }


@app.post("/api/devices/{device_id}/forecast", response_model=dict)
async def forecast(device_id: str, request: ForecastRequest):
    if device_id not in models:
        raise HTTPException(404, f"Device '{device_id}' not found")

    model = models[device_id]
    agg = device_agg_data[device_id]

    # Determine current time and observation
    if request.current_observation is not None:
        obs = request.current_observation.model_dump()
        if request.observation_time:
            current_time = pd.Timestamp(request.observation_time)
        else:
            current_time = agg.index[-1] + pd.Timedelta("1min")
    else:
        current_time = agg.index[-1]
        obs = agg.loc[current_time].to_dict()

    # Build observation row
    obs_ts = pd.Timestamp(current_time) if not isinstance(current_time, pd.Timestamp) else current_time
    obs_row = pd.Series({
        **obs,
        "Hour": obs_ts.hour,
        "DayOfWeek": obs_ts.dayofweek,
        "IsWeekend": int(obs_ts.dayofweek >= 5),
        "IsHoliday": is_vietnam_holiday(pd.Series([obs_ts])).values[0],
    })
    obs_exog = compute_exog_row(obs_row)

    steps = request.horizon // 5
    forecast_start = obs_ts + pd.Timedelta("5min")
    future_index = pd.date_range(start=forecast_start, periods=steps, freq="5min")

    # Build future exogenous data
    X_future_list = []
    for f_t in future_index:
        if f_t in agg.index:
            row = agg.loc[f_t]
        else:
            row = obs_row
        X_future_list.append(compute_exog_row(row))
    X_future = pd.DataFrame(X_future_list, index=future_index)

    # Extend model with current observation then forecast
    try:
        extended = model.extend(
            endog=pd.Series([obs[TARGET_COL]], index=[obs_ts]),
            exog=pd.DataFrame([obs_exog], index=[obs_ts]),
        )
        forecast_result = extended.forecast(steps=steps, exog=X_future).clip(lower=0)
    except Exception as e:
        # Fallback: try get_prediction on the original model
        try:
            start_idx = len(agg) - agg.index.get_indexer([obs_ts], method="nearest")[0]
            pred = model.get_prediction(start=start_idx + 1, end=start_idx + steps, exog=X_future)
            forecast_result = pred.predicted_mean.clip(lower=0)
        except Exception as e2:
            raise HTTPException(500, f"Forecast failed: {e} / {e2}")

    forecast_result.index = future_index

    # Build response
    forecast_points = []
    for f_t, val in zip(future_index, forecast_result):
        forecast_points.append({
            "time": f_t.strftime("%Y-%m-%d %H:%M"),
            "predicted": float(val),
        })

    # History (last 120 minutes before current)
    history = agg.loc[:obs_ts].tail(120)
    history_points = [
        {"time": t.strftime("%Y-%m-%d %H:%M"), "actual": float(r[TARGET_COL])}
        for t, r in history.iterrows()
    ]

    return {
        "device_id": device_id,
        "forecast": forecast_points,
        "history": history_points,
        "metrics": device_metrics_map.get(device_id, {}),
        "generated_at": pd.Timestamp.now().isoformat(),
    }


@app.post("/api/nodes/{node_id}/forecast", response_model=NodeForecastResponse)
async def forecast_node(node_id: str, request: NodeForecastRequest):
    node_id = node_id.upper()
    print(request)
    if node_id not in models:
        raise HTTPException(404, f"Node '{node_id}' not found")

    model = models[node_id]
    agg = device_agg_data[node_id]

    # Determine current time and observation
    if request.CurrentObservation is not None:
        obs = request.CurrentObservation.model_dump()
        # Map Confidence to MeanConfidence
        if "Confidence" in obs:
            obs["MeanConfidence"] = obs.pop("Confidence")
        
        if request.ObservationTime:
            current_time = pd.Timestamp(request.ObservationTime)
        else:
            current_time = agg.index[-1] + pd.Timedelta("1min")
    else:
        current_time = agg.index[-1]
        obs = agg.loc[current_time].to_dict()

    # Fill in any missing columns using historical last row or defaults
    last_row = agg.iloc[-1]
    for col in BASE_TRAFFIC_FEATURES:
        if col not in obs:
            if col == "MeanConfidence" and "Confidence" in obs:
                obs["MeanConfidence"] = obs["Confidence"]
            elif col == "MedianSpeed" and "AvgSpeed" in obs:
                obs["MedianSpeed"] = obs["AvgSpeed"]
            elif col in last_row:
                obs[col] = last_row[col]
            else:
                obs[col] = 0.0

    # Build observation row (strip timezone to match naive agg.index)
    obs_ts = pd.Timestamp(current_time) if not isinstance(current_time, pd.Timestamp) else current_time
    if getattr(obs_ts, 'tz', None) is not None:
        obs_ts = obs_ts.tz_localize(None)
    obs_row = pd.Series({
        **obs,
        "Hour": obs_ts.hour,
        "DayOfWeek": obs_ts.dayofweek,
        "IsWeekend": int(obs_ts.dayofweek >= 5),
        "IsHoliday": is_vietnam_holiday(pd.Series([obs_ts])).values[0],
    })
    obs_exog = compute_exog_row(obs_row)

    steps = request.Horizon // 5
    forecast_start = obs_ts + pd.Timedelta("5min")
    future_index = pd.date_range(start=forecast_start, periods=steps, freq="5min")

    # Build future exogenous data
    X_future_list = []
    for f_t in future_index:
        if f_t in agg.index:
            row = agg.loc[f_t]
        else:
            row = obs_row
        X_future_list.append(compute_exog_row(row))
    X_future = pd.DataFrame(X_future_list, index=future_index)

    # Extend model then forecast
    try:
        extended = model.extend(
            endog=pd.Series([obs[TARGET_COL]], index=[obs_ts]),
            exog=pd.DataFrame([obs_exog], index=[obs_ts]),
        )
        forecast_result = extended.forecast(steps=steps, exog=X_future).clip(lower=0)
    except Exception as e:
        try:
            start_idx = len(agg) - agg.index.get_indexer([obs_ts], method="nearest")[0]
            pred = model.get_prediction(start=start_idx + 1, end=start_idx + steps, exog=X_future)
            forecast_result = pred.predicted_mean.clip(lower=0)
        except Exception as e2:
            raise HTTPException(500, f"Forecast failed: {e} / {e2}")

    forecast_result.index = future_index

    # Build all forecast points
    forecast_points = []
    for i, (f_t, pred_val) in enumerate(zip(future_index, forecast_result)):
        forecast_time = f_t
        predicted_val = float(pred_val)
        
        # Retrieve forecast exogenous values from database at that minute if available, else fall back to request inputs
        if forecast_time in agg.index:
            forecast_row = agg.loc[forecast_time]
            resp_num_vehicles = float(forecast_row["NumVehicles"]) if TARGET_COL != "NumVehicles" else predicted_val
            resp_avg_speed = float(forecast_row["AvgSpeed"]) if TARGET_COL != "AvgSpeed" else predicted_val
            resp_occupancy = float(forecast_row["Occupancy"])
            resp_avg_density = float(forecast_row["AvgDensity"])
            resp_avg_headway = float(forecast_row["AvgHeadway"])
            resp_flow_rate = float(forecast_row["FlowRate"])
            resp_confidence = float(forecast_row["MeanConfidence"])
        else:
            resp_num_vehicles = float(obs.get("NumVehicles", 0.0)) if TARGET_COL != "NumVehicles" else predicted_val
            resp_avg_speed = float(obs.get("AvgSpeed", 0.0)) if TARGET_COL != "AvgSpeed" else predicted_val
            resp_occupancy = float(obs.get("Occupancy", 0.0))
            resp_avg_density = float(obs.get("AvgDensity", 0.0))
            resp_avg_headway = float(obs.get("AvgHeadway", 0.0))
            resp_flow_rate = float(obs.get("FlowRate", 0.0))
            resp_confidence = float(obs.get("MeanConfidence", 0.0))

        forecast_points.append(NodeForecastPoint(
            Time=forecast_time,
            NumVehicles=round(resp_num_vehicles, 1),
            AvgSpeed=round(resp_avg_speed, 2),
            Occupancy=round(resp_occupancy, 2),
            AvgDensity=round(resp_avg_density, 2),
            AvgHeadway=round(resp_avg_headway, 2),
            FlowRate=round(resp_flow_rate, 2),
            Confidence=round(resp_confidence, 2)
        ))

    db_metrics = device_metrics_map.get(node_id, {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "r2": 0.0})
    metrics = {
        "Mae": round(db_metrics.get("mae", 0.0), 2),
        "Rmse": round(db_metrics.get("rmse", 0.0), 2),
        "Mape": round(db_metrics.get("mape", 0.0), 2),
        "R2": round(db_metrics.get("r2", 0.0), 3),
    }

    return NodeForecastResponse(
        Forecast=forecast_points,
        Metrics=metrics
    )


@app.post("/api/devices/{device_id}/extend", response_model=ExtendResponse)
async def extend_model(device_id: str, request: ExtendRequest):

    if device_id not in models:
        raise HTTPException(404, f"Device '{device_id}' not found")

    obs_ts = pd.Timestamp(request.timestamp)
    obs = request.observation.model_dump()
    obs_row = pd.Series({
        **obs,
        "Hour": obs_ts.hour,
        "DayOfWeek": obs_ts.dayofweek,
        "IsWeekend": int(obs_ts.dayofweek >= 5),
        "IsHoliday": is_vietnam_holiday(pd.Series([obs_ts])).values[0],
    })
    obs_exog = compute_exog_row(obs_row)

    try:
        extended = models[device_id].extend(
            endog=pd.Series([obs[TARGET_COL]], index=[obs_ts]),
            exog=pd.DataFrame([obs_exog], index=[obs_ts]),
        )
        models[device_id] = extended
    except Exception as e:
        raise HTTPException(500, f"Extend failed: {e}")

    return ExtendResponse(
        device_id=device_id,
        status="ok",
        model_updated_at=obs_ts.isoformat(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=True)
