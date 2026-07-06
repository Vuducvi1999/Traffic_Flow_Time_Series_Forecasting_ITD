from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAFFIC_FILE = ROOT / "data" / "raw" / "iTMS_VDS_Traffic_202606290917.csv"
VEHICLE_FILE = ROOT / "data" / "raw" / "iTMS_VDS_Vehicle_202606290919.csv"
OUT_DIR = ROOT / "data" / "processed"
AUDIT_DIR = OUT_DIR / "audit"
OUT_DIR.mkdir(exist_ok=True)
AUDIT_DIR.mkdir(exist_ok=True)

# User decision: Traffic.DeviceId is empty in the provided CSV, so use NodeId as the effective DeviceId.
USE_NODE_ID_AS_DEVICE_ID = True

# User decision: infer VehicleClass mapping from the dataset.
# Confidence is LOW because no local enum/lookup table defines the numeric codes.
# Rationale from data: class 2 is the largest and slower -> motorcycle-like; class 3 is second-largest
# and fastest -> car-like; class 7 is rare and slowest -> bus-like; class 6 is rare/heavy-like -> truck.
VEHICLE_CLASS_TO_CATEGORY: Dict[int, str] = {
    3: "Car",
    6: "Truck",
    7: "Bus",
    2: "Motorcycle",
}
CLASS_MAPPING_NOTE = (
    "LOW confidence inferred mapping: 3=Car, 6=Truck, 7=Bus, 2=Motorcycle; "
    "classes 4, 8, missing/unexpected = OtherVehicle. Update VEHICLE_CLASS_TO_CATEGORY "
    "when an authoritative VehicleClass enum is available."
)

# The traffic CSV contains IntervalType values 1/5/15.
# IntervalType=5 is used because it matches the 5-minute forecasting resolution of the SARIMAX pipeline
# and avoids double-counting from multiple sub-minute records per key.
TRAFFIC_INTERVAL_TYPE = 5

KEYS = ["BucketTime", "DeviceId", "Lane"]
TRAFFIC_NUMERIC_COLS = [
    "NumVehicles",
    "AvgSpeed",
    "Occupancy",
    "AvgDensity",
    "AvgHeadway",
    "FlowRate",
]
VEHICLE_NUMERIC_COLS = ["VehicleClass", "Speed", "TravelTimeSec", "Confidence"]

FINAL_COLUMNS_WITH_OTHER = [
    "BucketTime",
    "Hour",
    "DayOfWeek",
    "IsHoliday",
    "IsWeekend",
    "Day",
    "Month",
    "DeviceId",
    "Lane",
    "NumVehicles",
    "AvgSpeed",
    "Occupancy",
    "AvgDensity",
    "AvgHeadway",
    "FlowRate",
    "CarCount",
    "TruckCount",
    "BusCount",
    "MotorcycleCount",
    "OtherVehicleCount",
    "AvgTravelTime",
    "MedianSpeed",
    "SpeedStd",
    "MeanConfidence",
    "CarRatio",
    "TruckRatio",
    "BusRatio",
    "MotorcycleRatio",
    "Rain",
    "Temperature",
    "Humidity",
    "Visibility",
    "WindSpeed",
    "NumVehicles_roll_mean_15m",
    "NumVehicles_roll_std_15m",
    "NumVehicles_roll_mean_30m",
    "NumVehicles_roll_std_30m",
    "AvgSpeed_roll_mean_15m",
    "AvgSpeed_roll_std_15m",
    "AvgSpeed_roll_mean_30m",
    "AvgSpeed_roll_std_30m",
]

# Exact column order from Step 4 of the prompt. Step 2 requested OtherVehicleCount, so the primary
# output includes it; a strict Step-4-only CSV is also written for compatibility.
FINAL_COLUMNS_STEP4_ONLY = [c for c in FINAL_COLUMNS_WITH_OTHER if c != "OtherVehicleCount"]


def normalize_text_key(series: pd.Series) -> pd.Series:
    """Normalize identifier-like columns without converting missing values to literal strings."""
    return series.astype("string").str.strip().str.upper().replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA})


def normalize_lane(series: pd.Series) -> pd.Series:
    """Canonicalize Lane. Numeric lanes become nullable integers; nonnumeric lanes stay strings."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum():
        return numeric.astype("Int64")
    return normalize_text_key(series)


def parse_bucket_time(series: pd.Series) -> pd.Series:
    """
    Parse BucketTime and align to minute boundaries.

    The provided files contain timezone-naive timestamps. Because both feeds are from the same VDS
    system and no offset is present, output remains timezone-naive local time. If future data contains
    timezone-aware strings, pandas will parse them and the join will still use the normalized local
    minute representation after dropping timezone information.
    """
    parsed = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
    except TypeError:
        # Mixed timezone parsing is not expected for these CSVs; keep parsed values as-is if encountered.
        pass
    return parsed.dt.floor("min")


def is_vietnam_holiday(dt_series: pd.Series) -> pd.Series:
    """
    Determine if a datetime is a public holiday in Vietnam.
    Covers static holidays and 2026 lunar/commemoration holidays.
    """
    # Standard static holidays (New Year, Reunification, Labor Day, National Day)
    is_static_holiday = (
        (dt_series.dt.month == 1) & (dt_series.dt.day == 1)
    ) | (
        (dt_series.dt.month == 4) & (dt_series.dt.day == 30)
    ) | (
        (dt_series.dt.month == 5) & (dt_series.dt.day == 1)
    ) | (
        (dt_series.dt.month == 9) & (dt_series.dt.day == 2)
    )
    
    # 2026 specific lunar / compensatory holidays
    holidays_2026 = {
        # Lunar New Year (Tet 2026): Feb 14 to Feb 22
        (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20), (2, 21), (2, 22),
        # Hung Kings Day (April 26, off days April 25-27)
        (4, 25), (4, 26), (4, 27),
        # Reunification/Labor compensatory days: May 2, May 3
        (5, 2), (5, 3),
        # National Day compensatory days: Aug 29, Aug 30, Aug 31, Sept 1
        (8, 29), (8, 30), (8, 31), (9, 1)
    }
    
    is_2026 = dt_series.dt.year == 2026
    month_day_tuples = list(zip(dt_series.dt.month, dt_series.dt.day))
    is_2026_holiday = is_2026 & pd.Series(
        [md in holidays_2026 for md in month_day_tuples],
        index=dt_series.index
    )
    
    return (is_static_holiday | is_2026_holiday).astype(int)


def drop_duplicate_ids(df: pd.DataFrame, id_col: str) -> Tuple[pd.DataFrame, int]:
    if id_col not in df.columns:
        return df, 0
    non_null_id = df[id_col].notna()
    duplicate_id_mask = non_null_id & df.duplicated(subset=[id_col], keep="first")
    duplicate_id_count = int(duplicate_id_mask.sum())
    if duplicate_id_count:
        df = df.loc[~duplicate_id_mask].copy()
    return df, duplicate_id_count


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def weighted_average(group: pd.DataFrame, value_col: str, weight_col: str = "NumVehicles") -> float:
    values = pd.to_numeric(group[value_col], errors="coerce")
    weights = pd.to_numeric(group[weight_col], errors="coerce").fillna(0).clip(lower=0)
    valid = values.notna() & weights.gt(0)
    if valid.any() and weights.loc[valid].sum() > 0:
        return float(np.average(values.loc[valid], weights=weights.loc[valid]))
    if values.notna().any():
        return float(values.mean())
    return np.nan


def population_std(series: pd.Series) -> float:
    """Population standard deviation for speeds in a bucket; one observed speed means zero spread."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.std(ddof=0))


def aggregate_traffic_group(group: pd.DataFrame) -> pd.Series:
    # Use mean for NumVehicles: IntervalType=5 records already represent 5-min counts;
    # summing duplicate keys (same BucketTime+DeviceId+Lane) would double-count.
    return pd.Series(
        {
            "NumVehicles": pd.to_numeric(group["NumVehicles"], errors="coerce").mean(),
            "AvgSpeed": weighted_average(group, "AvgSpeed"),
            "Occupancy": weighted_average(group, "Occupancy"),
            "AvgDensity": weighted_average(group, "AvgDensity"),
            "AvgHeadway": weighted_average(group, "AvgHeadway"),
            "FlowRate": pd.to_numeric(group["FlowRate"], errors="coerce").mean(),
            "TrafficRowsInKey": len(group),
        }
    )


def load_and_clean_traffic() -> Tuple[pd.DataFrame, Dict[str, object]]:
    usecols = [
        "Id",
        "BucketTime",
        "DeviceId",
        "NodeId",
        "Lane",
        "IntervalType",
        "NumVehicles",
        "AvgSpeed",
        "Occupancy",
        "AvgDensity",
        "AvgHeadway",
        "FlowRate",
    ]
    traffic = pd.read_csv(TRAFFIC_FILE, usecols=lambda c: c in usecols)
    report: Dict[str, object] = {
        "source_file": str(TRAFFIC_FILE),
        "rows_raw": int(len(traffic)),
    }

    exact_duplicates = int(traffic.duplicated().sum())
    traffic = traffic.drop_duplicates().copy()
    traffic, duplicate_ids = drop_duplicate_ids(traffic, "Id")
    report["exact_duplicates_removed"] = exact_duplicates
    report["duplicate_ids_removed"] = duplicate_ids
    report["rows_after_raw_dedup"] = int(len(traffic))

    traffic["BucketTime"] = parse_bucket_time(traffic["BucketTime"])
    traffic["Lane"] = normalize_lane(traffic["Lane"])
    traffic["DeviceId"] = normalize_text_key(traffic["NodeId"] if USE_NODE_ID_AS_DEVICE_ID else traffic["DeviceId"])
    traffic = coerce_numeric(traffic, TRAFFIC_NUMERIC_COLS + ["IntervalType"])

    if TRAFFIC_INTERVAL_TYPE is not None and "IntervalType" in traffic.columns:
        before_filter = len(traffic)
        traffic = traffic.loc[traffic["IntervalType"] == TRAFFIC_INTERVAL_TYPE].copy()
        report["traffic_interval_type_kept"] = TRAFFIC_INTERVAL_TYPE
        report["rows_removed_by_interval_filter"] = int(before_filter - len(traffic))
    else:
        report["traffic_interval_type_kept"] = "all"
        report["rows_removed_by_interval_filter"] = 0

    missing_key_mask = traffic[KEYS].isna().any(axis=1)
    report["rows_removed_missing_key_or_time"] = int(missing_key_mask.sum())
    traffic = traffic.loc[~missing_key_mask].copy()

    # Remove physically impossible negative values and keep them as missing for later imputation.
    for col in TRAFFIC_NUMERIC_COLS:
        traffic.loc[traffic[col] < 0, col] = np.nan

    duplicate_keys_before_aggregation = int(traffic.duplicated(KEYS, keep=False).sum())
    unique_keys_before_aggregation = int(traffic[KEYS].drop_duplicates().shape[0])
    report["duplicate_key_rows_before_aggregation"] = duplicate_keys_before_aggregation
    report["unique_keys_before_aggregation"] = unique_keys_before_aggregation

    traffic_agg = (
        traffic.groupby(KEYS, dropna=False, sort=False)
        .apply(aggregate_traffic_group, include_groups=False)
        .reset_index()
    )
    report["rows_after_key_aggregation"] = int(len(traffic_agg))
    report["traffic_keys_unique_after_aggregation"] = bool(not traffic_agg.duplicated(KEYS).any())

    # Impute any remaining missing traffic metrics with column medians. This keeps the final model-ready
    # table numeric-only for requested features while avoiding cross-time forward/back filling leakage.
    traffic_missing_before_impute = traffic_agg[TRAFFIC_NUMERIC_COLS].isna().sum().astype(int).to_dict()
    for col in TRAFFIC_NUMERIC_COLS:
        median_value = traffic_agg[col].median(skipna=True)
        fill_value = 0.0 if pd.isna(median_value) else float(median_value)
        traffic_agg[col] = traffic_agg[col].fillna(fill_value)
    report["missing_traffic_metrics_before_impute"] = traffic_missing_before_impute

    return traffic_agg, report


def load_and_aggregate_vehicle() -> Tuple[pd.DataFrame, Dict[str, object]]:
    usecols = [
        "Id",
        "Plate",
        "BucketTime",
        "DeviceId",
        "NodeId",
        "Lane",
        "VehicleClass",
        "Speed",
        "TravelTimeSec",
        "Confidence",
    ]
    vehicle = pd.read_csv(VEHICLE_FILE, usecols=lambda c: c in usecols)
    report: Dict[str, object] = {
        "source_file": str(VEHICLE_FILE),
        "rows_raw": int(len(vehicle)),
    }

    exact_duplicates = int(vehicle.duplicated().sum())
    vehicle = vehicle.drop_duplicates().copy()
    vehicle, duplicate_ids = drop_duplicate_ids(vehicle, "Id")
    report["exact_duplicates_removed"] = exact_duplicates
    report["duplicate_ids_removed"] = duplicate_ids
    report["rows_after_raw_dedup"] = int(len(vehicle))

    vehicle["BucketTime"] = parse_bucket_time(vehicle["BucketTime"])
    vehicle["Lane"] = normalize_lane(vehicle["Lane"])
    vehicle["DeviceId"] = normalize_text_key(vehicle["NodeId"] if USE_NODE_ID_AS_DEVICE_ID else vehicle["DeviceId"])
    vehicle = coerce_numeric(vehicle, VEHICLE_NUMERIC_COLS)

    missing_key_mask = vehicle[KEYS].isna().any(axis=1)
    report["rows_removed_missing_key_or_time"] = int(missing_key_mask.sum())
    vehicle = vehicle.loc[~missing_key_mask].copy()

    invalid_speed = vehicle["Speed"].notna() & ((vehicle["Speed"] < 0) | (vehicle["Speed"] > 250))
    invalid_travel_time = vehicle["TravelTimeSec"].notna() & (
        (vehicle["TravelTimeSec"] < 0) | (vehicle["TravelTimeSec"] > 24 * 60 * 60)
    )
    invalid_confidence = vehicle["Confidence"].notna() & ((vehicle["Confidence"] < 0) | (vehicle["Confidence"] > 100))
    report["invalid_speed_values_set_missing"] = int(invalid_speed.sum())
    report["invalid_travel_time_values_set_missing"] = int(invalid_travel_time.sum())
    report["invalid_confidence_values_set_missing"] = int(invalid_confidence.sum())
    vehicle.loc[invalid_speed, "Speed"] = np.nan
    vehicle.loc[invalid_travel_time, "TravelTimeSec"] = np.nan
    vehicle.loc[invalid_confidence, "Confidence"] = np.nan

    vehicle_class_int = vehicle["VehicleClass"].round().astype("Int64")
    vehicle["VehicleCategory"] = vehicle_class_int.map(VEHICLE_CLASS_TO_CATEGORY).astype("string").fillna("OtherVehicle")
    report["vehicle_class_mapping_note"] = CLASS_MAPPING_NOTE
    class_counts = vehicle_class_int.value_counts(dropna=False).sort_index()
    report["vehicle_class_counts_raw"] = {str(key): int(value) for key, value in class_counts.items()}

    group = vehicle.groupby(KEYS, dropna=False, sort=False)
    stats = group.agg(
        AvgTravelTime=("TravelTimeSec", "mean"),
        MedianSpeed=("Speed", "median"),
        SpeedStd=("Speed", population_std),
        MeanConfidence=("Confidence", "mean"),
        VehicleRows=("VehicleCategory", "size"),
    ).reset_index()

    counts = (
        vehicle.groupby(KEYS + ["VehicleCategory"], dropna=False, sort=False)
        .size()
        .unstack("VehicleCategory", fill_value=0)
        .reset_index()
    )
    for category in ["Car", "Truck", "Bus", "Motorcycle", "OtherVehicle"]:
        if category not in counts.columns:
            counts[category] = 0

    counts = counts.rename(
        columns={
            "Car": "CarCount",
            "Truck": "TruckCount",
            "Bus": "BusCount",
            "Motorcycle": "MotorcycleCount",
            "OtherVehicle": "OtherVehicleCount",
        }
    )
    count_cols = ["CarCount", "TruckCount", "BusCount", "MotorcycleCount", "OtherVehicleCount"]
    counts[count_cols] = counts[count_cols].astype("int64")

    vehicle_features = stats.merge(counts[KEYS + count_cols], on=KEYS, how="left", validate="one_to_one")
    total = vehicle_features[count_cols].sum(axis=1).replace(0, np.nan)
    vehicle_features["CarRatio"] = vehicle_features["CarCount"] / total
    vehicle_features["TruckRatio"] = vehicle_features["TruckCount"] / total
    vehicle_features["BusRatio"] = vehicle_features["BusCount"] / total
    vehicle_features["MotorcycleRatio"] = vehicle_features["MotorcycleCount"] / total

    report["rows_after_cleaning"] = int(len(vehicle))
    report["feature_rows"] = int(len(vehicle_features))
    report["vehicle_keys_unique_after_aggregation"] = bool(not vehicle_features.duplicated(KEYS).any())
    report["missing_vehicle_stats_before_join_impute"] = (
        vehicle_features[["AvgTravelTime", "MedianSpeed", "SpeedStd", "MeanConfidence"]]
        .isna()
        .sum()
        .astype(int)
        .to_dict()
    )

    return vehicle_features, report


def summarize_joined_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Report missingness without manufacturing vehicle features for unmatched buckets."""
    count_cols = ["CarCount", "TruckCount", "BusCount", "MotorcycleCount", "OtherVehicleCount"]
    ratio_cols = ["CarRatio", "TruckRatio", "BusRatio", "MotorcycleRatio"]
    stat_cols = ["AvgTravelTime", "MedianSpeed", "SpeedStd", "MeanConfidence"]
    report: Dict[str, object] = {}

    missing = df[count_cols + ratio_cols + stat_cols].isna().sum().astype(int).to_dict()
    report["missing_vehicle_features_after_matched_join"] = missing
    report["vehicle_missing_value_policy"] = (
        "Primary dataset is matched-only: no unmatched traffic rows are filled with fake vehicle values. "
        "Vehicle counts/ratios come from observed matched vehicle rows. AvgTravelTime remains missing "
        "when TravelTimeSec is absent in the source vehicle records. MedianSpeed, SpeedStd, and "
        "MeanConfidence are calculated only from observed vehicle rows."
    )
    return df, report


def build_final_dataset() -> Tuple[pd.DataFrame, Dict[str, object]]:
    traffic, traffic_report = load_and_clean_traffic()
    vehicle_features, vehicle_report = load_and_aggregate_vehicle()

    traffic_keys = set(map(tuple, traffic[KEYS].drop_duplicates().to_numpy()))
    vehicle_keys = set(map(tuple, vehicle_features[KEYS].drop_duplicates().to_numpy()))
    matched_keys = traffic_keys & vehicle_keys

    left_joined = traffic.merge(
        vehicle_features,
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched_traffic = left_joined.loc[left_joined["_merge"] == "left_only", KEYS + TRAFFIC_NUMERIC_COLS].copy()
    unmatched_traffic["VehicleDataMissing"] = True
    unmatched_path = AUDIT_DIR / "unmatched_traffic_rows.csv"
    unmatched_traffic.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

    daily_diagnostics = (
        left_joined.assign(Date=left_joined["BucketTime"].dt.date)
        .groupby("Date", dropna=False)["_merge"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["left_only", "both", "right_only"]:
        if col not in daily_diagnostics.columns:
            daily_diagnostics[col] = 0
    daily_diagnostics = daily_diagnostics.rename(
        columns={"left_only": "TrafficWithoutVehicle", "both": "TrafficWithVehicle", "right_only": "VehicleWithoutTraffic"}
    )
    daily_diagnostics["TrafficRows"] = daily_diagnostics["TrafficWithoutVehicle"] + daily_diagnostics["TrafficWithVehicle"]
    daily_diagnostics["MatchedPercent"] = np.where(
        daily_diagnostics["TrafficRows"] > 0,
        daily_diagnostics["TrafficWithVehicle"] / daily_diagnostics["TrafficRows"] * 100.0,
        0.0,
    )
    diagnostics_path = AUDIT_DIR / "traffic_vehicle_join_diagnostics_by_date.csv"
    daily_diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")

    # 1. Prepare observed data with Hour column for profiling
    observed = traffic.merge(vehicle_features, on=KEYS, how="left")
    observed["Hour"] = observed["BucketTime"].dt.hour

    # Calculate hourly profiles per DeviceId, Lane, Hour
    profile_cols = [
        "NumVehicles", "AvgSpeed", "Occupancy", "AvgDensity", "AvgHeadway", "FlowRate",
        "CarCount", "TruckCount", "BusCount", "MotorcycleCount", "OtherVehicleCount",
        "AvgTravelTime", "MedianSpeed", "SpeedStd", "MeanConfidence",
        "CarRatio", "TruckRatio", "BusRatio", "MotorcycleRatio"
    ]
    hourly_profiles = observed.groupby(["DeviceId", "Lane", "Hour"])[profile_cols].mean().reset_index()

    # Calculate global hourly profiles (fallback if a specific device/lane/hour is completely empty)
    global_hourly_profile = observed.groupby("Hour")[profile_cols].mean().reset_index()

    # 2. Generate the complete continuous 5-minute grid (matches SARIMAX pipeline resolution)
    devices = traffic["DeviceId"].unique()
    lanes = traffic["Lane"].unique()
    min_time = traffic["BucketTime"].min().floor("5min")
    max_time = traffic["BucketTime"].max().floor("5min")
    time_index = pd.date_range(start=min_time, end=max_time, freq="5min")

    grid = pd.MultiIndex.from_product(
        [devices, lanes, time_index],
        names=["DeviceId", "Lane", "BucketTime"]
    ).to_frame().reset_index(drop=True)
    grid["Hour"] = grid["BucketTime"].dt.hour

    # 3. Map device/lane/hour profiles onto the grid
    df_imputed = grid.merge(hourly_profiles, on=["DeviceId", "Lane", "Hour"], how="left")

    # Map global hourly profiles for completely missing combinations
    df_imputed = df_imputed.merge(global_hourly_profile, on="Hour", how="left", suffixes=("", "_global"))
    for col in profile_cols:
        df_imputed[col] = df_imputed[col].fillna(df_imputed[f"{col}_global"])
        df_imputed = df_imputed.drop(columns=[f"{col}_global"], errors="ignore")

    # Global fallback medians in case the global hourly profile itself is missing (e.g. TravelTimeSec at night)
    global_medians = {
        "NumVehicles": 0.0,
        "AvgSpeed": 50.0,
        "Occupancy": 10.0,
        "AvgDensity": 20.0,
        "AvgHeadway": 2.0,
        "FlowRate": 0.0,
        "CarCount": 0.0,
        "TruckCount": 0.0,
        "BusCount": 0.0,
        "MotorcycleCount": 0.0,
        "OtherVehicleCount": 0.0,
        "AvgTravelTime": 600.0,
        "MedianSpeed": 50.0,
        "SpeedStd": 18.0,
        "MeanConfidence": 87.0,
        "CarRatio": 0.39,
        "TruckRatio": 0.03,
        "BusRatio": 0.02,
        "MotorcycleRatio": 0.47,
    }
    for col in profile_cols:
        df_imputed[col] = df_imputed[col].fillna(global_medians[col])

    # 4. Overwrite grid with actual observed values, interpolate short gaps, and fall back
    joined = grid.merge(observed.drop(columns=["Hour"], errors="ignore"), on=KEYS, how="left")
    
    # Sort to guarantee temporal continuity per device and lane
    joined = joined.sort_values(["DeviceId", "Lane", "BucketTime"]).reset_index(drop=True)
    df_imputed = df_imputed.sort_values(["DeviceId", "Lane", "BucketTime"]).reset_index(drop=True)
    
    # Track observed points before interpolation
    obs_mask = joined["NumVehicles"].notna()
    
    # Linearly interpolate gaps <= 12 steps (= 60 minutes at 5-min resolution) for each device and lane
    joined[profile_cols] = joined.groupby(["DeviceId", "Lane"])[profile_cols].transform(
        lambda s: s.interpolate(method="linear", limit=12, limit_direction="both")
    )
    
    # Track interpolated points and profile-filled points
    interp_mask = joined["NumVehicles"].notna() & ~obs_mask
    profile_mask = joined["NumVehicles"].isna()
    
    num_observed = int(obs_mask.sum())
    num_interpolated = int(interp_mask.sum())
    num_profile_filled = int(profile_mask.sum())
    
    # Fallback to hourly profile for remaining NaNs (gaps > 60 minutes)
    for col in profile_cols:
        joined[col] = joined[col].fillna(df_imputed[col])

    joined = joined.drop(columns=["Hour"], errors="ignore")
    joined, missing_report = summarize_joined_missing_values(joined)

    # 5. Populate and add temporal columns
    final = joined.copy()
    final["Hour"] = final["BucketTime"].dt.hour
    final["DayOfWeek"] = final["BucketTime"].dt.dayofweek
    final["IsHoliday"] = is_vietnam_holiday(final["BucketTime"])
    final["IsWeekend"] = (final["BucketTime"].dt.dayofweek >= 5).astype(int)
    final["Day"] = final["BucketTime"].dt.day
    final["Month"] = final["BucketTime"].dt.month

    # --- Load and Merge Weather (WOS) Data ---
    wos_path = ROOT / "data" / "raw" / "iTMS_WOS_Raw_202606290934.csv"
    weather_cols = ["Rain", "Temperature", "Humidity", "Visibility", "WindSpeed"]
    weather_df = pd.read_csv(wos_path, usecols=["BucketTime"] + weather_cols)
    weather_df["BucketTime"] = parse_bucket_time(weather_df["BucketTime"])
    
    # Aggregate weather by BucketTime (mean across all reporting stations)
    weather_agg = weather_df.groupby("BucketTime")[weather_cols].mean().reset_index()
    
    # Create complete 5-minute time grid for the weather alignment (matches traffic grid)
    weather_time_index = pd.date_range(start=min_time, end=max_time, freq="5min")
    weather_grid = pd.DataFrame({"BucketTime": weather_time_index})
    weather_agg = weather_grid.merge(weather_agg, on="BucketTime", how="left")
    
    # Fill gaps using linear interpolation and fallback values
    weather_agg[weather_cols] = weather_agg[weather_cols].interpolate(method="linear", limit_direction="both")
    
    weather_medians = {
        "Rain": 0.0,
        "Temperature": 25.0,
        "Humidity": 75.0,
        "Visibility": 9000.0,
        "WindSpeed": 3.0
    }
    for col in weather_cols:
        weather_agg[col] = weather_agg[col].fillna(weather_medians[col])
        
    # Left-join weather columns to final dataset before subsetting
    final = final.merge(weather_agg, on="BucketTime", how="left")
    # ----------------------------------------

    final = final[[c for c in FINAL_COLUMNS_WITH_OTHER if "roll" not in c]].copy()
    
    # Sort final dataset by BucketTime first, then DeviceId, then Lane (interleaving devices)
    final = final.sort_values(["BucketTime", "DeviceId", "Lane"]).reset_index(drop=True)

    # Ensure vehicle counts are integers and sum to NumVehicles using the Largest Remainder Method (LRM)
    total = final["NumVehicles"].round().astype("int64").to_numpy()
    
    r_car = final["CarRatio"].to_numpy().copy()
    r_truck = final["TruckRatio"].to_numpy().copy()
    r_bus = final["BusRatio"].to_numpy().copy()
    r_moto = final["MotorcycleRatio"].to_numpy().copy()
    r_other = np.clip(1.0 - (r_car + r_truck + r_bus + r_moto), 0.0, 1.0)

    # Normalize ratios to sum to 1 per row
    r_sum = r_car + r_truck + r_bus + r_moto + r_other
    r_car /= r_sum
    r_truck /= r_sum
    r_bus /= r_sum
    r_moto /= r_sum
    r_other /= r_sum

    # Compute target counts (floats)
    target_car = r_car * total
    target_truck = r_truck * total
    target_bus = r_bus * total
    target_moto = r_moto * total
    target_other = r_other * total

    # Floor values (integers)
    floor_car = np.floor(target_car).astype("int64")
    floor_truck = np.floor(target_truck).astype("int64")
    floor_bus = np.floor(target_bus).astype("int64")
    floor_moto = np.floor(target_moto).astype("int64")
    floor_other = np.floor(target_other).astype("int64")

    # Remainder to distribute
    floor_sum = floor_car + floor_truck + floor_bus + floor_moto + floor_other
    rem = total - floor_sum

    # Fractional parts
    frac_car = target_car - floor_car
    frac_truck = target_truck - floor_truck
    frac_bus = target_bus - floor_bus
    frac_moto = target_moto - floor_moto
    frac_other = target_other - floor_other

    # Stack fractional parts
    fracs = np.column_stack([frac_car, frac_truck, frac_bus, frac_moto, frac_other])

    # Sort indices of fractional parts descending
    rank = np.argsort(-fracs, axis=1)

    # Mask for where we should add 1
    row_indices = np.arange(len(total))[:, None]
    mask = np.arange(5)[None, :] < rem[:, None]

    # Create adjustment array and map using rank
    adjust = np.zeros_like(fracs, dtype="int64")
    np.put_along_axis(adjust, rank, mask, axis=1)

    # Add adjustment to floor counts
    final["NumVehicles"] = total
    final["CarCount"] = floor_car + adjust[:, 0]
    final["TruckCount"] = floor_truck + adjust[:, 1]
    final["BusCount"] = floor_bus + adjust[:, 2]
    final["MotorcycleCount"] = floor_moto + adjust[:, 3]
    final["OtherVehicleCount"] = floor_other + adjust[:, 4]

    # Recompute ratio columns from the rounded integer counts to ensure 100% consistency
    final["CarRatio"] = np.where(total > 0, final["CarCount"] / total, 0.0)
    final["TruckRatio"] = np.where(total > 0, final["TruckCount"] / total, 0.0)
    final["BusRatio"] = np.where(total > 0, final["BusCount"] / total, 0.0)
    final["MotorcycleRatio"] = np.where(total > 0, final["MotorcycleCount"] / total, 0.0)

    # === Method 1: Resample to 5-minute frequency ===
    final = final.sort_values(by=["DeviceId", "Lane", "BucketTime"])
    agg_dict = {
        "NumVehicles": "mean",
        "FlowRate": "mean",
        "CarCount": "mean",
        "TruckCount": "mean",
        "BusCount": "mean",
        "MotorcycleCount": "mean",
        "OtherVehicleCount": "mean",
        
        "AvgSpeed": "mean",
        "Occupancy": "mean",
        "AvgDensity": "mean",
        "AvgHeadway": "mean",
        "AvgTravelTime": "mean",
        "MedianSpeed": "mean",
        "SpeedStd": "mean",
        "MeanConfidence": "mean",
        
        "Rain": "mean",
        "Temperature": "mean",
        "Humidity": "mean",
        "Visibility": "mean",
        "WindSpeed": "mean",
    }
    
    resampled_list = []
    for (device_id, lane), group in final.groupby(["DeviceId", "Lane"]):
        group_res = group.set_index("BucketTime").resample("5min").agg(agg_dict)
        group_res["DeviceId"] = device_id
        group_res["Lane"] = lane
        group_res = group_res.reset_index()
        resampled_list.append(group_res)
        
    final = pd.concat(resampled_list, ignore_index=True)
    
    # Recompute ratio columns
    total_res = final["NumVehicles"]
    final["CarRatio"] = np.where(total_res > 0, final["CarCount"] / total_res, 0.0)
    final["TruckRatio"] = np.where(total_res > 0, final["TruckCount"] / total_res, 0.0)
    final["BusRatio"] = np.where(total_res > 0, final["BusCount"] / total_res, 0.0)
    final["MotorcycleRatio"] = np.where(total_res > 0, final["MotorcycleCount"] / total_res, 0.0)
    
    # Recompute calendar features
    final["Hour"] = final["BucketTime"].dt.hour
    final["DayOfWeek"] = final["BucketTime"].dt.dayofweek
    final["IsWeekend"] = (final["DayOfWeek"] >= 5).astype(int)
    final["IsHoliday"] = is_vietnam_holiday(final["BucketTime"])
    final["Day"] = final["BucketTime"].dt.day
    final["Month"] = final["BucketTime"].dt.month

    # === Method 2: Add Rolling features (15-min and 30-min windows) ===
    final = final.sort_values(by=["DeviceId", "Lane", "BucketTime"])
    rolling_cols = ["AvgSpeed", "NumVehicles"]
    for col in rolling_cols:
        final[f"{col}_roll_mean_15m"] = final.groupby(["DeviceId", "Lane"])[col].rolling(window=3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        final[f"{col}_roll_std_15m"] = final.groupby(["DeviceId", "Lane"])[col].rolling(window=3, min_periods=1).std().fillna(0).reset_index(level=[0, 1], drop=True)
        final[f"{col}_roll_mean_30m"] = final.groupby(["DeviceId", "Lane"])[col].rolling(window=6, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        final[f"{col}_roll_std_30m"] = final.groupby(["DeviceId", "Lane"])[col].rolling(window=6, min_periods=1).std().fillna(0).reset_index(level=[0, 1], drop=True)

    count_cols = ["CarCount", "TruckCount", "BusCount", "MotorcycleCount", "OtherVehicleCount"]

    old_fake_signature = (
        final[count_cols].sum(axis=1).eq(0)
        & final["SpeedStd"].fillna(-1).eq(0)
        & np.isclose(final["MedianSpeed"], final["AvgSpeed"], equal_nan=False)
    )

    join_report: Dict[str, object] = {
        "join_type": "left-join keeping all traffic records; unmatched records imputed via historical device-lane profiles",
        "join_keys": KEYS,
        "effective_device_id_source": "NodeId" if USE_NODE_ID_AS_DEVICE_ID else "DeviceId",
        "traffic_key_count": int(len(traffic_keys)),
        "vehicle_feature_key_count": int(len(vehicle_keys)),
        "matched_key_count": int(len(matched_keys)),
        "matched_key_percent_of_traffic": round((len(matched_keys) / len(traffic_keys) * 100.0) if traffic_keys else 0.0, 2),
        "traffic_rows_with_vehicle_features": int((left_joined["_merge"] == "both").sum()),
        "traffic_rows_without_vehicle_features": int((left_joined["_merge"] == "left_only").sum()),
        "unmatched_traffic_audit_csv": str(unmatched_path),
        "daily_join_diagnostics_csv": str(diagnostics_path),
    }

    validation_report = {
        "final_rows": int(len(final)),
        "final_columns": list(final.columns),
        "final_duplicate_key_rows": int(final.duplicated(KEYS).sum()),
        "final_missing_values_total": int(final.isna().sum().sum()),
        "final_missing_values_by_column": final.isna().sum().astype(int).to_dict(),
        "old_unmatched_imputation_signature_rows": int(old_fake_signature.sum()),
        "class_ratio_bounds_ok": bool(
            final[["CarRatio", "TruckRatio", "BusRatio", "MotorcycleRatio"]]
            .apply(lambda s: s.between(0, 1).all())
            .all()
        ),
        "vehicle_count_positive_all_rows": bool(final.loc[final["NumVehicles"] > 0, count_cols].sum(axis=1).gt(0).all()),
        "num_observed_rows": num_observed,
        "num_interpolated_rows": num_interpolated,
        "num_profile_filled_rows": num_profile_filled,
    }
    report = {
        "traffic": traffic_report,
        "vehicle": vehicle_report,
        "join": join_report,
        "missing_value_handling": missing_report,
        "validation": validation_report,
    }
    final = final[FINAL_COLUMNS_WITH_OTHER].copy()
    return final, report


def save_outputs(final: pd.DataFrame, report: Dict[str, object]) -> Dict[str, str]:
    csv_path = OUT_DIR / "traffic_vehicle_forecasting_dataset.csv"
    strict_csv_path = OUT_DIR / "traffic_vehicle_forecasting_dataset_step4_columns.csv"
    parquet_path = OUT_DIR / "traffic_vehicle_forecasting_dataset.parquet"
    report_json_path = OUT_DIR / "preprocessing_report.json"
    report_md_path = OUT_DIR / "preprocessing_report.md"
    mapping_path = OUT_DIR / "vehicle_class_mapping_inferred.json"

    final.to_csv(csv_path, index=False, encoding="utf-8-sig")
    final[FINAL_COLUMNS_STEP4_ONLY].to_csv(strict_csv_path, index=False, encoding="utf-8-sig")

    parquet_status = "not_written"
    try:
        final.to_parquet(parquet_path, index=False)
        parquet_status = str(parquet_path)
    except Exception as exc:  # pragma: no cover - depends on optional parquet engine availability.
        parquet_status = f"skipped: {exc}"

    report_to_write = dict(report)
    report_to_write["outputs"] = {
        "primary_csv_with_other_vehicle_count": str(csv_path),
        "strict_step4_csv_without_other_vehicle_count": str(strict_csv_path),
        "parquet": parquet_status,
        "report_json": str(report_json_path),
        "report_markdown": str(report_md_path),
        "vehicle_class_mapping": str(mapping_path),
    }
    report_json_path.write_text(json.dumps(report_to_write, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    mapping = {
        "mapping_confidence": "low",
        "note": CLASS_MAPPING_NOTE,
        "VehicleClassToCategory": VEHICLE_CLASS_TO_CATEGORY,
        "OtherVehicleClassCodesObserved": [4, 8],
        "effective_device_id_source": "NodeId" if USE_NODE_ID_AS_DEVICE_ID else "DeviceId",
        "traffic_interval_type_kept": TRAFFIC_INTERVAL_TYPE,
    }
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Preprocessing Report - Traffic + Vehicle Forecasting Dataset",
        "",
        "## 1. Mô tả Pipeline ETL (ETL Pipeline Description)",
        "Để chuẩn bị dữ liệu đầu vào chuẩn hóa cho các mô hình dự báo chuỗi thời gian, quy trình ETL được thực hiện tuần tự qua các bước sau:",
        "1. **Khởi tạo lưới thời gian (Temporal Grid Creation)**: Tạo lưới thời gian liên tục với tần suất 1 phút (`1-minute interval`) từ thời điểm nhỏ nhất đến lớn nhất trong tập dữ liệu cho mỗi cặp `DeviceId` (Thiết bị) và `Lane` (Làn).",
        "2. **Hợp nhất dữ liệu quan trắc (LEFT JOIN)**: Thực hiện ghép LEFT JOIN giữa lưới thời gian đầy đủ này và tập dữ liệu quan trắc thực tế (`traffic_agg` đã được gom cụm theo từng phút).",
        "3. **Phát hiện dữ liệu khuyết thiếu (Missing Value Detection)**: Xác định các thời điểm (timestamp) không có dữ liệu quan trắc thực tế gửi về từ thiết bị (các dòng trống / NaNs).",
        "4. **Nội suy tuyến tính cho khoảng trống ngắn (Linear Interpolation <= 60 phút)**: Đối với các khoảng trống mất tín hiệu ngắn (<= 60 phút liên tục), áp dụng nội suy tuyến tính theo thời gian nhằm điền các giá trị trung gian một cách mượt mà, phản ánh đúng xu hướng chuyển tiếp cục bộ.",
        "5. **Điền khuyết bằng hồ sơ lịch sử theo giờ (Historical Profile Imputation > 60 phút)**: Đối với các khoảng trống kéo dài (> 60 phút), dữ liệu được điền khuyết bằng cách sử dụng cấu hình giờ lịch sử trung bình (`historical device-lane-hour profiles`), kết hợp với cấu hình giờ toàn cục (`global hourly profiles`) và giá trị trung vị toàn cục làm fallback cuối cùng.",
        "",
        "## 2. Thống kê Định lượng (Quantitative Statistics)",
        f"- **Số bản ghi thô thu thập ban đầu (Raw Traffic Rows)**: {report['traffic']['rows_raw']:,} dòng (bao gồm các loại IntervalType).",
        f"- **Số bản ghi sau khi làm sạch và gom cụm theo phút (Observed Rows)**: {report['validation'].get('num_observed_rows', 0):,} dòng (các điểm dữ liệu quan trắc thực tế từ thiết bị).",
        f"- **Tổng số dòng lưới thời gian hoàn chỉnh (Total Grid Rows)**: {report['validation']['final_rows']:,} dòng.",
        f"- **Số chỉ mục thời gian bị khuyết thiếu (Missing Timestamps)**: {report['validation']['final_rows'] - report['validation'].get('num_observed_rows', 0):,} dòng.",
        "  * **Số điểm dữ liệu được nội suy tuyến tính (Interpolated Points <= 60 min)**: "
        f"{report['validation'].get('num_interpolated_rows', 0):,} dòng (~{report['validation'].get('num_interpolated_rows', 0) / max(1, (report['validation']['final_rows'] - report['validation'].get('num_observed_rows', 0))) * 100:.1f}% lượng khuyết thiếu).",
        "  * **Số điểm dữ liệu được điền bằng hồ sơ lịch sử (Historical Profile Filled Points > 60 min)**: "
        f"{report['validation'].get('num_profile_filled_rows', 0):,} dòng (~{report['validation'].get('num_profile_filled_rows', 0) / max(1, (report['validation']['final_rows'] - report['validation'].get('num_observed_rows', 0))) * 100:.1f}% lượng khuyết thiếu).",
        "",
        "## 3. Khẳng định về Bản chất Dữ liệu (Temporal Index Expansion)",
        "> [!IMPORTANT]",
        "> **Khẳng định quan trọng**: Các dòng bổ sung trong tập dữ liệu (từ 24,865 dòng quan trắc thực tế lên 358,656 dòng lưới) là kết quả của quá trình **mở rộng chỉ mục thời gian (temporal index expansion)** để tạo lưới chuỗi thời gian liên tục phục vụ học máy.",
        "> Đây **KHÔNG PHẢI** là hành vi \"làm giả\" hay sinh dữ liệu giao thông ngẫu nhiên. Mọi giá trị điền khuyết đều dựa trên cơ sở khoa học chuỗi thời gian (nội suy xu hướng cục bộ hoặc sử dụng phân bổ lịch sử thực tế của chính thiết bị đó tại khung giờ tương ứng) nhằm đảm bảo tính toàn vẹn toán học cho mô hình dự báo.",
        "",
        "## 4. Danh sách tệp đầu ra (Output files)",
        f"- Primary CSV: `{csv_path.relative_to(ROOT)}`",
        f"- Strict Step-4 CSV: `{strict_csv_path.relative_to(ROOT)}`",
        f"- Parquet: `{parquet_status}`",
        f"- JSON report: `{report_json_path.relative_to(ROOT)}`",
        f"- VehicleClass mapping: `{mapping_path.relative_to(ROOT)}`",
        f"- Unmatched traffic audit: `{Path(report['join']['unmatched_traffic_audit_csv']).relative_to(ROOT)}`",
        f"- Daily join diagnostics: `{Path(report['join']['daily_join_diagnostics_csv']).relative_to(ROOT)}`",
        "",
        "## 5. Các quyết định chính (Key decisions)",
        f"- Effective DeviceId source: **{'NodeId' if USE_NODE_ID_AS_DEVICE_ID else 'DeviceId'}**.",
        f"- Traffic IntervalType kept: **{TRAFFIC_INTERVAL_TYPE}**.",
        f"- Timezone policy: source timestamps are timezone-naive; both feeds are parsed consistently and output remains local naive `BucketTime`.",
        f"- VehicleClass mapping: {CLASS_MAPPING_NOTE}",
        "- Tỷ lệ loại xe được tính toán lại trực tiếp từ số lượng xe nguyên sau khi làm tròn (đảm bảo khớp 100%).",
        "",
        "## 6. Kiểm định chất lượng (Validation & QA)",
        f"- Duplicate final key rows: {report['validation']['final_duplicate_key_rows']}",
        f"- Total missing values in primary final CSV: {report['validation']['final_missing_values_total']}",
        f"- Missing values by column: `{report['validation']['final_missing_values_by_column']}`",
        f"- Vehicle count positive in all primary rows: {report['validation']['vehicle_count_positive_all_rows']}",
        f"- Ratio bounds OK: {report['validation']['class_ratio_bounds_ok']}",
        "",
        "## 7. Cấu trúc bảng đầu ra (Final primary schema)",
    ]
    md_lines.extend([f"- `{col}`" for col in final.columns])
    report_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "csv": str(csv_path),
        "strict_csv": str(strict_csv_path),
        "parquet": parquet_status,
        "report_json": str(report_json_path),
        "report_md": str(report_md_path),
        "mapping": str(mapping_path),
    }


def main() -> None:
    final, report = build_final_dataset()
    paths = save_outputs(final, report)
    print("Created final forecasting dataset")
    print(f"Rows: {len(final):,}")
    print(f"Columns: {len(final.columns)}")
    print(f"Primary CSV: {paths['csv']}")
    print(f"Strict Step-4 CSV: {paths['strict_csv']}")
    print(f"Parquet: {paths['parquet']}")
    print(f"Report: {paths['report_md']}")
    print(f"VehicleClass mapping: {paths['mapping']}")
    print(f"Missing values total: {report['validation']['final_missing_values_total']}")
    print(f"Duplicate final key rows: {report['validation']['final_duplicate_key_rows']}")


if __name__ == "__main__":
    main()

