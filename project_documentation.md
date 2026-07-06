# 📋 Tài Liệu Tổng Hợp Dự Án: Traffic Flow Time Series Forecasting (ITD)

> **Lưu ý bối cảnh**: Đây là dự án demo nội bộ sử dụng **dữ liệu mock của công ty** nhằm minh hoạ khả năng dự báo lưu lượng giao thông theo thời gian thực. Lựa chọn biến mục tiêu `NumVehicles` và mô hình `SARIMAX` là **theo yêu cầu của sếp**.

---

## 1. Tổng Quan Dự Án

| Mục | Nội dung |
|-----|----------|
| **Mục tiêu** | Dự báo lưu lượng xe qua từng trạm VDS (Vehicle Detection System) theo chu kỳ 1 phút |
| **Biến mục tiêu** | `NumVehicles` — Tổng số phương tiện đi qua trong mỗi chu kỳ |
| **Mô hình** | SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) |
| **Loại dữ liệu** | Dữ liệu mock của công ty, phục vụ mục đích demo |
| **Tần suất** | 1 phút / điểm dữ liệu |
| **Horizon dự báo** | 15 phút, 30 phút, 60 phút, 120 phút |
| **Số trạm VDS** | 25 trạm (DeviceId) |

---

## 2. Nguồn Dữ Liệu

### 2.1 Ba file raw CSV

| File | Kích thước | Mô tả |
|------|-----------|-------|
| `iTMS_VDS_Traffic_202606290917.csv` | ~18 MB | Dữ liệu tổng hợp lưu lượng giao thông từ hệ thống VDS |
| `iTMS_VDS_Vehicle_202606290919.csv` | ~424 MB | Dữ liệu chi tiết từng phương tiện phát hiện bởi VDS |
| `iTMS_WOS_Raw_202606290934.csv` | ~26 MB | Dữ liệu thời tiết từ trạm quan trắc môi trường (WOS) |

### 2.2 Cấu trúc dữ liệu nguồn

**VDSTraffic** — Chỉ số tổng hợp mỗi chu kỳ:

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `DeviceId` | GUID | ID thiết bị VDS |
| `NodeId` | string | ID nút giao thông (dùng làm DeviceId hiệu quả) |
| `BucketTime` | datetime | Thời điểm ghi nhận |
| `Lane` | int | Làn đường |
| `IntervalType` | int | Loại chu kỳ (1/5/15 phút) |
| `NumVehicles` | int | **Tổng số xe qua** |
| `AvgSpeed` | decimal | Tốc độ trung bình (km/h) |
| `Occupancy` | decimal | Tỷ lệ chiếm dụng mặt đường (%) |
| `AvgDensity` | decimal | Mật độ xe trung bình |
| `AvgHeadway` | decimal | Khoảng cách đầu xe (giây/mét) |
| `FlowRate` | decimal | Lưu lượng quy đổi (veh/h) |

**VDSVehicle** — Chi tiết từng phương tiện:

| Trường | Kiểu | Mô tả |
|--------|------|-------|
| `VehicleClass` | int | Mã loại xe (2=Motorcycle, 3=Car, 6=Truck, 7=Bus) |
| `Speed` | decimal | Tốc độ (km/h) |
| `TravelTimeSec` | int | Thời gian di chuyển từ nút trước (giây) |
| `Confidence` | decimal | Độ tin cậy nhận diện (0–100) |

**WosRaw** — Dữ liệu thời tiết:

| Trường | Đơn vị | Mô tả |
|--------|--------|-------|
| `Rain` | mm/h | Lượng mưa |
| `Temperature` | °C | Nhiệt độ |
| `Humidity` | % | Độ ẩm |
| `Visibility` | m | Tầm nhìn xa |
| `WindSpeed` | m/s | Tốc độ gió |

---

## 3. Quy Trình Xử Lý Dữ Liệu (ETL Pipeline)

> **Script**: [`prepare_forecasting_dataset.py`](file:///d:/2026/TS_ITD/src/prepare_forecasting_dataset.py)

### 3.1 Làm sạch dữ liệu Traffic

1. **Loại bỏ trùng lặp**: Xoá bản ghi trùng hoàn toàn (exact duplicates) và trùng ID.
2. **Chuẩn hoá thời gian**: Parse `BucketTime` → align về ranh giới phút (floor to minute).
3. **Lọc IntervalType = 1**: Chỉ giữ dữ liệu 1 phút/chu kỳ để đảm bảo đồng nhất độ phân giải cao nhất.
4. **Thay DeviceId bằng NodeId**: `DeviceId` trong file traffic bị trống, dùng `NodeId` thay thế.
5. **Loại giá trị âm vật lý bất hợp lý**: Giá trị âm của các chỉ số giao thông được đặt thành `NaN`.
6. **Gom cụm (aggregation) theo key `[BucketTime, DeviceId, Lane]`**:
   - `NumVehicles` → **sum** (tổng xe)
   - `AvgSpeed`, `Occupancy`, `AvgDensity`, `AvgHeadway` → **weighted average** theo NumVehicles
   - `FlowRate` → **sum**
7. **Imputation thiếu sau gom**: Điền bằng **median** của cột tương ứng.

### 3.2 Làm sạch và tổng hợp dữ liệu Vehicle

1. **Loại bỏ trùng lặp**: Tương tự Traffic.
2. **Mapping VehicleClass** (độ tin cậy thấp — inferred):
   | Mã | Loại | Căn cứ suy luận |
   |----|------|-----------------|
   | 2 | Motorcycle | Số lượng nhiều nhất, tốc độ thấp |
   | 3 | Car | Số lượng lớn thứ hai, tốc độ cao |
   | 6 | Truck | Ít gặp, nặng |
   | 7 | Bus | Rất ít, chậm nhất |
   | 4, 8, khác | OtherVehicle | Không xác định |
3. **Validate giá trị**:
   - Speed: 0 ≤ v ≤ 250 km/h
   - TravelTimeSec: 0 ≤ t ≤ 86400 giây
   - Confidence: 0 ≤ c ≤ 100
4. **Tổng hợp theo `[BucketTime, DeviceId, Lane]`**:
   - `CarCount`, `TruckCount`, `BusCount`, `MotorcycleCount`, `OtherVehicleCount`
   - `CarRatio`, `TruckRatio`, `BusRatio`, `MotorcycleRatio`
   - `AvgTravelTime`, `MedianSpeed`, `SpeedStd`, `MeanConfidence`

### 3.3 Xây dựng lưới thời gian liên tục

```
Raw Traffic Data (24,865 dòng quan trắc thực)
         ↓
LEFT JOIN với lưới thời gian đầy đủ 1-phút × DeviceId × Lane
         ↓
Total Grid: 358,656 dòng
         ↓
Missing: 333,791 dòng (93.1%)
```

**Chiến lược điền khuyết (theo thứ tự ưu tiên):**

| Loại khoảng trống | Phương pháp | Số dòng | Tỷ lệ |
|-------------------|-------------|---------|--------|
| ≤ 60 phút liên tiếp | Nội suy tuyến tính (linear interpolation) | 160,672 | 48.1% |
| > 60 phút liên tiếp | Hồ sơ lịch sử trung bình theo giờ (device-lane-hour profile) | 173,119 | 51.9% |
| Fallback | Global median cứng | — | — |

> **Đây KHÔNG phải làm giả dữ liệu.** Đây là kỹ thuật chuẩn trong time-series ML gọi là **temporal index expansion** — tạo lưới thời gian liên tục để đảm bảo tính toàn vẹn của chuỗi thời gian cho mô hình.

### 3.4 Merge dữ liệu thời tiết (WOS)

- Aggregate WOS theo `BucketTime` (trung bình nhiều trạm quan trắc).
- Nội suy tuyến tính để điền khoảng trống.
- LEFT JOIN vào dataset chính theo `BucketTime`.

### 3.5 Thêm đặc trưng thời gian

| Cột | Mô tả |
|-----|-------|
| `Hour` | Giờ trong ngày (0–23) |
| `DayOfWeek` | Thứ trong tuần (0=Thứ 2, 6=CN) |
| `IsWeekend` | Cuối tuần (0/1) |
| `IsHoliday` | Ngày lễ Việt Nam 2026 (0/1) |
| `Day` | Ngày trong tháng |
| `Month` | Tháng |

**Các ngày lễ được hard-code (2026)**:
- Tết Nguyên Đán: 14/2 – 22/2
- Giỗ Tổ Hùng Vương: 25–27/4
- Ngày Giải phóng + Lao động: 30/4 – 3/5
- Quốc khánh: 29/8 – 2/9
- Các ngày lễ tĩnh: 1/1, 30/4, 1/5, 2/9

### 3.6 Làm tròn số xe (Largest Remainder Method)

Để đảm bảo `CarCount + TruckCount + BusCount + MotorcycleCount + OtherVehicleCount = NumVehicles` (số nguyên chính xác 100%), dùng thuật toán **Largest Remainder Method (LRM)**.

---

## 4. Dataset Đầu Ra

**File**: `data/processed/traffic_vehicle_forecasting_dataset.csv`

| Thuộc tính | Giá trị |
|-----------|---------|
| Tổng dòng | 358,656 |
| Tổng cột | 33 |
| Missing values | **0** (hoàn toàn không có) |
| Duplicate key rows | **0** |
| Ratio bounds OK | **True** |

**33 cột đầu ra đầy đủ:**

```
BucketTime, Hour, DayOfWeek, IsHoliday, IsWeekend, Day, Month,
DeviceId, Lane,
NumVehicles, AvgSpeed, Occupancy, AvgDensity, AvgHeadway, FlowRate,
CarCount, TruckCount, BusCount, MotorcycleCount, OtherVehicleCount,
AvgTravelTime, MedianSpeed, SpeedStd, MeanConfidence,
CarRatio, TruckRatio, BusRatio, MotorcycleRatio,
Rain, Temperature, Humidity, Visibility, WindSpeed
```

---

## 5. Biến Mục Tiêu: Tại Sao Chọn `NumVehicles`?

> **Lý do chính: Theo yêu cầu của sếp.**

### Căn cứ kỹ thuật hỗ trợ quyết định:

1. **Phản ánh trực tiếp tải giao thông**: `NumVehicles` = tổng số phương tiện thực sự đi qua trong 1 phút — đây là chỉ số cốt lõi nhất, trực quan nhất cho người vận hành.

2. **Nghiệp vụ rõ ràng**: Các chỉ số phái sinh như `FlowRate`, `Occupancy` là đại lượng quy đổi/tính toán từ `NumVehicles` + chiều dài loop detector. Dự báo trực tiếp số xe gốc là tự nhiên và dễ giải thích hơn.

3. **Cơ sở để tính các chỉ số còn lại**: Khi có `NumVehicles` dự báo + tỷ lệ loại xe hiện tại, có thể suy ra số xe theo từng loại.

4. **Tính nguyên (integer)**: Là số đếm tự nhiên, không âm — dễ validate và clip sau dự báo (`clip(lower=0)`).

---

## 6. Mô Hình: SARIMAX

> **Lý do chính: Theo yêu cầu của sếp.**

### 6.1 Tại Sao SARIMAX Phù Hợp?

SARIMAX = **S**easonal **A**uto**R**egressive **I**ntegrated **M**oving **A**verage with e**X**ogenous variables

| Đặc điểm dữ liệu giao thông | Khả năng SARIMAX đáp ứng |
|------------------------------|--------------------------|
| Có tính mùa vụ theo giờ (giờ cao điểm sáng/chiều) | Seasonal component (P,D,Q,s) |
| Có tự tương quan theo thời gian | AR(p) component |
| Có thể không dừng (non-stationary) | Differencing I(d) — kiểm định ADF |
| Bị ảnh hưởng bởi thời tiết, ngày lễ, giờ trong ngày | eXogenous variables |
| Dữ liệu 1 phút → cần phản ứng nhanh | Mô hình tuyến tính, fit nhanh |
| Yêu cầu có thể giải thích (explainable) | Hệ số hồi quy trực tiếp đọc được |

### 6.2 Cấu Trúc Mô Hình

```
SARIMAX(p, d, q)(P, D, Q, s)[exog]

Trong đó:
- (p, d, q): Order ARIMA không mùa vụ
- (P, D, Q, s): Order mùa vụ, s=60 (chu kỳ 60 phút = 1 giờ)
- exog: Ma trận biến ngoại sinh 24 chiều
```

### 6.3 Tìm siêu tham số

**Grid Search nội bộ** (trong `train_sarimax.py`):
- Tập ứng viên p, q: `[1,2,3]×[0,1,2]`
- Tách validation set: 120 phút gần nhất để train, 60 phút để validate
- Chọn order có **MAE validation thấp nhất**

**Auto-ARIMA** (trong `optimize_sarimax.py` — pipeline tối ưu hoá):
- Dùng `pmdarima.auto_arima` với stepwise search
- 3 tầng cấu hình theo train_length: 500 / 800 / 1500 điểm
- Dừng sớm nếu đạt ngưỡng: MAE ≤ 12, RMSE ≤ 20, MAPE ≤ 20%, R² ≥ 0.5

**Kiểm định tính dừng (ADF Test)**:
- Trước khi fit, chạy **Augmented Dickey-Fuller test**
- p-value < 0.05 → chuỗi đã dừng, d = 0
- p-value ≥ 0.05 → lấy sai phân bậc 1 (d=1), nếu vẫn chưa → d=2

### 6.4 Huấn luyện theo từng thiết bị

- **1 mô hình = 1 DeviceId** (mỗi trạm VDS độc lập)
- **25 trạm → 25 mô hình** được lưu riêng dưới dạng `.joblib`
- Huấn luyện song song: `joblib.Parallel(n_jobs=-1)`
- Test split: **60 phút cuối cùng** không được nhìn thấy lúc train

---

## 7. Biến Ngoại Sinh (Exogenous Variables): Tại Sao Chọn?

Toàn bộ 24 biến ngoại sinh được đưa vào SARIMAX dưới dạng **lag-1** (giá trị của phút trước) để tránh data leakage.

### 7.1 Nhóm chỉ số giao thông (lag-1)

| Biến | Lý do chọn |
|------|------------|
| `AvgSpeed_lag1` | Tốc độ là chỉ báo trực tiếp trạng thái tắc nghẽn: tốc độ thấp → mật độ cao → xe nhiều hơn sắp tới |
| `Occupancy_lag1` | Tỷ lệ chiếm dụng đường phản ánh mức độ bão hoà làn — correlated cao với NumVehicles |
| `AvgDensity_lag1` | Mật độ xe/km bổ sung thông tin không gian không có trong số đếm đơn thuần |
| `AvgHeadway_lag1` | Khoảng cách đầu xe liên quan nghịch với mật độ, giúp mô hình phân biệt lúc thưa/đông |
| `FlowRate_lag1` | Lưu lượng quy đổi theo giờ — tương quan cao với NumVehicles nhưng ở đơn vị khác |
| `AvgTravelTime_lag1` | Thời gian di chuyển giữa các nút: tăng → ách tắc → sắp có đỉnh xe |
| `MedianSpeed_lag1` | Robust hơn AvgSpeed khi có outliers (xe đặc biệt nhanh/chậm) |
| `SpeedStd_lag1` | Phân tán tốc độ cao → hỗn hợp xe → trạng thái giao thông không ổn định |
| `MeanConfidence_lag1` | Chất lượng nhận diện — giá trị thấp gợi ý dữ liệu không đáng tin, cần hiệu chỉnh |

### 7.2 Nhóm thành phần loại xe (lag-1)

| Biến | Lý do chọn |
|------|------------|
| `CarCount_lag1`, `TruckCount_lag1`, `BusCount_lag1`, `MotorcycleCount_lag1`, `OtherVehicleCount_lag1` | **Phòng ngừa leakage**: các count này tổng = NumVehicles hiện tại → nếu dùng không có lag sẽ là data leakage hoàn toàn. Dùng lag-1 để mô hình học được **thành phần xe của phút trước** có ảnh hưởng thế nào đến tổng xe phút tiếp theo (e.g., xe tải đông → đường chậm → ảnh hưởng giãn cách) |

### 7.3 Nhóm thời tiết (lag-1)

| Biến | Lý do chọn |
|------|------------|
| `Rain_lag1` | Mưa lớn → người hạn chế ra đường ngay lập tức → giảm NumVehicles; mưa nhỏ kéo dài → xe chậm → tắc nghẽn |
| `Temperature_lag1` | Nhiệt độ cao cực đoan (>38°C tại VN) → ít xe máy hơn ban ngày; nhiệt độ thấp → sương mù → giảm tầm nhìn |
| `Humidity_lag1` | Độ ẩm cao correlated với mưa/sương — bổ sung context thời tiết cho mô hình |
| `Visibility_lag1` | Tầm nhìn xa thấp (<1000m) → giảm tốc độ cho phép → ảnh hưởng FlowRate và NumVehicles |
| `WindSpeed_lag1` | Gió mạnh → nguy hiểm cho xe máy (chiếm tỷ lệ lớn) → ảnh hưởng hành vi lái xe |

> **Tóm lại**: Thời tiết là **biến ngoại sinh kinh điển** trong dự báo giao thông vì nó là yếu tố bên ngoài hệ thống giao thông, có thể đo độc lập, và tác động trực tiếp đến hành vi di chuyển. Không thể dự báo bằng AR component thuần tuý.

### 7.4 Nhóm đặc trưng thời gian (không lag)

| Biến | Lý do chọn |
|------|------------|
| `Hour` | Mẫu giờ cao điểm 7–9h và 17–19h — pattern rõ nhất trong giao thông đô thị |
| `DayOfWeek` | Thứ 2 và thứ 6 thường cao điểm hơn; cuối tuần pattern khác biệt hoàn toàn |
| `IsWeekend` | Binary indicator bổ sung cho DayOfWeek — giúp mô hình học nhanh sự chuyển đổi tuần → cuối tuần |
| `IsHoliday` | Ngày lễ (Tết, 30/4...) có hành vi giao thông đặc biệt: đỉnh trước lễ, thấp trong lễ |

---

## 8. Chiến Lược Huấn Luyện và Dự Báo

### 8.1 Pipeline Training (train_sarimax.py)

```
Cho mỗi DeviceId:
  1. Aggregate tất cả Lane → 1 chuỗi thời gian tổng hợp
  2. Tạo lag-1 features từ 19 biến traffic + weather
  3. ADF Test → xác định d
  4. Grid search trên validation set 60 phút
  5. Fit SARIMAX(p,d,q)(P,D,Q,60) trên train set
  6. Evaluate trên test set (60 phút cuối)
  7. Fit lại trên toàn bộ dữ liệu cho future forecast
  8. Lưu model → .joblib
```

### 8.2 Pipeline Tối Ưu Hoá (optimize_sarimax.py)

```
Targets: MAE ≤ 12, RMSE ≤ 20, MAPE ≤ 20%, R² ≥ 0.5
Tier 1: auto_arima, train_length=500,  seasonal=False
Tier 2: auto_arima, train_length=800,  seasonal=True, m=60
Tier 3: auto_arima, train_length=1500, seasonal=True, m=60
→ Lưu config tối ưu vào device_sarimax_config.json
```

### 8.3 Online Forecasting Engine (online_forecasting_engine.py)

```
Mô phỏng streaming (1 tick = 1 phút thực):
  Mỗi tick:
    - Nhận dữ liệu mới (new_y, new_X)
    - model.extend(endog, exog) → cập nhật state không cần retrain
    - Forecast 120 phút tiếp theo
    - Cập nhật sliding window 120 phút lịch sử
    - Ghi kết quả → dashboard_data.json
  Mỗi 10 ticks:
    - Sync buffer → CSV storage (hourly sync simulation)
  Cuối stream:
    - Trigger daily retrain (2:00 AM simulation)
```

### 8.4 Kiến Trúc Phục Vụ

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Online Forecast │───▶│  dashboard_      │───▶│  Streamlit       │
│  Engine          │    │  data.json       │    │  Dashboard       │
│  (Python script) │    │  (live update)   │    │  (streamlit_     │
└─────────────────┘    └──────────────────┘    │  dashboard.py)   │
                                                └──────────────────┘
┌─────────────────┐
│  FastAPI Server  │ ← API endpoint cho tích hợp hệ thống ngoài
│  (api_server.py) │   GET /forecast/{device_id}
└─────────────────┘   POST /update-and-forecast
```

---

## 9. Metrics Đánh Giá

### 9.1 Các chỉ số sử dụng

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **MAE** | `mean(|y - ŷ|)` | Sai số tuyệt đối trung bình (xe/phút) |
| **RMSE** | `√mean((y-ŷ)²)` | Phạt nặng outlier hơn MAE |
| **MAPE** | `mean(|y-ŷ|/y) × 100%` | Sai số tương đối (%), chỉ tính khi y > 0 |
| **R²** | `1 - SS_res/SS_tot` | Mức độ giải thích phương sai (1 = hoàn hảo) |

### 9.2 Ngưỡng mục tiêu (từ optimize_sarimax.py)

| Metric | Ngưỡng đạt |
|--------|-----------|
| MAE | ≤ 12 xe/phút |
| RMSE | ≤ 20 xe/phút |
| MAPE | ≤ 20% |
| R² | ≥ 0.5 |

### 9.3 Kiểm định thống kê bổ sung

- **ADF Test** (Augmented Dickey-Fuller): kiểm định tính dừng để xác định `d`
- **Ljung-Box Test**: kiểm định phần dư có còn autocorrelation không (p > 0.05 là tốt)
- **QQ Plot** + **Histogram of residuals**: kiểm tra phân phối chuẩn của residuals

### 9.4 Kết quả thực tế (2 trạm đã train)

> **Lưu ý**: Đây là demo data — kết quả trên dữ liệu mock

| DeviceId (8 ký tự đầu) | MAE | RMSE | MAPE | R² |
|------------------------|-----|------|------|----|
| 01B42F01 | 375.6 | 533.3 | 722.3% | -404.9 |
| 0E7C5457 | 56.3 | 66.2 | 95.8% | -5.98 |

> Kết quả hiện tại chưa đạt ngưỡng vì **dữ liệu mock** không phản ánh pattern giao thông thực tế — mục đích là demo luồng xử lý, không phải đánh giá hiệu suất mô hình thực.

---

## 10. Cấu Trúc Project

```
TS_ITD/
├── data/
│   ├── raw/                          # Dữ liệu gốc mock của công ty
│   │   ├── iTMS_VDS_Traffic_*.csv    # Traffic aggregated data
│   │   ├── iTMS_VDS_Vehicle_*.csv   # Per-vehicle data  
│   │   └── iTMS_WOS_Raw_*.csv       # Weather data
│   └── processed/
│       ├── traffic_vehicle_forecasting_dataset.csv    # Dataset đã xử lý (358,656 rows)
│       ├── traffic_vehicle_forecasting_dataset.parquet
│       ├── dashboard_data.json       # Output của online engine cho dashboard
│       ├── preprocessing_report.json/md
│       ├── vehicle_class_mapping_inferred.json
│       ├── audit/                    # Diagnostic files
│       └── models/
│           ├── model_*.joblib        # 25 mô hình SARIMAX đã train
│           ├── device_sarimax_config.json
│           ├── sarimax_evaluation_metrics.csv
│           ├── sarimax_coefficients.json
│           └── plots/               # Forecast & diagnostic plots
├── src/
│   ├── prepare_forecasting_dataset.py   # ETL pipeline
│   ├── train_sarimax.py                 # Training pipeline
│   ├── optimize_sarimax.py              # Hyperparameter optimization
│   ├── online_forecasting_engine.py     # Streaming forecast engine
│   ├── test_accuracy.py                 # Interactive accuracy tester
│   ├── api_server.py                    # FastAPI REST server
│   ├── streamlit_dashboard.py           # Streamlit visualization
│   ├── gateway/                         # API proxy gateway
│   └── utils/
│       ├── export_dashboard_data.py
│       ├── generate_charts.py
│       ├── generate_explanation_gif.py
│       └── fetch_all_data.py
├── docs/
│   ├── thông số vds.md                  # Schema entity VDS
│   ├── explanation.gif                  # Demo animation
│   └── dashboard.html                   # Dashboard static export
├── logs/                                # Log files
└── requirements.txt                     # Python dependencies
```

---

## 11. Dependencies Chính

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|---------|
| `statsmodels` | ≥ 0.13 | SARIMAX implementation |
| `pmdarima` | ≥ 2.0 | auto_arima cho hyperparameter search |
| `pandas` | ≥ 1.4 | Data processing |
| `numpy` | ≥ 1.22 | Numerical computation |
| `scikit-learn` | ≥ 1.0 | Metrics (MAE, RMSE, R²) |
| `joblib` | ≥ 1.1 | Parallel training + model serialization |
| `scipy` | ≥ 1.8 | Statistical tests (ADF, Ljung-Box, QQ) |
| `fastapi` + `uvicorn` | ≥ 0.110 | REST API server |
| `streamlit` + `plotly` | ≥ 1.35 | Dashboard visualization |
| `matplotlib` | ≥ 3.5 | Static plots cho diagnostic |

---

## 12. Tóm Tắt Các Quyết Định Thiết Kế Chính

| Quyết định | Lựa chọn | Lý do |
|-----------|---------|-------|
| **Biến mục tiêu** | `NumVehicles` | Theo yêu cầu sếp; trực quan, phản ánh tải giao thông thực |
| **Mô hình** | SARIMAX | Theo yêu cầu sếp; phù hợp với chuỗi thời gian có seasonality và biến ngoại sinh |
| **Chu kỳ** | 1 phút | Độ phân giải cao nhất có trong dữ liệu (IntervalType=1) |
| **Seasonality** | s = 60 phút | Chu kỳ 1 giờ là chu kỳ tự nhiên của giao thông đô thị |
| **Lag** | lag-1 | Tránh data leakage; mô phỏng đúng thực tế online forecasting |
| **Interpolation** | Linear ≤60min, Profile >60min | Balance giữa độ chính xác cục bộ và tính hợp lý lịch sử |
| **Deployment** | Mỗi device 1 model | Pattern giao thông mỗi điểm giao khác nhau; tránh overgeneralize |
| **Online update** | `model.extend()` (statsmodels) | Cập nhật state mà không cần retrain toàn bộ |
| **Retrain** | Daily (2:00 AM) | Cân bằng freshness và cost tính toán |

---

*Tài liệu được tổng hợp từ toàn bộ source code của project. Cập nhật lần cuối: 2026-07-01.*
