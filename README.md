# Hệ Thống Dự Báo Lưu Lượng Giao Thông VDS Sử Dụng SARIMAX
(VDS Traffic & Speed Forecasting System with SARIMAX, FastAPI & Streamlit)

Hệ thống này cung cấp giải pháp toàn diện từ đầu đến cuối (End-to-End) nhằm xử lý dữ liệu cảm biến giao thông (VDS), phân loại phương tiện và dữ liệu thời tiết (WOS), huấn luyện các mô hình dự báo chuỗi thời gian SARIMAX, và triển khai dưới dạng dịch vụ API thời gian thực cùng giao diện giám sát trực quan.

---

## Tính Năng Nổi Bật

1. **Pipeline ETL mạnh mẽ**: Tự động đồng bộ và gom cụm dữ liệu giao thông theo chu kỳ **5 phút** (khớp với tần suất thực tế của thiết bị cảm biến VDS để tránh sai lệch dữ liệu). Tạo lưới thời gian liên tục (Temporal Grid Expansion) và xử lý khuyết thiếu thông minh bằng nội suy tuyến tính (khoảng trống ngắn <= 12 bước - tương đương 60 phút) và cấu hình giờ lịch sử (Historical Profile Imputation cho khoảng trống lớn).
2. **Chuẩn hóa Phân loại Xe**: Sử dụng Thuật toán số dư lớn nhất (Largest Remainder Method - LRM) giúp làm tròn số lượng xe của từng phân loại (Car, Truck, Bus, Motorcycle, Other) sao cho tổng của chúng luôn khớp chính xác 100% với tổng số xe quan trắc (NumVehicles).
3. **Tránh Rò Rỉ Thông Tin (Mathematical Leakage) & Thiết Kế Bất Đối Xứng**: 
   - Tự động chuyển đổi các đặc trưng lưu lượng, tỷ lệ xe và thời tiết thành biến trễ 1 bước (Lag-1 ứng với 5 phút) để làm đầu vào ngoại sinh (Exogenous) cho mô hình SARIMAX.
   - **Cấu hình biến ngoại sinh bất đối xứng**:
     - Mô hình dự báo `NumVehicles` sử dụng các biến ngoại sinh đầy đủ (AvgSpeed, Occupancy, AvgDensity, các biến thời tiết).
     - Mô hình dự báo `AvgSpeed` chỉ sử dụng các biến ngoại sinh liên quan trực tiếp đến tốc độ và lưu thông vật lý (Occupancy, các biến thời tiết), loại bỏ `NumVehicles` và `AvgDensity` nhằm tăng tính chính xác, thực tế và tránh rò rỉ thông tin tương quan ngược.
4. **Tối ưu hóa Tự động**: Tự động kiểm tra tính dừng bằng kiểm định Augmented Dickey-Fuller (ADF) để tìm bậc sai phân d tối ưu. Hỗ trợ tìm kiếm siêu tham số tối ưu thông qua auto_arima và lưu cấu hình riêng cho từng thiết bị VDS.
5. **Dịch vụ API Cấp Công Nghiệp**: Xây dựng trên FastAPI kèm cơ chế cập nhật trạng thái cuộn thời gian thực thông qua API /extend. Đi kèm cổng Proxy Gateway bảo mật.
6. **Dashboard Đối Chất Trực Quan**: Ứng dụng Streamlit cho phép người dùng lựa chọn bất kỳ mốc thời gian lịch sử nào để chạy mô phỏng đối chất song song (Backtest Simulator) giữa Thực tế diễn ra và Đường dự báo kèm khoảng tin cậy 95%.

---

## Cấu Trúc Thư Mục Dự Án

```text
TS_ITD/
├── data/                            # Thư mục lưu trữ dữ liệu
├── data/raw/                        # Dữ liệu quan trắc thô đầu vào
│   ├── iTMS_VDS_Traffic_202606290917.csv
│   ├── iTMS_VDS_Vehicle_202606290919.csv
│   └── iTMS_WOS_Raw_202606290934.csv
└── data/processed/                  # Kết quả sau tiền xử lý & mô hình đã huấn luyện
    ├── traffic_vehicle_forecasting_dataset.csv
    ├── traffic_vehicle_forecasting_dataset_step4_columns.csv
    ├── traffic_vehicle_forecasting_dataset.parquet
    ├── preprocessing_report.md
    ├── vehicle_class_mapping_inferred.json
    ├── audit/
    │   ├── unmatched_traffic_rows.csv
    │   └── traffic_vehicle_join_diagnostics_by_date.csv
    └── models/
        ├── model_{target_col}_{device_id}.joblib
        ├── device_sarimax_config.json
        ├── sarimax_evaluation_metrics.csv
        ├── sarimax_coefficients.json
        └── plots/
            ├── forecast_{target_col}_{device_id}.png
            └── diagnostics_{target_col}_{device_id}.png

├── src/                             # Mã nguồn Python chính
│   ├── prepare_forecasting_dataset.py
│   ├── optimize_sarimax.py
│   ├── train_sarimax.py
│   ├── test_accuracy.py
│   ├── api_server.py
│   ├── online_forecasting_engine.py
│   ├── streamlit_dashboard.py
│   │
│   ├── gateway/                     # Proxy Gateway chuyển tiếp
│   │   ├── main.py
│   │   ├── .env
│   │   └── requirements.txt
│   │
│   └── utils/                       # Các tập lệnh tiện ích bổ trợ
│
├── requirements.txt                 # Tổng hợp tất cả các thư viện cần thiết ở cấp dự án
└── README.md                        # Hướng dẫn sử dụng hệ thống
```

---

## Hướng Dẫn Cài Đặt

### Yêu cầu hệ thống
* Python phiên bản 3.9 trở lên (Khuyến nghị 3.10 hoặc 3.11).
* RAM tối thiểu: 8GB (khuyến nghị 16GB).

### Bước 1: Khởi tạo môi trường ảo và cài đặt thư viện
Mở cửa sổ dòng lệnh (Terminal/PowerShell) tại thư mục dự án và chạy:

```bash
# Tạo môi trường ảo (tùy chọn nhưng khuyến nghị)
python -m venv venv
venv\Scripts\activate  # Trên Windows
source venv/bin/activate  # Trên Linux/macOS

# Nâng cấp pip và cài đặt toàn bộ thư viện cần thiết
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quy Trình Thực Thi Hệ Thống

Để vận hành hệ thống từ bước tiền xử lý dữ liệu thô đến khi hiển thị giao diện, hãy thực hiện tuần tự theo các bước dưới đây:

### Bước 1: Tiền xử lý dữ liệu (ETL Pipeline)
* **Thời gian thực thi dự kiến**: ~15 - 30 giây.
* **Mô tả**: Gộp dữ liệu lưu lượng VDS, phân loại phương tiện và thời tiết thô, thực hiện lưới hóa thời gian 5 phút, xử lý khuyết thiếu và áp dụng giải thuật chuẩn hóa phân bổ xe (LRM).

```bash
python src/prepare_forecasting_dataset.py
```
* **Đầu ra**: 
  * Tệp dữ liệu sạch: [traffic_vehicle_forecasting_dataset.csv](file:///d:/2026/TS_ITD/data/processed/traffic_vehicle_forecasting_dataset.csv)
  * Tệp báo cáo ETL: [preprocessing_report.md](file:///d:/2026/TS_ITD/data/processed/preprocessing_report.md).

### Bước 2: Tối ưu hóa siêu tham số (Tùy chọn)
* **Mô tả**: Chạy kịch bản tìm kiếm lưới để tìm bộ tham số (p, d, q) x (P, D, Q)s tối ưu cho từng trạm VDS cụ thể.

```bash
# Sử dụng 4 luồng song song để tìm kiếm nhanh
python src/optimize_sarimax.py --n-jobs 4 --target NumVehicles
python src/optimize_sarimax.py --n-jobs 4 --target AvgSpeed
```
* **Đầu ra**: Tệp cấu hình lưu tham số tối ưu nhất cho mỗi trạm tại [device_sarimax_config.json](file:///d:/2026/TS_ITD/data/processed/models/device_sarimax_config.json).

### Bước 3: Huấn luyện mô hình SARIMAX
* **Mô tả**: Huấn luyện mô hình chính thức trên toàn bộ dữ liệu lịch sử bằng cấu hình tối ưu.

```bash
# Chạy huấn luyện song song cho tất cả các thiết bị VDS
python src/train_sarimax.py --n-jobs -1
```
* **Tham số hỗ trợ**:
  * `--test-mode`: Chạy thử nghiệm nhanh (chỉ huấn luyện 2 trạm với lịch sử ngắn 300 dòng).
  * `--no-search`: Bỏ qua việc tìm kiếm tự động, ép buộc sử dụng cấu hình tĩnh từ tệp JSON.
* **Đầu ra**:
  * Các tệp mô hình đã huấn luyện: `data/processed/models/model_{target_col}_{device_id}.joblib`
  * Bảng sai số đánh giá: [sarimax_evaluation_metrics.csv](file:///d:/2026/TS_ITD/data/processed/models/sarimax_evaluation_metrics.csv)
  * Biểu đồ trực quan hóa dự báo và phân tích phần dư tại thư mục `data/processed/models/plots/`.

### Bước 4: Khởi chạy API Server & Proxy Gateway
Hệ thống sử dụng kiến trúc phân tách với một API gốc và một Proxy Gateway trung gian.

1. **Khởi chạy API gốc (cổng 8001)**:
```bash
python src/api_server.py
```
2. **Khởi chạy Proxy Gateway (cổng 8002)**:
Mở một Terminal mới, kích hoạt môi trường ảo và chạy:
```bash
cd src/gateway
python main.py
```

### Bước 5: Khởi chạy Streamlit Dashboard
```bash
streamlit run src/streamlit_dashboard.py
```
Trình duyệt sẽ tự động mở trang web tại địa chỉ `http://localhost:8501`.

---

## Tài Liệu API Endpoints (Cổng 8001 / Cổng 8002)

### 1. Kiểm tra trạng thái hệ thống
* **Endpoint**: `GET /api/health`
* **Mô tả**: Trả về tình trạng hoạt động của API và số lượng mô hình trạm đã tải thành công.
* **Response mẫu**:
  ```json
  {
    "status": "ok",
    "devices_loaded": 24,
    "dataset_period": "2026-06-19 01:55:00 → 2026-06-29 09:17:00"
  }
  ```

### 2. Lấy danh sách thiết bị trạm VDS
* **Endpoint**: `GET /api/devices`
* **Mô tả**: Trả về danh sách tất cả mã thiết bị (DeviceId) hiện có mô hình dự báo hoạt động.

### 3. Xem chỉ số đánh giá của trạm
* **Endpoint**: `GET /api/devices/{device_id}/metrics`
* **Mô tả**: Lấy các thông số sai số tính toán được của trạm trong quá trình kiểm thử (MAE, RMSE, MAPE, R2).

### 4. Dự báo lưu lượng giao thông
* **Endpoint**: `POST /api/devices/{device_id}/forecast`
* **Headers**: `Content-Type: application/json`
* **Body cấu trúc**:
  ```json
  {
    "horizon": 60,
    "observation_time": "2026-06-29 09:18",
    "current_observation": {
      "NumVehicles": 25,
      "AvgSpeed": 45.2,
      "Occupancy": 12.5,
      "AvgDensity": 8.2,
      "AvgHeadway": 2.1,
      "FlowRate": 1500,
      "AvgTravelTime": 180,
      "MedianSpeed": 44.0,
      "SpeedStd": 5.5,
      "MeanConfidence": 98.2,
      "CarCount": 10,
      "TruckCount": 2,
      "BusCount": 1,
      "MotorcycleCount": 12,
      "OtherVehicleCount": 0,
      "Rain": 0.0,
      "Temperature": 32.5,
      "Humidity": 78.0,
      "Visibility": 9000.0,
      "WindSpeed": 2.5
    }
  }
  ```
  (Lưu ý: Nếu bỏ trống current_observation, hệ thống sẽ tự động lấy bản ghi dữ liệu cuối cùng có trong cơ sở dữ liệu làm mốc dự báo).
* **Response**: Trả về chuỗi thời gian kết quả dự báo lưu lượng (predicted) cho các phút tiếp theo cùng với 120 phút dữ liệu lịch sử thực tế liền trước mốc dự báo để phục vụ hiển thị đồ thị.

### 5. Cập nhật dữ liệu quan trắc mới (Kéo dài chuỗi đầu vào)
* **Endpoint**: `POST /api/devices/{device_id}/extend`
* **Mô tả**: Đưa thêm một bản ghi quan trắc mới vào chuỗi lịch sử của mô hình mà không cần phải huấn luyện lại từ đầu. Thao tác này giúp cải thiện độ chính xác cho lần dự báo kế tiếp.

---

## Chi Tiết Giải Thuật & Thiết Kế Kỹ Thuật

### 1. Mở Rộng Chỉ Mục Thời Gian & Điền Khuyết (Temporal Grid & Imputation)
Trong các cảm biến giao thông thô, dữ liệu thường bị đứt quãng hoặc không gửi tín hiệu khi không có phương tiện. Để mô hình chuỗi thời gian hoạt động ổn định:
* Hệ thống xây dựng một lưới thời gian liên tục (frequency='5min') cho mọi làn xe và thiết bị.
* Khoảng trống <= 12 bước (tương đương 60 phút): Áp dụng nội suy tuyến tính (Linear Interpolation) để đảm bảo sự chuyển tiếp mượt mà của lưu lượng giao thông.
* Khoảng trống > 60 phút: Điền bằng Hồ sơ giờ lịch sử (Hourly Profile Imputation) trung bình của chính làn/thiết bị đó vào khung giờ tương ứng, giúp bảo toàn tính chu kỳ ngày-đêm mà không sinh nhiễu ngẫu nhiên.

### 2. Thuật Toán Số Dư Lớn Nhất (Largest Remainder Method - LRM)
Khi phân bổ tổng số lượng xe sang từng phân loại cụ thể theo tỷ lệ (ví dụ: 35.4% ô tô, 64.6% xe máy trên tổng số 3 xe), việc làm tròn thông thường sẽ dẫn đến tổng số lượng xe sau làm tròn không khớp với tổng số xe thực tế.
Hệ thống sử dụng giải thuật LRM:
1. Tính phần nguyên làm tròn xuống (floor) cho từng loại xe.
2. Tính phần dư thập phân còn lại của mỗi loại.
3. Tính tổng số xe còn thiếu so với thực tế (rem = NumVehicles - sum(floored)).
4. Phân bổ thêm từng đơn vị xe vào các loại xe có phần dư thập phân lớn nhất từ cao xuống thấp cho đến khi khớp hoàn toàn.
5. Tính toán lại cột tỷ lệ xe dựa trên số lượng xe nguyên đã chuẩn hóa để đảm bảo tính nhất quán dữ liệu 100%.

### 3. Biến Ngoại Sinh Lag-1 (Exogenous Lagging)
Để loại bỏ sự rò rỉ thông tin toán học (mathematical leakage) khi tổng số lượng xe chi tiết bằng chính xác NumVehicles tại cùng một thời điểm quan trắc, tất cả đặc trưng động bao gồm cả tốc độ trung bình, mật độ, số lượng phân loại xe, và các biến thời tiết (nhiệt độ, lượng mưa) đều được dịch trễ 1 bước (lùi 5 phút, tương ứng shift(1) trên lưới 5 phút).
* Do đó, dự báo lưu lượng tại bước t sẽ chỉ dựa trên dữ liệu lịch sử thực tế của các biến số này tại bước t-1, đảm bảo mô hình có thể triển khai dự báo tương lai thực tế.

### 4. Đo Lường Vi Phân Gia Tốc (First & Second Differences)
Để cung cấp cảnh báo ùn tắc sớm cho bộ phận điều hành giao thông, Dashboard Streamlit phân tích hai chỉ số vi phân tại mốc bắt đầu dự báo:
* Vi phân cấp 1 (First Difference - % Thay đổi): Phản ánh xu hướng lưu lượng tăng hay giảm so với phút trước.
* Vi phân cấp 2 (Second Difference - % Gia tốc): Đo tốc độ thay đổi của xu hướng. Gia tốc dương lớn biểu thị lưu lượng đang dồn toa nhanh đột biến (khả năng cao sắp xảy ra ùn tắc), hỗ trợ ra quyết định phân luồng kịp thời.
