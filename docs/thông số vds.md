


# VDSTraffic Entity

## Mô tả

Entity `VDSTraffic` lưu trữ các chỉ số giao thông thu thập tại 1 thời điểm 

## Thông tin các trường

| Field | Kiểu dữ liệu | Nullable | Mô tả | Ví dụ |
|--------|-------------|----------|-------|--------|
| DeviceId | Guid | ✔ | thiết bị sinh dữ liệu (bao gồm cả vị trí) | `9c9a3f1b-xxxx-xxxx-xxxx-xxxxxxxxxxxx` |
| NumVehicles | int | ✘ | Tổng số phương tiện đi qua trong chu kỳ. | `125` |
| AvgSpeed | decimal | ✘ | Tốc độ trung bình của các phương tiện (km/h). | `42.5` |
| Occupancy | decimal | ✘ | Tỷ lệ chiếm dụng mặt đường (%). | `38.7` |
| AvgDensity | decimal | ✔ | Mật độ phương tiện trung bình. | `24.5` |
| AvgHeadway | decimal | ✔ | Khoảng cách trung bình giữa hai đầu xe liên tiếp (giây hoặc mét tùy định nghĩa hệ thống). | `2.35` |
| FlowRate | decimal | ✔ | Lưu lượng phương tiện quy đổi theo giờ (veh/h). | `1350` |
| Confidence | decimal | ✔ | Độ tin cậy của dữ liệu nhận diện (0-100). | `98.5` |

---

# VDSVehicle Entity

## Mô tả

Entity `VDSVehicle` lưu trữ thông tin chi tiết của từng phương tiện phát hiện được tại một thời điểm.

## Thông tin các trường

| Field               | Kiểu dữ liệu        | Nullable | Mô tả                                                      | Ví dụ                                  |
| ------------------- | ------------------- | -------- | ---------------------------------------------------------- | -------------------------------------- |
| DeviceId            | Guid                | ✘        | Thiết bị ghi nhận phương tiện (bao gồm cả vị trí lắp đặt). | `9c9a3f1b-xxxx-xxxx-xxxx-xxxxxxxxxxxx` |
| Plate               | string              | ✘        | Biển số xe được nhận diện.                      
| Speed               | decimal             | ✔        | Tốc độ của phương tiện (km/h).                             | `62.5`                                 |
| Confidence          | decimal             | ✔        | Độ tin cậy của kết quả nhận diện biển số (0-100).          | `98.7`                                 |
| ConfidenceColor     | decimal             | ✔        | Độ tin cậy của kết quả nhận diện màu xe (0-100).           | `95.2`                                 |
| ConfidenceSpeed     | decimal             | ✔        | Độ tin cậy của kết quả nhận diện tốc độ (0-100).           | `99.1`                                 |
| ConfidenceDirection | decimal             | ✔        | Độ tin cậy của kết quả nhận diện hướng di chuyển (0-100).  | `97.4`                                 |
| VehicleColor        | string              | ✔        | Màu sắc của phương tiện.                                   | `White`                                |
| PlateColor          | string              | ✔        | Màu nền của biển số xe.                                    | `White`                                |
| VehicleLength       | decimal             | ✔        | Chiều dài phương tiện (m).                                 | `4.8`                                  |
| Direction           | VDSVehicleDirection | ✔        | Hướng di chuyển của phương tiện.                           | `forward` | `backward`                           |
| IsBookmark          | bool                | ✘        | Đánh dấu bản ghi cần theo dõi hoặc xử lý nghiệp vụ.        | `true`                                 |
| TravelTimeSec       | int                 | ✔        | Thời gian di chuyển từ nút trước đến nút hiện tại (giây) - không hiểu thì có thể hỏi anh cụ thể hơn  | `185`                                  |

---

# WosRaw Entity

## Mô tả

Entity `WosRaw` lưu trữ dữ liệu thô thu thập từ trạm quan trắc môi trường (WOS) tại một thời điểm.

## Thông tin các trường

| Field             | Kiểu dữ liệu    | Nullable | Mô tả                                                     | Ví dụ                                  |
| ----------------- | --------------- | -------- | --------------------------------------------------------- | -------------------------------------- |
| DeviceId          | Guid            | ✘        | Thiết bị/trạm quan trắc sinh dữ liệu (bao gồm cả vị trí). | `9c9a3f1b-xxxx-xxxx-xxxx-xxxxxxxxxxxx` |
| ParameterCode     | string          | ✘        | Mã tham số đo (tham chiếu tới bảng `ParameterType`).      | `TEMP`, `HUM`, `RAIN`                  |
| Value             | double          | ✘        | Giá trị đo chính của tham số.                             | `32.5`                                 |
| Rain              | double          | ✔        | Lượng mưa (mm/h).                                         | `12.8`                                 |
| Temperature       | double          | ✔        | Nhiệt độ (°C).                                            | `31.6`                                 |
| Humidity          | double          | ✔        | Độ ẩm không khí (%).                                      | `78.4`                                 |
| Visibility        | double          | ✔        | Tầm nhìn xa (m).                                          | `8500`                                 |
| WindSpeed         | double          | ✔        | Tốc độ gió (m/s).                                         | `4.2`                                  |
| Pressure          | double          | ✔        | Áp suất khí quyển (hPa).                                  | `1008.5`                               |
| WindDirection     | double          | ✔        | Góc hướng gió (0–360°).                                   | `135`                                  |
| WindDirectionName | WindDirection16 | ✔        | Tên hướng gió theo la bàn 16 hướng.                       | `SE`, `N`, `WNW`                       |



