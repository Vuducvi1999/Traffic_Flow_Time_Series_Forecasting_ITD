from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / 'docs' / 'charts'
OUT.mkdir(parents=True, exist_ok=True)
RAW_DIR = ROOT / "data" / "raw"
TRAFFIC_FILE = max(
    RAW_DIR.glob("iTMS_VDS_Traffic_*.csv"),
    key=lambda f: f.stat().st_mtime,
)

VEHICLE_FILE = max(
    RAW_DIR.glob("iTMS_VDS_Vehicle_*.csv"),
    key=lambda f: f.stat().st_mtime,
)

plt.rcParams.update({
    'figure.figsize': (12, 7),
    'figure.dpi': 120,
    'savefig.dpi': 180,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'axes.titlesize': 15,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
})

CHARTS = []
SUMMARY = []


def savefig(name: str, title: str):
    path = OUT / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    CHARTS.append((title, path.name))


def short_id(value):
    text = str(value)
    return text[:8] if len(text) > 8 else text


def add_bar_labels(ax, fmt='{:.0f}'):
    for patch in ax.patches:
        width = patch.get_width()
        if np.isfinite(width):
            ax.text(width, patch.get_y() + patch.get_height() / 2, ' ' + fmt.format(width),
                    va='center', ha='left', fontsize=8)


def correlation_heatmap(df: pd.DataFrame, columns, title, out_name):
    corr = df[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)), labels=corr.columns, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(corr.index)), labels=corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            color = 'white' if abs(value) > 0.55 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Hệ số tương quan')
    ax.set_title(title)
    savefig(out_name, title)


def build_traffic_charts():
    traffic = pd.read_csv(TRAFFIC_FILE)
    traffic['BucketTime'] = pd.to_datetime(traffic['BucketTime'], errors='coerce')
    traffic['OccurDate'] = pd.to_datetime(traffic['OccurDate'], errors='coerce')
    numeric_cols = ['IntervalType', 'NumVehicles', 'AvgSpeed', 'Occupancy', 'AvgDensity', 'AvgHeadway', 'Confidence', 'FlowRate']
    for col in numeric_cols:
        traffic[col] = pd.to_numeric(traffic[col], errors='coerce')

    SUMMARY.append('## iTMS_VDS_Traffic_202606290917.csv')
    SUMMARY.append(f'- Số dòng/cột: {traffic.shape[0]:,} dòng, {traffic.shape[1]} cột')
    SUMMARY.append(f"- Khoảng thời gian OccurDate: {traffic['OccurDate'].min()} → {traffic['OccurDate'].max()}")
    SUMMARY.append(f"- AvgSpeed trung bình: {traffic['AvgSpeed'].mean():.2f}; NumVehicles trung bình mỗi bản ghi: {traffic['NumVehicles'].mean():.2f}; FlowRate trung bình: {traffic['FlowRate'].mean():.2f}")

    # 1. Hourly volume by interval type
    hourly = (traffic.dropna(subset=['BucketTime'])
              .groupby([pd.Grouper(key='BucketTime', freq='h'), 'IntervalType'])['NumVehicles']
              .sum()
              .unstack('IntervalType')
              .sort_index())
    ax = hourly.plot(kind='line', marker='o', linewidth=1.3, markersize=2.5)
    ax.set_title('Traffic 1/5 - Tổng NumVehicles theo giờ và IntervalType')
    ax.set_xlabel('Thời gian theo giờ')
    ax.set_ylabel('Tổng NumVehicles')
    ax.legend(title='IntervalType')
    savefig('traffic_01_hourly_numvehicles_by_interval.png', 'Traffic 1/5 - Tổng NumVehicles theo giờ và IntervalType')

    # 2. Average speed by hour of day and interval type
    traffic['HourOfDay'] = traffic['BucketTime'].dt.hour
    speed_hour = traffic.groupby(['HourOfDay', 'IntervalType'])['AvgSpeed'].mean().unstack('IntervalType')
    ax = speed_hour.plot(kind='line', marker='o', linewidth=2)
    ax.set_title('Traffic 2/5 - AvgSpeed trung bình theo giờ trong ngày')
    ax.set_xlabel('Giờ trong ngày')
    ax.set_ylabel('AvgSpeed trung bình')
    ax.set_xticks(range(0, 24))
    ax.legend(title='IntervalType')
    savefig('traffic_02_avg_speed_by_hour.png', 'Traffic 2/5 - AvgSpeed trung bình theo giờ trong ngày')

    # 3. Speed vs occupancy scatter
    scatter_data = traffic[['AvgSpeed', 'Occupancy', 'IntervalType']].dropna()
    if len(scatter_data) > 25000:
        scatter_data = scatter_data.sample(n=25000, random_state=42)
    fig, ax = plt.subplots(figsize=(11, 7))
    for interval, part in scatter_data.groupby('IntervalType'):
        ax.scatter(part['Occupancy'], part['AvgSpeed'], s=12, alpha=0.35, label=f'Interval {int(interval)}')
    ax.set_title('Traffic 3/5 - Quan hệ Occupancy và AvgSpeed')
    ax.set_xlabel('Occupancy')
    ax.set_ylabel('AvgSpeed')
    ax.legend(title='IntervalType')
    savefig('traffic_03_speed_vs_occupancy.png', 'Traffic 3/5 - Quan hệ Occupancy và AvgSpeed')

    # 4. Top nodes by total volume
    node_volume = traffic.groupby('NodeId', dropna=False)['NumVehicles'].sum().sort_values(ascending=False).head(10)
    node_volume.index = [short_id(x) for x in node_volume.index]
    fig, ax = plt.subplots(figsize=(11, 7))
    node_volume.sort_values().plot(kind='barh', ax=ax, color='#4C78A8')
    ax.set_title('Traffic 4/5 - Top 10 NodeId theo tổng NumVehicles')
    ax.set_xlabel('Tổng NumVehicles')
    ax.set_ylabel('NodeId rút gọn')
    add_bar_labels(ax)
    savefig('traffic_04_top_nodes_numvehicles.png', 'Traffic 4/5 - Top 10 NodeId theo tổng NumVehicles')

    # 5. Correlation heatmap
    correlation_heatmap(
        traffic,
        ['NumVehicles', 'AvgSpeed', 'Occupancy', 'AvgDensity', 'AvgHeadway', 'Confidence', 'FlowRate'],
        'Traffic 5/5 - Ma trận tương quan các chỉ số giao thông',
        'traffic_05_correlation_heatmap.png',
    )


def build_vehicle_charts():
    vehicle = pd.read_csv(VEHICLE_FILE)
    vehicle['BucketTime'] = pd.to_datetime(vehicle['BucketTime'], errors='coerce')
    vehicle['OccurDate'] = pd.to_datetime(vehicle['OccurDate'], errors='coerce')
    numeric_cols = ['VehicleClass', 'Speed', 'Confidence', 'ConfidenceColor', 'ConfidenceDirection', 'ConfidenceSpeed', 'TravelTimeSec']
    for col in numeric_cols:
        if col in vehicle.columns:
            vehicle[col] = pd.to_numeric(vehicle[col], errors='coerce')

    SUMMARY.append('')
    SUMMARY.append('## iTMS_VDS_Vehicle_202606290919.csv')
    SUMMARY.append(f'- Số dòng/cột: {vehicle.shape[0]:,} dòng, {vehicle.shape[1]} cột')
    SUMMARY.append(f"- Khoảng thời gian OccurDate: {vehicle['OccurDate'].min()} → {vehicle['OccurDate'].max()}")
    SUMMARY.append(f"- Speed trung bình: {vehicle['Speed'].mean():.2f}; Confidence trung bình: {vehicle['Confidence'].mean():.2f}; số biển số duy nhất: {vehicle['Plate'].nunique():,}")

    # 1. Vehicles per hour
    hourly_count = vehicle.dropna(subset=['BucketTime']).groupby(pd.Grouper(key='BucketTime', freq='h')).size().sort_index()
    fig, ax = plt.subplots(figsize=(12, 7))
    hourly_count.plot(ax=ax, color='#F58518', linewidth=1.8, marker='o', markersize=2.5)
    ax.set_title('Vehicle 1/5 - Số lượt xe ghi nhận theo giờ')
    ax.set_xlabel('Thời gian theo giờ')
    ax.set_ylabel('Số lượt xe')
    savefig('vehicle_01_hourly_count.png', 'Vehicle 1/5 - Số lượt xe ghi nhận theo giờ')

    # 2. Speed distribution
    speed = vehicle['Speed'].dropna()
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.hist(speed, bins=60, color='#54A24B', alpha=0.85, edgecolor='white')
    mean_speed = speed.mean()
    median_speed = speed.median()
    ax.axvline(mean_speed, color='red', linestyle='--', linewidth=2, label=f'TB: {mean_speed:.1f}')
    ax.axvline(median_speed, color='black', linestyle=':', linewidth=2, label=f'Trung vị: {median_speed:.1f}')
    ax.set_title('Vehicle 2/5 - Phân phối tốc độ xe')
    ax.set_xlabel('Speed')
    ax.set_ylabel('Số lượt xe')
    ax.legend()
    savefig('vehicle_02_speed_distribution.png', 'Vehicle 2/5 - Phân phối tốc độ xe')

    # 3. Vehicle class counts
    class_counts = vehicle['VehicleClass'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 7))
    class_counts.plot(kind='bar', ax=ax, color='#B279A2')
    ax.set_title('Vehicle 3/5 - Phân bổ VehicleClass')
    ax.set_xlabel('VehicleClass')
    ax.set_ylabel('Số lượt xe')
    ax.bar_label(ax.containers[0], fmt='%.0f', fontsize=8, rotation=90, padding=3)
    savefig('vehicle_03_vehicle_class_counts.png', 'Vehicle 3/5 - Phân bổ VehicleClass')

    # 4. Vehicle color counts
    color_counts = vehicle['VehicleColor'].fillna('Không rõ').value_counts().head(12)
    fig, ax = plt.subplots(figsize=(11, 7))
    color_counts.sort_values().plot(kind='barh', ax=ax, color='#E45756')
    ax.set_title('Vehicle 4/5 - Phân bổ màu xe')
    ax.set_xlabel('Số lượt xe')
    ax.set_ylabel('VehicleColor')
    add_bar_labels(ax)
    savefig('vehicle_04_vehicle_color_counts.png', 'Vehicle 4/5 - Phân bổ màu xe')

    # 5. Speed by vehicle class boxplot
    box_source = vehicle[['VehicleClass', 'Speed']].dropna()
    if len(box_source) > 300000:
        box_source = box_source.sample(n=300000, random_state=42)
    classes = sorted(box_source['VehicleClass'].unique())
    data = [box_source.loc[box_source['VehicleClass'] == cls, 'Speed'].to_numpy() for cls in classes]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.boxplot(data, labels=[str(int(c)) if float(c).is_integer() else str(c) for c in classes], showfliers=False)
    ax.set_title('Vehicle 5/5 - Phân bố Speed theo VehicleClass')
    ax.set_xlabel('VehicleClass')
    ax.set_ylabel('Speed')
    savefig('vehicle_05_speed_by_vehicle_class.png', 'Vehicle 5/5 - Phân bố Speed theo VehicleClass')


def write_report():
    lines = ['# Báo cáo biểu đồ phân tích 2 file CSV', '']
    lines.extend(SUMMARY)
    lines.append('')
    lines.append('## Danh sách biểu đồ đã tạo')
    for title, file_name in CHARTS:
        lines.append(f'- {title}: `charts/{file_name}`')
    lines.append('')
    lines.append('Ghi chú: các NodeId trong biểu đồ top node được rút gọn 8 ký tự đầu để dễ đọc.')
    (OUT / 'chart_summary.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    build_traffic_charts()
    build_vehicle_charts()
    write_report()
    print(f'Created {len(CHARTS)} charts in: {OUT}')
    for title, file_name in CHARTS:
        print(f'- {file_name}: {title}')
    print(f'- chart_summary.md: summary report')


if __name__ == '__main__':
    main()
