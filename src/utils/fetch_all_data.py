import json
import requests
from pathlib import Path

BASE = "http://localhost:8000"
OUT = Path("fetched_data")
OUT.mkdir(exist_ok=True)

# 1. Health
r = requests.get(f"{BASE}/api/health")
(OUT / "health.json").write_text(json.dumps(r.json(), indent=2, ensure_ascii=False))
print(f"Health: {r.json()}")

# 2. Devices list
r = requests.get(f"{BASE}/api/devices")
devices = r.json()["devices"]
(OUT / "devices.json").write_text(json.dumps(r.json(), indent=2, ensure_ascii=False))
print(f"Devices: {len(devices)} devices")

# 3. Metrics + Forecast cho từng device
all_metrics = {}
all_forecasts = {}

for device_id in devices:
    # Metrics
    r = requests.get(f"{BASE}/api/devices/{device_id}/metrics")
    all_metrics[device_id] = r.json()
    print(f"  {device_id}: metrics OK")

    # Forecast 60 phút (không observation mới)
    r = requests.post(f"{BASE}/api/devices/{device_id}/forecast", json={"horizon": 60})
    if r.ok:
        all_forecasts[device_id] = r.json()
        print(f"  {device_id}: forecast OK")

(OUT / "all_metrics.json").write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False))
(OUT / "all_forecasts.json").write_text(json.dumps(all_forecasts, indent=2, ensure_ascii=False))

print(f"\nDone! All data saved to {OUT}/")
