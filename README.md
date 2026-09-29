# SkyGuard AI - Real-Time Anomaly Detection for Automatic Weather Stations
SIH Problem Statement 26073 (MoES / IMD). Uses only temperature, pressure and relative humidity.

## Quick start
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python run_demo.py              # train + detect + metrics + health + streaming replay
streamlit run app/dashboard.py  # interactive dashboard
python -m pytest -q             # tests
```

## Use your own data
CSV columns: `station,timestamp,temp,pres,rh` (regular step, e.g. 10 min). Train on healthy history:
```python
from src.skyguard import SkyGuard, load_csv
model = SkyGuard().fit(load_csv("healthy_history.csv"))
results = model.score(load_csv("new_data.csv"))
print(results["AWS_01"][["cause","severity","confidence","reason","fix_temp"]].query("cause!='normal'"))
```
See `docs/USE_CASES.md` for use cases, evaluation mapping and limitations; `edge/` for the ESP32 filter.
