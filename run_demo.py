"""python run_demo.py  ->  trains on healthy data, detects on anomaly-injected data, prints metrics."""
import time
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from src.simulate import make_network, make_test_network
from src.skyguard import SkyGuard, PARAMS

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# 1) Train on 30 days of healthy multi-station data
train = make_network(n_stations=5, n_days=30, seed=1)
model = SkyGuard().fit(train)
model.save("skyguard_model.joblib")

# 2) Test on a different month: genuine storms everywhere + injected faults on AWS_01 and AWS_03
test = make_test_network(n_stations=5, n_days=30, seed=7)
results = model.score(test)

# 3) Accuracy on anomaly-injected data
y_true = pd.concat([d["is_anomaly"] for d in test.values()]).to_numpy()
y_pred = pd.concat([r["anomaly"] for r in results.values()]).astype(int).to_numpy()
print("\n=== DETECTION QUALITY (all stations) ===")
print(f"precision={precision_score(y_true, y_pred):.3f}  recall={recall_score(y_true, y_pred):.3f}  "
      f"F1={f1_score(y_true, y_pred):.3f}")

ev = (pd.concat([d["genuine_event"] for d in test.values()]).to_numpy() == 1) & (y_true == 0)
print(f"genuine storm rows (no fault present) wrongly flagged: {int((y_pred[ev] == 1).sum())} of {int(ev.sum())}")

print("\nRecall by fault type:")
rows = []
for sid, d in test.items():
    r = results[sid]
    for t, g in d[d["is_anomaly"] == 1].groupby("anomaly_type"):
        rows.append((t, r.loc[g.index, "anomaly"].mean(), len(g)))
tab = pd.DataFrame(rows, columns=["type", "recall", "n"]).groupby("type").apply(
    lambda g: pd.Series({"recall": (g.recall * g.n).sum() / g.n.sum(), "samples": g.n.sum()}), include_groups=False)
print(tab.round(3))

# 4) Example alert: the '55 C, 100 % RH, pressure crash' case
r = results["AWS_01"] if (test["AWS_01"]["anomaly_type"] == "multi_parameter_fault").any() else results["AWS_03"]
d = test["AWS_01"] if r is results["AWS_01"] else test["AWS_03"]
ts = d.index[d["anomaly_type"] == "multi_parameter_fault"][0]
print("\n=== EXAMPLE ALERT ===")
print(r.loc[[ts], ["temp", "pres", "rh", "cause", "severity", "confidence", "reason", "fix_temp", "fix_pres", "fix_rh"]]
      .T.to_string())

# 5) Sensor health + maintenance prediction
print("\n=== SENSOR HEALTH ===")
print(model.health_report(results))

# 6) Real-time replay: feed the newest reading each step using a 72-sample sliding window
print("\n=== STREAMING REPLAY (5 stations, sliding window) ===")
ids = list(test)
lat, alerts = [], 0
t0 = 4000
for i in range(t0, t0 + 200):
    win = {s: test[s][PARAMS].iloc[i - 72:i + 1] for s in ids}
    t = time.perf_counter()
    res = model.score(win)
    lat.append((time.perf_counter() - t) * 1000)
    alerts += sum(bool(r["anomaly"].iloc[-1]) for r in res.values())
print(f"mean latency per update (all 5 stations): {sum(lat) / len(lat):.1f} ms  | alerts raised: {alerts}")

pd.concat({s: r for s, r in results.items()}, names=["station", "timestamp"]).to_csv("results.csv")
print("\nSaved: skyguard_model.joblib, results.csv")
