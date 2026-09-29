# SkyGuard AI - Use Cases & Design Notes

**Problem:** SIH 26073 - AI/ML-based anomaly detection for Automatic Weather Stations (MoES / IMD).
**Inputs:** temperature (°C), pressure (hPa), relative humidity (%) only.

## 1. How it works (one paragraph)
Five evidence layers vote on every reading: (1) physical rules, (2) the station's own temporal trend,
(3) agreement with neighbouring stations, (4) humidity-vs-temperature physical consistency, and
(5) an Isolation Forest over all of the above. A reading that is unusual **for the station but confirmed
by its neighbours** is labelled a *genuine weather event* and is NOT alarmed; one that neighbours
contradict is a *sensor/data fault*, classified by root cause, scored, explained and (optionally) corrected.

## 2. Use cases

| # | Situation | What SkyGuard does | Output |
|---|-----------|--------------------|--------|
| 1 | **The PS example:** 55 °C, 100 % RH, ~20 hPa pressure drop at one AWS while neighbours are normal | Spatial z-scores explode on all 3 parameters, abrupt step, humidity inconsistent with temperature | `critical`, confidence 0.99, cause `spike`/multi-parameter fault, corrected values from neighbours |
| 2 | **Thunderstorm / cold outflow** seen by every station (temp −8 °C in 4 h, pressure dip) | Own-trend deviation is large, but neighbours agree, so no alarm | Label `genuine_weather_event` (false alarm avoided) |
| 3 | **Frozen sensor** (identical values ≥ 6 readings) | Rule layer | `frozen_sensor`, medium+, alert after 6 samples |
| 4 | **Communication failure** (NaN or −999 sentinel) | Rule layer, sentinel masked from analytics | `communication_error`, imputed values |
| 5 | **Slow calibration drift** (e.g. +5 °C over ~3 days) | Rolling spatial residual keeps growing; health trend regression | `calibration_drift` + maintenance advice ("service within ~N days") |
| 6 | **Pressure level shift** (+/−15 hPa for a while) | Spatial + step detector | `spike`/`level_shift`, high severity |
| 7 | **Humidity sensor stuck at 5 % or 100 %** | Humidity–temperature consistency + spatial | `level_shift` / `cross_parameter_inconsistency` |
| 8 | **Data-fusion for forecasting** | Only `anomaly == False` (or corrected `fix_*`) values are passed downstream | Cleaner NWP/nowcasting inputs |
| 9 | **Network operations** | Health table ranks stations by risk | Prioritised maintenance visits, fewer truck rolls |

## 3. Mapping to the evaluation criteria

| Criterion (weight) | How the design addresses it |
|---|---|
| Innovation & novelty (25 %) | Genuine-event vs fault separation via spatial consensus; two-pass robust neighbour reference (faulty stations cannot poison the reference - "self-healing"); per-station auto-calibration offsets; drift-to-maintenance forecasting |
| Detection accuracy (20 %) | Hybrid rules + statistics + Isolation Forest; benchmark in `run_demo.py` (precision ≈ 1.0, recall ≈ 0.65 overall incl. slow-drift onset, ≈ 1.0 on spikes, jumps, dropouts, humidity faults) |
| Real-time (15 %) | Causal features, sliding-window scoring, ~0.15 s per update for a 5-station network on a laptop CPU |
| Explainability (10 %) | Human-readable reason per alert (`z` scores per cause), optional SHAP (`SkyGuard.explain_shap`) |
| Scalability (10 %) | Stateless per-window scoring; per-station models are tiny; shard by region and run in parallel |
| Practical deployability (10 %) | Pure Python + scikit-learn, `joblib` model file, CSV loader, Streamlit UI, ESP32 tier-1 filter |
| Visualization/UI (5 %) | Streamlit dashboard: health, explorer with anomaly + corrected points, alerts, explanations |
| Energy efficiency (5 %) | ESP32 sketch: on-device range/jump/frozen checks, radio only on anomaly or heartbeat, deep sleep between 10-min samples |

## 4. Architecture

```
ESP32 AWS  --(MQTT/HTTP)-->  Ingest  -->  SkyGuard engine  -->  Alerts / API / Dashboard
 tier-1 rules               (window)      rules+temporal+       corrected data -> forecasting
 deep sleep                               spatial+consistency   health & maintenance
                                          +IsolationForest
```

## 5. Honest limitations & future work
* Benchmarked on simulated data; validate on real IMD AWS archives (tune `z_thr`, `SP_FLOOR`, `FLAT_WINDOW`).
* Spatial reasoning needs ≥ 3 nearby stations; with fewer, the system falls back to temporal + physical checks only.
* Very early drift (below ~1.2 °C offset) is intentionally not alarmed; it is caught once it exceeds tolerance.
* Next steps: LSTM/autoencoder for seasonal dynamics, distance-weighted/elevation-aware neighbours, TinyML (quantised) model on ESP32, online drift-adaptive retraining, feedback loop from engineers.
