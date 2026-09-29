"""SkyGuard AI - hybrid, explainable anomaly detection for AWS (temperature, pressure, humidity).

Layers
  1. Physical rules      : missing / sentinel values, out-of-range, frozen values
  2. Temporal model      : deviation from own recent trend + rate of change (robust z-scores)
  3. Spatial consistency : agreement with neighbouring stations (separates sensor faults from real weather)
  4. Cross-parameter     : humidity must be physically consistent with temperature
  5. ML (IsolationForest): multivariate outliers over all of the above
Then: root-cause classification, severity, confidence, explanation, imputation, sensor health.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import HuberRegressor
import joblib

PARAMS = ["temp", "pres", "rh"]
LIMITS = {"temp": (-20.0, 55.0), "pres": (850.0, 1085.0), "rh": (0.0, 100.0)}
ZCOLS = [f"{k}_{p}" for k in ("dev", "step", "sp") for p in PARAMS] + ["rh_res"]
FLATCOLS = [f"flat_{p}" for p in PARAMS]
SP_FLOOR = {"sp_temp": 0.5, "sp_pres": 0.35, "sp_rh": 4.0}   # min. credible station-to-station tolerance
FLAT_WINDOW = 6          # identical consecutive readings => frozen sensor
LABEL = {"temp": "Temperature", "pres": "Pressure", "rh": "Humidity"}
NAMES = {"dev": "deviates from its recent trend", "step": "changed abruptly",
         "sp": "disagrees with neighbouring stations"}


def load_csv(path):
    """CSV columns: station,timestamp,temp,pres,rh  (regular time step, e.g. 10 min)."""
    raw = pd.read_csv(path, parse_dates=["timestamp"])
    return {sid: g.set_index("timestamp").sort_index()[PARAMS] for sid, g in raw.groupby("station")}


class SkyGuard:
    def __init__(self, z_thr=5.0, contamination=0.005, seed=42):
        self.z_thr = z_thr
        self.iso = IsolationForest(n_estimators=60, contamination=contamination, random_state=seed)
        self.zero = pd.Series(0.0, index=PARAMS + ["rh_res"])
        self._last = {}

    # ------------------------------------------------------------------ features
    @staticmethod
    def _net(dfs):
        return pd.concat([d[PARAMS] for d in dfs.values()]).groupby(level=0).median()

    def _features(self, df, net, off):
        x = df[PARAMS].astype(float).where(lambda d: d > -900)      # sentinel (-999) -> NaN
        net = net.reindex(df.index)
        f = pd.DataFrame(index=df.index)
        for p in PARAMS:
            f[f"x_{p}"] = x[p]
            med = x[p].shift(1).rolling(12, min_periods=4).median()   # causal => usable in streaming
            f[f"med_{p}"] = med
            f[f"dev_{p}"] = x[p] - med
            f[f"step_{p}"] = x[p].diff()
            f[f"sp_{p}"] = x[p] - net[p] - off[p]
            f[f"flat_{p}"] = (x[p].rolling(FLAT_WINDOW).std() < 1e-6).astype(float)
        t = x["temp"].fillna(self.t_med)
        f["rh_res"] = x["rh"] - self.rh_model.predict(t.to_frame("temp")) - off["rh_res"]
        f["net_temp"], f["net_pres"], f["net_rh"] = net["temp"], net["pres"], net["rh"]
        return f

    # ------------------------------------------------------------------ training
    def fit(self, train):
        """train: {station: DataFrame} of (mostly) healthy historical data."""
        pooled = pd.concat([d[PARAMS] for d in train.values()]).dropna()
        self.t_med = float(pooled["temp"].median())
        self.rh_model = HuberRegressor().fit(pooled[["temp"]], pooled["rh"])   # RH ~ f(T)
        net = self._net(train)

        self.offset = {}                       # per-station calibration offsets vs the network
        for sid, df in train.items():
            f = self._features(df, net, self.zero)
            self.offset[sid] = pd.Series({**{p: f[f"sp_{p}"].median() for p in PARAMS},
                                          "rh_res": f["rh_res"].median()})
        F = pd.concat([self._features(df, net, self.offset[sid]) for sid, df in train.items()])
        self.med = F[ZCOLS].median()
        self.scale = (1.4826 * (F[ZCOLS] - self.med).abs().median()).clip(lower=1e-3)
        for c, fl in SP_FLOOR.items():
            self.scale[c] = max(self.scale[c], fl)
        self.iso.fit(self._X(F))
        return self

    def _z(self, f):
        return ((f[ZCOLS] - self.med) / self.scale).abs()

    def _X(self, f):
        return pd.concat([self._z(f).fillna(0).clip(upper=100), f[FLATCOLS]], axis=1)

    def _robust_net(self, dfs):
        """Two-pass neighbour reference: drop readings that already disagree, then re-take the median,
        so a few faulty stations cannot drag the reference away from the truth."""
        net = self._net({s: d[PARAMS].astype(float).where(lambda x: x > -900) for s, d in dfs.items()})
        masked = {}
        for sid, df in dfs.items():
            off = self.offset.get(sid, self.zero)
            x = df[PARAMS].astype(float).where(lambda d: d > -900)
            bad = pd.DataFrame({p: (x[p] - net[p].reindex(x.index) - off[p] - self.med[f"sp_{p}"]).abs()
                                    / self.scale[f"sp_{p}"] > 3 for p in PARAMS})
            masked[sid] = x.mask(bad)
        return self._net(masked)

    # ------------------------------------------------------------------ inference
    def score(self, dfs):
        """dfs: {station: DataFrame}. Returns {station: result DataFrame}."""
        net = self._robust_net(dfs)
        has_nb = len(dfs) >= 3
        out = {}
        for sid, df in dfs.items():
            off = self.offset.get(sid, self.zero)
            f = self._features(df, net, off)
            z = self._z(f)
            sp_cols = [f"sp_{p}" for p in PARAMS]
            if not has_nb:
                z[sp_cols] = 0.0
            xs = f[[f"x_{p}" for p in PARAMS]]
            thr = self.z_thr

            tz = z[[c for c in ZCOLS if c.startswith(("dev", "step"))]].max(axis=1)
            stepz = z[[f"step_{p}" for p in PARAMS]].max(axis=1)
            sz = z[sp_cols].max(axis=1)
            szr = z[sp_cols].rolling(36, min_periods=12).mean().max(axis=1)
            cz = z["rh_res"]

            missing = xs.isna().any(axis=1)
            oor = pd.concat([(xs[f"x_{p}"] < LIMITS[p][0]) | (xs[f"x_{p}"] > LIMITS[p][1]) for p in PARAMS],
                            axis=1).any(axis=1)
            flat = f[FLATCOLS].max(axis=1) > 0
            rules = missing | oor | flat

            X = self._X(f)
            self._last[sid] = X
            iso_flag = pd.Series(self.iso.predict(X) == -1, index=df.index)

            spatial = sz > thr
            drift = (szr > 3) & (sz > 2.5)
            incons = cz > thr + 1
            ml = iso_flag & (tz > thr) & ((sz > 3.5) if has_nb else True)
            anom = rules | spatial | drift | incons | ml

            recovery = anom.shift(1, fill_value=False).astype(float).rolling(12, min_periods=1).max() > 0
            genuine = has_nb & (tz > thr) & (sz <= 2.5) & ~anom & ~recovery

            cause = np.select(
                [missing, oor, flat, (stepz > thr) & spatial, sz >= 12, drift, incons],
                ["communication_error", "out_of_range", "frozen_sensor", "spike",
                 "level_shift", "calibration_drift", "cross_parameter_inconsistency"],
                default="multivariate_outlier")
            cause = pd.Series(np.where(anom, cause, np.where(genuine, "genuine_weather_event", "normal")),
                              index=df.index)

            maxz = z.max(axis=1).fillna(0)
            ms = np.where(rules, np.maximum(maxz, 6.0), maxz)
            sev = np.select([ms >= 25, ms >= 12, ms >= 6], ["critical", "high", "medium"], default="low")
            conf = (1 - np.exp(-maxz / 4)).clip(0, 0.99)
            conf = np.where(rules, np.maximum(conf, 0.95), conf)

            res = pd.DataFrame({"temp": df["temp"], "pres": df["pres"], "rh": df["rh"],
                                "anomaly": anom, "cause": cause,
                                "severity": np.where(anom, sev, "none"),
                                "confidence": np.where(anom, conf, 0.0).round(3)}, index=df.index)
            for p in PARAMS:
                res[f"sp_{p}"] = f[f"sp_{p}"]
                # corrected value: neighbours' median + this station's offset (fallback: own recent median)
                expect = (f[f"net_{p}"] + off[p]) if has_nb else f[f"med_{p}"]
                expect = expect.fillna(f[f"med_{p}"])
                res[f"fix_{p}"] = np.where(anom, expect, xs[f"x_{p}"]).round(2)
            res["reason"] = ""
            idx = res.index[res["anomaly"]]
            res.loc[idx, "reason"] = [self._reason(z.loc[t], res.at[t, "cause"]) for t in idx]
            out[sid] = res
        return out

    @staticmethod
    def _reason(zrow, cause):
        parts = []
        for c, v in zrow.dropna().nlargest(2).items():
            if v < 3:
                continue
            if c == "rh_res":
                parts.append(f"humidity inconsistent with temperature (z={v:.1f})")
            else:
                k, p = c.split("_")
                parts.append(f"{LABEL[p]} {NAMES[k]} (z={v:.1f})")
        return "; ".join(parts) or cause.replace("_", " ")

    # ------------------------------------------------------------------ explainability
    def explain_shap(self, station, ts):
        """SHAP values of the IsolationForest for one timestamp (needs `pip install shap`)."""
        import shap
        X = self._last[station]
        sv = shap.TreeExplainer(self.iso).shap_values(X.loc[[ts]])
        return pd.Series(np.asarray(sv)[0], index=X.columns).sort_values(key=abs, ascending=False)

    # ------------------------------------------------------------------ sensor health / maintenance
    def health_report(self, results, tol=None):
        tol = tol or {"temp": 2.0, "pres": 3.0, "rh": 10.0}
        rows = []
        for sid, r in results.items():
            score = 100 - min(60.0, float(r.tail(144)["anomaly"].mean()) * 300)
            eta = {}
            for p in PARAMS:
                d = r[f"sp_{p}"].resample("D").median().dropna().tail(5)
                if len(d) < 3:
                    continue
                slope = float(np.polyfit(np.arange(len(d)), d.to_numpy(), 1)[0])
                last = float(d.iloc[-1])
                score -= 20 * min(1.0, abs(last) / tol[p])
                if slope * last > 0 and abs(slope) > 0.02 * tol[p]:
                    eta[p] = max(0.0, (tol[p] - abs(last)) / abs(slope))
            score = max(0.0, score)
            status = "HEALTHY" if score >= 80 else "DEGRADED" if score >= 55 else "FAULTY"
            days = min(eta.values()) if eta else None
            rows.append({"station": sid, "health_score": round(score, 1), "status": status,
                         "anomaly_rate_24h_%": round(100 * float(r.tail(144)["anomaly"].mean()), 2),
                         "maintenance_advice": (f"drift trend: service within ~{days:.0f} days"
                                                if days is not None and days < 14 else "none")})
        return pd.DataFrame(rows).set_index("station")

    # ------------------------------------------------------------------ persistence
    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
