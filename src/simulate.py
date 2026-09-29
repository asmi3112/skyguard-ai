"""Synthetic AWS network generator + anomaly injector (for training and testing SkyGuard AI).

Real IMD data can be dropped in through skyguard.load_csv(); this module lets you
develop and benchmark without waiting for a dataset.
"""
import numpy as np
import pandas as pd

PARAMS = ["temp", "pres", "rh"]


def _smooth(rng, n, std, win=144):
    """Slowly varying 'regional weather' signal shared by all stations."""
    x = pd.Series(np.cumsum(rng.normal(size=n))).rolling(win, min_periods=1).mean().to_numpy()
    x = x - x.mean()
    return std * x / x.std()


def make_network(n_stations=5, n_days=30, step_min=10, seed=0, with_events=False):
    """Return {station_id: DataFrame[temp, pres, rh, is_anomaly, anomaly_type, genuine_event]}.

    with_events=True adds thunderstorm-like GENUINE events seen by every station
    (big temperature drop, humidity surge, pressure dip). A good detector must NOT flag them.
    """
    rng = np.random.default_rng(seed)
    n = n_days * 24 * 60 // step_min
    idx = pd.date_range("2025-06-01", periods=n, freq=f"{step_min}min")
    h = idx.hour.to_numpy() + idx.minute.to_numpy() / 60

    reg_t, reg_p, reg_h = _smooth(rng, n, 1.5), _smooth(rng, n, 3.0), np.zeros(n)
    event = np.zeros(n, dtype=int)
    if with_events:
        starts = np.linspace(700, n - 200, 4).astype(int) + rng.integers(-100, 100, 4)
        bump = np.sin(np.linspace(0, np.pi, 24))
        for s in starts:
            reg_t[s:s + 24] -= 8 * bump
            reg_h[s:s + 24] += 6 * bump
            reg_p[s:s + 24] -= 6 * bump
            event[s:s + 24] = 1

    dfs = {}
    for k in range(n_stations):
        # each physical station keeps the SAME calibration offsets in every generated dataset
        o = np.random.default_rng(1000 + k)
        o_t, o_p, o_h = o.normal(0, 0.8), o.normal(0, 1.5), o.normal(0, 3)
        T = 28 + 6 * np.sin(2 * np.pi * (h - 9) / 24) + reg_t + o_t + rng.normal(0, 0.3, n)
        P = 1008 + 1.2 * np.sin(2 * np.pi * (h - 4) / 12) + reg_p + o_p + rng.normal(0, 0.15, n)
        H = 65 - 2.2 * (T - 28) + reg_h + o_h + rng.normal(0, 2, n)
        df = pd.DataFrame({"temp": T, "pres": P, "rh": np.clip(H, 5, 100)}, index=idx)
        df["is_anomaly"] = 0
        df["anomaly_type"] = "normal"
        df["genuine_event"] = event
        dfs[f"AWS_{k + 1:02d}"] = df
    return dfs


def inject_anomalies(df, seed=0, n_each=3):
    """Inject labelled faults into ONE station's data and return a new DataFrame."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    X = {p: df[p].to_numpy().copy() for p in PARAMS}
    lab = np.zeros(n, dtype=int)
    typ = np.array(["normal"] * n, dtype=object)
    used = np.zeros(n, dtype=bool)

    def pick(L):
        for _ in range(300):
            s = int(rng.integers(300, n - L - 10))
            if not used[s - 40:s + L + 40].any():
                used[s:s + L] = True
                return s
        return None

    def mark(s, L, t):
        lab[s:s + L] = 1
        typ[s:s + L] = t

    for _ in range(n_each):
        s = pick(1)                                   # 1) single-sample spike
        if s is not None:
            X["temp"][s] += rng.choice([-1, 1]) * rng.uniform(12, 20)
            mark(s, 1, "spike")

        s = pick(30)                                  # 2) frozen sensor
        if s is not None:
            p = str(rng.choice(PARAMS))
            X[p][s:s + 30] = X[p][s]
            mark(s + 1, 29, "frozen_sensor")

        s = pick(400)                                 # 3) slow calibration drift
        if s is not None:
            X["temp"][s:s + 400] += np.linspace(0, 5, 400)
            mark(s, 400, "calibration_drift")

        s = pick(8)                                   # 4) communication dropout
        if s is not None:
            sentinel = float(rng.choice([np.nan, -999.0]))
            for p in PARAMS:
                X[p][s:s + 8] = sentinel
            mark(s, 8, "communication_error")

        s = pick(12)                                  # 5) pressure level jump
        if s is not None:
            X["pres"][s:s + 12] += rng.choice([-1, 1]) * rng.uniform(12, 25)
            mark(s, 12, "pressure_jump")

        s = pick(15)                                  # 6) humidity stuck at an impossible level
        if s is not None:
            X["rh"][s:s + 15] = float(rng.choice([5.0, 100.0]))
            mark(s, 15, "humidity_fault")

        s = pick(5)                                   # 7) the "55 C + 100 % RH + pressure crash" case
        if s is not None:
            X["temp"][s:s + 5] = 55.0
            X["rh"][s:s + 5] = 100.0
            X["pres"][s:s + 5] -= 20.0
            mark(s, 5, "multi_parameter_fault")

    for p in PARAMS:
        df[p] = X[p]
    df["is_anomaly"] = lab
    df["anomaly_type"] = typ
    return df


def make_test_network(n_stations=5, n_days=30, seed=7, faulty=("AWS_01", "AWS_03")):
    """Test network: genuine regional events everywhere + injected faults on selected stations."""
    dfs = make_network(n_stations, n_days, seed=seed, with_events=True)
    for i, sid in enumerate(faulty):
        dfs[sid] = inject_anomalies(dfs[sid], seed=seed + 10 + i)
    return dfs
