import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.simulate import make_network, make_test_network
from src.skyguard import SkyGuard


def _run():
    model = SkyGuard().fit(make_network(5, 20, seed=1))
    test = make_test_network(5, 20, seed=7)
    return test, model.score(test)


def test_high_precision_and_key_faults_caught():
    test, res = _run()
    tp = fp = 0
    for sid, d in test.items():
        pred = res[sid]["anomaly"]
        tp += int((pred & (d.is_anomaly == 1)).sum())
        fp += int((pred & (d.is_anomaly == 0)).sum())
    assert tp / max(tp + fp, 1) > 0.9
    for sid, d in test.items():
        for t in ("spike", "multi_parameter_fault", "communication_error", "pressure_jump"):
            m = d.anomaly_type == t
            if m.any():
                assert res[sid].loc[m, "anomaly"].mean() > 0.9


def test_healthy_station_stays_clean():
    test, res = _run()
    assert res["AWS_05"]["anomaly"].mean() < 0.01
