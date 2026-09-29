"""SkyGuard AI dashboard.  Run from the project root:  streamlit run app/dashboard.py"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go # type: ignore
import streamlit as st
from plotly.subplots import make_subplots # type: ignore

from src.simulate import make_network, make_test_network
from src.skyguard import SkyGuard, PARAMS, load_csv

st.set_page_config(page_title="SkyGuard AI", page_icon="🛰️", layout="wide")
st.title("🛰️ SkyGuard AI - AWS Anomaly Detection")
st.caption("Temperature · Pressure · Humidity | temporal + spatial + physical + ML consistency")

src = st.sidebar.radio("Data source", ["Simulated network (with injected faults)", "Upload CSV"])
z_thr = st.sidebar.slider("Sensitivity threshold (robust z)", 3.0, 8.0, 5.0, 0.5)


@st.cache_resource
def get_model(z):
    return SkyGuard(z_thr=z).fit(make_network(5, 30, seed=1))


@st.cache_data
def get_sim():
    return make_test_network(5, 30, seed=7)


model = get_model(z_thr)
if src.startswith("Simulated"):
    data = get_sim()
else:
    up = st.sidebar.file_uploader("CSV: station,timestamp,temp,pres,rh", type="csv")
    if up is None:
        st.info("Upload a CSV (columns: station, timestamp, temp, pres, rh; regular time step) to continue.")
        st.stop()
    data = load_csv(up)

results = model.score(data)
health = model.health_report(results)

# ---- KPIs
allres = pd.concat(results.values())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stations", len(results))
c2.metric("Readings analysed", f"{len(allres):,}")
c3.metric("Anomalies flagged", f"{int(allres['anomaly'].sum()):,}")
c4.metric("Genuine weather events kept", f"{int((allres['cause'] == 'genuine_weather_event').sum()):,}")

tab1, tab2, tab3, tab4 = st.tabs(["Network health", "Station explorer", "Alerts & root cause", "Explain / fix"])

with tab1:
    st.subheader("Sensor health & maintenance prediction")
    st.dataframe(health, use_container_width=True)
    st.bar_chart(health["health_score"])

with tab2:
    sid = st.selectbox("Station", list(results))
    r = results[sid]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Temperature (°C)", "Pressure (hPa)", "Humidity (%)"])
    bad = r[r["anomaly"]]
    for i, p in enumerate(PARAMS, 1):
        fig.add_trace(go.Scatter(x=r.index, y=r[p].where(r[p] > -900), name=p, line=dict(width=1)), row=i, col=1)
        fig.add_trace(go.Scatter(x=bad.index, y=bad[p].where(bad[p] > -900), mode="markers",
                                 marker=dict(color="red", size=6), name=f"{p} anomaly"), row=i, col=1)
        fig.add_trace(go.Scatter(x=bad.index, y=bad[f"fix_{p}"], mode="markers",
                                 marker=dict(color="green", size=4), name=f"{p} corrected"), row=i, col=1)
    fig.update_layout(height=750, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    alerts = allres[allres["anomaly"]].copy()
    alerts["station"] = [s for s, r_ in results.items() for _ in range(int(r_["anomaly"].sum()))]
    sev = st.multiselect("Severity", ["critical", "high", "medium", "low"], default=["critical", "high"])
    show = alerts[alerts["severity"].isin(sev)].sort_index(ascending=False)
    st.write(f"{len(show):,} alerts")
    st.dataframe(show[["station", "temp", "pres", "rh", "cause", "severity", "confidence", "reason"]].head(500),
                 use_container_width=True)
    st.bar_chart(alerts["cause"].value_counts())

with tab4:
    sid2 = st.selectbox("Station ", list(results), key="s2")
    r2 = results[sid2]
    bad2 = r2[r2["anomaly"]]
    if bad2.empty:
        st.success("No anomalies on this station.")
    else:
        ts = st.selectbox("Anomalous timestamp", list(bad2.index))
        row = r2.loc[ts]
        st.write(f"**Cause:** {row['cause']} | **Severity:** {row['severity']} | **Confidence:** {row['confidence']:.0%}")
        st.write(f"**Why:** {row['reason']}")
        st.table(pd.DataFrame({"raw": [row[p] for p in PARAMS], "suggested correction": [row[f'fix_{p}'] for p in PARAMS]},
                              index=["Temperature °C", "Pressure hPa", "Humidity %"]))
        try:
            st.write("**SHAP contributions (ML layer):**")
            st.bar_chart(model.explain_shap(sid2, ts).head(8))
        except Exception:
            st.caption("Install `shap` (pip install shap) to see SHAP contributions for the ML layer.")
