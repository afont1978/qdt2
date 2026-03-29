from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.q_infratwin.engine import (
    TwinCORE,
    EdgeAgentSim,
    FeatureExtractor,
    ClassicalSolver,
    SimulatedCloudQPU,
    HybridOrchestrator,
)

st.set_page_config(page_title="Q-InfraTwin Control Room", layout="wide")
st.title("Q-InfraTwin — TRL4 Control Room (Hybrid Quantum–Classical)")

st.markdown(
    """
Start / Pause / Step / Reset

Speed modes: Real-time (fixed interval) and Max speed (batch per tick).
Choose total steps and auto-stop when reached.

Deployment-ready Streamlit app with direct package imports.
"""
)


def do_rerun() -> None:
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


ss = st.session_state
ss.setdefault("core", None)
ss.setdefault("orch", None)
ss.setdefault("agents", None)
ss.setdefault("twin_ids", None)
ss.setdefault("records", [])
ss.setdefault("step_id", 0)
ss.setdefault("running", False)
ss.setdefault("speed_mode", "Real-time")
ss.setdefault("interval_ms", 500)
ss.setdefault("batch_steps", 25)
ss.setdefault("policy", "bandit")
ss.setdefault("twins", 5)
ss.setdefault("seed", 42)
ss.setdefault("correlation_id", None)
ss.setdefault("target_steps", 500)
ss.setdefault("auto_stop", True)
ss.setdefault("ui_refresh_ms", 200)
ss.setdefault("last_step_ts", 0.0)


def build_runtime(twins: int, policy: str, seed: int):
    import numpy as np

    rng = np.random.default_rng(int(seed))
    core = TwinCORE()
    agents = {}
    twin_ids = []

    for i in range(int(twins)):
        tid = f"infra:asset:{i+1:03d}"
        core.create_twin(twin_id=tid, level="asset", topology_ref="graph://infra_demo_v1")
        agents[tid] = EdgeAgentSim(source_id=tid, seed=int(rng.integers(1, 10_000)))
        twin_ids.append(tid)

    extractor = FeatureExtractor(window=12)
    classical = ClassicalSolver()
    gateway = SimulatedCloudQPU(seed=int(seed))
    orch = HybridOrchestrator(
        gateway=gateway,
        classical=classical,
        extractor=extractor,
        policy=str(policy),
        seed=int(seed),
    )
    return core, orch, agents, twin_ids


def reset_runtime() -> None:
    ss.core, ss.orch, ss.agents, ss.twin_ids = build_runtime(ss.twins, ss.policy, ss.seed)
    ss.records = []
    ss.step_id = 0
    ss.correlation_id = f"run-{int(time.time())}"
    ss.running = False
    ss.last_step_ts = 0.0


def step_once() -> None:
    ss.step_id += 1
    tid = ss.twin_ids[ss.step_id % len(ss.twin_ids)]
    tel = ss.agents[tid].step()
    stt = ss.core.update_from_telemetry(tid, tel)

    action, rec = ss.orch.step(stt, step_id=int(ss.step_id), correlation_id=str(ss.correlation_id))
    ss.core.apply_action(tid, action)
    ss.core.append_record(rec)
    ss.records.append(rec.__dict__)


if ss.core is None:
    reset_runtime()


def run_some_steps() -> None:
    remaining = ss.target_steps - ss.step_id if ss.auto_stop else 10**12
    if remaining <= 0:
        ss.running = False
        return

    if ss.speed_mode == "Real-time":
        step_once()
    else:
        n = min(int(ss.batch_steps), int(remaining))
        for _ in range(n):
            step_once()

    if ss.auto_stop and ss.step_id >= ss.target_steps:
        ss.running = False


with st.sidebar:
    st.header("Config")
    ss.twins = st.slider("Twins", 1, 20, int(ss.twins))
    ss.policy = st.selectbox("Policy", ["rule", "bandit"], index=1 if ss.policy == "bandit" else 0)
    ss.seed = st.number_input("Seed", 1, 1_000_000, int(ss.seed))

    st.divider()
    st.header("Run control")
    ss.target_steps = st.number_input(
        "Total steps to run", min_value=1, max_value=1_000_000, value=int(ss.target_steps), step=50
    )
    ss.auto_stop = st.checkbox("Auto-stop at total steps", value=bool(ss.auto_stop))
    ss.ui_refresh_ms = st.slider("UI refresh (ms)", 50, 2000, int(ss.ui_refresh_ms), step=50)

    st.divider()
    st.header("Speed")
    ss.speed_mode = st.selectbox("Mode", ["Real-time", "Max speed"], index=0 if ss.speed_mode == "Real-time" else 1)
    if ss.speed_mode == "Real-time":
        ss.interval_ms = st.slider("Step interval (ms)", 50, 5000, int(ss.interval_ms), step=50)
    else:
        ss.batch_steps = st.slider("Steps per tick", 5, 500, int(ss.batch_steps), step=5)

    st.divider()
    st.header("Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start", use_container_width=True):
            reset_runtime()
            ss.running = True
            do_rerun()
    with col2:
        if st.button("Pause", use_container_width=True):
            ss.running = False
            do_rerun()

    col3, col4 = st.columns(2)
    with col3:
        if st.button("Step", use_container_width=True):
            step_once()
            do_rerun()
    with col4:
        if st.button("Reset", use_container_width=True):
            reset_runtime()
            do_rerun()

if ss.running:
    now = time.time()
    last = ss.get("last_step_ts", 0.0)
    if ss.speed_mode == "Real-time":
        if (now - last) * 1000.0 >= float(ss.interval_ms):
            run_some_steps()
            ss.last_step_ts = time.time()
    else:
        run_some_steps()
        ss.last_step_ts = time.time()

    time.sleep(float(ss.ui_refresh_ms) / 1000.0)
    do_rerun()


df = pd.DataFrame(ss.records)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Steps", f"{ss.step_id} / {ss.target_steps}" if ss.auto_stop else str(ss.step_id))
c2.metric("Running", "Yes" if ss.running else "No")
c3.metric("Mode", ss.speed_mode)
c4.metric("Twins", str(ss.twins))
c5.metric("Policy", ss.policy)
c6.metric("Auto-stop", "On" if ss.auto_stop else "Off")

if df.empty:
    st.info("Ready. Use Reset → Start or Step.")
    st.stop()

st.divider()
left, right = st.columns([2, 1])

with left:
    st.subheader("Objective and latency")
    dfp = df.copy()
    dfp["step"] = range(1, len(dfp) + 1)
    st.line_chart(dfp.set_index("step")[["objective_value"]], height=220)
    st.line_chart(dfp.set_index("step")[["exec_ms"]], height=220)

with right:
    st.subheader("Routing distribution")
    st.bar_chart(df["route"].value_counts())

    st.subheader("Fallback reasons")
    reasons = []
    for x in df["fallback_reasons"].fillna(""):
        if isinstance(x, list):
            reasons.extend(x)
        elif isinstance(x, str) and x.startswith("["):
            try:
                reasons.extend(json.loads(x.replace("'", '"')))
            except Exception:
                reasons.extend([r.strip().strip('"').strip("'") for r in x.strip("[]").split(",") if r.strip()])
        elif isinstance(x, str) and x:
            reasons.append(x)
    if reasons:
        st.bar_chart(pd.Series(reasons).value_counts().head(10))
    else:
        st.caption("No fallback reasons recorded.")

st.divider()
st.subheader("Per-twin drill-down")
twin_ids = sorted(df["twin_id"].unique().tolist())
sel = st.selectbox("Select twin_id", twin_ids, index=0)

dft = df[df["twin_id"] == sel].copy()
dft["step"] = range(1, len(dft) + 1)
a1, a2 = st.columns(2)
with a1:
    st.line_chart(dft.set_index("step")[["objective_value"]], height=220)
    st.line_chart(dft.set_index("step")[["exec_ms"]], height=220)
with a2:
    st.bar_chart(dft["route"].value_counts())
    if "qpu_queue_ms" in dft and dft["qpu_queue_ms"].notna().any():
        st.line_chart(dft.set_index("step")[["qpu_queue_ms"]], height=220)

st.divider()
st.subheader("Audit inspector")
cols = [
    "step_id",
    "ts",
    "twin_id",
    "route",
    "exec_ms",
    "qpu_queue_ms",
    "noise_proxy",
    "cost_eur",
    "latency_breach",
]
st.dataframe(df[cols].tail(50), use_container_width=True, height=260)

row_idx = st.number_input(
    "Pick a row index (0-based)", min_value=0, max_value=max(0, len(df) - 1), value=max(0, len(df) - 1), step=1
)
row = df.iloc[int(row_idx)]


def safe_json(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


qre_obj = safe_json(row.get("qre_json"))
res_obj = safe_json(row.get("result_json"))

b1, b2 = st.columns(2)
with b1:
    st.markdown("**QRE**")
    st.json(qre_obj) if qre_obj else st.caption("No QRE (classical step).")
with b2:
    st.markdown("**Quantum Result**")
    st.json(res_obj) if res_obj else st.caption("No quantum result recorded.")

st.divider()
st.subheader("Export current run")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download CSV",
    data=csv_bytes,
    file_name="q_infratwin_live_run.csv",
    mime="text/csv",
    use_container_width=True,
)
