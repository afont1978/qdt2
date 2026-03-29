from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from src.q_infratwin.engine import (
    ClassicalSolver,
    EdgeAgentSim,
    FeatureExtractor,
    HybridOrchestrator,
    SimulatedCloudQPU,
    TwinCORE,
)

st.set_page_config(
    page_title="Q-InfraTwin Control Room",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 1.5rem;}
    .demo-card {
        padding: 1rem 1.15rem;
        border: 1px solid rgba(120,120,120,0.22);
        border-radius: 0.9rem;
        background: rgba(250,250,250,0.02);
        margin-bottom: 0.75rem;
    }
    .small-note {
        font-size: 0.92rem;
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def do_rerun() -> None:
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass


def build_runtime(twins: int, policy: str, seed: int):
    import numpy as np

    rng = np.random.default_rng(int(seed))
    core = TwinCORE()
    agents: dict[str, EdgeAgentSim] = {}
    twin_ids: list[str] = []

    for i in range(int(twins)):
        twin_id = f"infra:asset:{i+1:03d}"
        core.create_twin(twin_id=twin_id, level="asset", topology_ref="graph://infra_demo_v1")
        agents[twin_id] = EdgeAgentSim(source_id=twin_id, seed=int(rng.integers(1, 10_000)))
        twin_ids.append(twin_id)

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


def extract_reasons(series: pd.Series) -> list[str]:
    reasons: list[str] = []
    for value in series.fillna(""):
        if isinstance(value, list):
            reasons.extend(value)
        elif isinstance(value, str) and value.startswith("["):
            try:
                reasons.extend(json.loads(value.replace("'", '"')))
            except Exception:
                parsed = [x.strip().strip('"').strip("'") for x in value.strip("[]").split(",") if x.strip()]
                reasons.extend(parsed)
        elif isinstance(value, str) and value:
            reasons.append(value)
    return reasons


def compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "quantum_share": 0.0,
            "fallback_share": 0.0,
            "avg_latency": 0.0,
            "mean_confidence": 0.0,
            "mean_objective": 0.0,
            "avg_queue": 0.0,
            "latency_breach_rate": 0.0,
        }

    quantum_share = float((df["route"] == "QUANTUM").mean())
    fallback_share = float((df["route"] == "FALLBACK_CLASSICAL").mean())
    avg_latency = float(df["exec_ms"].mean())
    mean_confidence = float(df["confidence"].mean())
    mean_objective = float(df["objective_value"].mean())
    avg_queue = float(df["qpu_queue_ms"].dropna().mean()) if df["qpu_queue_ms"].notna().any() else 0.0
    latency_breach_rate = float(df["latency_breach"].mean())
    return {
        "quantum_share": quantum_share,
        "fallback_share": fallback_share,
        "avg_latency": avg_latency,
        "mean_confidence": mean_confidence,
        "mean_objective": mean_objective,
        "avg_queue": avg_queue,
        "latency_breach_rate": latency_breach_rate,
    }


def get_twin_snapshot(core: TwinCORE) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for twin_id, state in sorted(core.registry.items()):
        rows.append(
            {
                "twin_id": twin_id,
                "level": state.level,
                "health": round(float(state.health), 4),
                "x1": round(float(state.x1), 4),
                "x2": round(float(state.x2), 4),
                "x3": round(float(state.x3), 4),
                "last_mode": state.last_action.get("mode", "-"),
                "last_damp": round(float(state.last_action.get("damp", 0.0)), 4),
                "timestamp": state.ts,
            }
        )
    return pd.DataFrame(rows)


def make_markdown_report(df: pd.DataFrame, summary: dict[str, Any], run_id: str, twins: int, policy: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""# Q-InfraTwin Run Summary

Generated: {timestamp}
Run ID: {run_id}
Twins: {twins}
Policy: {policy}
Steps: {len(df)}

## Core KPIs
- Quantum share: {summary['quantum_share']:.1%}
- Fallback share: {summary['fallback_share']:.1%}
- Average latency: {summary['avg_latency']:.1f} ms
- Mean confidence: {summary['mean_confidence']:.3f}
- Mean objective: {summary['mean_objective']:.3f}
- Latency breach rate: {summary['latency_breach_rate']:.1%}
- Average queue time: {summary['avg_queue']:.1f} ms

## Route distribution
{df['route'].value_counts().to_string() if not df.empty else 'No data'}
"""


def reset_runtime() -> None:
    ss.core, ss.orch, ss.agents, ss.twin_ids = build_runtime(ss.twins, ss.policy, ss.seed)
    ss.records = []
    ss.step_id = 0
    ss.correlation_id = f"run-{int(time.time())}"
    ss.running = False
    ss.last_step_ts = 0.0


def step_once() -> None:
    ss.step_id += 1
    twin_id = ss.twin_ids[ss.step_id % len(ss.twin_ids)]
    telemetry = ss.agents[twin_id].step()
    twin_state = ss.core.update_from_telemetry(twin_id, telemetry)
    action, record = ss.orch.step(twin_state, step_id=int(ss.step_id), correlation_id=str(ss.correlation_id))
    ss.core.apply_action(twin_id, action)
    ss.core.append_record(record)
    ss.records.append(record.__dict__)


def run_some_steps() -> None:
    remaining = ss.target_steps - ss.step_id if ss.auto_stop else 10**12
    if remaining <= 0:
        ss.running = False
        return

    if ss.speed_mode == "Real-time":
        step_once()
    else:
        n_steps = min(int(ss.batch_steps), int(remaining))
        for _ in range(n_steps):
            step_once()

    if ss.auto_stop and ss.step_id >= ss.target_steps:
        ss.running = False


def apply_preset(name: str) -> None:
    presets = {
        "Balanced demo": {"twins": 6, "policy": "bandit", "target_steps": 180, "speed_mode": "Real-time", "interval_ms": 250},
        "Fast operator view": {"twins": 8, "policy": "rule", "target_steps": 250, "speed_mode": "Max speed", "batch_steps": 40},
        "Quantum stress": {"twins": 10, "policy": "bandit", "target_steps": 300, "speed_mode": "Max speed", "batch_steps": 60},
    }
    for key, value in presets[name].items():
        ss[key] = value
    reset_runtime()


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
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

if ss.core is None:
    reset_runtime()


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Public demo controls")
    st.caption("Prepared for a public-facing QDT demonstration with deterministic presets and exportable results.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Balanced demo", use_container_width=True):
            apply_preset("Balanced demo")
            do_rerun()
    with col_b:
        if st.button("Fast operator view", use_container_width=True):
            apply_preset("Fast operator view")
            do_rerun()
    if st.button("Quantum stress", use_container_width=True):
        apply_preset("Quantum stress")
        do_rerun()

    st.divider()
    st.markdown("## Runtime configuration")
    ss.twins = st.slider("Number of twins", 1, 20, int(ss.twins), help="Applied after Reset or Start.")
    ss.policy = st.selectbox("Routing policy", ["rule", "bandit"], index=1 if ss.policy == "bandit" else 0)
    ss.seed = st.number_input("Random seed", 1, 1_000_000, int(ss.seed))
    ss.target_steps = st.number_input("Target steps", min_value=1, max_value=1_000_000, value=int(ss.target_steps), step=50)
    ss.auto_stop = st.checkbox("Auto-stop at target steps", value=bool(ss.auto_stop))
    ss.ui_refresh_ms = st.slider("UI refresh (ms)", 50, 2000, int(ss.ui_refresh_ms), step=50)

    st.divider()
    st.markdown("## Execution speed")
    ss.speed_mode = st.selectbox("Mode", ["Real-time", "Max speed"], index=0 if ss.speed_mode == "Real-time" else 1)
    if ss.speed_mode == "Real-time":
        ss.interval_ms = st.slider("Step interval (ms)", 50, 5000, int(ss.interval_ms), step=50)
    else:
        ss.batch_steps = st.slider("Steps per tick", 5, 500, int(ss.batch_steps), step=5)

    st.divider()
    st.markdown("## Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Start", use_container_width=True):
            reset_runtime()
            ss.running = True
            do_rerun()
    with c2:
        if st.button("⏸ Pause", use_container_width=True):
            ss.running = False
            do_rerun()

    c3, c4 = st.columns(2)
    with c3:
        if st.button("⏭ Step", use_container_width=True):
            step_once()
            do_rerun()
    with c4:
        if st.button("⏹ Reset", use_container_width=True):
            reset_runtime()
            do_rerun()

    st.divider()
    st.markdown("## Demo notes")
    st.caption(
        "Use Balanced demo for stakeholder presentations. Use Fast operator view for dense runs. Use Quantum stress to surface routing and fallback behaviour."
    )


# -----------------------------------------------------------------------------
# Live execution loop
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
df = pd.DataFrame(ss.records)
summary = compute_summary(df)
progress = min(float(ss.step_id) / float(max(ss.target_steps, 1)), 1.0) if ss.auto_stop else 0.0
route_counts = df["route"].value_counts() if not df.empty else pd.Series(dtype=int)
fallback_counts = pd.Series(extract_reasons(df.get("fallback_reasons", pd.Series(dtype=object)))).value_counts() if not df.empty else pd.Series(dtype=int)

st.title("Q-InfraTwin — Hybrid Quantum-Classical Control Room")
st.markdown(
    """
    <div class="demo-card">
    <strong>Public demo</strong> of a TRL4-style Quantum Digital Twin orchestration layer. The app simulates telemetry,
    hybrid routing, queue-aware quantum requests, fallback governance, and auditable result inspection for multiple infrastructure twins.
    </div>
    """,
    unsafe_allow_html=True,
)

headline_left, headline_right = st.columns([3, 1])
with headline_left:
    st.caption(
        f"Run ID: {ss.correlation_id} · Policy: {ss.policy} · Twins: {ss.twins} · Mode: {ss.speed_mode}"
    )
with headline_right:
    st.caption("Suitable for Streamlit Cloud, Docker, Render and GitHub-hosted demos.")

if ss.auto_stop:
    st.progress(progress, text=f"Progress: {ss.step_id}/{ss.target_steps} steps")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Steps", f"{ss.step_id} / {ss.target_steps}" if ss.auto_stop else str(ss.step_id))
m2.metric("Quantum share", f"{summary['quantum_share']:.1%}")
m3.metric("Fallback rate", f"{summary['fallback_share']:.1%}")
m4.metric("Avg latency", f"{summary['avg_latency']:.0f} ms")
m5.metric("Mean confidence", f"{summary['mean_confidence']:.2f}")
m6.metric("Running", "Yes" if ss.running else "No")

if df.empty:
    st.info("Ready to run. Choose a preset or press Start/Step from the sidebar.")
    twin_snapshot = get_twin_snapshot(ss.core)
    st.subheader("Initial twin snapshot")
    st.dataframe(twin_snapshot, use_container_width=True, hide_index=True)
    st.stop()

if summary["fallback_share"] > 0.35:
    st.warning("Fallback rate is elevated. This is useful for demoing governance, but it may indicate aggressive quantum routing or queue pressure.")
elif summary["quantum_share"] > 0.45:
    st.success("Quantum routing is active across a significant share of steps, which makes this run well suited for showcasing hybrid orchestration.")
else:
    st.info("This run is currently conservative. Switch to a more aggressive preset if you want more visible quantum activity.")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Twin drill-down", "Audit inspector", "About"])

with tab1:
    row1_left, row1_right = st.columns([2, 1])
    with row1_left:
        st.subheader("Objective and latency evolution")
        plot_df = df.copy()
        plot_df["step"] = range(1, len(plot_df) + 1)
        st.line_chart(plot_df.set_index("step")[["objective_value"]], height=240)
        st.line_chart(plot_df.set_index("step")[["exec_ms"]], height=240)

    with row1_right:
        st.subheader("Routing distribution")
        st.bar_chart(route_counts)
        st.subheader("Fallback reasons")
        if not fallback_counts.empty:
            st.bar_chart(fallback_counts.head(10))
        else:
            st.caption("No fallback reasons recorded.")

    row2_left, row2_right = st.columns([1.1, 1])
    with row2_left:
        st.subheader("Operational twin snapshot")
        twin_snapshot = get_twin_snapshot(ss.core)
        st.dataframe(twin_snapshot, use_container_width=True, hide_index=True, height=280)

    with row2_right:
        st.subheader("Run health summary")
        health_summary = pd.DataFrame(
            {
                "metric": [
                    "Mean objective",
                    "Average queue",
                    "Latency breach rate",
                    "Recorded routes",
                ],
                "value": [
                    f"{summary['mean_objective']:.3f}",
                    f"{summary['avg_queue']:.0f} ms",
                    f"{summary['latency_breach_rate']:.1%}",
                    str(int(route_counts.sum())),
                ],
            }
        )
        st.dataframe(health_summary, use_container_width=True, hide_index=True)

        recent_events = df[["step_id", "twin_id", "route", "exec_ms", "latency_breach"]].tail(10).copy()
        st.markdown("**Recent events**")
        st.dataframe(recent_events, use_container_width=True, hide_index=True, height=220)

with tab2:
    st.subheader("Per-twin analysis")
    twin_ids = sorted(df["twin_id"].unique().tolist())
    selected_twin = st.selectbox("Select twin", twin_ids, index=0)
    twin_df = df[df["twin_id"] == selected_twin].copy()
    twin_df["step"] = range(1, len(twin_df) + 1)

    c_left, c_right = st.columns(2)
    with c_left:
        st.line_chart(twin_df.set_index("step")[["objective_value"]], height=240)
        st.line_chart(twin_df.set_index("step")[["exec_ms"]], height=240)
    with c_right:
        st.bar_chart(twin_df["route"].value_counts())
        if twin_df["qpu_queue_ms"].notna().any():
            st.line_chart(twin_df.set_index("step")[["qpu_queue_ms"]], height=240)
        else:
            st.caption("No queue metrics available for this twin yet.")

    twin_state = ss.core.registry[selected_twin]
    twin_state_df = pd.DataFrame(
        [
            {"field": "health", "value": round(float(twin_state.health), 4)},
            {"field": "x1", "value": round(float(twin_state.x1), 4)},
            {"field": "x2", "value": round(float(twin_state.x2), 4)},
            {"field": "x3", "value": round(float(twin_state.x3), 4)},
            {"field": "last_mode", "value": twin_state.last_action.get("mode", "-")},
            {"field": "last_damp", "value": round(float(twin_state.last_action.get("damp", 0.0)), 4)},
            {"field": "timestamp", "value": twin_state.ts},
        ]
    )
    st.markdown("**Current state**")
    st.dataframe(twin_state_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Audit table")
    audit_cols = [
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
    st.dataframe(df[audit_cols].tail(50), use_container_width=True, height=280, hide_index=True)

    row_idx = st.number_input(
        "Select row index for inspection",
        min_value=0,
        max_value=max(0, len(df) - 1),
        value=max(0, len(df) - 1),
        step=1,
    )
    row = df.iloc[int(row_idx)]

    def safe_json(value: Any) -> Any:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return json.loads(value)
        except Exception:
            return None

    qre_obj = safe_json(row.get("qre_json"))
    result_obj = safe_json(row.get("result_json"))

    q_col, r_col = st.columns(2)
    with q_col:
        st.markdown("**Quantum Request Envelope**")
        st.json(qre_obj) if qre_obj else st.caption("No QRE available for this step.")
    with r_col:
        st.markdown("**Quantum result envelope**")
        st.json(result_obj) if result_obj else st.caption("No quantum result recorded for this step.")

    st.subheader("Export run")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")
    report_md = make_markdown_report(df, summary, str(ss.correlation_id), int(ss.twins), str(ss.policy)).encode("utf-8")

    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("Download CSV", csv_bytes, file_name="q_infratwin_live_run.csv", mime="text/csv", use_container_width=True)
    with e2:
        st.download_button("Download JSON", json_bytes, file_name="q_infratwin_live_run.json", mime="application/json", use_container_width=True)
    with e3:
        st.download_button("Download run summary", report_md, file_name="q_infratwin_run_summary.md", mime="text/markdown", use_container_width=True)

with tab4:
    st.subheader("What this demo shows")
    st.markdown(
        """
        This application presents a deployable demonstration of a Quantum Digital Twin orchestration loop for infrastructure assets.
        The demo focuses on five ideas: synthetic telemetry generation, hybrid classical-quantum routing, queue-aware quantum execution,
        fallback governance, and audit-ready result inspection.
        """
    )

    st.markdown("**Suggested public demo flow**")
    st.markdown(
        """
        1. Launch **Balanced demo** and let it run for 100–180 steps.
        2. Show the **Overview** tab to explain objective, latency and routing.
        3. Open **Twin drill-down** to focus on one asset.
        4. Use **Audit inspector** to expose the QRE and result envelopes.
        5. Export the run summary as a stakeholder-ready attachment.
        """
    )

    st.markdown("**Deployment notes**")
    st.markdown(
        """
        - Streamlit Cloud: simplest public demo path.
        - Docker/Render/Railway: best when you want a more controlled web deployment.
        - GitHub: recommended source of truth for versioning and updates.
        """
    )
