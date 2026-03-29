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
    page_title="Hybrid Quantum-Classical Control Room",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

VISUAL_STYLES = {
    "Mission Control": {
        "font_import": "@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');",
        "heading_font": "'Rajdhani', sans-serif",
        "body_font": "'Inter', sans-serif",
        "mono_font": "'JetBrains Mono', monospace",
        "title_letter_spacing": "0.04em",
    },
    "Cyber Minimal": {
        "font_import": "@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600;700&display=swap');",
        "heading_font": "'Space Grotesk', sans-serif",
        "body_font": "'Inter', sans-serif",
        "mono_font": "'JetBrains Mono', monospace",
        "title_letter_spacing": "0.02em",
    },
    "Research Console": {
        "font_import": "@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');",
        "heading_font": "'Chakra Petch', sans-serif",
        "body_font": "'Inter', sans-serif",
        "mono_font": "'JetBrains Mono', monospace",
        "title_letter_spacing": "0.03em",
    },
}

ACCENT_PALETTES = {
    "Cobalt Mint": {
        "primary": "#69e7d5",
        "secondary": "#7aa2ff",
        "muted": "#a9bedf",
        "chip_bg": "rgba(105, 231, 213, 0.14)",
        "hero_glow": "rgba(122, 162, 255, 0.22)",
        "hero_glow_two": "rgba(105, 231, 213, 0.14)",
        "sidebar_top": "#f7f9fc",
        "sidebar_bottom": "#edf2f8",
        "sidebar_border": "rgba(15, 23, 36, 0.08)",
    },
    "Graphite Cyan": {
        "primary": "#7ae6ff",
        "secondary": "#8ea6ff",
        "muted": "#bdd3e6",
        "chip_bg": "rgba(122, 230, 255, 0.14)",
        "hero_glow": "rgba(122, 230, 255, 0.16)",
        "hero_glow_two": "rgba(142, 166, 255, 0.16)",
        "sidebar_top": "#f7f9fc",
        "sidebar_bottom": "#edf2f8",
        "sidebar_border": "rgba(15, 23, 36, 0.08)",
    },
    "Violet Ice": {
        "primary": "#b6c7ff",
        "secondary": "#8af0dc",
        "muted": "#cad6f1",
        "chip_bg": "rgba(182, 199, 255, 0.14)",
        "hero_glow": "rgba(182, 199, 255, 0.18)",
        "hero_glow_two": "rgba(138, 240, 220, 0.12)",
        "sidebar_top": "#f7f9fc",
        "sidebar_bottom": "#edf2f8",
        "sidebar_border": "rgba(15, 23, 36, 0.08)",
    },
}


def build_theme_css(style_name: str, palette_name: str) -> str:
    style = VISUAL_STYLES.get(style_name, VISUAL_STYLES["Mission Control"])
    palette = ACCENT_PALETTES.get(palette_name, ACCENT_PALETTES["Cobalt Mint"])
    return f"""
    <style>
    {style['font_import']}
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@500;700&display=swap');

    :root {{
        --primary: {palette['primary']};
        --secondary: {palette['secondary']};
        --muted: {palette['muted']};
        --chip-bg: {palette['chip_bg']};
    }}

    html, body, [class*="css"] {{
        font-family: {style['body_font']};
        color: #edf4ff !important;
    }}

    p, li, div, label, span, small, .stMarkdown, .stCaption {{
        color: #eaf2ff !important;
    }}

    .stApp {{
        background:
            radial-gradient(circle at top right, {palette['hero_glow']}, transparent 30%),
            radial-gradient(circle at top left, {palette['hero_glow_two']}, transparent 25%),
            linear-gradient(180deg, #02060d 0%, #07101b 55%, #040913 100%);
    }}

    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1.7rem;
        max-width: 1500px;
    }}

    h1, h2, h3, .tech-title, .kpi-label, .brand-wordmark, .section-heading {{
        font-family: {style['heading_font']} !important;
        letter-spacing: {style['title_letter_spacing']};
        color: #f7fbff !important;
    }}

    .brand-shell {{
        display: grid;
        grid-template-columns: minmax(0, 1.6fr) minmax(300px, 1fr);
        gap: 1rem;
        margin-bottom: 0.95rem;
    }}

    .hero-card {{
        padding: 1.2rem 1.3rem 1.15rem 1.3rem;
        border: 1px solid rgba(122, 146, 190, 0.28);
        border-radius: 1.15rem;
        background: linear-gradient(180deg, rgba(14, 20, 33, 0.96) 0%, rgba(8, 13, 23, 0.96) 100%);
        box-shadow: 0 0 0 1px rgba(99, 102, 241, 0.06), 0 14px 38px rgba(0, 0, 0, 0.28);
    }}

    .hero-topline {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--primary) !important;
        padding: 0.35rem 0.62rem;
        border-radius: 999px;
        background: var(--chip-bg);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 0.75rem;
    }}

    .brand-row {{
        display: flex;
        align-items: flex-start;
        gap: 0.9rem;
        margin-bottom: 0.7rem;
    }}

    .brand-badge {{
        width: 3.1rem;
        height: 3.1rem;
        border-radius: 0.95rem;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.45rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: inset 0 0 18px rgba(255,255,255,0.025);
    }}

    .brand-wordmark {{
        font-size: 2.1rem;
        line-height: 1;
        margin-bottom: 0.35rem;
    }}

    .brand-subtitle {{
        font-size: 1rem;
        line-height: 1.5;
        color: #edf5ff !important;
        max-width: 54rem;
    }}

    .hero-chip-row, .status-chip-row {{
        display: flex;
        gap: 0.55rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }}

    .hero-chip, .status-chip {{
        padding: 0.44rem 0.68rem;
        border-radius: 999px;
        font-size: 0.8rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        color: #edf5ff !important;
    }}

    .status-panel {{
        display: grid;
        gap: 0.75rem;
    }}

    .status-card {{
        padding: 1rem 1.05rem;
        border-radius: 1rem;
        border: 1px solid rgba(124, 146, 183, 0.22);
        background: linear-gradient(180deg, rgba(12, 18, 29, 0.98), rgba(7, 12, 21, 0.98));
    }}

    .status-label {{
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted) !important;
        margin-bottom: 0.35rem;
    }}

    .status-value {{
        font-family: {style['heading_font']};
        font-size: 1.15rem;
        color: #ffffff !important;
    }}

    .architecture-card {{
        margin-top: 0.95rem;
        margin-bottom: 0.9rem;
        padding: 1rem 1.1rem 1.05rem 1.1rem;
        border: 1px solid rgba(124, 146, 183, 0.22);
        border-radius: 1.1rem;
        background: linear-gradient(180deg, rgba(11, 16, 27, 0.96), rgba(7, 12, 22, 0.96));
    }}

    .architecture-title {{
        font-family: {style['heading_font']};
        font-size: 1rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #f7fbff !important;
        margin-bottom: 0.8rem;
    }}

    .architecture-grid {{
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 0.7rem;
        align-items: stretch;
    }}

    .arch-node {{
        position: relative;
        min-height: 112px;
        padding: 0.9rem 0.85rem;
        border-radius: 0.95rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.08);
        overflow: hidden;
    }}

    .arch-node::before {{
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: linear-gradient(180deg, var(--primary), var(--secondary));
        border-radius: 4px;
    }}

    .arch-node-title {{
        font-family: {style['heading_font']};
        font-size: 0.86rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #f8fbff !important;
        margin-bottom: 0.45rem;
    }}

    .arch-node-text {{
        font-size: 0.84rem;
        line-height: 1.45;
        color: #dce8fb !important;
    }}

    .arch-node-arrow {{
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.05rem;
        color: var(--primary) !important;
        opacity: 0.82;
    }}

    .kpi-card {{
        border: 1px solid rgba(95, 117, 162, 0.28);
        border-radius: 1rem;
        padding: 0.95rem 1rem 0.85rem 1rem;
        min-height: 132px;
        background: linear-gradient(180deg, rgba(14, 21, 34, 0.97), rgba(7, 13, 24, 0.97));
        box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.04), 0 10px 24px rgba(0,0,0,0.22);
    }}

    .kpi-label {{
        color: var(--muted) !important;
        font-size: 0.81rem;
        text-transform: uppercase;
        margin-bottom: 0.28rem;
    }}

    .kpi-value {{
        font-family: {style['heading_font']};
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.08;
        color: #ffffff !important;
        margin-bottom: 0.42rem;
    }}

    .kpi-trend {{
        font-size: 0.98rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.35rem;
        color: #dbe8ff !important;
    }}

    .kpi-sub {{
        font-size: 0.91rem;
        color: #c7d6ef !important;
        margin-top: 0.35rem;
    }}

    .section-note {{
        color: #d7e6ff !important;
        opacity: 0.92;
        margin-top: -0.35rem;
        margin-bottom: 0.5rem;
    }}

    .info-panel {{
        padding: 0.95rem 1rem;
        border-radius: 1rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, rgba(11, 17, 28, 0.96), rgba(7, 12, 21, 0.96));
        margin-bottom: 0.8rem;
    }}

    .info-title {{
        font-family: {style['heading_font']};
        font-size: 0.96rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #f7fbff !important;
        margin-bottom: 0.45rem;
    }}

    .info-body {{
        color: #dce8fb !important;
        line-height: 1.55;
        font-size: 0.92rem;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.4rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        color: #eaf2ff !important;
        background: rgba(255,255,255,0.03);
        border-radius: 999px;
        padding-left: 1rem;
        padding-right: 1rem;
        border: 1px solid rgba(255,255,255,0.06);
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0.04)) !important;
        border-color: rgba(255,255,255,0.12) !important;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {palette['sidebar_top']} 0%, {palette['sidebar_bottom']} 100%) !important;
        border-right: 1px solid {palette['sidebar_border']};
    }}

    section[data-testid="stSidebar"] * {{
        color: #111827 !important;
    }}

    .sidebar-brand {{
        padding: 1rem 1rem 0.95rem 1rem;
        border-radius: 1rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(255,255,255,0.70));
        border: 1px solid rgba(15, 23, 36, 0.06);
        box-shadow: 0 8px 22px rgba(15, 23, 36, 0.08);
        margin-bottom: 0.85rem;
    }}

    .sidebar-brand-title {{
        font-family: {style['heading_font']};
        font-size: 1.15rem;
        color: #0f1724 !important;
        margin-bottom: 0.28rem;
    }}

    .sidebar-brand-text {{
        color: #334155 !important;
        font-size: 0.89rem;
        line-height: 1.45;
    }}

    .sidebar-section {{
        margin-top: 0.35rem;
        margin-bottom: 0.35rem;
        padding: 0.78rem 0.85rem 0.72rem 0.85rem;
        border-radius: 0.95rem;
        border: 1px solid rgba(15, 23, 36, 0.07);
        background: rgba(255,255,255,0.58);
    }}

    .sidebar-section-title {{
        font-family: {style['heading_font']};
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 0.86rem;
        color: #0f1724 !important;
        margin-bottom: 0.2rem;
    }}

    .sidebar-section-copy {{
        font-size: 0.84rem;
        color: #475569 !important;
        line-height: 1.4;
    }}

    section[data-testid="stSidebar"] [data-baseweb="select"] > div,
    section[data-testid="stSidebar"] [data-baseweb="input"] > div,
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea {{
        background: rgba(255,255,255,0.88) !important;
        color: #0f1724 !important;
        border-color: rgba(15, 23, 36, 0.15) !important;
    }}

    section[data-testid="stSidebar"] button {{
        border-radius: 0.8rem !important;
        border: 1px solid rgba(15, 23, 36, 0.10) !important;
        box-shadow: none !important;
    }}

    .stSlider label, .stSelectbox label, .stNumberInput label, .stCheckbox label {{
        color: #edf4ff !important;
    }}

    .stDataFrame, .stTable {{
        color: #edf4ff !important;
    }}

    .stCodeBlock, code, pre {{
        font-family: {style['mono_font']} !important;
    }}

    .trend-up {{ color: #43f08f !important; }}
    .trend-down {{ color: #ff7a92 !important; }}
    .trend-flat {{ color: #ffd166 !important; }}
    .trend-live {{ color: var(--primary) !important; }}
    .trend-pause {{ color: #ffd166 !important; }}
    .trend-idle {{ color: #afc3e8 !important; }}

    @media (max-width: 1200px) {{
        .brand-shell {{ grid-template-columns: 1fr; }}
        .architecture-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
        .arch-node-arrow {{ display: none; }}
    }}

    @media (max-width: 780px) {{
        .architecture-grid {{ grid-template-columns: 1fr; }}
    }}
    </style>
    """


def hero_panel(summary: dict[str, Any], running: bool, style_name: str, palette_name: str, route_counts: pd.Series) -> str:
    run_state = "Live" if running else "Standby"
    routes = ", ".join(f"{k}: {v}" for k, v in route_counts.to_dict().items()) if not route_counts.empty else "No routes yet"
    return f"""
    <div class="brand-shell">
      <div class="hero-card">
        <div class="hero-topline">⚛ Operational Demonstrator · Hybrid orchestration</div>
        <div class="brand-row">
          <div class="brand-badge">⚛</div>
          <div>
            <div class="brand-wordmark">Hybrid Quantum-Classical Control Room</div>
            <div class="brand-subtitle">
              Live control-room view for a multi-twin orchestration layer combining classical solving, cloud-accessed quantum execution,
              queue-aware governance and audit-ready result tracing in a single operational interface.
            </div>
          </div>
        </div>
        <div class="hero-chip-row">
          <div class="hero-chip">Style: {style_name}</div>
          <div class="hero-chip">Palette: {palette_name}</div>
          <div class="hero-chip">Quantum share: {summary['quantum_share']:.1%}</div>
          <div class="hero-chip">Fallback rate: {summary['fallback_share']:.1%}</div>
        </div>
      </div>
      <div class="status-panel">
        <div class="status-card">
          <div class="status-label">System state</div>
          <div class="status-value">{run_state}</div>
          <div class="status-chip-row">
            <div class="status-chip">Mean confidence {summary['mean_confidence']:.2f}</div>
            <div class="status-chip">Avg latency {summary['avg_latency']:.0f} ms</div>
          </div>
        </div>
        <div class="status-card">
          <div class="status-label">Routing mix</div>
          <div class="status-value">{routes}</div>
        </div>
      </div>
    </div>
    """


def architecture_panel() -> str:
    nodes = [
        ("Edge telemetry", "Synthetic infrastructure signals and state changes emitted per asset twin."),
        ("Twin runtime", "TwinCORE registry updates health, state vectors and last applied actions."),
        ("Hybrid orchestrator", "Routing policy evaluates eligibility, SLA, queue pressure and expected value."),
        ("Classical / QPU", "Classical baseline or simulated cloud quantum execution produces candidate decisions."),
        ("Governance", "Fallback checks, validation, latency control and breach handling decide final action."),
        ("Control room", "KPIs, live trends, per-twin drill-down and audit inspector expose the full trace."),
    ]
    items = []
    for i, (title, text) in enumerate(nodes):
        items.append(
            f'<div class="arch-node"><div class="arch-node-title">{title}</div><div class="arch-node-text">{text}</div></div>'
        )
    grid = ''.join(items)
    return f"""
    <div class="architecture-card">
      <div class="architecture-title">System architecture</div>
      <div class="architecture-grid">{grid}</div>
    </div>
    """


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
    return f"""# Hybrid Quantum-Classical Control Room — Run Summary

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
        "Balanced demo": {"twins": 6, "policy": "bandit", "target_steps": 180, "speed_mode": "Real-time", "interval_ms": 180},
        "Fast operator view": {"twins": 8, "policy": "rule", "target_steps": 250, "speed_mode": "Max speed", "batch_steps": 40},
        "Quantum stress": {"twins": 10, "policy": "bandit", "target_steps": 300, "speed_mode": "Real-time", "interval_ms": 120},
    }
    for key, value in presets[name].items():
        ss[key] = value
    reset_runtime()


def trend_descriptor(current: float, previous: float, better: str = "up", fmt: str = ".1f", suffix: str = "") -> dict[str, str]:
    delta = current - previous
    ref = max(abs(previous), 1e-9)
    tol = max(ref * 0.03, 1e-9)
    if abs(delta) <= tol:
        return {
            "icon": "●",
            "label": "Stable",
            "css": "trend-flat",
            "delta": f"{delta:{fmt}}{suffix}",
        }

    improving = (better == "up" and delta > 0) or (better == "down" and delta < 0)
    return {
        "icon": "▲" if improving else "▼",
        "label": "Improving" if improving else "Worsening",
        "css": "trend-up" if improving else "trend-down",
        "delta": f"{delta:+{fmt}}{suffix}",
    }


def metric_trends(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    if len(df) < 8:
        baseline = {"icon": "●", "label": "Collecting baseline", "css": "trend-flat", "delta": "—"}
        return {
            "quantum": baseline,
            "fallback": baseline,
            "latency": baseline,
            "confidence": baseline,
        }

    window = max(4, min(16, len(df) // 3))
    prev = df.iloc[-2 * window:-window] if len(df) >= 2 * window else df.iloc[:window]
    recent = df.iloc[-window:]

    quantum_prev = float((prev["route"] == "QUANTUM").mean())
    quantum_recent = float((recent["route"] == "QUANTUM").mean())
    fallback_prev = float((prev["route"] == "FALLBACK_CLASSICAL").mean())
    fallback_recent = float((recent["route"] == "FALLBACK_CLASSICAL").mean())
    latency_prev = float(prev["exec_ms"].mean())
    latency_recent = float(recent["exec_ms"].mean())
    conf_prev = float(prev["confidence"].mean())
    conf_recent = float(recent["confidence"].mean())

    return {
        "quantum": trend_descriptor(quantum_recent, quantum_prev, better="up", fmt=".1%"),
        "fallback": trend_descriptor(fallback_recent, fallback_prev, better="down", fmt=".1%"),
        "latency": trend_descriptor(latency_recent, latency_prev, better="down", fmt=".0f", suffix=" ms"),
        "confidence": trend_descriptor(conf_recent, conf_prev, better="up", fmt=".2f"),
    }


def kpi_card(label: str, value: str, icon: str, trend_label: str, css_class: str, sub: str) -> str:
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-trend {css_class}">{icon} {trend_label}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """


ss = st.session_state
ss.setdefault("core", None)
ss.setdefault("orch", None)
ss.setdefault("agents", None)
ss.setdefault("twin_ids", None)
ss.setdefault("records", [])
ss.setdefault("step_id", 0)
ss.setdefault("running", False)
ss.setdefault("speed_mode", "Real-time")
ss.setdefault("interval_ms", 180)
ss.setdefault("batch_steps", 25)
ss.setdefault("policy", "bandit")
ss.setdefault("twins", 5)
ss.setdefault("seed", 42)
ss.setdefault("correlation_id", None)
ss.setdefault("target_steps", 500)
ss.setdefault("auto_stop", True)
ss.setdefault("ui_refresh_ms", 90)
ss.setdefault("last_step_ts", 0.0)
ss.setdefault("live_window", 60)
ss.setdefault("visual_style", "Mission Control")
ss.setdefault("accent_palette", "Cobalt Mint")

st.markdown(build_theme_css(ss.visual_style, ss.accent_palette), unsafe_allow_html=True)

if ss.core is None:
    reset_runtime()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-brand-title">Hybrid Quantum-Classical Control Room</div>
          <div class="sidebar-brand-text">Elegant demo shell for live hybrid orchestration, quantum routing visibility and stakeholder-facing auditability.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Demo presets</div><div class="sidebar-section-copy">Choose a ready-made execution mood before starting the run.</div></div>', unsafe_allow_html=True)
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

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Visual identity</div><div class="sidebar-section-copy">Select a typography family and accent palette for the control-room presentation.</div></div>', unsafe_allow_html=True)
    ss.visual_style = st.selectbox("Visual style", list(VISUAL_STYLES.keys()), index=list(VISUAL_STYLES.keys()).index(ss.visual_style))
    ss.accent_palette = st.selectbox("Accent palette", list(ACCENT_PALETTES.keys()), index=list(ACCENT_PALETTES.keys()).index(ss.accent_palette))

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Runtime configuration</div><div class="sidebar-section-copy">Adjust orchestration scope, reproducibility and live observation window.</div></div>', unsafe_allow_html=True)
    ss.twins = st.slider("Number of twins", 1, 20, int(ss.twins), help="Applied after Reset or Start.")
    ss.policy = st.selectbox("Routing policy", ["rule", "bandit"], index=1 if ss.policy == "bandit" else 0)
    ss.seed = st.number_input("Random seed", 1, 1_000_000, int(ss.seed))
    ss.target_steps = st.number_input("Target steps", min_value=1, max_value=1_000_000, value=int(ss.target_steps), step=50)
    ss.auto_stop = st.checkbox("Auto-stop at target steps", value=bool(ss.auto_stop))
    ss.ui_refresh_ms = st.slider("UI refresh (ms)", 50, 1000, int(ss.ui_refresh_ms), step=10)
    ss.live_window = st.slider("Live chart window (steps)", 20, 200, int(ss.live_window), step=10)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Execution speed</div><div class="sidebar-section-copy">Use real-time for visible live evolution or max speed for dense runs.</div></div>', unsafe_allow_html=True)
    ss.speed_mode = st.selectbox("Mode", ["Real-time", "Max speed"], index=0 if ss.speed_mode == "Real-time" else 1)
    if ss.speed_mode == "Real-time":
        ss.interval_ms = st.slider("Step interval (ms)", 50, 2500, int(ss.interval_ms), step=10)
    else:
        ss.batch_steps = st.slider("Steps per tick", 5, 500, int(ss.batch_steps), step=5)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Controls</div><div class="sidebar-section-copy">Reset starts a fresh run. Step is useful for staged presentations.</div></div>', unsafe_allow_html=True)
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

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Presentation note</div><div class="sidebar-section-copy">Balanced demo + Mission Control + Cobalt Mint is the most institutional combination.</div></div>', unsafe_allow_html=True)

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
summary = compute_summary(df)
trends = metric_trends(df) if not df.empty else None
progress = min(float(ss.step_id) / float(max(ss.target_steps, 1)), 1.0) if ss.auto_stop else 0.0
route_counts = df["route"].value_counts() if not df.empty else pd.Series(dtype=int)
fallback_counts = pd.Series(extract_reasons(df.get("fallback_reasons", pd.Series(dtype=object)))).value_counts() if not df.empty else pd.Series(dtype=int)

st.markdown(hero_panel(summary, ss.running, ss.visual_style, ss.accent_palette, route_counts), unsafe_allow_html=True)
st.markdown(architecture_panel(), unsafe_allow_html=True)

headline_left, headline_right = st.columns([3, 1])
with headline_left:
    st.caption(f"Run ID: {ss.correlation_id} · Policy: {ss.policy} · Twins: {ss.twins} · Mode: {ss.speed_mode}")
with headline_right:
    st.caption(f"Public demo view · {ss.visual_style} · {ss.accent_palette}")

if ss.auto_stop:
    st.progress(progress, text=f"Progress: {ss.step_id}/{ss.target_steps} steps")

k1, k2, k3, k4, k5 = st.columns(5)

if not df.empty:
    quantum_sub = f"Recent change: {trends['quantum']['delta']}"
    fallback_sub = f"Recent change: {trends['fallback']['delta']}"
    latency_sub = f"Recent change: {trends['latency']['delta']}"
    confidence_sub = f"Recent change: {trends['confidence']['delta']}"
else:
    quantum_sub = fallback_sub = latency_sub = confidence_sub = "Waiting for live data"

steps_css = "trend-live" if ss.running else ("trend-pause" if ss.step_id > 0 else "trend-idle")
steps_icon = "◉" if ss.running else ("⏸" if ss.step_id > 0 else "○")
steps_label = "Live evolution" if ss.running else ("Paused" if ss.step_id > 0 else "Ready")
steps_sub = f"{ss.speed_mode} · refresh {int(ss.ui_refresh_ms)} ms"

with k1:
    st.markdown(kpi_card("Steps", f"{ss.step_id} / {ss.target_steps}" if ss.auto_stop else str(ss.step_id), steps_icon, steps_label, steps_css, steps_sub), unsafe_allow_html=True)
with k2:
    if not df.empty:
        t = trends["quantum"]
        st.markdown(kpi_card("Quantum share", f"{summary['quantum_share']:.1%}", t["icon"], t["label"], t["css"], quantum_sub), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("Quantum share", "0.0%", "●", "Collecting baseline", "trend-flat", quantum_sub), unsafe_allow_html=True)
with k3:
    if not df.empty:
        t = trends["fallback"]
        st.markdown(kpi_card("Fallback rate", f"{summary['fallback_share']:.1%}", t["icon"], t["label"], t["css"], fallback_sub), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("Fallback rate", "0.0%", "●", "Collecting baseline", "trend-flat", fallback_sub), unsafe_allow_html=True)
with k4:
    if not df.empty:
        t = trends["latency"]
        st.markdown(kpi_card("Avg latency", f"{summary['avg_latency']:.0f} ms", t["icon"], t["label"], t["css"], latency_sub), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("Avg latency", "0 ms", "●", "Collecting baseline", "trend-flat", latency_sub), unsafe_allow_html=True)
with k5:
    if not df.empty:
        t = trends["confidence"]
        st.markdown(kpi_card("Mean confidence", f"{summary['mean_confidence']:.2f}", t["icon"], t["label"], t["css"], confidence_sub), unsafe_allow_html=True)
    else:
        st.markdown(kpi_card("Mean confidence", "0.00", "●", "Collecting baseline", "trend-flat", confidence_sub), unsafe_allow_html=True)

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
    st.markdown("<div class='section-note'>The live charts below update continuously while the run is active, so the evolution can be observed in real time.</div>", unsafe_allow_html=True)
    plot_df = df.copy()
    plot_df["step"] = range(1, len(plot_df) + 1)
    live_df = plot_df.tail(int(ss.live_window)).copy()

    st.markdown("<div class='info-panel'><div class='info-title'>Presentation framing</div><div class='info-body'>Use the top architecture block to explain the system in one minute, then move into the live evolution charts to show the orchestration actually behaving under runtime conditions.</div></div>", unsafe_allow_html=True)

    live_left, live_right = st.columns([2, 1])
    with live_left:
        st.subheader("Live evolution window")
        st.line_chart(live_df.set_index("step")[["objective_value"]], height=235)
        st.line_chart(live_df.set_index("step")[["exec_ms"]], height=235)
    with live_right:
        st.subheader("Routing distribution")
        st.bar_chart(route_counts)
        st.subheader("Fallback reasons")
        if not fallback_counts.empty:
            st.bar_chart(fallback_counts.head(10))
        else:
            st.caption("No fallback reasons recorded.")

    full_left, full_right = st.columns([1.2, 1])
    with full_left:
        st.subheader("Full-run evolution")
        st.line_chart(plot_df.set_index("step")[["objective_value", "exec_ms"]], height=260)
    with full_right:
        st.subheader("Operational twin snapshot")
        twin_snapshot = get_twin_snapshot(ss.core)
        st.dataframe(twin_snapshot, use_container_width=True, hide_index=True, height=260)

    info_left, info_right = st.columns([1.1, 1])
    with info_left:
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
    with info_right:
        recent_events = df[["step_id", "twin_id", "route", "exec_ms", "latency_breach"]].tail(10).copy()
        st.subheader("Recent events")
        st.dataframe(recent_events, use_container_width=True, hide_index=True, height=210)

with tab2:
    st.subheader("Per-twin analysis")
    twin_ids = sorted(df["twin_id"].unique().tolist())
    selected_twin = st.selectbox("Select twin", twin_ids, index=0)
    twin_df = df[df["twin_id"] == selected_twin].copy()
    twin_df["step"] = range(1, len(twin_df) + 1)
    twin_live_df = twin_df.tail(int(min(ss.live_window, max(len(twin_df), 1))))

    c_left, c_right = st.columns(2)
    with c_left:
        st.line_chart(twin_live_df.set_index("step")[["objective_value"]], height=240)
        st.line_chart(twin_live_df.set_index("step")[["exec_ms"]], height=240)
    with c_right:
        st.bar_chart(twin_df["route"].value_counts())
        if twin_df["qpu_queue_ms"].notna().any():
            st.line_chart(twin_live_df.set_index("step")[["qpu_queue_ms"]], height=240)
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
        st.download_button("Download CSV", csv_bytes, file_name="hybrid_control_room_run.csv", mime="text/csv", use_container_width=True)
    with e2:
        st.download_button("Download JSON", json_bytes, file_name="hybrid_control_room_run.json", mime="application/json", use_container_width=True)
    with e3:
        st.download_button("Download run summary", report_md, file_name="hybrid_control_room_run_summary.md", mime="text/markdown", use_container_width=True)

with tab4:
    st.subheader("What this demo shows")
    st.markdown(
        """
        This application presents a deployable demonstration of a hybrid quantum-classical orchestration loop for infrastructure assets.
        The demo focuses on five ideas: synthetic telemetry generation, hybrid routing, queue-aware quantum execution,
        fallback governance, and audit-ready result inspection.
        """
    )

    st.markdown("**Suggested public demo flow**")
    st.markdown(
        """
        1. Launch **Balanced demo** and let it run in real time for 100–180 steps.
        2. Use the **architecture block** to explain the system before entering the data.
        3. Show the **Overview** tab to explain objective, latency and routing live.
        4. Open **Twin drill-down** to focus on one asset.
        5. Use **Audit inspector** to expose the QRE and result envelopes.
        6. Export the run summary as a stakeholder-ready attachment.
        """
    )

    st.markdown("**Visual updates included**")
    st.markdown(
        """
        - Refined identity with a product-style hero section.
        - Elegant light sidebar with black typography.
        - Architecture overview block for faster storytelling.
        - Fine accent palettes for institutional or technical demos.
        - Live-update charts with a configurable recent-step window.
        - KPI cards with icon-based evolution indicators.
        """
    )
