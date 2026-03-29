from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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

    .panel-intro {{
        margin-bottom: 0.5rem;
    }}

    .panel-kicker {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        color: var(--primary) !important;
        margin-bottom: 0.18rem;
    }}

    .panel-title {{
        font-family: {style['heading_font']};
        font-size: 1.02rem;
        letter-spacing: 0.05em;
        color: #f7fbff !important;
        margin-bottom: 0.18rem;
    }}

    .panel-subtitle {{
        font-size: 0.88rem;
        line-height: 1.45;
        color: #c7d6ef !important;
        margin-bottom: 0.55rem;
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
    ss.focus_twin = ss.twin_ids[0] if ss.twin_ids else None


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


def panel_intro(kicker: str, title: str, subtitle: str) -> str:
    return f"""
    <div class="panel-intro">
        <div class="panel-kicker">{kicker}</div>
        <div class="panel-title">{title}</div>
        <div class="panel-subtitle">{subtitle}</div>
    </div>
    """


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return f"rgba(255,255,255,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def chart_tokens(palette_name: str) -> dict[str, str]:
    palette = ACCENT_PALETTES.get(palette_name, ACCENT_PALETTES["Cobalt Mint"])
    return {
        "primary": palette["primary"],
        "secondary": palette["secondary"],
        "muted": palette["muted"],
        "grid": "rgba(193, 214, 255, 0.10)",
        "panel": "rgba(10, 15, 25, 0.62)",
        "danger": "#ff7a92",
        "warning": "#ffd166",
        "success": "#43f08f",
        "text": "#edf4ff",
        "fill_primary": hex_to_rgba(palette["primary"], 0.16),
        "fill_secondary": hex_to_rgba(palette["secondary"], 0.12),
    }


def apply_plotly_layout(fig: go.Figure, palette_name: str, height: int, show_legend: bool = False) -> go.Figure:
    tokens = chart_tokens(palette_name)
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=16, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=tokens["panel"],
        font=dict(color=tokens["text"], family="Inter, sans-serif", size=12),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=tokens["grid"],
        zeroline=False,
        linecolor=tokens["grid"],
        tickfont=dict(color=tokens["muted"]),
        title=None,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=tokens["grid"],
        zeroline=False,
        linecolor=tokens["grid"],
        tickfont=dict(color=tokens["muted"]),
        title=None,
    )
    return fig


def build_system_performance_figure(df: pd.DataFrame, palette_name: str, window: int) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    tokens = chart_tokens(palette_name)
    if df.empty:
        fig.add_annotation(text="Waiting for live data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=tokens["muted"], size=15))
        return apply_plotly_layout(fig, palette_name, 410)

    plot_df = df.copy()
    plot_df["step"] = range(1, len(plot_df) + 1)
    live_df = plot_df.tail(int(window)).copy()
    live_df["latency_ma"] = live_df["exec_ms"].rolling(5, min_periods=1).mean()

    fig.add_trace(
        go.Scatter(
            x=live_df["step"], y=live_df["objective_value"], mode="lines",
            line=dict(color=tokens["primary"], width=3),
            fill="tozeroy", fillcolor=tokens["fill_primary"],
            name="Objective value",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=live_df["step"], y=live_df["exec_ms"], mode="lines",
            line=dict(color=tokens["secondary"], width=2.5),
            name="Latency",
        ), row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=live_df["step"], y=live_df["latency_ma"], mode="lines",
            line=dict(color=tokens["warning"], width=2, dash="dot"),
            name="Latency MA(5)",
        ), row=2, col=1,
    )
    fig.update_yaxes(title_text="Objective", row=1, col=1)
    fig.update_yaxes(title_text="ms", row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    return apply_plotly_layout(fig, palette_name, 410, show_legend=True)


def build_donut_figure(values: pd.Series, palette_name: str, title: str) -> go.Figure:
    tokens = chart_tokens(palette_name)
    fig = go.Figure()
    if values.empty:
        fig.add_annotation(text=f"No {title.lower()} yet", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=tokens["muted"], size=14))
        return apply_plotly_layout(fig, palette_name, 240)
    colors = [tokens["primary"], tokens["secondary"], tokens["warning"], tokens["danger"], tokens["success"]]
    fig.add_trace(go.Pie(labels=list(values.index), values=list(values.values), hole=0.58, marker=dict(colors=colors[:len(values)]), textinfo="percent", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
    fig.update_layout(showlegend=True, legend=dict(orientation="v", x=1.02, y=0.5))
    return apply_plotly_layout(fig, palette_name, 240, show_legend=True)


def build_reason_bar_figure(values: pd.Series, palette_name: str) -> go.Figure:
    tokens = chart_tokens(palette_name)
    fig = go.Figure()
    if values.empty:
        fig.add_annotation(text="No fallback reasons recorded", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=tokens["muted"], size=14))
        return apply_plotly_layout(fig, palette_name, 240)
    plot_values = values.sort_values(ascending=True).tail(8)
    fig.add_trace(go.Bar(x=plot_values.values, y=plot_values.index, orientation="h", marker=dict(color=tokens["secondary"]), hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_xaxes(title_text="Count")
    return apply_plotly_layout(fig, palette_name, 240)


def build_twin_focus_figure(twin_df: pd.DataFrame, palette_name: str, window: int) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    tokens = chart_tokens(palette_name)
    if twin_df.empty:
        fig.add_annotation(text="Twin has no activity yet", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=tokens["muted"], size=15))
        return apply_plotly_layout(fig, palette_name, 360)
    plot_df = twin_df.copy()
    if "step" not in plot_df.columns:
        plot_df["step"] = range(1, len(plot_df) + 1)
    live_df = plot_df.tail(int(window)).copy()
    fig.add_trace(go.Scatter(x=live_df["step"], y=live_df["objective_value"], mode="lines+markers", line=dict(color=tokens["primary"], width=2.8), marker=dict(size=6), name="Twin objective"), row=1, col=1)
    fig.add_trace(go.Scatter(x=live_df["step"], y=live_df["exec_ms"], mode="lines+markers", line=dict(color=tokens["secondary"], width=2.2), marker=dict(size=5), name="Twin latency"), row=2, col=1)
    fig.update_yaxes(title_text="Objective", row=1, col=1)
    fig.update_yaxes(title_text="ms", row=2, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    return apply_plotly_layout(fig, palette_name, 360, show_legend=False)


def build_health_figure(snapshot_df: pd.DataFrame, palette_name: str) -> go.Figure:
    tokens = chart_tokens(palette_name)
    fig = go.Figure()
    if snapshot_df.empty:
        fig.add_annotation(text="No twins registered", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(color=tokens["muted"], size=14))
        return apply_plotly_layout(fig, palette_name, 360)
    plot_df = snapshot_df.sort_values("health", ascending=True)
    colors = [tokens["danger"] if h < 0.35 else tokens["warning"] if h < 0.65 else tokens["success"] for h in plot_df["health"]]
    fig.add_trace(go.Bar(x=plot_df["health"], y=plot_df["twin_id"], orientation="h", marker=dict(color=colors), hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Health score", range=[0, 1])
    return apply_plotly_layout(fig, palette_name, 360)


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
ss.setdefault("ui_refresh_ms", 250)
ss.setdefault("last_step_ts", 0.0)
ss.setdefault("live_window", 60)
ss.setdefault("visual_style", "Mission Control")
ss.setdefault("accent_palette", "Cobalt Mint")
ss.setdefault("focus_twin", None)

st.markdown(build_theme_css(ss.visual_style, ss.accent_palette), unsafe_allow_html=True)

if ss.core is None:
    reset_runtime()

if not ss.focus_twin and ss.twin_ids:
    ss.focus_twin = ss.twin_ids[0]

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
    with col_b:
        if st.button("Fast operator view", use_container_width=True):
            apply_preset("Fast operator view")
    if st.button("Quantum stress", use_container_width=True):
        apply_preset("Quantum stress")

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Visual identity</div><div class="sidebar-section-copy">Select a typography family and accent palette for the control-room presentation.</div></div>', unsafe_allow_html=True)
    ss.visual_style = st.selectbox("Visual style", list(VISUAL_STYLES.keys()), index=list(VISUAL_STYLES.keys()).index(ss.visual_style))
    ss.accent_palette = st.selectbox("Accent palette", list(ACCENT_PALETTES.keys()), index=list(ACCENT_PALETTES.keys()).index(ss.accent_palette))

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Runtime configuration</div><div class="sidebar-section-copy">Adjust orchestration scope, reproducibility and live observation window.</div></div>', unsafe_allow_html=True)
    ss.twins = st.slider("Number of twins", 1, 20, int(ss.twins), help="Applied after Reset or Start.")
    ss.policy = st.selectbox("Routing policy", ["rule", "bandit"], index=1 if ss.policy == "bandit" else 0)
    ss.seed = st.number_input("Random seed", 1, 1_000_000, int(ss.seed))
    ss.target_steps = st.number_input("Target steps", min_value=1, max_value=1_000_000, value=int(ss.target_steps), step=50)
    ss.auto_stop = st.checkbox("Auto-stop at target steps", value=bool(ss.auto_stop))
    ss.ui_refresh_ms = st.slider("Live refresh (ms)", 100, 2000, int(ss.ui_refresh_ms), step=50)
    ss.live_window = st.slider("Live chart window (steps)", 20, 200, int(ss.live_window), step=10)

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Execution speed</div><div class="sidebar-section-copy">Use real-time for visible live evolution or max speed for dense runs.</div></div>', unsafe_allow_html=True)
    ss.speed_mode = st.selectbox("Mode", ["Real-time", "Max speed"], index=0 if ss.speed_mode == "Real-time" else 1)
    if ss.speed_mode == "Real-time":
        ss.interval_ms = st.slider("Step interval (ms)", 100, 2500, int(ss.interval_ms), step=20)
    else:
        ss.batch_steps = st.slider("Steps per tick", 5, 500, int(ss.batch_steps), step=5)

    focus_options = ss.twin_ids if ss.twin_ids else []
    if focus_options:
        if ss.focus_twin not in focus_options:
            ss.focus_twin = focus_options[0]
        ss.focus_twin = st.selectbox("Focus twin", focus_options, index=focus_options.index(ss.focus_twin))

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Controls</div><div class="sidebar-section-copy">Reset starts a fresh run. Step is useful for staged presentations.</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶ Start", use_container_width=True):
            reset_runtime()
            ss.focus_twin = ss.twin_ids[0] if ss.twin_ids else None
            ss.running = True
            ss.last_step_ts = 0.0
    with c2:
        if st.button("⏸ Pause", use_container_width=True):
            ss.running = False

    c3, c4 = st.columns(2)
    with c3:
        if st.button("⏭ Step", use_container_width=True):
            ss.running = False
            if ss.core is None:
                reset_runtime()
            step_once()
    with c4:
        if st.button("⏹ Reset", use_container_width=True):
            reset_runtime()
            ss.focus_twin = ss.twin_ids[0] if ss.twin_ids else None

    st.markdown('<div class="sidebar-section"><div class="sidebar-section-title">Presentation note</div><div class="sidebar-section-copy">Balanced demo + Mission Control + Cobalt Mint is the most institutional combination.</div></div>', unsafe_allow_html=True)

static_df = pd.DataFrame(ss.records)
static_summary = compute_summary(static_df)
static_routes = static_df["route"].value_counts() if not static_df.empty else pd.Series(dtype=int)
st.markdown(hero_panel(static_summary, ss.running, ss.visual_style, ss.accent_palette, static_routes), unsafe_allow_html=True)

overview_tab, twin_tab, audit_tab, about_tab = st.tabs(["Overview", "Twin drill-down", "Audit inspector", "About"])

with overview_tab:
    st.markdown(architecture_panel(), unsafe_allow_html=True)
    st.markdown("<div class='section-note'>Overview is the only continuously refreshed view. The other tabs remain more stable so you recover navigation without reintroducing noticeable blinking.</div>", unsafe_allow_html=True)

    status_slot = st.empty()
    progress_slot = st.empty()
    kpi_slot = st.empty()
    notice_slot = st.empty()
    charts_slot = st.empty()
    focus_slot = st.empty()

    def render_live_area() -> None:
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

        df = pd.DataFrame(ss.records)
        summary = compute_summary(df)
        trends = metric_trends(df) if not df.empty else None
        progress = min(float(ss.step_id) / float(max(ss.target_steps, 1)), 1.0) if ss.auto_stop else 0.0
        route_counts = df["route"].value_counts() if not df.empty else pd.Series(dtype=int)
        fallback_counts = pd.Series(extract_reasons(df.get("fallback_reasons", pd.Series(dtype=object)))).value_counts() if not df.empty else pd.Series(dtype=int)

        with status_slot.container():
            left, right = st.columns([3, 1])
            with left:
                st.caption(f"Run ID: {ss.correlation_id} · Policy: {ss.policy} · Twins: {ss.twins} · Mode: {ss.speed_mode}")
            with right:
                live_state = "LIVE" if ss.running else "STATIC"
                st.caption(f"{live_state} · refresh {int(ss.ui_refresh_ms)} ms · window {int(ss.live_window)}")

        with progress_slot.container():
            if ss.auto_stop:
                st.progress(progress, text=f"Progress: {ss.step_id}/{ss.target_steps} steps")
            else:
                st.empty()

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

        with kpi_slot.container():
            k1, k2, k3, k4, k5 = st.columns(5)
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

        with notice_slot.container():
            if df.empty:
                st.info("Ready to run. Choose a preset or press Start/Step from the sidebar.")
            elif summary["fallback_share"] > 0.35:
                st.warning("Fallback rate is elevated. This is useful for demoing governance, but it may indicate aggressive quantum routing or queue pressure.")
            elif summary["quantum_share"] > 0.45:
                st.success("Quantum routing is active across a significant share of steps, which makes this run well suited for showcasing hybrid orchestration.")
            else:
                st.info("This run is currently conservative. Switch to a more aggressive preset if you want more visible quantum activity.")

        with charts_slot.container():
            st.markdown(panel_intro("Live monitoring", "System performance overview", "A more executive visual layout: one primary performance timeline, one governance column, and one operational layer for fleet status and event review."), unsafe_allow_html=True)

            perf_left, perf_right = st.columns([1.7, 1], gap="large")
            with perf_left:
                st.plotly_chart(
                    build_system_performance_figure(df, ss.accent_palette, int(ss.live_window)),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                    key="overview_system_performance_chart",
                )
            with perf_right:
                st.markdown(panel_intro("Routing", "Decision mix", "Current composition of classical, quantum and governed fallback routes."), unsafe_allow_html=True)
                st.plotly_chart(
                    build_donut_figure(route_counts, ss.accent_palette, "Routes"),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                    key="overview_routes_donut_chart",
                )
                st.markdown(panel_intro("Governance", "Fallback reasons", "Highest-frequency causes behind governed fallback decisions."), unsafe_allow_html=True)
                st.plotly_chart(
                    build_reason_bar_figure(fallback_counts, ss.accent_palette),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                    key="overview_fallback_reasons_chart",
                )

            focus_twin = ss.focus_twin if ss.focus_twin in ss.core.registry else next(iter(ss.core.registry.keys()))
            twin_state = ss.core.registry[focus_twin]
            twin_df = df[df["twin_id"] == focus_twin].copy() if not df.empty else pd.DataFrame()
            if not twin_df.empty:
                twin_df["step"] = range(1, len(twin_df) + 1)

            twin_snapshot = get_twin_snapshot(ss.core)
            ops_left, ops_right = st.columns([1.45, 1], gap="large")
            with ops_left:
                st.markdown(panel_intro("Focused twin", f"Activity timeline · {focus_twin}", "Objective and latency evolution for the selected twin across the live window."), unsafe_allow_html=True)
                st.plotly_chart(
                    build_twin_focus_figure(twin_df, ss.accent_palette, int(ss.live_window)),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                    key="overview_focused_twin_chart",
                )
            with ops_right:
                st.markdown(panel_intro("Fleet health", "Operational health map", "Fast comparison of current health scores across all twins in the registry."), unsafe_allow_html=True)
                st.plotly_chart(
                    build_health_figure(twin_snapshot, ss.accent_palette),
                    use_container_width=True,
                    config={"displayModeBar": False, "responsive": True},
                    key="overview_fleet_health_chart",
                )

        with focus_slot.container():
            twin_state_df = pd.DataFrame(
                [
                    {"field": "twin_id", "value": focus_twin},
                    {"field": "health", "value": round(float(twin_state.health), 4)},
                    {"field": "x1", "value": round(float(twin_state.x1), 4)},
                    {"field": "x2", "value": round(float(twin_state.x2), 4)},
                    {"field": "x3", "value": round(float(twin_state.x3), 4)},
                    {"field": "last_mode", "value": twin_state.last_action.get("mode", "-")},
                    {"field": "last_damp", "value": round(float(twin_state.last_action.get("damp", 0.0)), 4)},
                    {"field": "timestamp", "value": twin_state.ts},
                ]
            )
            recent_events = df[["step_id", "twin_id", "route", "exec_ms", "latency_breach"]].tail(12).copy() if not df.empty else pd.DataFrame(columns=["step_id", "twin_id", "route", "exec_ms", "latency_breach"])

            bottom_left, bottom_right = st.columns([1.1, 1], gap="large")
            with bottom_left:
                st.markdown(panel_intro("State vector", "Focused twin state", "Current values for the selected twin, including the last applied control decision."), unsafe_allow_html=True)
                st.dataframe(twin_state_df, use_container_width=True, hide_index=True, height=300)
            with bottom_right:
                st.markdown(panel_intro("Operations log", "Recent events", "Most recent routing outcomes, latency observations and breach flags in the live run."), unsafe_allow_html=True)
                st.dataframe(recent_events, use_container_width=True, hide_index=True, height=300)

    _fragment = getattr(st, "fragment", None)
    if _fragment is None:
        _fragment = getattr(st, "experimental_fragment", None)

    run_every = float(ss.ui_refresh_ms) / 1000.0 if ss.running else None

    if _fragment is not None:
        @_fragment(run_every=run_every)
        def live_dashboard_fragment() -> None:
            render_live_area()
    else:
        def live_dashboard_fragment() -> None:
            render_live_area()

    live_dashboard_fragment()

with twin_tab:
    st.markdown(panel_intro("Twin drill-down", "Focused asset analysis", "This tab is more stable than Overview. Use it for closer reading of one twin without continuous full-tab refreshing."), unsafe_allow_html=True)
    current_df = pd.DataFrame(ss.records)
    if not ss.core.registry:
        st.info("No twins initialised yet.")
    else:
        twin_options = list(ss.core.registry.keys())
        selected_twin = st.selectbox("Twin for detailed analysis", twin_options, index=twin_options.index(ss.focus_twin) if ss.focus_twin in twin_options else 0, key="drill_twin")
        ss.focus_twin = selected_twin
        selected_state = ss.core.registry[selected_twin]
        twin_df = current_df[current_df["twin_id"] == selected_twin].copy() if not current_df.empty else pd.DataFrame()
        if not twin_df.empty:
            twin_df["step"] = range(1, len(twin_df) + 1)

        top_left, top_right = st.columns([1.5, 1], gap="large")
        with top_left:
            st.plotly_chart(
                build_twin_focus_figure(twin_df, ss.accent_palette, int(ss.live_window)),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
                key="twin_tab_focus_chart",
            )
        with top_right:
            route_counts = twin_df["route"].value_counts() if not twin_df.empty else pd.Series(dtype=int)
            st.plotly_chart(
                build_donut_figure(route_counts, ss.accent_palette, f"Routes · {selected_twin}"),
                use_container_width=True,
                config={"displayModeBar": False, "responsive": True},
                key="twin_tab_routes_donut_chart",
            )

        state_df = pd.DataFrame(
            [
                {"Field": "Twin ID", "Value": selected_twin},
                {"Field": "Health", "Value": round(float(selected_state.health), 4)},
                {"Field": "x1", "Value": round(float(selected_state.x1), 4)},
                {"Field": "x2", "Value": round(float(selected_state.x2), 4)},
                {"Field": "x3", "Value": round(float(selected_state.x3), 4)},
                {"Field": "Last mode", "Value": selected_state.last_action.get("mode", "-")},
                {"Field": "Last damp", "Value": round(float(selected_state.last_action.get("damp", 0.0)), 4)},
                {"Field": "Timestamp", "Value": selected_state.ts},
            ]
        )
        events_df = twin_df[["step_id", "route", "exec_ms", "latency_breach", "confidence"]].tail(20).copy() if not twin_df.empty else pd.DataFrame(columns=["step_id", "route", "exec_ms", "latency_breach", "confidence"])
        bottom_left, bottom_right = st.columns([1, 1.2], gap="large")
        with bottom_left:
            st.dataframe(state_df, use_container_width=True, hide_index=True, height=320)
        with bottom_right:
            st.dataframe(events_df, use_container_width=True, hide_index=True, height=320)

with audit_tab:
    st.markdown(panel_intro("Audit inspector", "Run evidence and exports", "This tab keeps audit records, exports and per-row inspection outside the live fragment so it remains stable and reliable."), unsafe_allow_html=True)
    current_df = pd.DataFrame(ss.records)
    current_summary = compute_summary(current_df)
    if current_df.empty:
        st.caption("No audit data yet.")
    else:
        audit_cols = ["step_id", "ts", "twin_id", "route", "exec_ms", "qpu_queue_ms", "noise_proxy", "cost_eur", "latency_breach"]
        st.dataframe(current_df[audit_cols].tail(80), use_container_width=True, height=280, hide_index=True)

        row_idx = st.number_input("Inspect row", min_value=0, max_value=max(0, len(current_df) - 1), value=max(0, len(current_df) - 1), step=1, key="audit_row_idx")
        row = current_df.iloc[int(row_idx)]

        def safe_json(s: Any) -> Any:
            if s is None or (isinstance(s, float) and pd.isna(s)):
                return None
            try:
                return json.loads(s)
            except Exception:
                return None

        qre_obj = safe_json(row.get("qre_json"))
        res_obj = safe_json(row.get("result_json"))
        inspect_left, inspect_right = st.columns(2, gap="large")
        with inspect_left:
            st.markdown("**QRE**")
            st.json(qre_obj) if qre_obj else st.caption("No QRE recorded for this step.")
        with inspect_right:
            st.markdown("**Quantum result**")
            st.json(res_obj) if res_obj else st.caption("No quantum result recorded for this step.")

        export_left, export_right, export_third = st.columns(3)
        with export_left:
            csv_bytes = current_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="hybrid_control_room_run.csv", mime="text/csv", use_container_width=True, key="dl_csv")
        with export_right:
            json_bytes = current_df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button("Download JSON", data=json_bytes, file_name="hybrid_control_room_run.json", mime="application/json", use_container_width=True, key="dl_json")
        with export_third:
            report = make_markdown_report(current_df, current_summary, str(ss.correlation_id), int(ss.twins), str(ss.policy)).encode("utf-8")
            st.download_button("Download Markdown summary", data=report, file_name="hybrid_control_room_summary.md", mime="text/markdown", use_container_width=True, key="dl_md")

with about_tab:
    st.markdown(panel_intro("About", "Hybrid control-room narrative", "A static explanatory view for presentations, so the audience can understand the architecture and value proposition without watching live charts."), unsafe_allow_html=True)
    st.markdown(architecture_panel(), unsafe_allow_html=True)
    about_left, about_right = st.columns([1.2, 1], gap="large")
    with about_left:
        st.markdown("""
**Hybrid Quantum-Classical Control Room** is a presentation layer for a simulated orchestration stack in which a classical baseline, a simulated cloud-QPU gateway and governance rules coexist inside the same operational loop.

The interface is organised so that **Overview** behaves like the live wall, while the other tabs act as stable analytical surfaces for inspection, explanation and export. This preserves the original navigation logic without sacrificing the visual stability needed for a convincing real-time demo.

The architecture links synthetic telemetry, twin state updates, feature extraction, route selection, quantum request envelopes, governed fallback, audit logging and exportable evidence. In a presentation context, this lets you move naturally from live monitoring to explainability and then to governance evidence.
        """)
    with about_right:
        latest_df = pd.DataFrame(ss.records)
        summary = compute_summary(latest_df)
        summary_df = pd.DataFrame(
            [
                {"Metric": "Total steps", "Value": ss.step_id},
                {"Metric": "Quantum share", "Value": f"{summary['quantum_share']:.1%}"},
                {"Metric": "Fallback rate", "Value": f"{summary['fallback_share']:.1%}"},
                {"Metric": "Average latency", "Value": f"{summary['avg_latency']:.0f} ms"},
                {"Metric": "Mean confidence", "Value": f"{summary['mean_confidence']:.2f}"},
                {"Metric": "Policy", "Value": ss.policy},
                {"Metric": "Twins", "Value": ss.twins},
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True, height=290)
