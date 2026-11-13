"""SIMGuard AI analyst console. Run with: streamlit run app.py"""

from datetime import datetime, timezone
import html
import json
from uuid import uuid4

import pandas as pd
import streamlit as st

import api_client
from browser_store import browser_store
from theme import *  # noqa: F403 - design tokens are intentionally centralized

st.set_page_config(
    page_title="SIMGuard AI | Analyst Console",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

EVENT_FIELDS = [
    "sim_swap_time_gap_minutes", "device_change_flag", "sim_type_change_flag",
    "imsi_change_flag", "iccid_change_flag", "otp_and_sim_change_geo_hash_length",
    "recent_sim_activation_days", "num_sim_changes_last_30d",
    "previous_sim_holder_tenure_days", "account_age_days", "ip_change_flag",
]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700;800&display=swap');

:root {{
    --bg: {PAGE_BG};
    --surface: {SURFACE};
    --surface-raised: {SOFT_SURFACE};
    --surface-inset: #0F1116;
    --text: {TEXT};
    --muted: {MUTED};
    --yellow: {PRIMARY};
    --yellow-hover: {PRIMARY_HOVER};
    --yellow-dark: {YELLOW_TINT};
    --silver: {SILVER};
    --silver-muted: {SILVER_MUTED};
    --signal: {SIGNAL};
    --border-soft: {SOFT_BORDER};
    --border-hard: {HARD_BORDER};
    --font-display: 'Space Grotesk', 'Bahnschrift SemiCondensed', 'Bahnschrift', 'Segoe UI Semibold', -apple-system, sans-serif;
    --font-body: 'Plus Jakarta Sans', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace;
}}

html, body, [class*="css"] {{
    font-family: var(--font-body);
    color: var(--text);
    background-color: var(--bg);
    -webkit-font-smoothing: antialiased;
}}

.stApp {{
    background-color: var(--bg);
    background-image:
        linear-gradient(to right, rgba(226, 232, 240, 0.025) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(226, 232, 240, 0.025) 1px, transparent 1px);
    background-size: 32px 32px;
    background-attachment: fixed;
}}

.block-container {{
    max-width: 1520px;
    padding-top: 1.2rem;
    padding-bottom: 4rem;
}}

section[data-testid="stMain"] {{
    background: transparent;
}}

header[data-testid="stHeader"] {{
    background: transparent;
}}

#MainMenu, footer, [data-testid="stDeployButton"] {{
    visibility: hidden;
}}

[data-testid="stSidebar"] {{
    display: none;
}}

/* Native alert overrides into block banners */
div[data-testid="stAlert"] {{
    background-color: var(--surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}}

ul[role="listbox"] {{
    background-color: var(--surface-raised) !important;
    border: 1px solid var(--border-hard) !important;
}}

li[role="option"] {{
    color: var(--silver) !important;
    font-family: var(--font-body) !important;
}}

li[role="option"][aria-selected="true"] {{
    background-color: var(--surface) !important;
    color: var(--yellow) !important;
    font-weight: 700 !important;
}}

li[role="option"]:hover {{
    background-color: var(--border-soft) !important;
    color: #FFFFFF !important;
}}

/* =========================================================
   BLOCK HIGHLIGHT SYSTEM
   ========================================================= */
.hl-yellow {{
    background: var(--yellow);
    color: #0A0B0E !important;
    font-weight: 800;
    padding: 2px 7px;
    border-radius: 3px;
    display: inline-block;
    letter-spacing: 0.02em;
    line-height: 1.25;
    vertical-align: baseline;
}}

.hl-silver {{
    background: var(--silver);
    color: #0F172A !important;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 3px;
    display: inline-block;
    letter-spacing: 0.02em;
    line-height: 1.25;
    vertical-align: baseline;
}}

.hl-gray {{
    background: #232833;
    color: #E2E8F0 !important;
    border: 1px solid #3E4756;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 3px;
    display: inline-block;
    letter-spacing: 0.02em;
    line-height: 1.25;
    vertical-align: baseline;
}}

.hl-dark {{
    background: #0B0D10;
    color: var(--yellow) !important;
    border: 1.5px solid var(--yellow);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 800;
    padding: 1px 6px;
    border-radius: 3px;
    display: inline-block;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    vertical-align: baseline;
}}

/* =========================================================
   MASTHEAD & BRANDING
   ========================================================= */
.masthead-wrap {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0 16px;
    border-bottom: 1px solid var(--border-soft);
    margin-bottom: 12px;
}}

.brand-wrap {{
    display: flex;
    align-items: center;
    gap: 14px;
}}

.brand-mark-block {{
    width: 42px;
    height: 42px;
    background: var(--yellow);
    color: #0A0B0E;
    display: grid;
    place-items: center;
    font-size: 1.35rem;
    font-weight: 900;
    border-radius: 4px;
    box-shadow: 0 0 20px rgba(255, 214, 0, 0.28);
    transition: transform 0.2s ease;
}}

.brand-mark-block:hover {{
    transform: scale(1.05);
}}

.brand-title {{
    font-family: var(--font-display);
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    color: #FFFFFF;
    line-height: 1.1;
}}

.brand-divider {{
    color: var(--yellow);
    font-weight: 900;
    margin: 0 2px;
}}

.brand-subtitle {{
    font-family: var(--font-mono);
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    color: var(--silver-muted);
    margin-top: 2px;
}}

.status-pill-block {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 4px;
    padding: 7px 12px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    color: var(--text);
}}

.dot-live {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--yellow);
    box-shadow: 0 0 10px var(--yellow);
    animation: livePulse 2s ease-in-out infinite;
}}

@keyframes livePulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(0.85); }}
}}

/* =========================================================
   TACTILE BLOCK TAB NAVIGATION (st.radio styling)
   ========================================================= */
div[role="radiogroup"] {{
    display: flex !important;
    flex-direction: row !important;
    gap: 8px !important;
    padding: 6px 0 16px !important;
    border-bottom: 1px solid var(--border-soft) !important;
    margin-bottom: 18px !important;
    overflow-x: auto !important;
    white-space: nowrap !important;
}}

div[role="radiogroup"] > label {{
    background: var(--surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: 4px !important;
    padding: 10px 22px !important;
    margin: 0 !important;
    cursor: pointer !important;
    transition: all 0.16s ease !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
}}

/* Hide the default radio circle */
div[role="radiogroup"] > label > div:first-child {{
    display: none !important;
}}

/* Radio label typography */
div[role="radiogroup"] > label p,
div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p {{
    font-family: var(--font-display) !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: var(--silver-muted) !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}}

div[role="radiogroup"] > label:hover {{
    background: var(--surface-raised) !important;
    border-color: var(--silver) !important;
}}

div[role="radiogroup"] > label:hover p,
div[role="radiogroup"] > label:hover div[data-testid="stMarkdownContainer"] p {{
    color: #FFFFFF !important;
}}

/* Checked state: Solid Yellow Block */
div[role="radiogroup"] > label:has(input:checked) {{
    background: var(--yellow) !important;
    border-color: var(--yellow) !important;
    box-shadow: 0 3px 14px rgba(255, 214, 0, 0.3) !important;
    transform: translateY(-1px) !important;
}}

div[role="radiogroup"] > label:has(input:checked) p,
div[role="radiogroup"] > label:has(input:checked) div[data-testid="stMarkdownContainer"] p {{
    color: #0A0B0E !important;
    font-weight: 800 !important;
}}

/* =========================================================
   SAFETY & PROTOCOL BANNER
   ========================================================= */
.notice-banner {{
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-left: 3.5px solid var(--silver);
    padding: 10px 16px;
    border-radius: 4px;
    font-size: 0.82rem;
    color: var(--silver-muted);
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 22px;
}}

/* =========================================================
   ATTRACTIVE TAB HERO / HEADING BANNER
   ========================================================= */
.hero {{
    position: relative;
    overflow: hidden;
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-left: 4px solid var(--yellow);
    border-radius: 6px;
    padding: 24px 28px;
    margin: 10px 0 22px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.28);
}}

.hero:before {{
    content: "";
    position: absolute;
    right: 0;
    top: 0;
    bottom: 0;
    width: 320px;
    pointer-events: none;
    opacity: 0.08;
    background-image:
        linear-gradient(to right, var(--silver) 1px, transparent 1px),
        linear-gradient(to bottom, var(--silver) 1px, transparent 1px);
    background-size: 14px 14px;
    mask-image: linear-gradient(90deg, transparent, #000 70%);
}}

.hero-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    position: relative;
    z-index: 1;
}}

.hero-tag-wrap {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

.block-tag-silver {{
    background: #232833;
    color: #E2E8F0;
    border: 1px solid #3E4756;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 2px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}

.block-tag-status {{
    background: var(--yellow-dark);
    color: var(--yellow);
    border: 1px solid var(--yellow);
    font-family: var(--font-mono);
    font-size: 0.68rem;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 2px;
    letter-spacing: 0.08em;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}}

.hero-meta-chip {{
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--silver-muted);
    letter-spacing: 0.06em;
}}

.hero-meta-chip strong {{
    color: var(--yellow);
}}

.hero-title-wrap {{
    position: relative;
    z-index: 1;
}}

.hero-heading {{
    font-family: var(--font-display);
    font-size: clamp(2rem, 3.4vw, 2.75rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.01em;
    text-transform: uppercase;
    color: #FFFFFF;
    margin: 0 0 10px;
}}

.hero-subtitle {{
    font-family: var(--font-body);
    font-size: 0.95rem;
    color: #CBD5E1;
    max-width: 860px;
    line-height: 1.5;
    margin: 0;
    position: relative;
    z-index: 1;
}}

/* =========================================================
   KPI METRIC CARDS
   ========================================================= */
.kpi-card {{
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-top: 3.5px solid var(--kpi-color);
    border-radius: 6px;
    padding: 18px 20px 16px;
    position: relative;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    min-height: 160px;
}}

.kpi-card:hover {{
    transform: translateY(-3px);
    border-color: var(--kpi-color);
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.35);
}}

.kpi-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}}

.kpi-tag {{
    font-family: var(--font-mono);
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--silver-muted);
    letter-spacing: 0.08em;
}}

.kpi-glyph {{
    font-family: var(--font-mono);
    font-size: 1.15rem;
    color: var(--kpi-color);
    line-height: 1;
}}

.kpi-value {{
    font-family: var(--font-display);
    font-size: 2.75rem;
    font-weight: 800;
    line-height: 1;
    color: #FFFFFF;
    margin-bottom: 8px;
    font-variant-numeric: tabular-nums;
}}

.kpi-label {{
    font-family: var(--font-display);
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--silver);
    letter-spacing: 0.02em;
    text-transform: uppercase;
}}

.kpi-note {{
    font-family: var(--font-body);
    font-size: 0.72rem;
    color: var(--silver-muted);
    margin-top: 4px;
    line-height: 1.35;
}}

/* Tone colors for KPI cards */
.kpi-total {{ --kpi-color: {SILVER}; }}
.kpi-pending {{ --kpi-color: {PRIMARY}; }}
.kpi-flagged {{ --kpi-color: {PRIMARY}; }}
.kpi-confirmed {{ --kpi-color: {SILVER}; }}
.kpi-false {{ --kpi-color: #94A3B8; }}

/* =========================================================
   SECTION HEADERS
   ========================================================= */
.section-block-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 16px;
}}

.section-block-pip {{
    width: 6px;
    height: 24px;
    background: var(--yellow);
    border-radius: 2px;
    flex-shrink: 0;
}}

.section-pip-silver {{
    background: var(--silver);
}}

.section-block-title {{
    font-family: var(--font-display);
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #FFFFFF;
    margin: 0;
    line-height: 1.1;
}}

.section-block-sub {{
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--silver-muted);
    margin: 3px 0 0;
}}

/* =========================================================
   PANELS & CALLOUTS
   ========================================================= */
.panel {{
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 6px;
    padding: 22px;
    margin: 16px 0;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
}}

.panel-title {{
    font-family: var(--font-display);
    font-size: 1.2rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #FFFFFF;
    margin-bottom: 6px;
}}

.subtle {{
    font-size: 0.82rem;
    color: var(--silver-muted);
    margin: 4px 0 0;
    line-height: 1.45;
}}

/* Clean, non-block disclaimer line for footer */
.footer-disclaimer {{
    font-size: 0.82rem;
    color: var(--silver-muted);
    margin: 20px 0 8px;
    padding: 8px 2px;
    line-height: 1.4;
    border-top: 1px solid var(--border-soft);
}}

/* =========================================================
   CASE DOSSIER / DETAIL
   ========================================================= */
.dossier-box {{
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 6px;
    padding: 24px;
    margin: 18px 0;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
}}

.dossier-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-soft);
    padding-bottom: 14px;
    margin-bottom: 18px;
}}

.case-id-code {{
    font-family: var(--font-mono);
    font-size: 0.85rem;
    font-weight: 700;
    color: #FFFFFF;
    background: var(--surface-raised);
    border: 1px solid var(--border-hard);
    padding: 5px 12px;
    border-radius: 3px;
}}

.dossier-score-val {{
    font-family: var(--font-display);
    font-size: 3.5rem;
    font-weight: 800;
    color: var(--yellow);
    line-height: 1;
    margin: 10px 0 14px;
}}

.dossier-score-max {{
    font-size: 1.15rem;
    color: var(--silver-muted);
    font-weight: 600;
}}

.evidence-box {{
    background: var(--surface-inset);
    border: 1px solid var(--border-soft);
    border-radius: 4px;
    padding: 14px 16px;
    margin: 10px 0;
}}

.evidence-row {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    font-size: 0.88rem;
    color: #E2E8F0;
}}

.evidence-row:last-child {{
    border-bottom: none;
}}

.evidence-pip {{
    color: var(--yellow);
    font-size: 0.8rem;
    line-height: 1.4;
}}

/* =========================================================
   SYSTEM HEALTH ROWS
   ========================================================= */
.health-block-item {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 14px 18px;
    background: var(--surface);
    border: 1px solid var(--border-soft);
    border-radius: 4px;
    margin-bottom: 10px;
    transition: border-color 0.15s ease;
}}

.health-block-item:hover {{
    border-color: var(--silver);
}}

.health-name {{
    font-family: var(--font-display);
    font-size: 0.92rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}}

.badge-online {{
    background: #0E2419;
    color: #34D399;
    border: 1px solid #059669;
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 0.72rem;
    padding: 4px 10px;
    border-radius: 2px;
    letter-spacing: 0.05em;
}}

.badge-mock {{
    background: var(--yellow-dark);
    color: var(--yellow);
    border: 1px solid var(--yellow);
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 0.72rem;
    padding: 4px 10px;
    border-radius: 2px;
    letter-spacing: 0.05em;
}}

.badge-offline {{
    background: #2D1417;
    color: #F87171;
    border: 1px solid #DC2626;
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 0.72rem;
    padding: 4px 10px;
    border-radius: 2px;
    letter-spacing: 0.05em;
}}

/* =========================================================
   STREAMLIT COMPONENTS OVERRIDES
   ========================================================= */
div.stButton > button {{
    font-family: var(--font-display) !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    border: 1px solid var(--border-hard) !important;
    background: var(--surface-raised) !important;
    color: var(--silver) !important;
    min-height: 42px !important;
    padding: 8px 18px !important;
    transition: all 0.16s ease !important;
}}

div.stButton > button:hover {{
    border-color: var(--silver) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
}}

div.stButton > button[kind="primary"],
div.stFormSubmitButton > button {{
    background: var(--yellow) !important;
    color: #0A0B0E !important;
    border: 1px solid var(--yellow) !important;
    font-weight: 800 !important;
}}

div.stButton > button[kind="primary"]:hover,
div.stFormSubmitButton > button:hover {{
    background: var(--yellow-hover) !important;
    color: #000000 !important;
    border-color: var(--yellow-hover) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(255, 214, 0, 0.38) !important;
}}

/* Inputs & Form Controls */
input, textarea, select {{
    border-radius: 4px !important;
}}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {{
    background-color: var(--surface-inset) !important;
    border-color: var(--border-soft) !important;
    border-radius: 4px !important;
}}

div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="select"]:focus-within > div {{
    border-color: var(--yellow) !important;
    box-shadow: 0 0 0 2px rgba(255, 214, 0, 0.25) !important;
}}

button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible,
a:focus-visible, [role="radio"]:focus-visible, [role="tab"]:focus-visible {{
    outline: 2px solid var(--yellow) !important;
    outline-offset: 2px !important;
}}

/* Streamlit Tabs */
button[data-baseweb="tab"] {{
    font-family: var(--font-display) !important;
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: var(--silver-muted) !important;
    padding: 10px 18px !important;
    border-radius: 4px 4px 0 0 !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
}}

button[data-baseweb="tab"]:hover {{
    color: var(--silver) !important;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--yellow) !important;
    border-bottom: 3px solid var(--yellow) !important;
    background: rgba(255, 214, 0, 0.04) !important;
}}

div[data-baseweb="tab-highlight"] {{
    background-color: var(--yellow) !important;
}}

/* Metrics */
div[data-testid="stMetric"] {{
    background: var(--surface) !important;
    border: 1px solid var(--border-soft) !important;
    border-left: 3.5px solid var(--yellow) !important;
    padding: 14px 18px !important;
    border-radius: 4px !important;
}}

div[data-testid="stMetricLabel"] {{
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--silver-muted) !important;
}}

div[data-testid="stMetricValue"] {{
    font-family: var(--font-display) !important;
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    color: #FFFFFF !important;
    font-variant-numeric: tabular-nums !important;
}}

/* Dataframe Container */
div[data-testid="stDataFrame"] {{
    border: 1px solid var(--border-soft) !important;
    border-radius: 4px !important;
    background: var(--surface) !important;
}}

/* Expander - clean and minimal */
details[data-testid="stExpander"] {{
    background: transparent !important;
    border: none !important;
    border-top: 1px solid var(--border-soft) !important;
    border-radius: 0 !important;
    color: var(--silver-muted) !important;
    margin-top: 24px !important;
}}

details[data-testid="stExpander"] summary {{
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    color: var(--silver-muted) !important;
    padding: 10px 0 !important;
}}

details[data-testid="stExpander"] summary:hover {{
    color: var(--silver) !important;
}}

@media (max-width: 760px) {{
    .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    .hero {{ padding: 18px; }}
    .hero-heading {{ font-size: 2rem; }}
    .masthead-wrap {{ align-items: flex-start; gap: 10px; flex-direction: column; }}
    div[role="radiogroup"] {{ overflow-x: auto; white-space: nowrap; }}
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Overview"
PAGES = ["Overview", "Cases", "Analyze", "Data & metrics", "System status"]
if st.session_state.get("page") not in PAGES:
    st.session_state.page = "Overview"
if st.session_state.get("top_workspace") not in PAGES:
    st.session_state.top_workspace = st.session_state.page
if "selected_case" not in st.session_state:
    st.session_state.selected_case = None


def fmt(value, digits=1):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def tier(case):
    final = case.get("final") or {}
    return str(final.get("tier", "unknown")).lower()


def score(case):
    return (case.get("final") or {}).get("score")


def flag(case):
    return bool(case.get("fraud_detected"))


def pill(label, style=""):
    return f"<span class='hl-gray' style='font-family:var(--font-mono);font-size:0.7rem;'>{label}</span>"


def case_frame(cases):
    rows = []
    for case in cases:
        final = case.get("final") or {}
        rows.append({
            "Case ID": str(case.get("case_id", ""))[:12],
            "Created": str(case.get("created_at", "—")).replace("T", " ")[:19],
            "Risk Score": fmt(final.get("score")),
            "Risk Tier": str(final.get("tier", "unknown")).title(),
            "Automated outcome": "Flagged for review" if flag(case) else "No risk signal",
            "Review status": str(case.get("status", "pending")).replace("_", " ").title(),
            "Model agreement": str(case.get("agreement", "unknown")).replace("_", " ").title(),
            "Report": "Available" if case.get("report") else "None",
        })
    return pd.DataFrame(rows)


def load_data(force=False):
    if force or "api_data" not in st.session_state:
        try:
            st.session_state.api_data = {"cases": api_client.get_cases(), "health": api_client.get_health(), "error": None}
        except Exception as exc:
            st.session_state.api_data = {"cases": [], "health": None, "error": str(exc)}
    return st.session_state.api_data


def go(page):
    st.session_state.page = page
    st.session_state.top_workspace = page
    st.rerun()


FEATURE_LABELS = {
    "sim_swap_time_gap_minutes": "OTP → SIM change gap (min)",
    "device_change_flag": "Device changed",
    "sim_type_change_flag": "SIM type changed",
    "imsi_change_flag": "IMSI changed",
    "iccid_change_flag": "ICCID changed",
    "otp_and_sim_change_geo_hash_length": "OTP / SIM location similarity",
    "recent_sim_activation_days": "Days since SIM activation",
    "num_sim_changes_last_30d": "SIM changes · last 30 days",
    "previous_sim_holder_tenure_days": "Previous holder tenure (days)",
    "account_age_days": "Account age (days)",
    "ip_change_flag": "IP address changed",
}
GATE_TEXT = {"block": "BLOCK", "allow": "ALLOW", "human_review": "HUMAN REVIEW"}
ID_COLUMNS = {"msisdn_hash", "msisdn"}


def fresh_case(case):
    """The newest copy of a case: background Qwen reports, reviews and regenerations update it."""
    return st.session_state.setdefault("case_updates", {}).get(case.get("case_id"), case)


def remember_case(case):
    st.session_state.setdefault("case_updates", {})[case.get("case_id")] = case


def unsafe_id_fields(records):
    """Subscriber-ID fields whose values are not generated SYNTH- tokens (privacy guard)."""
    bad = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            text = "" if value is None else str(value).strip()
            if normalized in ID_COLUMNS and text and not text.upper().startswith("SYNTH-"):
                bad.add(str(key))
    return sorted(bad)


def json_records(text):
    """Records inside pasted / uploaded JSON or JSONL, for the SYNTH- guard. [] if not JSON."""
    try:
        data = json.loads(text)
    except ValueError:
        try:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        except ValueError:
            return []
    if isinstance(data, dict):
        for key in ("events", "rows", "data", "records", "cases"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data.get("event", data)]
    return data if isinstance(data, list) else []


def section(title, silver=False):
    pip = "section-block-pip section-pip-silver" if silver else "section-block-pip"
    st.markdown(f"""
    <div class='section-block-header'>
        <div class='{pip}'></div>
        <div><h2 class='section-block-title'>{title}</h2></div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# BRAND MASTHEAD
# =========================================================
mast_html = """
<div class='masthead-wrap'>
    <div class='brand-wrap'>
        <div class='brand-mark-block' aria-hidden='true'>◈</div>
        <div>
            <div class='brand-title'>SIMGUARD <span class='brand-divider'>/</span> AI</div>
            <div class='brand-subtitle'><span class='hl-gray'>TELECOM FRAUD DEFENSE CONSOLE</span></div>
        </div>
    </div>
    <div>
        <span class='status-pill-block'>
            <span class='dot-live'></span>
            <span><span class='hl-yellow'>ACTIVE SHIELD</span> · <span class='hl-silver'>SYNTHETIC RUNTIME</span></span>
        </span>
    </div>
</div>
"""
st.markdown(mast_html, unsafe_allow_html=True)

# Workspace Selector
selected = st.radio("Workspace", PAGES, index=PAGES.index(st.session_state.top_workspace), horizontal=True, label_visibility="collapsed", key="top_workspace")
st.session_state.page = selected

# Topline Protocol Banner
banner_html = """
<div class='notice-banner'>
    <span class='hl-dark'>SAFETY PROTOCOL</span>
    <span>Automated classifications represent <span class='hl-yellow'>decision support</span>. Enforcement actions require <span class='hl-silver'>certified human analyst</span> verification.</span>
</div>
"""
st.markdown(banner_html, unsafe_allow_html=True)

QWEN_LOGIN_KEY = "qwen_login"  # {"endpoint", "user", "password"} in this browser's localStorage


def store_request(action, value=None):
    """Ask the browser to get / set / clear the saved Qwen login (answered on a later rerun)."""
    st.session_state.qwen_store_nonce = st.session_state.get("qwen_store_nonce", 0) + 1
    st.session_state.qwen_store_req = {"action": action, "value": value, "nonce": st.session_state.qwen_store_nonce}


def saved_qwen_login():
    """The login saved in this browser ({} if none). Also hands it to the API once per session."""
    request = st.session_state.get("qwen_store_req") or {"action": "get", "value": None, "nonce": 0}
    answer = browser_store(request["action"], QWEN_LOGIN_KEY, request["value"], request["nonce"],
                           widget_key="qwen_login_store")
    if isinstance(answer, dict) and answer.get("nonce") == request["nonce"]:
        st.session_state.qwen_saved_login = answer.get("value") or {}
        if not st.session_state.get("qwen_login_synced"):
            st.session_state.qwen_login_synced = True
            login = st.session_state.qwen_saved_login
            if login.get("password") or login.get("endpoint"):
                try:  # the API keeps the login in memory, so re-send it on every new visit
                    api_client.set_qwen_settings(login.get("endpoint"), login.get("user"), login.get("password"),
                                                 save_even_if_unreachable=True)
                except Exception:
                    pass
    return st.session_state.get("qwen_saved_login") or {}


saved_qwen_login()
data = load_data()
cases = data["cases"]
health = data["health"]


def render_hero(tag, title_html, subtitle_html, badge="ACTIVE ENGINE"):
    html_content = f"""
    <div class='hero'>
        <div class='hero-top'>
            <div class='hero-tag-wrap'>
                <span class='block-tag-silver'>{tag}</span>
                <span class='block-tag-status'><span class='dot-live'></span> {badge}</span>
            </div>
            <div class='hero-meta-chip'>SECURITY POSTURE: <strong>TIER-1 DEFENSE</strong></div>
        </div>
        <div class='hero-title-wrap'>
            <h1 class='hero-heading'>{title_html}</h1>
        </div>
        <div>
            <p class='hero-subtitle'>{subtitle_html}</p>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


def render_kpis(filtered):
    total = len(filtered)
    pending = sum(c.get("status", "pending") == "pending" for c in filtered)
    flagged = sum(flag(c) for c in filtered)
    confirmed = sum(c.get("status") == "confirmed" for c in filtered)
    false_alarm = sum(c.get("status") == "false_alarm" for c in filtered)
    metrics = [
        ("total", "01 / VOLUME", "Total cases", total, "Stored in selected range", "◫"),
        ("pending", "02 / QUEUE", "Pending review", pending, "Awaiting analyst decision", "◷"),
        ("flagged", "03 / MODEL", "Fraud flagged", flagged, "Automated flags · unconfirmed", "⌁"),
        ("confirmed", "04 / REVIEW", "Confirmed fraud", confirmed, "Human reviewed as confirmed", "✓"),
        ("false", "05 / OUTCOME", "False alarms", false_alarm, "Reviewed as false alarms", "↗"),
    ]
    cols = st.columns(5)
    for index, (col, (tone, label, name, value, note, glyph)) in enumerate(zip(cols, metrics)):
        card_html = f"""
        <section class='kpi-card kpi-{tone}' role='group' aria-label='{name}: {value:,}. {note}'>
            <div class='kpi-head'>
                <span class='kpi-tag'>{label}</span>
                <span class='kpi-glyph' aria-hidden='true'>{glyph}</span>
            </div>
            <div class='kpi-value'>{value:,}</div>
            <div class='kpi-label'>{name}</div>
            <div class='kpi-note'>{note}</div>
        </section>
        """
        col.markdown(card_html, unsafe_allow_html=True)


def render_recent(filtered):
    st.markdown("""
    <div class='section-block-header'>
        <div class='section-block-pip'></div>
        <div>
            <h2 class='section-block-title'>RECENT AUDIT CASES</h2>
            <p class='section-block-sub'>SORTED BY FUSED RISK SCORE · REAL-TIME TELEMETRY REPOSITORY</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if not filtered:
        st.info("No cases are stored yet. Screen an event to initiate the first audit case.")
        return
    frame = case_frame(filtered)
    st.dataframe(frame, width="stretch", hide_index=True, height=min(380, 42 + 36 * len(frame)))
    choices = {f"{str(c.get('case_id',''))[:12]} · {fmt(score(c))}/100 · {c.get('status','pending')}": c.get("case_id") for c in filtered}
    chosen = st.selectbox("Open a case", ["Select a case…", *choices], label_visibility="collapsed")
    if chosen != "Select a case…" and st.button("View case details", key="open_case"):
        st.session_state.selected_case = choices[chosen]
        go("Cases")


def render_overview():
    render_hero(
        tag="WORKSPACE // 01",
        title_html="FRAUD OPERATIONS <span class='hl-yellow'>OVERVIEW</span>",
        subtitle_html="Continuous surveillance of <span class='hl-yellow'>telecom anomalies</span>, automated <span class='hl-silver'>risk fusion</span>, and operator review triage.",
        badge="SYS_ONLINE",
    )
    if data["error"]:
        st.error(f"Could not reach the API: {data['error']}. Check the API URL and service, then refresh.")
    
    filt, refresh = st.columns([5, 1])
    with filt:
        dates = st.date_input("Date range", value=(), help="Filters dashboard counts and cases by created date.")
    with refresh:
        st.write("")
        if st.button("↻ Refresh", width="stretch"):
            load_data(force=True)
            st.rerun()

    filtered = cases
    if len(dates) == 2:
        start, end = dates
        filtered = [c for c in cases if start.isoformat() <= str(c.get("created_at", ""))[:10] <= end.isoformat()]

    render_kpis(filtered)

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("""
        <div class='section-block-header'>
            <div class='section-block-pip section-pip-silver'></div>
            <div>
                <h2 class='section-block-title'>CASES OVER TIME</h2>
                <p class='section-block-sub'>HISTORICAL CASE DISPATCH BY ANALYST OUTCOME</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if filtered:
            frame = pd.DataFrame([{"date": str(c.get("created_at", ""))[:10], "status": c.get("status", "pending"), "count": 1} for c in filtered])
            chart = frame.pivot_table(index="date", columns="status", values="count", aggfunc="sum", fill_value=0).sort_index()
            st.line_chart(chart, width="stretch", height=250)
            totals = frame.groupby("status")["count"].sum().to_dict()
            st.caption("Chart summary: " + ", ".join(f"{status.replace('_', ' ')} {count}" for status, count in sorted(totals.items())) + f" across {frame['date'].nunique()} date(s).")
        else:
            st.info("Case trend will appear when API records are available.")
    
    with right:
        st.markdown("""
        <div class='section-block-header'>
            <div class='section-block-pip'></div>
            <div>
                <h2 class='section-block-title'>RISK SCORE DISTRIBUTION</h2>
                <p class='section-block-sub'>FUSED MULTI-MODEL PROBABILITY SPREAD · 0–100 SCALE</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        scores = [float(score(c)) for c in filtered if score(c) is not None]
        if scores:
            buckets = pd.cut(scores, bins=[-0.01, 39.99, 69.99, 100], labels=["Low · 0–39", "Medium · 40–69", "High · 70–100"])
            bucket_counts = pd.Series(buckets).value_counts().sort_index()
            st.bar_chart(bucket_counts, width="stretch", height=250)
            st.caption("Chart summary: " + ", ".join(f"{label} {count}" for label, count in bucket_counts.items()) + ".")
        else:
            st.info("Risk distribution will appear when cases have been analyzed.")

    render_recent(filtered)
    st.markdown("<p class='footer-disclaimer'>Operational notice: Automated flagged cases are model risk scores. Certified analyst review is required to confirm fraud or record a false alarm.</p>", unsafe_allow_html=True)


def render_cases():
    render_hero(
        tag="WORKSPACE // 02",
        title_html="CASE AUDIT & <span class='hl-yellow'>ADJUDICATION</span>",
        subtitle_html="Audit and adjudicate <span class='hl-yellow'>flagged SIM-swap vectors</span> with dual-model telemetry and <span class='hl-silver'>forensic dossiers</span>.",
        badge="QUEUE ACTIVE",
    )
    if not cases:
        st.info("There are no stored cases yet. Use Analyze to screen an event.")
    query = st.text_input("Search case ID", placeholder="Enter a case ID")
    status_filter = st.selectbox("Review status", ["All statuses", "pending", "confirmed", "false_alarm"])
    filtered = [c for c in cases if (not query or query.lower() in str(c.get("case_id", "")).lower()) and (status_filter == "All statuses" or c.get("status", "pending") == status_filter)]
    render_recent(filtered)
    case_id = st.session_state.selected_case
    if case_id:
        try:
            case = api_client.get_case(case_id)
        except Exception as exc:
            st.error(f"Could not load case details: {exc}")
            return
        remember_case(case)
        render_case_detail(case, key="cases")


def render_case_detail(case, key="case"):
    case = fresh_case(case)
    case_id = case.get("case_id")
    final, ml, laya = case.get("final") or {}, case.get("ml") or {}, case.get("laya") or {}
    case_tier = str(final.get("tier", "unknown")).lower()
    tier_pill = f"<span class='hl-yellow'>TIER: {case_tier.upper()}</span>" if case_tier == "high" else f"<span class='hl-silver'>TIER: {case_tier.upper()}</span>"
    flag_pill = "<span class='hl-dark'>FLAGGED FOR AUDIT</span>" if flag(case) else "<span class='hl-gray'>NO ELEVATED RISK</span>"
    gate = laya.get("gate")
    gate_html = f"<span class='hl-silver'>LAYA GATE: {GATE_TEXT.get(gate, 'N/A')}</span>"

    dossier_html = f"""
    <div class='dossier-box'>
        <div class='dossier-top'>
            <div>
                <span class='block-tag-silver'>FORENSIC CASE DOSSIER</span>
                <span style='margin-left:8px;font-family:var(--font-mono);font-size:0.75rem;color:var(--silver-muted);'>CREATED: {html.escape(str(case.get('created_at','—')))} · INPUT: {html.escape(str((case.get('input') or {}).get('source', 'api')).upper())}</span>
            </div>
            <div class='case-id-code'>{html.escape(str(case_id))}</div>
        </div>
        <div style='display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:20px;'>
            <div>
                <div style='font-family:var(--font-mono);font-size:0.72rem;letter-spacing:0.08em;color:var(--silver-muted);text-transform:uppercase;'>FUSED RISK SCORE</div>
                <div class='dossier-score-val'>{fmt(final.get('score'))}<span class='dossier-score-max'> / 100</span></div>
                <div style='display:flex;gap:8px;align-items:center;flex-wrap:wrap;'>{tier_pill} &nbsp; {flag_pill} &nbsp; {gate_html}</div>
            </div>
            <div style='text-align:right;'>
                <div style='font-family:var(--font-mono);font-size:0.75rem;color:var(--silver-muted);margin-bottom:6px;'>
                    ANALYST DISPOSITION: <span class='hl-gray'>{str(case.get('status','pending')).replace('_',' ').upper()}</span>
                </div>
                <div style='font-family:var(--font-mono);font-size:0.75rem;color:var(--silver-muted);'>
                    MODEL AGREEMENT: <span class='hl-silver'>{str(case.get('agreement','unknown')).replace('_',' ').upper()}</span>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(dossier_html, unsafe_allow_html=True)

    if case.get("agreement") in {"disagree", "laya_unavailable"}:
        st.warning("Model signals disagree or a model is unavailable. Review the available evidence before deciding.")
    if laya.get("error"):
        st.caption(f"Laya was unavailable for this case: {laya.get('error')}")

    cols = st.columns(4)
    cols[0].metric("LightGBM fraud probability", fmt(ml.get("lgbm_prob"), 3) if ml.get("lgbm_prob") is not None else "Unknown",
                   help="Gradient-boosted trees on the 11 features. 70% of the ML risk score.")
    cols[1].metric("Isolation Forest anomaly", fmt(ml.get("anomaly"), 2) if ml.get("anomaly") is not None else "Unknown",
                   help="0 = typical of normal traffic, 1 = very unusual. 30% of the ML risk score.")
    cols[2].metric("Laya p(fraud)", fmt(laya.get("p_fraud"), 3) if laya.get("p_fraud") is not None else "Unknown",
                   help="Fine-tuned Laya (ModernBERT-large). Gate: block above 0.88, allow below 0.12, otherwise human review.")
    cols[3].metric("Final fused score", fmt(final.get("score")), help="Weighted log-odds of Laya and the ML risk score, 0-100.")

    left, right = st.columns([1, 1])
    with left:
        section("WHY THE SCORE IS WHAT IT IS")
        contributions = ml.get("contributions") or {}
        if contributions:
            event = case.get("event") or {}
            labels = []
            for name in contributions:
                value = event.get(name)
                shown = ("yes" if value else "no") if name.endswith("_flag") else value
                labels.append(f"{FEATURE_LABELS.get(name, name)} = {shown}")
            frame = pd.DataFrame({
                "Raises fraud risk": [max(v, 0.0) for v in contributions.values()],
                "Lowers fraud risk": [min(v, 0.0) for v in contributions.values()],
            }, index=labels)
            st.bar_chart(frame, horizontal=True, color=[SIGNAL, SILVER_MUTED], height=230, width="stretch")
            st.caption("Top LightGBM per-feature contributions (log-odds) for this case.")
        else:
            st.caption("No feature contributions were returned (mock mode).")
    with right:
        section("TELEMETRY EVIDENCE")
        evidence = case.get("evidence") or []
        if evidence:
            evidence_rows = "".join(f"<div class='evidence-row'><span class='evidence-pip'>▲</span><span>{html.escape(item)}</span></div>" for item in evidence)
            st.markdown(f"<div class='evidence-box'>{evidence_rows}</div>", unsafe_allow_html=True)
        else:
            st.caption("No rule-based indicators fired for this event.")

    render_report_section(case, key)

    with st.expander("Input → parsed features"):
        info = case.get("input") or {}
        event = case.get("event") or {}
        rows = [{"Field": name, "Value": event.get(name), "Meaning": FEATURE_LABELS[name]} for name in EVENT_FIELDS]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        ids = {k: event.get(k) for k in ("event_id", "msisdn_hash", "timestamp") if event.get(k)}
        if ids:
            st.caption("Identifiers: " + " · ".join(f"{k} = {v}" for k, v in ids.items()))
        if info.get("extra"):
            st.caption("Context kept with the case (not used by the models): " + ", ".join(f"{k} = {v}" for k, v in info["extra"].items()))
        for note in info.get("notes") or []:
            st.caption(f"Parser: {note}")
        if info.get("raw"):
            st.code(info["raw"], language="text")
    with st.expander("What Qwen received"):
        if case.get("report_payload"):
            st.json(case["report_payload"], expanded=False)
        else:
            st.caption("Nothing was sent to Qwen for this case.")
    with st.expander("Stage latency"):
        latency = {
            "LightGBM + Isolation Forest": f"{fmt(ml.get('latency_ms'), 2)} ms",
            "Laya": f"{fmt(laya.get('latency_ms'), 1)} ms",
            "Qwen report": "—" if case.get("report_latency_ms") is None else f"{fmt(case['report_latency_ms'] / 1000, 1)} s",
            "Laya checkpoint": str(laya.get("model", "unknown")),
        }
        st.dataframe(pd.DataFrame([latency]), hide_index=True, width="stretch")

    section("HUMAN REVIEW DECISION")
    current = case.get("status", "pending")
    if current == "pending":
        decision = st.radio("Choose a human review outcome", ["No decision", "Confirm fraud", "Mark as false alarm"], horizontal=True, key=f"{key}_decision_{case_id}")
        if decision != "No decision":
            st.warning(f"Confirm recording ‘{decision}’ for this case.")
            if st.button("Confirm review decision", type="primary", key=f"{key}_confirm_{case_id}"):
                try:
                    updated = api_client.review_case(case_id, "confirmed" if decision == "Confirm fraud" else "false_alarm")
                    remember_case({**case, **updated})
                    st.session_state.selected_case = case_id
                    load_data(force=True)
                    st.success("Review decision saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Review was not saved: {exc}")
    else:
        st.success(f"Analyst decision recorded: {current.replace('_',' ').title()}")


def render_report_section(case, key):
    case_id = case.get("case_id")
    source = case.get("report_source", "none")
    section("INVESTIGATION REPORT", silver=True)
    if source == "pending":
        st.info("Qwen 3.5 2B is writing the report (about 15–40 s). The template report below is shown until it arrives.")
        _poll_pending_report(case_id)
    if case.get("report"):
        if source == "qwen":
            meta = f"Written by Qwen 3.5 2B ({case.get('report_model') or 'Qwen/Qwen3.5-2B'})"
            if case.get("report_latency_ms") is not None:
                meta += f" in {fmt(case['report_latency_ms'] / 1000, 1)} s"
        elif source == "template":
            meta = "Template report (Qwen was not used or not reachable)"
        else:
            meta = "Template report, shown while Qwen writes"
        st.caption(meta + " · The report explains the supplied scores and evidence; it does not calculate the risk score.")
        if case.get("report_error") and source == "template":
            st.caption(f"Why not Qwen: {case['report_error']}")
        with st.container(border=True):
            st.markdown(case["report"])  # Markdown only; raw HTML in model output is not rendered
    elif not flag(case):
        st.info("No elevated risk signal and Laya did not ask for human review, so no report was written automatically.")
    if source != "pending":
        label = "Regenerate with Qwen" if source == "qwen" else "Ask Qwen 3.5 2B for a report"
        if st.button(label, key=f"{key}_report_{case_id}"):
            with st.spinner("Qwen 3.5 2B is writing the report…"):
                try:
                    remember_case(api_client.regenerate_report(case_id))
                    load_data(force=True)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Qwen report failed: {exc}")
                    if "unreachable" in str(exc).lower() or "no qwen link" in str(exc).lower():
                        st.caption(f"Qwen is not running. {START_QWEN_HINT}")


@st.fragment(run_every=3)
def _poll_pending_report(case_id):
    """Re-check a pending case every 3 s; redraw the page once the Qwen report is in."""
    try:
        latest = api_client.get_case(case_id)
    except Exception:
        return
    if latest.get("report_source") != "pending":
        remember_case(latest)
        load_data(force=True)
        st.rerun()


PROMPT_EXAMPLE = (
    "SIM changed 45 minutes after an OTP request. Device changed, IMSI changed, ICCID changed, "
    "SIM type not changed. Geohash similarity 0.12. SIM activated 3 days ago, 2 SIM changes in last "
    "30 days. Previous holder tenure 40 days, account age 120 days. IP changed."
)


def run_analysis(call, key):
    """Run one API analysis call and keep its response for the results panel."""
    try:
        response = call()
    except Exception as exc:
        st.error(f"Analysis failed: {exc}. No mock score was substituted.")
        return
    st.session_state[f"{key}_response"] = response
    for item in response.get("results", []):
        remember_case(item)
    load_data(force=True)


def render_analysis_result(key):
    response = st.session_state.get(f"{key}_response")
    if not response:
        return
    results = response.get("results", [])
    flagged = sum(flag(r) for r in results)
    st.success(f"{(response.get('source') or '').upper()} input · analyzed {response.get('rows_analysed', len(results))} of "
               f"{response.get('rows_received', len(results))} · {flagged} flagged · saved to the case store.")
    for note in response.get("notes", []):
        st.info(note)
    errors = response.get("errors", [])
    if errors:
        st.warning(f"{len(errors)} row(s) could not be analyzed.")
        st.dataframe(pd.DataFrame(errors), width="stretch", hide_index=True)
    if not results:
        return
    if len(results) == 1:
        render_case_detail(results[0], key=key)
        return
    current = [fresh_case(r) for r in results]
    st.dataframe(case_frame(current), width="stretch", hide_index=True, height=min(500, 42 + 36 * min(len(current), 12)))
    options = {f"{str(item.get('case_id', ''))[:12]} · {fmt(score(item))}/100 · {'flagged' if flag(item) else 'not flagged'}": item for item in current}
    chosen = st.selectbox("Inspect an analyzed case", ["Select a case…", *options], key=f"{key}_pick")
    if chosen != "Select a case…":
        render_case_detail(options[chosen], key=key)


def display_value(field, value):
    """A value as an analyst reads it: yes/no for flags, units for durations."""
    if field.endswith("_flag"):
        return "Yes" if value else "No"
    if field == "sim_swap_time_gap_minutes":
        return f"{value:g} min"
    if field in ("recent_sim_activation_days", "previous_sim_holder_tenure_days", "account_age_days"):
        return f"{value:g} days"
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def clear_gap_state():
    st.session_state.pop("text_gap", None)
    for key in [k for k in st.session_state if str(k).startswith("gap_")]:
        del st.session_state[key]


def read_then_analyze(prompt):
    """Read the text first. Analyze only when all 11 values were found; otherwise say what is missing."""
    try:
        parsed = api_client.parse_text(prompt, "text")
    except Exception as exc:
        st.error(f"Could not read the text: {exc}")
        return
    row = parsed["rows"][0]
    if row["ok"]:
        clear_gap_state()
        run_analysis(lambda: api_client.analyze_input(prompt, "text"), "text")
        return
    st.session_state.pop("text_response", None)
    st.session_state.text_gap = {"prompt": prompt, "row": row, "fields": parsed.get("fields", [])}


def render_text_gap(gap):
    row = gap["row"]
    labels = {item["field"]: item["label"] for item in gap["fields"]}
    missing = row["missing"]
    if not missing:  # the text was read but a value is not valid (e.g. a similarity above 1)
        st.error(f"I read your text but a value is not usable: {row.get('error')}")
    else:
        count = len(missing)
        st.warning(f"I understood {len(row['found'])} of the 11 values. {count} {'is' if count == 1 else 'are'} still "
                   "missing: add them below, or reword your text and press Analyze again.")
    if row["found"]:
        st.markdown("**What I understood**")
        table = pd.DataFrame([{"Value": labels.get(f, f), "Understood as": display_value(f, v)} for f, v in row["found"].items()])
        st.dataframe(table, hide_index=True, width="stretch")
    if row.get("understood"):
        with st.expander("How I read your text"):
            for note in row["understood"]:
                st.caption(note)
    if not missing:
        return
    with st.form("text_gap_form"):
        st.markdown("**What I could not find**")
        answers = {}
        for item in missing:
            key = f"gap_{item['field']}"
            if item["kind"] == "yes/no":
                answers[item["field"]] = st.selectbox(item["label"], ["Select…", "Yes", "No"], key=key)
            else:
                answers[item["field"]] = st.text_input(item["label"], key=key, placeholder=item["example"],
                                                       help="Numbers with units are fine, e.g. 45 minutes, 3 days, 40%.")
        submitted = st.form_submit_button("Analyze with these values", type="primary", width="stretch")
    if not submitted:
        return
    blank = [labels.get(f, f) for f, v in answers.items() if v in ("", "Select…")]
    if blank:
        st.error("Still needed: " + "; ".join(blank))
        return
    # the answers go in as `key: value` lines under the original text, so the case keeps your own words
    extra = [f"{field}: {value.lower() if value in ('Yes', 'No') else value.strip()}" for field, value in answers.items()]
    combined = gap["prompt"].rstrip() + "\n" + "\n".join(extra)
    st.session_state.pop("text_response", None)
    run_analysis(lambda: api_client.analyze_input(combined, "text"), "text")
    if st.session_state.get("text_response"):
        clear_gap_state()
        st.rerun()


def render_words_tab():
    st.caption("Write the event however you like: full sentences, short notes, a list or `key: value` lines. SIMGuard reads "
               "the 11 values out of your text. If something is missing it tells you exactly what, and you can add it right here.")
    prompt = st.text_area("Describe the event", height=170, placeholder=PROMPT_EXAMPLE, key="prompt_text")
    c1, c2 = st.columns([1, 4])
    if c1.button("Analyze text", type="primary", key="analyze_text"):
        clear_gap_state()
        if not prompt.strip():
            st.error("Write a description first.")
        else:
            read_then_analyze(prompt)
    if c2.button("Use the example", key="prompt_example"):
        st.session_state.prompt_text = PROMPT_EXAMPLE
        clear_gap_state()
        st.rerun()
    gap = st.session_state.get("text_gap")
    if gap:
        render_text_gap(gap)
    render_analysis_result("text")


def render_analyze():
    render_hero(
        tag="WORKSPACE // 03",
        title_html="TELEMETRY <span class='hl-yellow'>EVENT SCREENER</span>",
        subtitle_html="Evaluate synthetic <span class='hl-yellow'>11-feature SIM-swap events</span> through <span class='hl-silver'>LightGBM, Isolation Forest, and Laya</span>, with <span class='hl-yellow'>Qwen 3.5 2B</span> reports for flagged cases.",
        badge="INFERENCE ENGINE",
    )
    guided, words, upload, paste = st.tabs(["Guided questions", "Describe in words", "Upload file", "Paste CSV / JSON"])
    with guided:
        st.markdown("<div class='callout'><span class='hl-dark'>SAFETY NOTICE</span> Synthetic test data only. The form matches SIMGuard's current <span class='hl-yellow'>11-feature model schema</span>; real subscriber IDs and phone numbers are not accepted.</div>", unsafe_allow_html=True)
        with st.form("event_form"):
            st.markdown("""
            <div class='section-block-header' style='margin:14px 0 10px;'>
                <span class='hl-yellow'>01</span>
                <div><h3 class='section-block-title' style='font-size:1.05rem;'>EVENT IDENTIFIERS</h3></div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            event_id = c1.text_input("Event ID", value=f"SG-{datetime.now().strftime('%y%m%d-%H%M%S')}")
            if "synthetic_subscriber_token" not in st.session_state:
                st.session_state.synthetic_subscriber_token = f"SYNTH-{uuid4().hex[:12].upper()}"
            subscriber_hash = c2.text_input("Synthetic subscriber token", value=st.session_state.synthetic_subscriber_token, disabled=True, help="Generated test token. Real subscriber IDs and hashes are not accepted in this prototype.")
            timestamp = st.text_input("Event timestamp (ISO 8601)", value=datetime.now(timezone.utc).isoformat(timespec="seconds"))

            st.markdown("""
            <div class='section-block-header' style='margin:20px 0 10px;'>
                <span class='hl-silver'>02</span>
                <div><h3 class='section-block-title' style='font-size:1.05rem;'>TELEMETRY SIGNALS</h3></div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            time_gap = c1.number_input("OTP → SIM change gap (minutes)", min_value=0.0, value=45.0, step=1.0)
            device_changed = c2.selectbox("Device changed?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            sim_type_changed = c3.selectbox("SIM type changed?", [0, 1], format_func=lambda x: "Yes" if x else "No")

            c1, c2, c3 = st.columns(3)
            imsi_changed = c1.selectbox("IMSI changed?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            iccid_changed = c2.selectbox("ICCID changed?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            geo_similarity = c3.number_input("OTP / SIM location similarity (0–1)", min_value=0.0, max_value=1.0, value=0.12, step=0.01)

            c1, c2, c3 = st.columns(3)
            activation_days = c1.number_input("Days since SIM activation", min_value=0.0, value=3.0, step=1.0)
            sim_changes = c2.number_input("SIM changes · last 30 days", min_value=0.0, value=2.0, step=1.0)
            holder_tenure = c3.number_input("Previous SIM holder tenure (days)", min_value=0.0, value=40.0, step=1.0)

            c1, c2 = st.columns(2)
            account_age = c1.number_input("Account age (days)", min_value=0.0, value=120.0, step=1.0)
            ip_changed = c2.selectbox("IP address changed?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            submitted = st.form_submit_button("Analyze event", type="primary", width="stretch")
        if submitted:
            if not event_id.strip() or not subscriber_hash.strip().upper().startswith("SYNTH-") or not timestamp.strip():
                st.error("Enter an event ID, a generated SYNTH- token, and a timestamp.")
            else:
                payload = {
                    "event_id": event_id.strip(),
                    "msisdn_hash": subscriber_hash.strip(),
                    "timestamp": timestamp.strip(),
                    "sim_swap_time_gap_minutes": time_gap,
                    "device_change_flag": device_changed,
                    "sim_type_change_flag": sim_type_changed,
                    "imsi_change_flag": imsi_changed,
                    "iccid_change_flag": iccid_changed,
                    "otp_and_sim_change_geo_hash_length": geo_similarity,
                    "recent_sim_activation_days": activation_days,
                    "num_sim_changes_last_30d": sim_changes,
                    "previous_sim_holder_tenure_days": holder_tenure,
                    "account_age_days": account_age,
                    "ip_change_flag": ip_changed,
                }
                # through the input endpoint so the Qwen report is written in the background
                run_analysis(lambda: api_client.analyze_input(json.dumps(payload), "json"), "guided")
        render_analysis_result("guided")

    with words:
        render_words_tab()

    with upload:
        st.markdown("#### Analyze a file")
        st.caption("CSV / TSV with a header row, JSON (an object, a list, or {\"events\": [...]}), JSONL, or a TXT description. "
                   "The API analyzes up to 2,000 rows. Required model features (common aliases are accepted):")
        st.code(",".join(EVENT_FIELDS), language="text")
        file = st.file_uploader("Choose a file", type=["csv", "tsv", "json", "jsonl", "txt"], key="file_upload")
        if file:
            content = file.getvalue()
            st.caption(f"{len(content):,} bytes · {file.name}")
            name = file.name.lower()
            unsafe = []
            if name.endswith((".csv", ".tsv")):
                try:
                    raw = pd.read_csv(file, dtype=str, sep="\t" if name.endswith(".tsv") else ",")
                    unsafe = unsafe_id_fields(raw.to_dict("records"))
                except Exception as exc:
                    st.error(f"Could not read this file: {exc}")
                    unsafe = None
            elif name.endswith((".json", ".jsonl")):
                unsafe = unsafe_id_fields(json_records(content.decode("utf-8", errors="replace")))
            if unsafe:
                st.error("Subscriber identifier fields must contain only generated SYNTH- tokens: " + ", ".join(unsafe))
            elif unsafe is not None and st.button("Analyze file", type="primary", key="analyze_file"):
                run_analysis(lambda: api_client.analyze_upload(file.name, content), "file")
        render_analysis_result("file")

    with paste:
        st.caption("Paste CSV (with a header row) or JSON. The format is detected automatically.")
        pasted = st.text_area("CSV or JSON", height=180, key="pasted_text")
        if st.button("Analyze pasted data", type="primary", key="analyze_paste"):
            unsafe = unsafe_id_fields(json_records(pasted))
            if not unsafe and not pasted.lstrip().startswith(("{", "[")):
                try:
                    from io import StringIO
                    unsafe = unsafe_id_fields(pd.read_csv(StringIO(pasted), dtype=str).to_dict("records"))
                except Exception:
                    unsafe = []
            if not pasted.strip():
                st.error("Paste CSV or JSON first.")
            elif unsafe:
                st.error("Subscriber identifier fields must contain only generated SYNTH- tokens: " + ", ".join(unsafe))
            else:
                run_analysis(lambda: api_client.analyze_input(pasted), "paste")
        render_analysis_result("paste")


def render_data_metrics():
    render_hero(
        tag="WORKSPACE // 04",
        title_html="OPERATIONAL <span class='hl-yellow'>METRICS & DATA</span>",
        subtitle_html="Comparative benchmarks between <span class='hl-yellow'>live case storage</span> and <span class='hl-silver'>100K synthetic baselines</span>.",
        badge="ANALYTICS CORE",
    )
    st.markdown("<div class='panel'><div class='panel-title'>OPERATIONAL <span class='hl-yellow'>CASE REPOSITORY</span></div><p class='subtle'>Counts below come from GET /cases and reflect records stored in the SQLite database.</p></div>", unsafe_allow_html=True)
    render_kpis(cases)
    st.markdown("<div class='panel'><div class='panel-title'>SYNTHETIC <span class='hl-silver'>DATASET PROFILE</span></div><div style='margin:8px 0;'><span class='hl-dark'>BENCHMARK</span> <span class='hl-gray'>100,000 SYNTHETIC ROWS</span></div><p class='subtle'>The referenced dataset contains <span class='hl-yellow'>73,907 positive</span> and <span class='hl-silver'>26,093 negative</span> sim_swap_flag labels. These are dataset labels, not operational live traffic.</p></div>", unsafe_allow_html=True)

    section("MODEL BENCHMARKS · SYNTHETIC HELD-OUT DATA")
    ml_metrics = ((health or {}).get("ml") or {}).get("metrics") or {}
    left, right = st.columns(2)
    with left:
        st.markdown("**LightGBM + Isolation Forest** (risk score ≥ 70 = flagged)")
        if ml_metrics:
            split = ml_metrics.get("split_sizes", {})
            matrix = (ml_metrics.get("confusion_matrix") or {}).get("matrix") or [[None, None], [None, None]]
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", fmt(ml_metrics.get("precision"), 4))
            c2.metric("Recall", fmt(ml_metrics.get("recall"), 4))
            c3.metric("PR-AUC", fmt(ml_metrics.get("pr_auc"), 5))
            st.caption(f"Test split of {split.get('test', '—'):,} rows · missed fraud {matrix[1][0]} · false alarms {matrix[0][1]} · "
                       f"{ml_metrics.get('lgbm_rounds', '—')} LightGBM trees, 200 Isolation Forest trees.")
        else:
            st.caption("Scoring-engine metrics are unavailable (API offline or mock mode).")
    with right:
        st.markdown("**Fine-tuned Laya** (ModernBERT-large, 421M parameters)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Test accuracy", "0.9917", "+0.735 vs base")
        c2.metric("Hard cases", "0.8333", "+0.283 vs base")
        c3.metric("GPU latency", "39 ms", "13 ms batched", delta_color="off")
        st.caption("Base Laya: 0.2565 test / 0.5500 hard (no signal). Model and full card: "
                   "[Kyoiske/simguard-laya-fraud](https://huggingface.co/Kyoiske/simguard-laya-fraud).")

    section("CURRENT BACKEND ARCHITECTURE", silver=True)
    st.write("• Inputs: guided form, plain-English description, CSV / TSV / JSON / JSONL / TXT files or pasted CSV / JSON — all parsed by the API (POST /analyze/input, /analyze/upload; up to 2,000 rows). Missing features are reported, never guessed.")
    st.write("• Every event is scored by LightGBM + Isolation Forest and by Laya; rule-based evidence lists the indicators.")
    st.write("• Qwen 3.5 2B (Kaggle) writes reports for flagged / human-review cases in the background (up to 3 per upload; the rest on demand). Each report is checked for invented numbers and over-claims; otherwise the template report is kept.")
    st.write("• No analytics aggregation endpoint; dashboard charts aggregate GET /cases in the UI.")


def render_status():
    render_hero(
        tag="WORKSPACE // 05",
        title_html="INFRASTRUCTURE & <span class='hl-yellow'>SERVICE HEALTH</span>",
        subtitle_html="Real-time heartbeat verification across <span class='hl-yellow'>FastAPI endpoints</span>, SQLite persistence, and <span class='hl-silver'>model inference services</span>.",
        badge="HEARTBEAT 1.0",
    )
    if not health:
        st.error(f"Health data unavailable. {data['error'] or 'Check that the API is running and API_URL is correct.'}")
        return
    if st.button("↻ Re-check services", key="recheck_health"):
        load_data(force=True)
        st.rerun()
    config = health
    mocked = bool(config.get("mock_models"))
    ml = config.get("ml") or {}
    laya = config.get("laya") or {}
    qwen = config.get("qwen") or {}

    def state(ok, ready="Available", down="Unavailable", pending=None):
        if mocked:
            return "Mock", "badge-mock"
        if ok is None and pending:
            return pending, "badge-mock"
        return (ready, "badge-online") if ok else (down, "badge-offline")

    laya_ok = True if laya.get("loaded") else (False if laya.get("error") else None)
    entries = [
        ("FastAPI Application Gateway", ("Available", "badge-online")),
        ("SQLite Case Persistence Layer", state(config.get("db_ok"))),
        ("LightGBM + Isolation Forest Pipeline", state(ml.get("ok"))),
        ("Laya Fine-tuned Decision Model", state(laya_ok, pending="Loading")),
        ("Qwen 3.5 2B Report Agent (Kaggle)", state(qwen.get("reachable"), down="Offline · template reports")),
    ]
    for label, (value, badge_class) in entries:
        st.markdown(f"<div class='health-block-item'><span class='health-name'>{label}</span><span class='{badge_class}'>{html.escape(value.upper())}</span></div>", unsafe_allow_html=True)
    if ml.get("ok"):
        st.caption(f"Scoring engine: {ml.get('lgbm_trees', '—')} LightGBM trees · {ml.get('iforest_trees', '—')} Isolation Forest trees.")
    elif ml.get("error"):
        st.caption(f"Scoring engine error: {ml['error']}")
    gate = laya.get("gate") or {}
    st.caption(f"Laya checkpoint: {config.get('laya_checkpoint', 'unknown')} · gate allow ≤ {fmt(gate.get('lo'), 3)} / block ≥ {fmt(gate.get('hi'), 3)}"
               + (f" · error: {laya['error']}" if laya.get("error") else ""))
    st.caption(f"Fusion uses weighted log-odds. Laya weight: {config.get('laya_weight', 'unknown')}; threshold: {config.get('laya_fraud_threshold', 'unknown')}.")

    section("QWEN 3.5 2B ON KAGGLE", silver=True)
    render_qwen_panel()


KAGGLE_NOTEBOOK_URL = "https://www.kaggle.com/code/rudraksh12/qwen3-5"
START_QWEN_HINT = (f"Start it from the Kaggle editor: open [the notebook]({KAGGLE_NOTEBOOK_URL}) → Edit → "
                   "Save Version → Save & Run All. It is ready in about 2–4 minutes and is found automatically.")


def render_qwen_panel():
    st.caption("Qwen runs on a Kaggle GPU only while you need it. The notebook shuts itself down after "
               "10 minutes without a report request, so it never burns GPU hours idle.")
    _qwen_live_status()

    saved = st.session_state.get("qwen_saved_login") or {}  # read once per run, at the top of the page
    with st.form("qwen_login_form"):
        st.markdown("**Qwen login & link** · saved in this browser")
        link = st.text_input("Kaggle notebook link (optional)", value=saved.get("endpoint", ""),
                             placeholder="https://xxxxxxxxxxxxxxxx.gradio.live",
                             help="Leave empty: the link is found automatically each time the notebook starts.")
        c1, c2 = st.columns(2)
        user = c1.text_input("Username", value=saved.get("user", "simguard"),
                             help="The notebook's QWEN_USER Kaggle secret (default simguard).")
        password = c2.text_input("Password", value=saved.get("password", ""), type="password",
                                 help="The notebook's QWEN_PASSWORD Kaggle secret.")
        remember = st.checkbox("Remember in this browser", value=True,
                               help="Stored in this browser's local storage only. The API keeps it in memory, never on disk.")
        save_col, forget_col = st.columns(2)
        do_save = save_col.form_submit_button("Save login & link", type="primary", width="stretch")
        do_forget = forget_col.form_submit_button("Forget saved login & link", width="stretch")
    flash = st.session_state.get("qwen_flash")  # stays visible until the next save / forget
    if flash:
        getattr(st, flash[0])(flash[1])
    if do_save:
        login = {"endpoint": link.strip(), "user": user.strip() or "simguard", "password": password}
        if remember:
            store_request("set", login)
        try:
            result = api_client.set_qwen_settings(login["endpoint"], login["user"], login["password"])
        except Exception as exc:
            # keep the login even when the link test fails (e.g. the notebook is not running yet)
            try:
                api_client.set_qwen_settings(None, login["user"], login["password"])
            except Exception:
                pass
            flash = ("warning", f"Login saved, but the link was not: {exc}")
        else:
            check = result.get("check") or {}
            if check.get("reachable"):
                flash = ("success", f"Saved. Qwen answers at {check.get('endpoint')}.")
            else:
                flash = ("info", "Login saved. Qwen is not running right now. " + START_QWEN_HINT)
        st.session_state.qwen_login_synced = True
        st.session_state.qwen_saved_login = login if remember else saved
        st.session_state.qwen_flash = flash
        load_data(force=True)
        st.rerun()  # redraw so the browser-storage component receives the "set" request
    if do_forget:
        store_request("clear")
        st.session_state.qwen_saved_login = {}
        try:
            api_client.clear_qwen_endpoint()
            api_client.set_qwen_settings(None, "", "")
            st.session_state.qwen_flash = ("success", "Removed from this browser and from the API. The API falls back to its .env settings.")
        except Exception as exc:
            st.session_state.qwen_flash = ("error", f"Could not reset the API: {exc}")
        load_data(force=True)
        st.rerun()


@st.fragment(run_every=10)
def _qwen_live_status():
    """Whether Qwen answers, refreshed every 10 s while this page is open."""
    try:
        status = api_client.qwen_status()
    except Exception as exc:
        st.error(f"Could not read the Qwen status: {exc}")
        return
    online = bool(status.get("reachable"))
    c1, c2 = st.columns(2)
    c1.metric("Qwen 3.5 2B", "Online" if online else "Offline")
    c2.metric("Login used", "from this browser" if status.get("credentials_from") == "frontend" else ".env on the server")
    if online:
        st.caption(f"Answering at {status.get('endpoint')} as user {status.get('user')}.")
    else:
        st.markdown(f"Qwen is not running. {START_QWEN_HINT} Until then, reports use the template.")
        if status.get("detail"):
            st.caption(f"Last check: {status.get('detail')}"[:300])


def render_privacy_use():
    st.markdown("## Privacy & prototype use")
    st.caption("A plain-language account of the data this local prototype handles and its limits.")
    st.warning("Synthetic data only. Do not enter real subscriber information, real subscriber hashes, or live case records.")

    st.markdown("### Data and purpose")
    st.write("The analysis form accepts an event ID, a generated synthetic subscriber token, timestamp, and the 11 SIMGuard model features. File uploads (CSV, TSV, JSON, JSONL, TXT), pasted CSV / JSON and typed descriptions are parsed by the API and may include these features plus optional synthetic IDs and context. The API returns and stores analyzed events with risk scores, evidence, report text, case status, and timestamps. This data is used to screen demo events, show case results, and record a human review.")

    st.markdown("### Where data goes and how long it stays")
    st.write(f"Submitting one event sends it from this browser to the configured API at `{api_client.API_URL}`. The API stores the full response in SQLite (`DB_PATH`; `setup.bat` sets `data/simguard.db`, otherwise `simguard.db`); review actions also create an audit record. The prototype has no automatic retention limit, erasure workflow, or delete endpoint. Stored rows remain until the database is manually cleared by its operator. The API has no authentication, so keep both services on a trusted local machine and do not expose them to the public internet.")
    st.write("Uploaded or pasted contents and typed descriptions are sent to the configured API for parsing and analysis. The API stores each analyzed case, including the original input (up to 2,000 characters), in SQLite.")

    st.markdown("### Analytics and external services")
    st.write("No Google Analytics, advertising pixels, or iframe embeds are present in the UI. Streamlit usage statistics are disabled in the app-local configuration.")
    st.write("**Qwen 3.5 2B runs outside this machine**, in the operator's Kaggle notebook (`rudraksh12/qwen3-5`) behind a password-protected Gradio link. For flagged and human-review cases, and whenever an analyst asks for a report, the API sends that service the case verdict, the 11 event features, the model scores and contributions, the evidence list, non-label input context, and, for typed descriptions, the prompt text (up to 1,500 characters). The dataset label is never sent. If the service is unreachable, the local template report is used instead.")
    st.write("To find the notebook's current link, the API reads the operator's discovery topic on the public ntfy.sh service (only the link is published there, never case data). On first setup, the fine-tuned Laya model is downloaded from Hugging Face (`Kyoiske/simguard-laya-fraud`); no case data is sent there.")

    st.markdown("### Prototype use conditions")
    st.write("Use only synthetic demonstration data. Risk scores are decision support and do not establish that a person committed fraud. A human analyst must review any model flag. This prototype is not a production service or a binding service agreement; public operation requires an identified operator, access controls, support and grievance contact, retention and deletion rules, and a reviewed privacy policy.")

    st.markdown("### DPDP readiness")
    st.write("A subscriber hash is not automatically anonymous: if it can identify or be linked to a person, treat it as personal data. As of 14 November 2026, the staged commencement notification has brought section 6(9), including publication of a contact for data-principal questions, into force; the app has no operator contact or rights/grievance channel. Most notice, consent, and fiduciary obligations are scheduled to commence on 13 May 2027. The current prototype is therefore for synthetic data only and is not represented as DPDP-compliant.")
    st.markdown("Official references: [Digital Personal Data Protection Act, 2023 (MeitY)](https://www.meity.gov.in/static/uploads/2024/02/Digital-Personal-Data-Protection-Act-2023.pdf) · [DPDP Rules, 2025 and commencement dates (MeitY Gazette)](https://www.meity.gov.in/static/uploads/2025/11/53450e6e5dc0bfa85ebd78686cadad39.pdf)")


if st.session_state.page == "Overview":
    render_overview()
elif st.session_state.page == "Cases":
    render_cases()
elif st.session_state.page == "Analyze":
    render_analyze()
elif st.session_state.page == "Data & metrics":
    render_data_metrics()
elif st.session_state.page == "System status":
    render_status()

with st.expander("Privacy & prototype use", expanded=False):
    render_privacy_use()
