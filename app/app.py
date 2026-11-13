"""SIMGuard AI analyst console. Run with: streamlit run app.py"""

from datetime import datetime, timezone
import html
from uuid import uuid4

import pandas as pd
import streamlit as st

import api_client
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
    st.dataframe(frame, use_container_width=True, hide_index=True, height=min(380, 42 + 36 * len(frame)))
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
        if st.button("↻ Refresh", use_container_width=True):
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
            st.line_chart(chart, use_container_width=True, height=250)
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
            st.bar_chart(bucket_counts, use_container_width=True, height=250)
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
        render_case_detail(case)


def render_case_detail(case):
    final, ml, laya = case.get("final") or {}, case.get("ml") or {}, case.get("laya") or {}
    case_tier = str(final.get("tier", "unknown")).lower()
    tier_pill = f"<span class='hl-yellow'>TIER: {case_tier.upper()}</span>" if case_tier == "high" else f"<span class='hl-silver'>TIER: {case_tier.upper()}</span>"
    flag_pill = "<span class='hl-dark'>FLAGGED FOR AUDIT</span>" if flag(case) else "<span class='hl-gray'>NO ELEVATED RISK</span>"

    dossier_html = f"""
    <div class='dossier-box'>
        <div class='dossier-top'>
            <div>
                <span class='block-tag-silver'>FORENSIC CASE DOSSIER</span>
                <span style='margin-left:8px;font-family:var(--font-mono);font-size:0.75rem;color:var(--silver-muted);'>CREATED: {case.get('created_at','—')}</span>
            </div>
            <div class='case-id-code'>{case.get('case_id')}</div>
        </div>
        <div style='display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:20px;'>
            <div>
                <div style='font-family:var(--font-mono);font-size:0.72rem;letter-spacing:0.08em;color:var(--silver-muted);text-transform:uppercase;'>FUSED RISK SCORE</div>
                <div class='dossier-score-val'>{fmt(final.get('score'))}<span class='dossier-score-max'> / 100</span></div>
                <div style='display:flex;gap:8px;align-items:center;'>{tier_pill} &nbsp; {flag_pill}</div>
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

    cols = st.columns(4)
    for col, label, val in zip(cols, ["LightGBM probability", "Anomaly score", "Laya probability", "Final fused score"], [ml.get("lgbm_prob"), ml.get("anomaly"), laya.get("p_fraud"), final.get("score")]):
        col.metric(label, fmt(val, 2) if val is not None else "Unknown")

    st.markdown("""
    <div class='section-block-header'>
        <div class='section-block-pip'></div>
        <div><h2 class='section-block-title'>TELEMETRY EVIDENCE</h2></div>
    </div>
    """, unsafe_allow_html=True)
    evidence = case.get("evidence") or []
    if evidence:
        evidence_rows = "".join(f"<div class='evidence-row'><span class='evidence-pip'>▲</span><span>{html.escape(item)}</span></div>" for item in evidence)
        st.markdown(f"<div class='evidence-box'>{evidence_rows}</div>", unsafe_allow_html=True)
    else:
        st.caption("No evidence details were returned by the API.")

    report = case.get("report")
    if report:
        st.markdown("""
        <div class='section-block-header'>
            <div class='section-block-pip section-pip-silver'></div>
            <div><h2 class='section-block-title'>INVESTIGATION REPORT</h2></div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Source: {case.get('report_source','unknown')} · Report explains provided evidence; it does not calculate the risk score.")
        st.markdown(f"<div class='panel' style='font-size:0.92rem;line-height:1.6;'>{html.escape(report)}</div>", unsafe_allow_html=True)
    elif not flag(case):
        st.info("No elevated risk signal detected; no investigation report was generated.")

    st.markdown("""
    <div class='section-block-header'>
        <div class='section-block-pip'></div>
        <div><h2 class='section-block-title'>HUMAN REVIEW DECISION</h2></div>
    </div>
    """, unsafe_allow_html=True)
    current = case.get("status", "pending")
    if current == "pending":
        decision = st.radio("Choose a human review outcome", ["No decision", "Confirm fraud", "Mark as false alarm"], horizontal=True, key=f"decision_{case.get('case_id')}")
        if decision != "No decision":
            st.warning(f"Confirm recording ‘{decision}’ for this case.")
            if st.button("Confirm review decision", type="primary", key=f"confirm_{case.get('case_id')}"):
                try:
                    api_client.review_case(case["case_id"], "confirmed" if decision == "Confirm fraud" else "false_alarm")
                    st.session_state.selected_case = case["case_id"]
                    load_data(force=True)
                    st.success("Review decision saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Review was not saved: {exc}")
    else:
        st.success(f"Analyst decision recorded: {current.replace('_',' ').title()}")


def render_analyze():
    render_hero(
        tag="WORKSPACE // 03",
        title_html="TELEMETRY <span class='hl-yellow'>EVENT SCREENER</span>",
        subtitle_html="Evaluate synthetic <span class='hl-yellow'>11-feature SIM-swap events</span> through <span class='hl-silver'>LightGBM, Isolation Forest, and Laya</span>.",
        badge="INFERENCE ENGINE",
    )
    guided, upload = st.tabs(["Guided questions", "Upload CSV"])
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
            submitted = st.form_submit_button("Analyze event", type="primary", use_container_width=True)
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
                try:
                    result = api_client.analyze(payload)
                    st.session_state.selected_case = result.get("case_id")
                    load_data(force=True)
                    st.success("Analysis completed and saved to the case store.")
                    render_case_detail(result)
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}. No mock score was substituted. Check API health and retry.")
    with upload:
        st.markdown("#### Analyze a CSV batch")
        st.caption("The API parses and analyzes up to 2,000 rows. Required model features and supported column aliases are listed below.")
        st.code(",".join(EVENT_FIELDS), language="text")
        file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_upload")
        if file:
            content = file.getvalue()
            st.caption(f"{len(content):,} bytes · {file.name}")
            try:
                raw = pd.read_csv(file, dtype=str)
            except Exception as exc:
                st.error(f"Could not read this CSV: {exc}")
            else:
                id_columns = [
                    name for name in raw.columns
                    if name.strip().lower().replace("-", "_").replace(" ", "_") in {"msisdn_hash", "msisdn"}
                ]
                unsafe_ids = []
                for column in id_columns:
                    values = raw[column].dropna().str.strip()
                    values = values[values.ne("")]
                    if not values.str.upper().str.startswith("SYNTH-").all():
                        unsafe_ids.append(column)
                if unsafe_ids:
                    st.error("Subscriber identifier columns must contain only generated SYNTH- tokens: " + ", ".join(unsafe_ids))
                elif st.button("Analyze CSV", type="primary", key="analyze_csv"):
                    try:
                        response = api_client.analyze_upload(file.name, content)
                        st.session_state.last_batch_response = response
                        load_data(force=True)
                        st.success(f"Analyzed {response['rows_analysed']} of {response['rows_received']} rows.")
                    except Exception as exc:
                        st.error(f"Batch analysis failed: {exc}. Check the API connection and review any row-level errors.")

        response = st.session_state.get("last_batch_response")
        if response:
            for note in response.get("notes", []):
                st.info(note)
            errors = response.get("errors", [])
            if errors:
                st.warning(f"{len(errors)} row(s) could not be analyzed.")
                st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
            results = response.get("results", [])
            if results:
                st.dataframe(case_frame(results), use_container_width=True, hide_index=True, height=min(500, 42 + 36 * min(len(results), 12)))
                case_options = {
                    f"{str(item.get('case_id', ''))[:12]} · {fmt(score(item))}/100": item
                    for item in results
                }
                selected = st.selectbox("Inspect an analyzed case", ["Select a case…", *case_options], key="batch_case_detail")
                if selected != "Select a case…":
                    render_case_detail(case_options[selected])


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
    st.markdown("""
    <div class='section-block-header'>
        <div class='section-block-pip'></div>
        <div><h2 class='section-block-title'>CURRENT BACKEND ARCHITECTURE</h2></div>
    </div>
    """, unsafe_allow_html=True)
    st.write("• CSV batches use POST /analyze/upload; the API caps processing at 2,000 rows.")
    st.write("• No analytics aggregation endpoint; dashboard charts aggregate GET /cases in the UI.")
    st.write("• The guided form and CSV columns use the same 11 model features as the live inference pipeline.")


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
    config = health
    mocked = bool(config.get("mock_models"))
    checkpoint = str(config.get("laya_checkpoint") or "unknown")
    entries = [
        ("FastAPI Application Gateway", "Available", "badge-online"),
        ("SQLite Case Persistence Layer", "Available" if config.get("db_ok") else "Unavailable", "badge-online" if config.get("db_ok") else "badge-offline"),
        ("LightGBM + Isolation Forest Pipeline", "Mock" if mocked else "Unknown", "badge-mock" if mocked else "badge-offline"),
        ("Laya / OpenLaya Neural Engine", "Mock" if mocked else "Available" if checkpoint.lower() != "unknown" else "Unknown", "badge-mock" if mocked else "badge-online" if checkpoint.lower() != "unknown" else "badge-offline"),
        ("Qwen Report Generation Agent", "Mock" if mocked else "Available" if config.get("ollama_ok") else "Unavailable", "badge-mock" if mocked else "badge-online" if config.get("ollama_ok") else "badge-offline"),
    ]
    for label, value, badge_class in entries:
        st.markdown(f"<div class='health-block-item'><span class='health-name'>{label}</span><span class='{badge_class}'>{value.upper()}</span></div>", unsafe_allow_html=True)
    st.caption(f"Fusion uses weighted log-odds. Laya checkpoint: {checkpoint}. Current Laya weight: {config.get('laya_weight','unknown')}; threshold: {config.get('laya_fraud_threshold','unknown')}.")


def render_privacy_use():
    st.markdown("## Privacy & prototype use")
    st.caption("A plain-language account of the data this local prototype handles and its limits.")
    st.warning("Synthetic data only. Do not enter real subscriber information, real subscriber hashes, or live case records.")

    st.markdown("### Data and purpose")
    st.write("The analysis form accepts an event ID, a generated synthetic subscriber token, timestamp, and the 11 SIMGuard model features. CSV uploads are parsed by the API and may include these features plus optional synthetic IDs and context. The API returns and stores analyzed events with risk scores, evidence, report text, case status, and timestamps. This data is used to screen demo events, show case results, and record a human review.")

    st.markdown("### Where data goes and how long it stays")
    st.write(f"Submitting one event sends it from this browser to the configured API at `{api_client.API_URL}`. The API stores the full response in SQLite (`DB_PATH`, or `simguard.db` by default); review actions also create an audit record. The prototype has no automatic retention limit, erasure workflow, or delete endpoint. Stored rows remain until the database is manually cleared by its operator. The API has no authentication, so keep both services on a trusted local machine and do not expose them to the public internet.")
    st.write("CSV file contents are sent to the configured API for parsing and batch analysis. The API stores each analyzed case in SQLite.")

    st.markdown("### Analytics and external services")
    st.write("No Google Analytics, advertising pixels, or iframe embeds are present in the UI. Streamlit usage statistics are disabled in the app-local configuration. The API health check requests `/api/tags` from the configured `OLLAMA_URL` (default: localhost); if an operator points that setting to a remote service, that connection must be disclosed before real data is used. The Qwen agent is not included in this repository, so the current pipeline uses its local template fallback for reports.")

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
