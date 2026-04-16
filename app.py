import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from modules.market_structure import analyze_market_structure
from modules.trend_analysis import analyze_trend
from modules.support_resistance import find_support_resistance
from modules.demand_supply import find_demand_supply_zones
from modules.candlestick_patterns import detect_candlestick_patterns
from modules.breakouts import detect_breakouts
from modules.volume_analysis import analyze_volume
from modules.chart_patterns import detect_chart_patterns
from modules.multi_timeframe import multi_timeframe_analysis
from modules.trade_planning import generate_trade_plan
from modules.risk_management import calculate_risk
from modules.indicators import compute_indicators
from modules.utils import format_currency, get_color
from modules.darvas_box import detect_darvas_boxes
from modules.trading_signal import compute_trading_signal
from modules.fundamentals import get_fundamentals
from modules.crossover_tab import render_crossover_tab

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TA Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;600;700&display=swap');

/* ── Root ── */
:root {
    --bg: #ffffff;
    --panel: #f6f8fa;
    --border: #d0d7de;
    --green: #1a7f37;
    --red: #cf222e;
    --amber: #9a6700;
    --blue: #0969da;
    --text: #24292f;
    --dim: #6e7781;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
    background-color: #ffffff !important;
    color: #24292f !important;
}

.stApp { background: #ffffff; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #f6f8fa !important;
    border-right: 1px solid #d0d7de !important;
}
[data-testid="stSidebar"] * { color: #24292f !important; }

/* ── Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #d0d7de !important;
    color: #24292f !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 6px !important;
}

.stSlider [data-baseweb="thumb"] { background: #0969da !important; }
.stSlider [data-baseweb="track-background"] { background: #d0d7de !important; }
.stSlider [data-baseweb="track"] { background: #0969da !important; }

/* ── Buttons ── */
.stButton > button {
    background: #0969da !important;
    border: 1px solid #0969da !important;
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    border-radius: 6px !important;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #0550ae !important;
    border-color: #0550ae !important;
}

/* ── Metric Cards ── */
.metric-card {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-top: 3px solid #0969da;
    padding: 12px 12px 10px 12px;
    margin-bottom: 4px;
    overflow: hidden;
    border-radius: 0 0 6px 6px;
}
.metric-label {
    font-family: 'Inter', sans-serif !important;
    font-size: 10px !important;
    color: #6e7781 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    margin-bottom: 4px !important;
    display: block !important;
}
.metric-prefix {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    color: #6e7781 !important;
    display: block !important;
    margin-bottom: 1px !important;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #0969da !important;
    margin-bottom: 3px !important;
    display: block !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 100% !important;
}
.metric-delta {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #6e7781 !important;
    display: block !important;
}
.metric-delta.positive { color: #1a7f37 !important; }
.metric-delta.negative { color: #cf222e !important; }
 
/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: #f6f8fa !important;
    border-bottom: 1px solid #d0d7de !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #6e7781 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.2px !important;
    border-radius: 0 !important;
    border-right: 1px solid #d0d7de !important;
    padding: 10px 18px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #ffffff !important;
    color: #0969da !important;
    border-top: 2px solid #0969da !important;
    font-weight: 600 !important;
}
[data-baseweb="tab-panel"] { background: #ffffff !important; padding-top: 16px !important; }
 
/* ── Expander ── */
.streamlit-expanderHeader {
    background: #f6f8fa !important;
    border: 1px solid #d0d7de !important;
    border-radius: 6px !important;
    color: #24292f !important;
    font-family: 'Inter', sans-serif !important;
}
.streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid #d0d7de !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #d0d7de !important; border-radius: 6px; }
.dataframe { background: #ffffff !important; color: #24292f !important; }
th { background: #f6f8fa !important; color: #24292f !important;
     font-family: 'Inter', sans-serif !important;
     font-weight: 600 !important; font-size: 12px !important;
     border-bottom: 2px solid #d0d7de !important; }
td { color: #24292f !important; border-bottom: 1px solid #eaecef !important;
     font-size: 12px !important; }

/* ── TA Cards ── */
.ta-card {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-top: 3px solid #0969da;
    padding: 14px;
    margin-bottom: 8px;
    border-radius: 0 0 6px 6px;
    font-family: 'Inter', sans-serif;
}
.ta-card-title {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 700;
    color: #0969da;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.ta-card-danger { border-top-color: #cf222e; }
.ta-card-danger .ta-card-title { color: #cf222e; }
.ta-card-warn { border-top-color: #9a6700; }
.ta-card-warn .ta-card-title { color: #9a6700; }
.ta-card-cyan { border-top-color: #0550ae; }
.ta-card-cyan .ta-card-title { color: #0550ae; }
 
.ta-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    border-left: 3px solid #d0d7de;
    margin: 4px 0;
    font-size: 12px;
    color: #24292f;
    font-family: 'JetBrains Mono', monospace;
}
.ta-item-bullet { color: #6e7781; }
 
.ta-motto {
    font-size: 11px;
    color: #9a6700;
    border-top: 1px solid #d0d7de;
    padding-top: 8px;
    margin-top: 8px;
    font-style: italic;
}
 
.module-header {
    font-family: 'Inter', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #0969da;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.module-sub {
    font-size: 11px;
    color: #6e7781;
    letter-spacing: 1px;
    margin-bottom: 16px;
}
 
.signal-bull {
    background: #dafbe1;
    border: 1px solid #1a7f37;
    color: #1a7f37;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 4px;
    display: inline-block;
}
.signal-bear {
    background: #ffd7d9;
    border: 1px solid #cf222e;
    color: #cf222e;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 4px;
    display: inline-block;
}
.signal-neutral {
    background: #fff8c5;
    border: 1px solid #9a6700;
    color: #9a6700;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 4px;
    display: inline-block;
}
 
.divider {
    border: none;
    border-top: 1px solid #d0d7de;
    margin: 14px 0;
}
 
/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f6f8fa; }
::-webkit-scrollbar-thumb { background: #d0d7de; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #0969da; }
 
/* ── Radio / Checkbox ── */
.stRadio > div { gap: 8px; }
.stRadio label { color: #24292f !important; font-size: 12px !important; }
.stCheckbox label { color: #24292f !important; font-size: 12px !important; }
 
/* ── Spinner ── */
.stSpinner > div { border-top-color: #0969da !important; }
 
/* ── Progress bar ── */
.stProgress > div > div { background: #0969da !important; }
</style>
""", unsafe_allow_html=True)


/* ── TA Cards ── */
.ta-card {
    background: #f6f8fa;
    border: 1px solid #d0d7de;
    border-top: 3px solid #0969da;
    padding: 14px;
    margin-bottom: 8px;
    border-radius: 0 0 6px 6px;
    font-family: 'Inter', sans-serif;
}
.ta-card-title {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 700;
    color: #0969da;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.ta-card-danger { border-top-color: #cf222e; }
.ta-card-danger .ta-card-title { color: #cf222e; }
.ta-card-warn { border-top-color: #9a6700; }
.ta-card-warn .ta-card-title { color: #9a6700; }
.ta-card-cyan { border-top-color: #0550ae; }
.ta-card-cyan .ta-card-title { color: #0550ae; }
 
.ta-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    border-left: 3px solid #d0d7de;
    margin: 4px 0;
    font-size: 12px;
    color: #24292f;
    font-family: 'JetBrains Mono', monospace;
}
.ta-item-bullet { color: #6e7781; }
 
.ta-motto {
    font-size: 11px;
    color: #9a6700;
    border-top: 1px solid #d0d7de;
    padding-top: 8px;
    margin-top: 8px;
    font-style: italic;
}
 
.module-header {
    font-family: 'Inter', sans-serif;
    font-size: 22px;
    font-weight: 800;
    color: #0969da;
    letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.module-sub {
    font-size: 11px;
    color: #6e7781;
    letter-spacing: 1px;
    margin-bottom: 16px;
}
 
.signal-bull {
    background: #dafbe1;
    border: 1px solid #1a7f37;
    color: #1a7f37;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 4px;
    display: inline-block;
}
.signal-bear {
    background: #ffd7d9;
    border: 1px solid #cf222e;
    color: #cf222e;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 4px;
    display: inline-block;
}
.signal-neutral {
    background: #fff8c5;
    border: 1px solid #9a6700;
    color: #9a6700;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    border-radius: 4px;
    display: inline-block;
}
 
.divider {
    border: none;
    border-top: 1px solid #d0d7de;
    margin: 14px 0;
}
 
/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f6f8fa; }
::-webkit-scrollbar-thumb { background: #d0d7de; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #0969da; }
 
/* ── Radio / Checkbox ── */
.stRadio > div { gap: 8px; }
.stRadio label { color: #24292f !important; font-size: 12px !important; }
.stCheckbox label { color: #24292f !important; font-size: 12px !important; }
 
/* ── Spinner ── */
.stSpinner > div { border-top-color: #0969da !important; }
 
/* ── Progress bar ── */
.stProgress > div > div { background: #0969da !important; }
</style>
""", unsafe_allow_html=True)
