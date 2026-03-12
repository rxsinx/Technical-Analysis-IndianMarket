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
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

/* ── Root ── */
:root {
    --bg: #020c06;
    --panel: #040f08;
    --green: #00ff6a;
    --green2: #00cc55;
    --green3: #007733;
    --red: #ff3355;
    --amber: #ffaa00;
    --cyan: #00ffcc;
    --text: #a0ffc0;
    --dim: #3a6648;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace !important;
    background-color: #020c06 !important;
    color: #a0ffc0 !important;
}

.stApp { background: #020c06; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #020c06 !important;
    border-right: 1px solid #0d3318 !important;
}
[data-testid="stSidebar"] * { color: #a0ffc0 !important; }

/* ── Inputs ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #040f08 !important;
    border: 1px solid #0d3318 !important;
    color: #00ff6a !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 0 !important;
}

.stSlider [data-baseweb="slider"] { }
.stSlider [data-baseweb="thumb"] { background: #00ff6a !important; }
.stSlider [data-baseweb="track-background"] { background: #0d3318 !important; }
.stSlider [data-baseweb="track"] { background: #00ff6a !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00ff6a !important;
    color: #00ff6a !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    border-radius: 0 !important;
    text-transform: uppercase;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #00ff6a22 !important;
    box-shadow: 0 0 12px #00ff6a55 !important;
}

/* ── Metric Cards ── */
.metric-card {
    background: #040f08;
    border: 1px solid #0d3318;
    border-top: 2px solid #00ff6a;
    padding: 12px 14px;
    margin-bottom: 4px;
}
.metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: #3a6648;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 5px;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
    color: #00ff6a;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.metric-delta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: #ffaa00;
    letter-spacing: 1px;
}
.metric-delta.positive { color: #00ff6a; }
.metric-delta.negative { color: #ff3355; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { background: #040f08 !important; border-bottom: 1px solid #0d3318 !important; gap: 0 !important; }
[data-baseweb="tab"] {
    background: transparent !important;
    color: #3a6648 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 2px !important;
    border-radius: 0 !important;
    border-right: 1px solid #0d3318 !important;
    padding: 8px 20px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: #061510 !important;
    color: #00ff6a !important;
    border-top: 2px solid #00ff6a !important;
}
[data-baseweb="tab-panel"] { background: #020c06 !important; padding-top: 16px !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #040f08 !important;
    border: 1px solid #0d3318 !important;
    border-radius: 0 !important;
    color: #00ff6a !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 1px;
}
.streamlit-expanderContent { background: #020c06 !important; border: 1px solid #0d3318 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #0d3318 !important; }
.dataframe { background: #040f08 !important; color: #a0ffc0 !important; }
th { background: #061510 !important; color: #00ff6a !important; font-family: 'Share Tech Mono', monospace !important; border-bottom: 1px solid #0d3318 !important; }
td { color: #a0ffc0 !important; border-bottom: 1px solid #0a1a0e !important; }

/* ── Custom cards ── */
.ta-card {
    background: #040f08;
    border: 1px solid #0d3318;
    border-top: 2px solid #007733;
    padding: 14px;
    margin-bottom: 8px;
    font-family: 'Share Tech Mono', monospace;
}
.ta-card-title {
    font-family: 'Orbitron', monospace;
    font-size: 10px;
    color: #00ff6a;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 10px;
    text-shadow: 0 0 8px #00ff6a55;
}
.ta-card-danger { border-top-color: #ff3355; }
.ta-card-danger .ta-card-title { color: #ff3355; text-shadow: 0 0 8px #ff335555; }
.ta-card-warn { border-top-color: #ffaa00; }
.ta-card-warn .ta-card-title { color: #ffaa00; }
.ta-card-cyan { border-top-color: #00ffcc; }
.ta-card-cyan .ta-card-title { color: #00ffcc; }

.ta-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 6px;
    border-left: 2px solid #004422;
    margin: 3px 0;
    font-size: 12px;
}
.ta-item-bullet { color: #007733; }

.ta-motto {
    font-size: 10px;
    color: #ffaa00;
    border-top: 1px solid #0d3318;
    padding-top: 8px;
    margin-top: 8px;
    font-style: italic;
    letter-spacing: 1px;
}

.module-header {
    font-family: 'Orbitron', monospace;
    font-size: 20px;
    font-weight: 900;
    color: #00ff6a;
    text-shadow: 0 0 12px #00ff6a99, 0 0 30px #00ff6a44;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.module-sub {
    font-size: 10px;
    color: #3a6648;
    letter-spacing: 6px;
    margin-bottom: 16px;
}

.signal-bull {
    background: #002211;
    border: 1px solid #007733;
    color: #00ff6a;
    padding: 4px 10px;
    font-size: 11px;
    letter-spacing: 2px;
    display: inline-block;
}
.signal-bear {
    background: #1a0008;
    border: 1px solid #ff3355;
    color: #ff3355;
    padding: 4px 10px;
    font-size: 11px;
    letter-spacing: 2px;
    display: inline-block;
}
.signal-neutral {
    background: #1a1000;
    border: 1px solid #ffaa00;
    color: #ffaa00;
    padding: 4px 10px;
    font-size: 11px;
    letter-spacing: 2px;
    display: inline-block;
}

.status-pill {
    display: inline-block;
    padding: 2px 8px;
    font-size: 10px;
    letter-spacing: 2px;
    margin: 2px;
}

.divider {
    border: none;
    border-top: 1px solid #0d3318;
    margin: 12px 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #020c06; }
::-webkit-scrollbar-thumb { background: #007733; }
::-webkit-scrollbar-thumb:hover { background: #00ff6a; }

/* ── Radio buttons ── */
.stRadio > div { gap: 8px; }
.stRadio label { color: #a0ffc0 !important; font-size: 11px !important; }

/* ── Checkbox ── */
.stCheckbox label { color: #a0ffc0 !important; font-size: 11px !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00ff6a !important; }

/* ── Plotly chart background ── */
.js-plotly-plot { background: #040f08 !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: #00ff6a !important; }
</style>
""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────────────────────
def render_header():
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown("""
        <div class="module-header">TA // TERMINAL</div>
        <div class="module-sub">TECHNICAL ANALYSIS SYSTEM v2.4 &nbsp;|&nbsp; INDIAN EQUITY &nbsp;|&nbsp; 16 MODULES LOADED</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="text-align:right; padding-top:8px;">
            <span style="color:#00ff6a; font-size:10px; letter-spacing:2px;">● LIVE</span><br>
            <span style="color:#3a6648; font-size:10px;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="text-align:right; padding-top:8px;">
            <span style="color:#3a6648; font-size:10px; letter-spacing:2px;">RISK: </span>
            <span style="color:#00ff6a; font-size:10px;">MANAGED</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #0d3318;margin:0 0 16px 0;'>", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
def render_sidebar():
    st.sidebar.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:14px;color:#00ff6a;
    letter-spacing:3px;text-shadow:0 0 8px #00ff6a55;margin-bottom:16px;">
    ⬡ CONTROL PANEL
    </div>
    """, unsafe_allow_html=True)

    # Symbol input
    st.sidebar.markdown("<div style='font-size:10px;color:#3a6648;letter-spacing:2px;'>SYMBOL</div>", unsafe_allow_html=True)
    symbol = st.sidebar.text_input("", value="RELIANCE.NS", key="symbol_input", label_visibility="collapsed").upper().strip()

    # Timeframe
    st.sidebar.markdown("<div style='font-size:10px;color:#3a6648;letter-spacing:2px;margin-top:12px;'>PRIMARY TIMEFRAME</div>", unsafe_allow_html=True)
    timeframe = st.sidebar.selectbox("", ["Daily", "Weekly", "Monthly", "4H", "1H"], key="tf", label_visibility="collapsed")

    # Period
    st.sidebar.markdown("<div style='font-size:10px;color:#3a6648;letter-spacing:2px;margin-top:12px;'>LOOKBACK PERIOD</div>", unsafe_allow_html=True)
    period = st.sidebar.selectbox("", ["6mo", "1y", "2y", "5y", "max"], index=1, key="period", label_visibility="collapsed")

    # Risk %
    st.sidebar.markdown("<div style='font-size:10px;color:#3a6648;letter-spacing:2px;margin-top:12px;'>RISK PER TRADE (%)</div>", unsafe_allow_html=True)
    risk_pct = st.sidebar.slider("", 0.5, 5.0, 1.0, 0.5, key="risk_pct", label_visibility="collapsed")

    # Capital
    st.sidebar.markdown("<div style='font-size:10px;color:#3a6648;letter-spacing:2px;margin-top:12px;'>CAPITAL (INR)</div>", unsafe_allow_html=True)
    capital = st.sidebar.number_input("", value=100000, step=10000, key="capital", label_visibility="collapsed")

    # Indicators
    st.sidebar.markdown("<hr style='border:none;border-top:1px solid #0d3318;margin:16px 0;'>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='font-size:10px;color:#3a6648;letter-spacing:2px;'>OVERLAYS</div>", unsafe_allow_html=True)
    show_ema = st.sidebar.checkbox("EMA (20/50/200)", value=True)
    show_bb  = st.sidebar.checkbox("Bollinger Bands", value=True)
    show_vwap= st.sidebar.checkbox("VWAP", value=False)
    show_sr  = st.sidebar.checkbox("Support / Resistance", value=True)
    show_dz  = st.sidebar.checkbox("Demand / Supply Zones", value=True)

    st.sidebar.markdown("<hr style='border:none;border-top:1px solid #0d3318;margin:16px 0;'>", unsafe_allow_html=True)
    analyze = st.sidebar.button("▶  RUN ANALYSIS", use_container_width=True)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "period": period,
        "risk_pct": risk_pct,
        "capital": capital,
        "show_ema": show_ema,
        "show_bb": show_bb,
        "show_vwap": show_vwap,
        "show_sr": show_sr,
        "show_dz": show_dz,
        "analyze": analyze,
    }


# ── DATA FETCH ────────────────────────────────────────────────────────────────
TF_MAP = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo", "4H": "1h", "1H": "1h"}

@st.cache_data(ttl=300)
def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df

@st.cache_data(ttl=300)
def fetch_info(symbol: str) -> dict:
    try:
        t = yf.Ticker(symbol)
        info = t.info
        return info
    except:
        return {}


# ── MAIN CHART ────────────────────────────────────────────────────────────────
def build_main_chart(df: pd.DataFrame, indicators: dict, cfg: dict) -> go.Figure:
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=["", "", ""],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing=dict(line=dict(color="#00ff6a", width=1), fillcolor="rgba(0,255,106,0.15)"),
        decreasing=dict(line=dict(color="#ff3355", width=1), fillcolor="rgba(255,51,85,0.15)"),
        name="Price",
    ), row=1, col=1)

    # EMAs
    if cfg["show_ema"]:
        colors = {"ema20": "#00ffcc", "ema50": "#ffaa00", "ema200": "#ff3355"}
        for key, color in colors.items():
            if key in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators[key],
                    mode="lines", name=key.upper(),
                    line=dict(color=color, width=1, dash="solid"),
                    opacity=0.8,
                ), row=1, col=1)

    # Bollinger Bands
    if cfg["show_bb"] and "bb_upper" in indicators:
        fig.add_trace(go.Scatter(
            x=df.index, y=indicators["bb_upper"],
            mode="lines", name="BB Upper",
            line=dict(color="#3a6648", width=1, dash="dot"), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=indicators["bb_lower"],
            mode="lines", name="BB Lower",
            line=dict(color="#3a6648", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(0,255,106,0.03)", showlegend=False,
        ), row=1, col=1)

    # S/R Levels
    if cfg["show_sr"] and "sr_levels" in indicators:
        for level in indicators["sr_levels"][:6]:
            fig.add_hline(
                y=level["price"], row=1, col=1,
                line=dict(color="rgba(255,170,0,0.4)" if level["type"] == "R" else "rgba(0,255,106,0.4)", width=1, dash="dot"),
                annotation_text=f"{level['type']} {level['price']:.2f}",
                annotation_font_size=9, annotation_font_color="#ffaa00" if level["type"] == "R" else "#00ff6a",
            )

    # Demand/Supply Zones
    if cfg["show_dz"] and "zones" in indicators:
        for z in indicators["zones"][:4]:
            fig.add_hrect(
                y0=z["low"], y1=z["high"], row=1, col=1,
                fillcolor="rgba(0,255,106,0.07)" if z["type"] == "demand" else "rgba(255,51,85,0.07)",
                line_width=0,
                annotation_text=z["type"].upper(),
                annotation_font_size=8,
                annotation_font_color="#00ff6a" if z["type"] == "demand" else "#ff3355",
            )

    # Volume
    colors_vol = ["rgba(0,255,106,0.33)" if c >= o else "rgba(255,51,85,0.33)"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=colors_vol, name="Volume", showlegend=False,
    ), row=2, col=1)

    # Volume MA
    vol_ma = df["Volume"].rolling(20).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=vol_ma,
        mode="lines", name="Vol MA20",
        line=dict(color="#ffaa00", width=1), showlegend=False,
    ), row=2, col=1)

    # RSI
    if "rsi" in indicators:
        fig.add_trace(go.Scatter(
            x=df.index, y=indicators["rsi"],
            mode="lines", name="RSI",
            line=dict(color="#00ffcc", width=1.5),
        ), row=3, col=1)
        fig.add_hline(y=70, row=3, col=1, line=dict(color="rgba(255,51,85,0.27)", width=1, dash="dot"))
        fig.add_hline(y=30, row=3, col=1, line=dict(color="rgba(0,255,106,0.27)", width=1, dash="dot"))
        fig.add_hline(y=50, row=3, col=1, line=dict(color="#3a6648", width=1, dash="dot"))

    # Layout
    fig.update_layout(
        paper_bgcolor="#020c06",
        plot_bgcolor="#040f08",
        font=dict(family="Share Tech Mono", color="#a0ffc0", size=10),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(
            bgcolor="#040f08", bordercolor="#0d3318", borderwidth=1,
            font=dict(size=9), orientation="h", yanchor="bottom", y=1.01,
        ),
        height=600,
    )
    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor="#0d3318", showgrid=True, zeroline=False,
            tickfont=dict(size=9), row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor="#0d3318", showgrid=True, zeroline=False,
            tickfont=dict(size=9), row=i, col=1,
        )
    return fig


# ── MODULE CARDS ──────────────────────────────────────────────────────────────
def ta_card(num: str, title: str, items: list, motto: str, style: str = ""):
    cls = f"ta-card-{style}" if style else ""
    items_html = "".join([
        f'<div class="ta-item"><span class="ta-item-bullet">▸</span>{item}</div>'
        for item in items
    ])
    st.markdown(f"""
    <div class="ta-card {cls}">
        <div style="font-size:9px;color:#3a6648;letter-spacing:3px;margin-bottom:6px;">{num}</div>
        <div class="ta-card-title">{title}</div>
        {items_html}
        <div class="ta-motto">> {motto}</div>
    </div>
    """, unsafe_allow_html=True)


def signal_badge(label: str, signal: str):
    cls = {"BULLISH": "signal-bull", "BEARISH": "signal-bear"}.get(signal.upper(), "signal-neutral")
    st.markdown(f'<span class="{cls}">{label}: {signal}</span>', unsafe_allow_html=True)


# ── TABS ──────────────────────────────────────────────────────────────────────
def render_overview_tab(df, info, indicators, cfg):
    # Top metrics
    last = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]
    chg  = last - prev
    chg_pct = chg / prev * 100
    vol  = df["Volume"].iloc[-1]
    avg_vol = df["Volume"].rolling(20).mean().iloc[-1]

    metrics = [
        ("LAST PRICE", f"INR {last:,.2f}",  f"{chg:+.2f}",              "positive" if chg >= 0 else "negative"),
        ("CHANGE",     f"{chg_pct:+.2f}%",  None,                       "positive" if chg_pct >= 0 else "negative"),
        ("HIGH (D)",   f"INR {df['High'].iloc[-1]:,.2f}", None,          ""),
        ("LOW (D)",    f"INR {df['Low'].iloc[-1]:,.2f}",  None,          ""),
        ("VOLUME",     f"{vol/1e6:.2f}M",    f"avg {avg_vol/1e6:.2f}M", ""),
        ("52W RANGE",  f"INR {df['Low'].tail(252).min():,.0f} – INR {df['High'].tail(252).max():,.0f}", None, ""),
    ]
    cols = st.columns(6)
    for col, (lbl, val, delta, dcls) in zip(cols, metrics):
        with col:
            delta_html = f'<div class="metric-delta {dcls}">{delta}</div>' if delta else ""
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{lbl}</div>'
                f'<div class="metric-value">{val}</div>'
                f'{delta_html}'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.plotly_chart(build_main_chart(df, indicators, cfg), use_container_width=True)


def render_structure_tab(df, analysis):
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:13px;color:#00ff6a;
    letter-spacing:3px;margin-bottom:16px;">MODULE 02–03 &nbsp;|&nbsp; MARKET STRUCTURE & TREND</div>
    """, unsafe_allow_html=True)

    ms = analysis.get("market_structure", {})
    tr = analysis.get("trend", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ta_card("MODULE_02", "Market Structure", [
            f"Swing Highs: {ms.get('swing_highs', 'N/A')}",
            f"Swing Lows:  {ms.get('swing_lows', 'N/A')}",
            f"Structure:   {ms.get('structure', 'N/A')}",
            f"Last Break:  {ms.get('last_break', 'N/A')}",
        ], "Trend before trade")
    with c2:
        ta_card("MODULE_03", "Trend Analysis", [
            f"Trend (D):   {tr.get('daily', 'N/A')}",
            f"Trend (W):   {tr.get('weekly', 'N/A')}",
            f"ADX:         {tr.get('adx', 'N/A')}",
            f"Direction:   {tr.get('direction', 'N/A')}",
        ], "Do not fight price", style="cyan")
    with c3:
        ta_card("MODULE_04", "Support & Resistance", [
            f"Major S:   INR{ms.get('major_support', 0):.2f}",
            f"Major R:   INR{ms.get('major_resistance', 0):.2f}",
            f"Nearest S: INR{ms.get('nearest_support', 0):.2f}",
            f"Nearest R: INR{ms.get('nearest_resistance', 0):.2f}",
        ], "Levels matter", style="warn")
    with c4:
        ta_card("MODULE_05", "Demand & Supply", [
            f"Demand:  INR{ms.get('demand_low', 0):.2f} – INR{ms.get('demand_high', 0):.2f}",
            f"Supply:  INR{ms.get('supply_low', 0):.2f} – INR{ms.get('supply_high', 0):.2f}",
            f"Strength: {ms.get('zone_strength', 'N/A')}",
            f"Tested:   {ms.get('zone_tested', 'N/A')}",
        ], "Where institutions may act")

    # Structure chart
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    _plot_structure(df, ms)


def _plot_structure(df, ms):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="Close",
        line=dict(color="#00ff6a", width=1.5),
    ))
    # Swing points
    if "swing_high_idx" in ms:
        for idx in ms["swing_high_idx"]:
            if idx < len(df):
                fig.add_trace(go.Scatter(
                    x=[df.index[idx]], y=[df["High"].iloc[idx]],
                    mode="markers+text", marker=dict(color="#ff3355", size=8, symbol="triangle-down"),
                    text=["H"], textfont=dict(size=8, color="#ff3355"),
                    textposition="top center", showlegend=False,
                ))
    if "swing_low_idx" in ms:
        for idx in ms["swing_low_idx"]:
            if idx < len(df):
                fig.add_trace(go.Scatter(
                    x=[df.index[idx]], y=[df["Low"].iloc[idx]],
                    mode="markers+text", marker=dict(color="#00ff6a", size=8, symbol="triangle-up"),
                    text=["L"], textfont=dict(size=8, color="#00ff6a"),
                    textposition="bottom center", showlegend=False,
                ))
    fig.update_layout(
        paper_bgcolor="#020c06", plot_bgcolor="#040f08",
        font=dict(family="Share Tech Mono", color="#a0ffc0", size=10),
        height=300, margin=dict(l=0,r=0,t=10,b=0),
        showlegend=False,
        xaxis=dict(gridcolor="#0d3318"),
        yaxis=dict(gridcolor="#0d3318"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_patterns_tab(df, analysis):
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:13px;color:#00ff6a;
    letter-spacing:3px;margin-bottom:16px;">MODULE 06–09 &nbsp;|&nbsp; PATTERNS & SIGNALS</div>
    """, unsafe_allow_html=True)

    cp  = analysis.get("candlestick_patterns", {})
    bk  = analysis.get("breakouts", {})
    vol = analysis.get("volume", {})
    chp = analysis.get("chart_patterns", {})

    c1, c2 = st.columns(2)
    with c1:
        ta_card("MODULE_06", "Candlestick Behaviour", [
            f"Last Pattern: {cp.get('last_pattern', 'None')}",
            f"Signal:       {cp.get('signal', 'Neutral')}",
            f"Engulfing:    {cp.get('engulfing', 'None')}",
            f"Pin Bar:      {cp.get('pin_bar', 'None')}",
            f"Inside Bar:   {cp.get('inside_bar', 'No')}",
        ], "Close tells the truth")
    with c2:
        ta_card("MODULE_07", "Breakouts & Breakdowns", [
            f"Status:       {bk.get('status', 'N/A')}",
            f"Type:         {bk.get('type', 'N/A')}",
            f"Level:        INR{bk.get('level', 0):.2f}",
            f"Retest:       {bk.get('retest', 'N/A')}",
            f"Confirmed:    {bk.get('confirmed', 'N/A')}",
        ], "Real or trap?", style="warn")

    c3, c4 = st.columns(2)
    with c3:
        ta_card("MODULE_08", "Volume Analysis", [
            f"Today Vol:  {vol.get('today', 'N/A')}",
            f"Avg Vol:    {vol.get('avg', 'N/A')}",
            f"Vol Ratio:  {vol.get('ratio', 'N/A')}",
            f"Signal:     {vol.get('signal', 'N/A')}",
            f"Climax:     {vol.get('climax', 'No')}",
        ], "Volume confirms price", style="cyan")
    with c4:
        ta_card("MODULE_09", "Chart Patterns", [
            f"Pattern:      {chp.get('pattern', 'None')}",
            f"Completion:   {chp.get('completion', 'N/A')}",
            f"Target:       INR{chp.get('target', 0):.2f}",
            f"Invalidation: INR{chp.get('invalidation', 0):.2f}",
        ], "Pattern with context only")

    # Recent candle signals table
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#3a6648;letter-spacing:2px;margin-bottom:8px;'>RECENT CANDLESTICK SIGNALS</div>", unsafe_allow_html=True)
    if "history" in cp and cp["history"]:
        hist_df = pd.DataFrame(cp["history"])
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


def render_mtf_tab(df, analysis, symbol, period):
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:13px;color:#00ff6a;
    letter-spacing:3px;margin-bottom:16px;">MODULE 10 &nbsp;|&nbsp; MULTI-TIMEFRAME ANALYSIS</div>
    """, unsafe_allow_html=True)

    mtf = analysis.get("mtf", {})

    cols = st.columns(4)
    tfs  = [("MONTHLY", "monthly"), ("WEEKLY", "weekly"), ("DAILY", "daily"), ("4H", "h4")]
    for col, (label, key) in zip(cols, tfs):
        with col:
            d = mtf.get(key, {})
            bias = d.get("bias", "NEUTRAL")
            style = "danger" if "BEAR" in bias else ("" if "BULL" not in bias else "")
            ta_card(f"TF: {label}", f"Bias: {bias}", [
                f"Trend:    {d.get('trend', 'N/A')}",
                f"EMA200:   {d.get('ema200_pos', 'N/A')}",
                f"RSI:      {d.get('rsi', 'N/A')}",
                f"Volume:   {d.get('volume', 'N/A')}",
            ], d.get("note", "—"), style=style)

    # Alignment summary
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    alignment = mtf.get("alignment", "MIXED")
    color = "#00ff6a" if "BULL" in alignment else ("#ff3355" if "BEAR" in alignment else "#ffaa00")
    st.markdown(f"""
    <div style="text-align:center;padding:20px;border:1px solid #0d3318;background:#040f08;">
        <div style="font-size:10px;color:#3a6648;letter-spacing:4px;margin-bottom:8px;">TIMEFRAME ALIGNMENT</div>
        <div style="font-family:'Orbitron',monospace;font-size:28px;font-weight:900;
        color:{color};text-shadow:0 0 20px {color}77;letter-spacing:6px;">{alignment}</div>
        <div style="font-size:10px;color:#3a6648;letter-spacing:2px;margin-top:8px;">Alignment is power</div>
    </div>
    """, unsafe_allow_html=True)


def render_trade_tab(df, analysis, cfg):
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:13px;color:#00ff6a;
    letter-spacing:3px;margin-bottom:16px;">MODULE 11–13 &nbsp;|&nbsp; TRADE PLANNING & EXECUTION</div>
    """, unsafe_allow_html=True)

    tp = analysis.get("trade_plan", {})
    rm = analysis.get("risk", {})

    c1, c2 = st.columns(2)
    with c1:
        last = df["Close"].iloc[-1]
        entry = tp.get("entry", last)
        sl    = tp.get("stop_loss", last * 0.97)
        t1    = tp.get("target1", last * 1.03)
        t2    = tp.get("target2", last * 1.06)
        rr    = tp.get("rr_ratio", 0)

        ta_card("MODULE_11", "Trade Planning", [
            f"Entry Zone:  INR{entry:.2f}",
            f"Stop Loss:   INR{sl:.2f}  ({((sl-entry)/entry*100):+.1f}%)",
            f"Target 1:    INR{t1:.2f}  ({((t1-entry)/entry*100):+.1f}%)",
            f"Target 2:    INR{t2:.2f}  ({((t2-entry)/entry*100):+.1f}%)",
            f"R:R Ratio:   1 : {rr:.1f}",
        ], "Plan before price", style="cyan")
    with c2:
        ta_card("MODULE_12", "Risk Management", [
            f"Capital:     INR{cfg['capital']:,.0f}",
            f"Risk %:      {cfg['risk_pct']}%",
            f"Risk INR:      INR{rm.get('risk_amount', 0):,.0f}",
            f"Position Sz: {rm.get('position_size', 0):.0f} qty",
            f"Notional:    INR{rm.get('notional', 0):,.0f}",
        ], "First survive", style="danger")

    # Risk-Reward visual
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    _plot_rr(df, tp)

    # Execution checklist
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    ta_card("MODULE_13", "Execution Checklist", [
        f"✓ Wait for confirmation close",
        f"✓ No early entry before signal",
        f"✓ Follow the plan — no deviation",
        f"✓ Exit without ego if wrong",
        f"✓ Journal every trade",
    ], "Discipline is edge")


def _plot_rr(df, tp):
    last_n = min(60, len(df))
    dff = df.tail(last_n)
    entry = tp.get("entry", df["Close"].iloc[-1])
    sl    = tp.get("stop_loss", entry * 0.97)
    t1    = tp.get("target1",  entry * 1.03)
    t2    = tp.get("target2",  entry * 1.06)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=dff.index, open=dff["Open"], high=dff["High"],
        low=dff["Low"], close=dff["Close"],
        increasing=dict(line=dict(color="#00ff6a", width=1), fillcolor="rgba(0,255,106,0.15)"),
        decreasing=dict(line=dict(color="#ff3355", width=1), fillcolor="rgba(255,51,85,0.15)"),
        name="Price",
    ))
    for level, color, label in [
        (entry, "#00ffcc", "ENTRY"),
        (sl,    "#ff3355", "STOP LOSS"),
        (t1,    "#00ff6a", "TARGET 1"),
        (t2,    "#007733", "TARGET 2"),
    ]:
        fig.add_hline(
            y=level, line=dict(color=color, width=1.5, dash="dash"),
            annotation_text=f"{label} INR{level:.2f}",
            annotation_font_size=9, annotation_font_color=color,
        )
    # Zones
    fig.add_hrect(y0=sl, y1=entry, fillcolor="rgba(255,51,85,0.07)", line_width=0)
    fig.add_hrect(y0=entry, y1=t2, fillcolor="rgba(0,255,106,0.07)", line_width=0)

    fig.update_layout(
        paper_bgcolor="#020c06", plot_bgcolor="#040f08",
        font=dict(family="Share Tech Mono", color="#a0ffc0", size=10),
        height=320, margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(gridcolor="#0d3318", rangeslider_visible=False),
        yaxis=dict(gridcolor="#0d3318"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_review_tab(analysis):
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:13px;color:#00ff6a;
    letter-spacing:3px;margin-bottom:16px;">MODULE 14–16 &nbsp;|&nbsp; REVIEW & FINAL RESULT</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        ta_card("MODULE_14", "Common Traps — Check", [
            f"Fake Breakout?  {analysis.get('traps', {}).get('fake_breakout', 'Monitor')}",
            f"Late Entry?     {analysis.get('traps', {}).get('late_entry', 'Check')}",
            f"Overtrading?    Avoid",
            f"Volume Ignored? {analysis.get('traps', {}).get('volume_ignored', 'No')}",
        ], "Market loves teaching fees", style="danger")
    with c2:
        ta_card("MODULE_15", "Post-Trade Review", [
            "□  Screenshot saved",
            "□  Mistake logged",
            "□  Setup reviewed",
            "□  Lesson noted",
            "□  Journal updated",
        ], "Journal or repeat pain", style="warn")

    # Final result
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    final = analysis.get("final_result", {})
    setup   = final.get("setup",   "PENDING")
    entry_q = final.get("entry",   "PENDING")
    risk_q  = final.get("risk",    "MANAGED")
    exit_q  = final.get("exit",    "PENDING")
    overall = final.get("overall", "NEUTRAL")

    color = "#00ff6a" if overall in ("BULLISH","LONG") else ("#ff3355" if overall in ("BEARISH","SHORT") else "#ffaa00")

    st.markdown(f"""
    <div style="background:#040f08;border:1px solid #007733;padding:24px;margin-top:8px;">
        <div style="font-family:'Orbitron',monospace;font-size:11px;color:#3a6648;
        letter-spacing:4px;margin-bottom:16px;">MODULE_16 // FINAL RESULT</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;">
            <div style="border:1px solid #0d3318;padding:12px;text-align:center;background:#020c06;">
                <div style="font-size:9px;color:#3a6648;letter-spacing:2px;margin-bottom:6px;">SETUP</div>
                <div style="font-family:'Orbitron',monospace;font-size:13px;color:#00ff6a;">{setup}</div>
            </div>
            <div style="border:1px solid #0d3318;padding:12px;text-align:center;background:#020c06;">
                <div style="font-size:9px;color:#3a6648;letter-spacing:2px;margin-bottom:6px;">ENTRY</div>
                <div style="font-family:'Orbitron',monospace;font-size:13px;color:#00ff6a;">{entry_q}</div>
            </div>
            <div style="border:1px solid #0d3318;padding:12px;text-align:center;background:#020c06;">
                <div style="font-size:9px;color:#3a6648;letter-spacing:2px;margin-bottom:6px;">RISK</div>
                <div style="font-family:'Orbitron',monospace;font-size:13px;color:#00ff6a;">{risk_q}</div>
            </div>
            <div style="border:1px solid #0d3318;padding:12px;text-align:center;background:#020c06;">
                <div style="font-size:9px;color:#3a6648;letter-spacing:2px;margin-bottom:6px;">EXIT</div>
                <div style="font-family:'Orbitron',monospace;font-size:13px;color:#00ff6a;">{exit_q}</div>
            </div>
        </div>
        <div style="text-align:center;padding:16px;border-top:1px solid #0d3318;">
            <div style="font-size:10px;color:#3a6648;letter-spacing:4px;margin-bottom:8px;">OVERALL BIAS</div>
            <div style="font-family:'Orbitron',monospace;font-size:36px;font-weight:900;
            color:{color};text-shadow:0 0 20px {color}77;letter-spacing:8px;">{overall}</div>
            <div style="font-size:10px;color:#3a6648;margin-top:8px;letter-spacing:3px;">
            Stop Loss Also Respected &nbsp;|&nbsp; Process Over Outcome</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    render_header()
    cfg = render_sidebar()

    # Initial state
    if "analysis" not in st.session_state:
        st.session_state.analysis = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "info" not in st.session_state:
        st.session_state.info = {}

    # Auto-load on first run
    if st.session_state.df is None or cfg["analyze"]:
        with st.spinner("LOADING DATA..."):
            interval = TF_MAP.get(cfg["timeframe"], "1d")
            df = fetch_data(cfg["symbol"], interval, cfg["period"])
            info = fetch_info(cfg["symbol"])

            if df.empty:
                st.error(f"⚠ No data found for symbol: {cfg['symbol']}")
                st.stop()

            # Run all modules
            indicators = compute_indicators(df)
            ms   = analyze_market_structure(df)
            tr   = analyze_trend(df, indicators)
            sr   = find_support_resistance(df)
            dz   = find_demand_supply_zones(df)
            cp   = detect_candlestick_patterns(df)
            bk   = detect_breakouts(df, sr)
            vol  = analyze_volume(df)
            chp  = detect_chart_patterns(df)
            mtf  = multi_timeframe_analysis(cfg["symbol"], cfg["period"])
            tp   = generate_trade_plan(df, ms, sr, tr)
            rm   = calculate_risk(df, tp, cfg["capital"], cfg["risk_pct"])

            # Merge SR & DZ into indicators for chart
            indicators["sr_levels"] = sr.get("levels", [])
            indicators["zones"]     = dz.get("zones", [])

            # Compose traps & final
            bias = tr.get("daily", "NEUTRAL")
            traps = {
                "fake_breakout": "Low Risk" if vol.get("ratio_raw", 1) > 1.2 else "Monitor",
                "late_entry":    "YES – Extended" if abs(df["Close"].iloc[-1] - tp.get("entry", df["Close"].iloc[-1])) / df["Close"].iloc[-1] > 0.03 else "NO",
                "volume_ignored": "NO" if vol.get("confirmed", False) else "YES – Low Vol",
            }
            final = {
                "setup":   "CLEAN" if chp.get("pattern", "None") != "None" or bk.get("confirmed") else "PENDING",
                "entry":   "PATIENT" if not traps["late_entry"].startswith("YES") else "EXTENDED",
                "risk":    "MANAGED",
                "exit":    "PLAN SET",
                "overall": bias.upper() if bias != "N/A" else "NEUTRAL",
            }

            st.session_state.df   = df
            st.session_state.info = info
            st.session_state.analysis = {
                "market_structure": {**ms, **sr.get("summary", {}), **dz.get("summary", {})},
                "trend":            tr,
                "candlestick_patterns": cp,
                "breakouts":        bk,
                "volume":           vol,
                "chart_patterns":   chp,
                "mtf":              mtf,
                "trade_plan":       tp,
                "risk":             rm,
                "traps":            traps,
                "final_result":     final,
            }
            st.session_state.indicators = indicators

    df         = st.session_state.df
    analysis   = st.session_state.analysis
    info       = st.session_state.info
    indicators = st.session_state.indicators

    # ── Symbol label
    name = info.get("shortName", cfg["symbol"])
    exchange = info.get("exchange", "")
    st.markdown(f"""
    <div style="font-family:'Orbitron',monospace;font-size:16px;font-weight:700;
    color:#00ff6a;letter-spacing:4px;margin-bottom:12px;">
    {cfg['symbol']}
    <span style="font-size:11px;color:#3a6648;letter-spacing:2px;"> {name} &nbsp;|&nbsp; {exchange}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs
    tabs = st.tabs([
        "📊  OVERVIEW",
        "🏗  STRUCTURE",
        "🕯  PATTERNS",
        "🔭  MULTI-TF",
        "⚡  TRADE PLAN",
        "📋  REVIEW",
    ])

    with tabs[0]: render_overview_tab(df, info, indicators, cfg)
    with tabs[1]: render_structure_tab(df, analysis)
    with tabs[2]: render_patterns_tab(df, analysis)
    with tabs[3]: render_mtf_tab(df, analysis, cfg["symbol"], cfg["period"])
    with tabs[4]: render_trade_tab(df, analysis, cfg)
    with tabs[5]: render_review_tab(analysis)


if __name__ == "__main__":
    main()
