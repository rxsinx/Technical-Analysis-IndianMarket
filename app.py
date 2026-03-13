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
from modules.fundamental import get_fundamentals

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
    padding: 10px 10px 8px 10px;
    margin-bottom: 4px;
    overflow: hidden;
}
.metric-label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 9px !important;
    color: #3a6648 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
    display: block !important;
}
.metric-prefix {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 9px !important;
    color: #007733 !important;
    letter-spacing: 1px !important;
    display: block !important;
    margin-bottom: 1px !important;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 15px !important;
    font-weight: bold !important;
    color: #00ff6a !important;
    letter-spacing: 0.5px !important;
    margin-bottom: 3px !important;
    display: block !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 100% !important;
}
.metric-delta {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 10px !important;
    color: #ffaa00 !important;
    letter-spacing: 0.5px !important;
    display: block !important;
}
.metric-delta.positive { color: #00ff6a !important; }
.metric-delta.negative { color: #ff3355 !important; }

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
    st.sidebar.markdown("<div style='font-size:10px;color:#3a6648;letter-spacing:2px;margin-top:12px;'>CAPITAL (₹)</div>", unsafe_allow_html=True)
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

    def _m(lbl, prefix, val, delta=None, dcls=""):
        d_html = f'<span class="metric-delta {dcls}">{delta}</span>' if delta else ""
        p_html = f'<span class="metric-prefix">{prefix}</span>' if prefix else ""
        st.markdown(
            f'<div class="metric-card">'
            f'<span class="metric-label">{lbl}</span>'
            f'{p_html}'
            f'<span class="metric-value">{val}</span>'
            f'{d_html}'
            f'</div>',
            unsafe_allow_html=True
        )

    low52         = df['Low'].tail(252).min()
    high52        = df['High'].tail(252).max()
    pct_from_low  = (last - low52)  / low52  * 100
    pct_from_high = (last - high52) / high52 * 100   # always negative

    # CHANGE card: value color follows sign
    chg_color = "#00ff6a" if chg_pct >= 0 else "#ff3355"

    cols = st.columns(6)
    with cols[0]: _m("LAST PRICE", "INR", f"{last:,.2f}",   f"{chg:+.2f}", "positive" if chg>=0 else "negative")
    with cols[1]:
        st.markdown(
            f'<div class="metric-card">'
            f'<span class="metric-label">CHANGE</span>'
            f'<span class="metric-value" style="color:{chg_color} !important;">{chg_pct:+.2f}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    with cols[2]: _m("HIGH (D)",   "INR", f"{df['High'].iloc[-1]:,.2f}")
    with cols[3]: _m("LOW (D)",    "INR", f"{df['Low'].iloc[-1]:,.2f}")
    with cols[4]: _m("VOLUME",     "",    f"{vol/1e6:.2f}M", f"avg {avg_vol/1e6:.2f}M")
    with cols[5]:
        st.markdown(
            f'<div class="metric-card">'
            f'<span class="metric-label">52W RANGE</span>'
            f'<span class="metric-prefix">INR</span>'
            f'<span class="metric-value">{low52:,.0f} – {high52:,.0f}</span>'
            f'<span class="metric-delta positive">▲ {pct_from_low:.1f}% from 52W Low</span>'
            f'<span class="metric-delta negative">▼ {abs(pct_from_high):.1f}% from 52W High</span>'
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
            f"Major S:   ₹{ms.get('major_support', 0):.2f}",
            f"Major R:   ₹{ms.get('major_resistance', 0):.2f}",
            f"Nearest S: ₹{ms.get('nearest_support', 0):.2f}",
            f"Nearest R: ₹{ms.get('nearest_resistance', 0):.2f}",
        ], "Levels matter", style="warn")
    with c4:
        ta_card("MODULE_05", "Demand & Supply", [
            f"Demand:  ₹{ms.get('demand_low', 0):.2f} – ₹{ms.get('demand_high', 0):.2f}",
            f"Supply:  ₹{ms.get('supply_low', 0):.2f} – ₹{ms.get('supply_high', 0):.2f}",
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
    dvb = analysis.get("darvas_box", {})

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
            f"Level:        ₹{bk.get('level', 0):.2f}",
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
            f"Target:       ₹{chp.get('target', 0):.2f}",
            f"Invalidation: ₹{chp.get('invalidation', 0):.2f}",
        ], "Pattern with context only")

    # Darvas Box
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:11px;color:#00ffcc;letter-spacing:3px;margin-bottom:12px;">MODULE 09B &nbsp;|&nbsp; DARVAS BOX (ADVANCED)</div>', unsafe_allow_html=True)
    _render_darvas_box(dvb)

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
            f"Entry Zone:  ₹{entry:.2f}",
            f"Stop Loss:   ₹{sl:.2f}  ({((sl-entry)/entry*100):+.1f}%)",
            f"Target 1:    ₹{t1:.2f}  ({((t1-entry)/entry*100):+.1f}%)",
            f"Target 2:    ₹{t2:.2f}  ({((t2-entry)/entry*100):+.1f}%)",
            f"R:R Ratio:   1 : {rr:.1f}",
        ], "Plan before price", style="cyan")
    with c2:
        ta_card("MODULE_12", "Risk Management", [
            f"Capital:     ₹{cfg['capital']:,.0f}",
            f"Risk %:      {cfg['risk_pct']}%",
            f"Risk ₹:      ₹{rm.get('risk_amount', 0):,.0f}",
            f"Position Sz: {rm.get('position_size', 0):.0f} qty",
            f"Notional:    ₹{rm.get('notional', 0):,.0f}",
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
            annotation_text=f"{label} ₹{level:.2f}",
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



def _render_darvas_box(dvb: dict):
    """Render Darvas Box analysis with table."""
    import pandas as _pd
    if not dvb or dvb.get("box_top", 0) == 0:
        st.markdown("<div style='color:#3a6648;font-size:11px;'>No Darvas Box detected in current data.</div>", unsafe_allow_html=True)
        return

    box_top    = dvb.get("box_top", 0)
    box_bottom = dvb.get("box_bottom", 0)
    width      = dvb.get("width", 0)
    width_pct  = dvb.get("width_pct", 0)
    entry      = dvb.get("entry", 0)
    sl         = dvb.get("stop_loss", 0)
    target     = dvb.get("target", 0)
    rr         = dvb.get("rr", 0)
    status     = dvb.get("status", "N/A")
    brk        = dvb.get("breakout", False)
    brk_price  = dvb.get("breakout_price")
    brk_date   = dvb.get("breakout_date", "—")
    total      = dvb.get("total_boxes", 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        ta_card("DARVAS // BOX", "Box Boundaries", [
            f"Box Top:     INR{box_top:,.2f}",
            f"Box Bottom:  INR{box_bottom:,.2f}",
            f"Width:       INR{width:,.2f}  ({width_pct:.1f}%)",
            f"Boxes Found: {total}",
        ], "Box = institutional accumulation zone", style="cyan")
    with c2:
        ta_card("DARVAS // TRADE", "Entry & Exits", [
            f"Entry:       INR{entry:,.2f}  (above box top)",
            f"Stop Loss:   INR{sl:,.2f}  (below box bottom)",
            f"Target:      INR{target:,.2f}  (1x width proj.)",
            f"R:R Ratio:   1:{rr}",
        ], "Buy strength, not weakness", style="warn")
    with c3:
        brk_text = f"INR{brk_price:,.2f} on {brk_date}" if brk and brk_price else "Not yet triggered"
        ta_card("DARVAS // STATUS", "Breakout Monitor", [
            f"Status:      {status}",
            f"Breakout At: {brk_text}",
            f"R:R:         1:{rr}",
        ], "Breakout + Volume = Edge")

    history = dvb.get("history", [])
    if history:
        st.markdown("<div style='font-size:9px;color:#3a6648;letter-spacing:2px;margin:8px 0 4px;'>RECENT BOX HISTORY</div>", unsafe_allow_html=True)
        hist_rows = []
        for b in reversed(history):
            hist_rows.append({
                "Top Date":   b.get("top_date", "—"),
                "Box Top":    f"INR{b['box_top']:,.2f}",
                "Box Bottom": f"INR{b['box_bottom']:,.2f}",
                "Width %":    f"{b['width_pct']}%",
                "Entry":      f"INR{b['entry']:,.2f}",
                "Stop Loss":  f"INR{b['stop_loss']:,.2f}",
                "Target":     f"INR{b['target']:,.2f}",
                "Breakout":   "YES" if b['breakout'] else "—",
                "Brk Price":  f"INR{b['breakout_price']:,.2f}" if b['breakout_price'] else "—",
            })
        st.dataframe(_pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)


def render_signal_tab(analysis: dict):
    """Trading Signal & Analysis panel."""
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:13px;color:#00ff6a;
    letter-spacing:3px;margin-bottom:16px;">MODULE 17 &nbsp;|&nbsp; TRADING SIGNAL & ANALYSIS</div>
    """, unsafe_allow_html=True)

    sig = analysis.get("trading_signal", {})
    if not sig:
        st.markdown("<div style='color:#3a6648;'>Signal data unavailable.</div>", unsafe_allow_html=True)
        return

    score      = sig.get("score", 0)
    label      = sig.get("label", "N/A")
    sig_color  = sig.get("color", "#ffaa00")
    signals    = sig.get("signals", [])
    bull_count = sig.get("bull_count", 0)
    bear_count = sig.get("bear_count", 0)
    neu_count  = sig.get("neutral_count", 0)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        # Signal label
        st.markdown(f"""
        <div style="background:#040f08;border:1px solid {sig_color};
        border-top:3px solid {sig_color};padding:16px;text-align:center;margin-bottom:10px;">
            <div style="font-size:9px;color:#3a6648;letter-spacing:3px;margin-bottom:8px;">COMPOSITE SIGNAL</div>
            <div style="font-family:'Orbitron',monospace;font-size:26px;font-weight:900;
            color:{sig_color};text-shadow:0 0 16px {sig_color}88;letter-spacing:4px;">{label}</div>
            <div style="font-size:12px;color:#3a6648;margin-top:8px;letter-spacing:1px;">
            Score: {score:+.1f} / 100</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(font=dict(family="Orbitron", size=32, color=sig_color), suffix=""),
            title=dict(text="Signal Strength", font=dict(family="Share Tech Mono", size=11, color="#3a6648")),
            gauge=dict(
                axis=dict(
                    range=[-100, 100],
                    tickvals=[-100, -60, -25, 0, 25, 60, 100],
                    ticktext=["-100", "-60", "-25", "0", "+25", "+60", "+100"],
                    tickfont=dict(size=8, color="#3a6648"),
                    tickcolor="#0d3318",
                ),
                bar=dict(color=sig_color, thickness=0.2),
                bgcolor="#040f08",
                borderwidth=1,
                bordercolor="#0d3318",
                steps=[
                    dict(range=[-100, -60], color="#330011"),
                    dict(range=[-60,  -25], color="#220008"),
                    dict(range=[-25,   10], color="#151005"),
                    dict(range=[ 10,   25], color="#001a0a"),
                    dict(range=[ 25,   60], color="#003311"),
                    dict(range=[ 60,  100], color="#004d1a"),
                ],
                threshold=dict(
                    line=dict(color=sig_color, width=3),
                    thickness=0.8,
                    value=score,
                ),
            ),
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#020c06",
            plot_bgcolor="#020c06",
            font=dict(family="Share Tech Mono", color="#a0ffc0"),
            height=230,
            margin=dict(l=10, r=10, t=30, b=0),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Bull / Bear / Neutral counts
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-top:6px;">
            <div style="background:#002211;border:1px solid #007733;padding:8px;text-align:center;">
                <div style="font-size:8px;color:#3a6648;letter-spacing:1px;margin-bottom:2px;">BULL</div>
                <div style="font-family:'Orbitron',monospace;font-size:20px;color:#00ff6a;">{bull_count}</div>
            </div>
            <div style="background:#1a0008;border:1px solid #ff3355;padding:8px;text-align:center;">
                <div style="font-size:8px;color:#3a6648;letter-spacing:1px;margin-bottom:2px;">BEAR</div>
                <div style="font-family:'Orbitron',monospace;font-size:20px;color:#ff3355;">{bear_count}</div>
            </div>
            <div style="background:#1a1000;border:1px solid #ffaa00;padding:8px;text-align:center;">
                <div style="font-size:8px;color:#3a6648;letter-spacing:1px;margin-bottom:2px;">NEUT</div>
                <div style="font-family:'Orbitron',monospace;font-size:20px;color:#ffaa00;">{neu_count}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Detailed signals
        type_icon  = {"bull": "▲", "bear": "▼", "warn": "⚠", "neutral": "●"}
        type_color = {"bull": "#00ff6a", "bear": "#ff3355", "warn": "#ffaa00", "neutral": "#3a6648"}
        type_bg    = {"bull": "#001a0a", "bear": "#1a0008", "warn": "#1a1000", "neutral": "#040f08"}

        items_html = ""
        for s in signals:
            t     = s.get("type", "neutral")
            icon  = type_icon.get(t, "●")
            color = type_color.get(t, "#3a6648")
            bg    = type_bg.get(t, "#040f08")
            items_html += (
                f'<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;'
                f'background:{bg};border-left:3px solid {color};margin-bottom:3px;font-size:11px;letter-spacing:0.5px;">'
                f'<span style="color:{color};font-size:14px;min-width:16px;">{icon}</span>'
                f'<span style="color:#a0ffc0;">{s["text"]}</span>'
                f'</div>'
            )

        st.markdown(f"""
        <div style="background:#040f08;border:1px solid #0d3318;border-top:2px solid #00ffcc;padding:14px;">
            <div style="font-family:'Orbitron',monospace;font-size:10px;color:#00ffcc;
            letter-spacing:3px;margin-bottom:12px;">▶ DETAILED ANALYSIS SIGNALS</div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

    # Signal bar
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:9px;color:#3a6648;letter-spacing:2px;margin-bottom:6px;'>SIGNAL BREAKDOWN</div>", unsafe_allow_html=True)

    fig_bar = go.Figure(go.Bar(
        x=["Bullish Signals", "Bearish Signals", "Neutral / Watch"],
        y=[bull_count, bear_count, neu_count],
        marker_color=["#00ff6a", "#ff3355", "#ffaa00"],
        text=[bull_count, bear_count, neu_count],
        textposition="outside",
        textfont=dict(color="#a0ffc0", size=12, family="Orbitron"),
        width=0.4,
    ))
    fig_bar.update_layout(
        paper_bgcolor="#020c06", plot_bgcolor="#040f08",
        font=dict(family="Share Tech Mono", color="#a0ffc0", size=10),
        height=200, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor="#0d3318", showgrid=False),
        yaxis=dict(gridcolor="#0d3318", showgrid=True, zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


def render_fundamental_tab(analysis: dict):
    """Complete Financial Analysis panel — MarketSmith-style in terminal aesthetic."""
    st.markdown("""
    <div style="font-family:'Orbitron',monospace;font-size:13px;color:#00ff6a;
    letter-spacing:3px;margin-bottom:16px;">MODULE 18 &nbsp;|&nbsp; COMPLETE FINANCIAL ANALYSIS</div>
    """, unsafe_allow_html=True)

    fund = analysis.get("fundamentals", {})
    if not fund or "error" in fund:
        st.markdown(f"<div style='color:#ff3355;'>Data unavailable: {fund.get('error','')}</div>",
                    unsafe_allow_html=True)
        return

    scores    = fund.get("scores",     {})
    ratios    = fund.get("ratios",     {})
    quarterly = fund.get("quarterly",  [])
    annual    = fund.get("annual",     [])
    fins      = fund.get("financials", {})
    canslim   = fund.get("canslim",    [])
    info      = fund.get("info",       {})

    # ── SECTION 1: Master Score Row ──────────────────────────────────────────
    _score_row(scores)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── SECTION 2: Price Performance vs Key Ratios ───────────────────────────
    col_l, col_r = st.columns([1, 1])
    with col_l:
        _price_performance(scores, info)
    with col_r:
        _valuation_ratios(ratios)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── SECTION 3: Quarterly Earnings + Annual EPS/Price ─────────────────────
    col_q, col_a = st.columns([3, 2])
    with col_q:
        _quarterly_table(quarterly)
    with col_a:
        _annual_table(annual)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── SECTION 4: Financials Bar Charts ─────────────────────────────────────
    _financials_charts(fins)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── SECTION 5: Key Metrics Grid ──────────────────────────────────────────
    _metrics_grid(ratios)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── SECTION 6: CAN SLIM Checklist ────────────────────────────────────────
    _canslim_panel(canslim)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _score_row(scores: dict):
    """Top row: Master Score + sub-ratings."""
    master  = scores.get("master",    0)
    eps_r   = scores.get("eps_rating", 0)
    rs_r    = scores.get("rs_rating",  0)
    acc_dis = scores.get("acc_dis",    "C")
    gr      = scores.get("group_rank", "N/A")

    def score_color(v):
        if isinstance(v, int):
            if v >= 80: return "#00ff6a"
            if v >= 60: return "#ffaa00"
            return "#ff3355"
        return "#ffaa00"

    acc_color = {"A":"#00ff6a","B":"#00cc55","C":"#ffaa00","D":"#ff8844","E":"#ff3355"}.get(acc_dis,"#ffaa00")

    items = [
        ("MASTER SCORE",       str(master),  score_color(master),  "Composite rating"),
        ("EPS RATING",         str(eps_r),   score_color(eps_r),   "Earnings power"),
        ("PRICE STRENGTH",     str(rs_r),    score_color(rs_r),    "Relative strength"),
        ("ACC/DIS RATING",     acc_dis,      acc_color,            "Institutional flow"),
        ("GROUP RANK",         str(gr),      "#3a6648",            "Sector rank (proxy)"),
        ("EPS GROWTH",         f"{scores.get('eps_growth_rate',0):+.1f}%",
                               "#00ff6a" if scores.get('eps_growth_rate',0)>0 else "#ff3355", "TTM EPS growth"),
    ]

    cols = st.columns(len(items))
    for col, (lbl, val, color, note) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div style="background:#040f08;border:1px solid #0d3318;border-top:3px solid {color};
            padding:12px 10px;text-align:center;">
                <div style="font-size:8px;color:#3a6648;letter-spacing:2px;margin-bottom:6px;">{lbl}</div>
                <div style="font-family:'Orbitron',monospace;font-size:26px;font-weight:900;
                color:{color};text-shadow:0 0 12px {color}66;">{val}</div>
                <div style="font-size:9px;color:#3a6648;margin-top:4px;">{note}</div>
            </div>
            """, unsafe_allow_html=True)


def _price_performance(scores: dict, info: dict):
    """Price performance across timeframes."""
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:10px;color:#00ffcc;letter-spacing:3px;margin-bottom:10px;">PRICE PERFORMANCE</div>', unsafe_allow_html=True)

    rows = [
        ("4-Week Return",  scores.get("ret_4w")),
        ("13-Week Return", scores.get("ret_13w")),
        ("26-Week Return", scores.get("ret_26w")),
        ("52-Week Return", scores.get("ret_52w")),
    ]

    html = '<div style="font-family:\'Share Tech Mono\',monospace;">'
    for label, val in rows:
        if val is None:
            continue
        color  = "#00ff6a" if val >= 0 else "#ff3355"
        arrow  = "▲" if val >= 0 else "▼"
        bar_w  = min(100, abs(val) * 1.5)
        bar_c  = "#00ff6a" if val >= 0 else "#ff3355"
        html += f"""
        <div style="display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #0a1a0e;">
            <div style="width:120px;font-size:10px;color:#3a6648;">{label}</div>
            <div style="flex:1;background:#0d3318;height:6px;border-radius:0;">
                <div style="width:{bar_w}%;background:{bar_c};height:100%;"></div>
            </div>
            <div style="width:70px;text-align:right;font-size:11px;color:{color};">
                {arrow} {abs(val):.1f}%
            </div>
        </div>"""
    html += '</div>'

    # Also show 52W Hi/Lo distance
    hi52 = info.get("fiftyTwoWeekHigh")
    lo52 = info.get("fiftyTwoWeekLow")
    curr = info.get("currentPrice") or info.get("regularMarketPrice")
    if hi52 and lo52 and curr:
        pos_pct = (curr - lo52) / (hi52 - lo52) * 100 if hi52 != lo52 else 50
        html += f"""
        <div style="margin-top:12px;padding:10px;background:#040f08;border:1px solid #0d3318;">
            <div style="font-size:9px;color:#3a6648;letter-spacing:2px;margin-bottom:6px;">52W POSITION</div>
            <div style="display:flex;justify-content:space-between;font-size:10px;color:#3a6648;margin-bottom:4px;">
                <span>LOW ₹{lo52:,.0f}</span><span>HIGH ₹{hi52:,.0f}</span>
            </div>
            <div style="background:#0d3318;height:8px;position:relative;">
                <div style="width:{pos_pct:.0f}%;background:#00ff6a;height:100%;"></div>
                <div style="position:absolute;left:{pos_pct:.0f}%;top:-4px;transform:translateX(-50%);
                color:#00ff6a;font-size:14px;">▼</div>
            </div>
            <div style="text-align:center;font-size:10px;color:#00ff6a;margin-top:8px;">
                {pos_pct:.1f}% of 52W range &nbsp;·&nbsp; CMP ₹{curr:,.2f}
            </div>
        </div>"""

    st.markdown(html, unsafe_allow_html=True)


def _valuation_ratios(ratios: dict):
    """Valuation ratios grid."""
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:10px;color:#00ffcc;letter-spacing:3px;margin-bottom:10px;">VALUATION & QUALITY</div>', unsafe_allow_html=True)

    items = [
        ("Market Cap",      ratios.get("market_cap", "N/A")),
        ("P/E (TTM)",       ratios.get("pe",         "N/A")),
        ("P/E (Forward)",   ratios.get("fwd_pe",     "N/A")),
        ("P/B Ratio",       ratios.get("pb",         "N/A")),
        ("P/S Ratio",       ratios.get("ps",         "N/A")),
        ("EV/EBITDA",       ratios.get("ev_ebitda",  "N/A")),
        ("EPS (TTM)",       f"₹{ratios.get('eps_ttm','N/A')}"),
        ("EPS (Fwd)",       f"₹{ratios.get('eps_fwd','N/A')}"),
        ("Book Value",      f"₹{ratios.get('book_value','N/A')}"),
        ("ROE",             ratios.get("roe",        "N/A")),
        ("ROA",             ratios.get("roa",        "N/A")),
        ("Net Margin",      ratios.get("net_margin", "N/A")),
        ("Op Margin",       ratios.get("op_margin",  "N/A")),
        ("Gross Margin",    ratios.get("gross_margin","N/A")),
        ("Rev Growth",      ratios.get("rev_growth", "N/A")),
        ("Earn Growth",     ratios.get("earn_growth","N/A")),
    ]

    html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:3px;font-family:\'Share Tech Mono\',monospace;">'
    for lbl, val in items:
        color = "#a0ffc0"
        if isinstance(val, str) and val.endswith("%"):
            try:
                v = float(val.replace("%",""))
                color = "#00ff6a" if v > 0 else "#ff3355"
            except Exception:
                pass
        html += f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
        padding:4px 8px;background:#040f08;border-bottom:1px solid #0a1a0e;">
            <span style="font-size:9px;color:#3a6648;">{lbl}</span>
            <span style="font-size:10px;color:{color};font-weight:bold;">{val}</span>
        </div>"""
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _quarterly_table(quarterly: list):
    """Quarterly EPS + Revenue table."""
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:10px;color:#ffaa00;letter-spacing:3px;margin-bottom:10px;">QUARTERLY EARNINGS</div>', unsafe_allow_html=True)

    if not quarterly:
        st.markdown("<div style='color:#3a6648;font-size:11px;'>No quarterly data available.</div>", unsafe_allow_html=True)
        return

    hdr = '<div style="display:grid;grid-template-columns:80px 60px 55px 80px 55px;gap:2px;font-family:\'Orbitron\',monospace;font-size:8px;color:#3a6648;letter-spacing:1px;padding:4px 6px;background:#061510;margin-bottom:2px;">'
    hdr += '<div>QUARTER</div><div style="text-align:right;">EPS (₹)</div><div style="text-align:right;">CHG%</div><div style="text-align:right;">REV (Cr)</div><div style="text-align:right;">CHG%</div></div>'

    rows_html = ""
    for i, q in enumerate(quarterly):
        bg = "#040f08" if i % 2 == 0 else "#030b06"
        eps_str  = f"₹{q['eps']:,.2f}"    if q['eps']    is not None else "—"
        rev_str  = f"₹{q['rev_cr']:,.1f}" if q['rev_cr'] is not None else "—"

        def chg_html(v):
            if v is None: return '<div style="text-align:right;color:#3a6648;">—</div>'
            c = "#00ff6a" if v >= 0 else "#ff3355"
            arrow = "▲" if v >= 0 else "▼"
            return f'<div style="text-align:right;color:{c};">{arrow}{abs(v):.1f}%</div>'

        rows_html += f"""
        <div style="display:grid;grid-template-columns:80px 60px 55px 80px 55px;gap:2px;
        font-family:'Share Tech Mono',monospace;font-size:10px;color:#a0ffc0;
        padding:5px 6px;background:{bg};border-bottom:1px solid #0a1a0e;">
            <div style="color:#3a6648;">{q['date']}</div>
            <div style="text-align:right;">{eps_str}</div>
            {chg_html(q['eps_chg'])}
            <div style="text-align:right;">{rev_str}</div>
            {chg_html(q['rev_chg'])}
        </div>"""

    st.markdown(hdr + rows_html, unsafe_allow_html=True)


def _annual_table(annual: list):
    """Annual EPS with Hi/Lo price."""
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:10px;color:#ffaa00;letter-spacing:3px;margin-bottom:10px;">ANNUAL EPS & PRICE</div>', unsafe_allow_html=True)

    if not annual:
        st.markdown("<div style='color:#3a6648;font-size:11px;'>No annual data available.</div>", unsafe_allow_html=True)
        return

    hdr = '<div style="display:grid;grid-template-columns:50px 70px 75px 75px;gap:2px;font-family:\'Orbitron\',monospace;font-size:8px;color:#3a6648;letter-spacing:1px;padding:4px 6px;background:#061510;margin-bottom:2px;">'
    hdr += '<div>YEAR</div><div style="text-align:right;">EPS (₹)</div><div style="text-align:right;">HIGH</div><div style="text-align:right;">LOW</div></div>'

    rows_html = ""
    for i, a in enumerate(annual):
        bg      = "#040f08" if i % 2 == 0 else "#030b06"
        eps_str = f"₹{a['eps']:,.2f}" if a['eps'] is not None else "—"
        hi_str  = f"₹{a['hi']:,.0f}"  if a['hi']  is not None else "—"
        lo_str  = f"₹{a['lo']:,.0f}"  if a['lo']  is not None else "—"
        rows_html += f"""
        <div style="display:grid;grid-template-columns:50px 70px 75px 75px;gap:2px;
        font-family:'Share Tech Mono',monospace;font-size:10px;color:#a0ffc0;
        padding:5px 6px;background:{bg};border-bottom:1px solid #0a1a0e;">
            <div style="color:#3a6648;">{a['year']}</div>
            <div style="text-align:right;color:#00ff6a;">{eps_str}</div>
            <div style="text-align:right;color:#ffaa00;">{hi_str}</div>
            <div style="text-align:right;color:#ff3355;">{lo_str}</div>
        </div>"""

    st.markdown(hdr + rows_html, unsafe_allow_html=True)


def _financials_charts(fins: dict):
    """4-year bar charts: Revenue, Net Income, Operating CF, Free CF."""
    if not fins or "error" in fins:
        st.markdown("<div style='color:#3a6648;font-size:11px;'>Financial chart data unavailable.</div>", unsafe_allow_html=True)
        return

    years  = fins.get("years", ["Y1","Y2","Y3","Y4"])
    rev    = fins.get("revenue",      [])
    ni     = fins.get("net_income",   [])
    ocf    = fins.get("op_cashflow",  [])
    fcf    = fins.get("free_cashflow",[])

    def clean(lst):
        return [v if isinstance(v, (int, float)) else 0 for v in lst]

    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:10px;color:#00ffcc;letter-spacing:3px;margin-bottom:10px;">FINANCIAL TRENDS (₹ Cr)</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    chart_defs = [
        (c1, "Revenue",         rev,  "#00ff6a"),
        (c2, "Net Income",      ni,   "#00ffcc"),
        (c3, "Operating CF",    ocf,  "#ffaa00"),
        (c4, "Free Cash Flow",  fcf,  "#ff3355"),
    ]

    for col, title, data, color in chart_defs:
        with col:
            d = clean(data)[:4]
            yr = years[:len(d)]
            if not any(v != 0 for v in d):
                st.markdown(f"<div style='font-size:9px;color:#3a6648;text-align:center;padding:20px;'>{title}<br>N/A</div>", unsafe_allow_html=True)
                continue
            bar_colors = [color if v >= 0 else "#ff3355" for v in d]
            fig = go.Figure(go.Bar(
                x=yr, y=d,
                marker_color=bar_colors,
                text=[f"₹{v:,.0f}" for v in d],
                textposition="outside",
                textfont=dict(size=8, color="#a0ffc0"),
                width=0.6,
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(family="Orbitron", size=9, color="#3a6648"), x=0.5),
                paper_bgcolor="#020c06", plot_bgcolor="#040f08",
                font=dict(family="Share Tech Mono", color="#a0ffc0", size=9),
                height=180, margin=dict(l=0,r=0,t=28,b=0),
                xaxis=dict(gridcolor="#0d3318", showgrid=False, tickfont=dict(size=8)),
                yaxis=dict(gridcolor="#0d3318", showgrid=True, zeroline=True,
                           zerolinecolor="#3a6648", tickfont=dict(size=8)),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


def _metrics_grid(ratios: dict):
    """Bottom grid: shareholding + debt + cashflow metrics."""
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:10px;color:#00ffcc;letter-spacing:3px;margin-bottom:10px;">SHAREHOLDING & CAPITAL STRUCTURE</div>', unsafe_allow_html=True)

    sections = {
        "SHAREHOLDING": [
            ("Promoter Holding",   ratios.get("prom_hold","N/A")),
            ("Institutional (FII/DII)", ratios.get("inst_hold","N/A")),
            ("Shares Outstanding", ratios.get("shares_out","N/A")),
            ("Float Shares",       ratios.get("float_shares","N/A")),
            ("Dividend Yield",     ratios.get("div_yield","N/A")),
            ("Payout Ratio",       ratios.get("payout_ratio","N/A")),
        ],
        "BALANCE SHEET": [
            ("Debt / Equity",      ratios.get("debt_equity","N/A")),
            ("Current Ratio",      ratios.get("current_ratio","N/A")),
            ("Quick Ratio",        ratios.get("quick_ratio","N/A")),
            ("Enterprise Value",   ratios.get("ev","N/A")),
            ("Book Value / Share", f"₹{ratios.get('book_value','N/A')}"),
            ("Beta",               ratios.get("beta","N/A")),
        ],
        "CASHFLOW & MARGIN": [
            ("Operating CF",       ratios.get("op_cashflow","N/A")),
            ("Free Cash Flow",     ratios.get("free_cashflow","N/A")),
            ("Gross Margin",       ratios.get("gross_margin","N/A")),
            ("Operating Margin",   ratios.get("op_margin","N/A")),
            ("Net Margin",         ratios.get("net_margin","N/A")),
            ("Revenue",            ratios.get("revenue","N/A")),
        ],
    }

    cols = st.columns(3)
    for col, (sec_title, items) in zip(cols, sections.items()):
        with col:
            html = f'<div style="background:#040f08;border:1px solid #0d3318;border-top:2px solid #007733;padding:10px;">'
            html += f'<div style="font-family:\'Orbitron\',monospace;font-size:9px;color:#00ff6a;letter-spacing:2px;margin-bottom:8px;">{sec_title}</div>'
            for lbl, val in items:
                color = "#a0ffc0"
                if isinstance(val, str) and val.endswith("%"):
                    try:
                        v = float(val.replace("%",""))
                        color = "#00ff6a" if v > 15 else ("#ffaa00" if v > 5 else "#a0ffc0")
                    except Exception: pass
                html += f"""
                <div style="display:flex;justify-content:space-between;
                padding:4px 0;border-bottom:1px solid #0a1a0e;font-family:'Share Tech Mono',monospace;">
                    <span style="font-size:9px;color:#3a6648;">{lbl}</span>
                    <span style="font-size:10px;color:{color};">{val}</span>
                </div>"""
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)


def _canslim_panel(canslim: list):
    """CAN SLIM checklist."""
    st.markdown('<div style="font-family:\'Orbitron\',monospace;font-size:10px;color:#ffaa00;letter-spacing:3px;margin-bottom:10px;">CAN SLIM CHECKLIST</div>', unsafe_allow_html=True)

    pass_count = sum(1 for c in canslim if c.get("pass") is True)
    fail_count = sum(1 for c in canslim if c.get("pass") is False)
    total      = len([c for c in canslim if c.get("pass") is not None])

    score_pct = pass_count / total * 100 if total > 0 else 0
    score_color = "#00ff6a" if score_pct >= 70 else ("#ffaa00" if score_pct >= 40 else "#ff3355")

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:16px;padding:10px 14px;
    background:#040f08;border:1px solid #0d3318;margin-bottom:10px;">
        <div style="font-family:'Orbitron',monospace;font-size:22px;font-weight:900;
        color:{score_color};">{pass_count}/{total}</div>
        <div>
            <div style="font-size:9px;color:#3a6648;letter-spacing:2px;">CHECKS PASSED</div>
            <div style="font-size:10px;color:{score_color};">{score_pct:.0f}% — {'STRONG' if score_pct>=70 else ('MODERATE' if score_pct>=40 else 'WEAK')}</div>
        </div>
        <div style="flex:1;background:#0d3318;height:6px;margin-left:16px;">
            <div style="width:{score_pct:.0f}%;background:{score_color};height:100%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for i, item in enumerate(canslim):
        with cols[i % 2]:
            p = item.get("pass")
            if p is True:
                icon, color, bg = "✓", "#00ff6a", "#001a0a"
            elif p is False:
                icon, color, bg = "✗", "#ff3355", "#1a0008"
            else:
                icon, color, bg = "◎", "#ffaa00", "#1a1000"
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:10px;padding:7px 10px;
            background:{bg};border-left:3px solid {color};margin-bottom:3px;
            font-family:'Share Tech Mono',monospace;">
                <span style="color:{color};font-size:14px;min-width:16px;margin-top:1px;">{icon}</span>
                <div>
                    <div style="font-size:10px;color:#a0ffc0;">{item['label']}</div>
                    <div style="font-size:11px;color:{color};font-weight:bold;">{item['value']}</div>
                    <div style="font-size:9px;color:#3a6648;">{item.get('note','')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

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

            dvb  = detect_darvas_boxes(df)
            sig  = compute_trading_signal(df, indicators)
            fund = get_fundamentals(cfg["symbol"], df)

            st.session_state.df   = df
            st.session_state.info = info
            st.session_state.analysis = {
                "market_structure": {**ms, **sr.get("summary", {}), **dz.get("summary", {})},
                "trend":            tr,
                "candlestick_patterns": cp,
                "breakouts":        bk,
                "volume":           vol,
                "chart_patterns":   chp,
                "darvas_box":       dvb,
                "mtf":              mtf,
                "trade_plan":       tp,
                "risk":             rm,
                "traps":            traps,
                "final_result":     final,
                "trading_signal":   sig,
                "fundamentals":     fund,
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
        "📡  SIGNAL",
        "🏦  FUNDAMENTALS",
        "📋  REVIEW",
    ])

    with tabs[0]: render_overview_tab(df, info, indicators, cfg)
    with tabs[1]: render_structure_tab(df, analysis)
    with tabs[2]: render_patterns_tab(df, analysis)
    with tabs[3]: render_mtf_tab(df, analysis, cfg["symbol"], cfg["period"])
    with tabs[4]: render_trade_tab(df, analysis, cfg)
    with tabs[5]: render_signal_tab(analysis)
    with tabs[6]: render_fundamental_tab(analysis)
    with tabs[7]: render_review_tab(analysis)


if __name__ == "__main__":
    main()
