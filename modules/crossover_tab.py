"""
crossover_tab.py — MODULE 17 EMA × SMA CROSSOVER MATRIX (white theme)
FIX: cache key now includes symbol to prevent stale results on symbol change
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from modules.crossover_matrix import (
    compute_crossover_matrix,
    EMA_PERIODS,
    SMA_PERIODS,
    _ema,
    _sma,
)

# ── white theme palette ───────────────────────────────────────────────────────
_BG     = "#ffffff"
_PANEL  = "#f8f9fa"
_BORDER = "#dee2e6"
_GREEN  = "#1a7f37"
_GREEN2 = "#2da44e"
_RED    = "#cf222e"
_AMBER  = "#9a6700"
_BLUE   = "#0969da"
_CYAN   = "#0550ae"
_DIM    = "#6e7781"
_TEXT   = "#24292f"
_MONO   = "JetBrains Mono, Consolas, monospace"
_ORBI   = "Inter, Segoe UI, sans-serif"


# ── helpers ───────────────────────────────────────────────────────────────────

def _cell_color(t: float) -> tuple[str, str]:
    """(background, text) for heatmap cell, t in [0,1]."""
    if t < 0.20: return ("#ffd7d9", _RED)
    if t < 0.40: return ("#fff8c5", _AMBER)
    if t < 0.60: return ("#f6f8fa", _DIM)
    if t < 0.80: return ("#dafbe1", _GREEN)
    return ("#acf2bd", _GREEN)


def _metric_card(label: str, value: str, color: str = _GREEN, note: str = "") -> None:
    st.markdown(f"""
    <div style="background:{_PANEL};border:1px solid {_BORDER};
    border-top:3px solid {color};padding:12px 14px;text-align:center;border-radius:0 0 4px 4px;">
        <div style="font-size:10px;color:{_DIM};letter-spacing:1px;margin-bottom:4px;
        font-family:{_ORBI};">{label}</div>
        <div style="font-family:{_ORBI};font-size:18px;font-weight:700;
        color:{color};">{value}</div>
        {"" if not note else f'<div style="font-size:10px;color:{_DIM};margin-top:3px;font-family:{_MONO};">{note}</div>'}
    </div>""", unsafe_allow_html=True)


def _section(title: str, color: str = _BLUE) -> None:
    st.markdown(f"""
    <div style="font-family:{_ORBI};font-size:11px;font-weight:600;color:{color};
    letter-spacing:1px;margin:18px 0 8px;text-transform:uppercase;
    border-left:3px solid {color};padding-left:8px;">{title}</div>
    """, unsafe_allow_html=True)


def _divider() -> None:
    st.markdown("<hr style='border:none;border-top:1px solid #dee2e6;margin:16px 0;'>",
                unsafe_allow_html=True)


def _chart_defaults() -> dict:
    return dict(
        paper_bgcolor=_BG, plot_bgcolor=_PANEL,
        font=dict(family=_MONO, color=_TEXT, size=10),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor=_BORDER, showgrid=True, tickfont=dict(size=9),
                   linecolor=_BORDER),
        yaxis=dict(gridcolor=_BORDER, showgrid=True, tickfont=dict(size=9),
                   linecolor=_BORDER),
    )


# ── confluence card ───────────────────────────────────────────────────────────

def _confluence_card(conf: dict) -> None:
    def _row(label, value, good: bool | None, fmt=""):
        if value is None:
            val_str, color, bg = "N/A", _DIM, _PANEL
        else:
            val_str = f"{value}{fmt}"
            if good is None:
                color, bg = _AMBER, "#fff8c5"
            elif good:
                color, bg = _GREEN, "#dafbe1"
            else:
                color, bg = _RED, "#ffd7d9"
        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 12px;border-bottom:1px solid {_BORDER};background:{bg};">'
            f'<span style="font-size:11px;color:{_DIM};font-family:{_MONO};">{label}</span>'
            f'<span style="font-size:11px;color:{color};font-weight:600;'
            f'font-family:{_MONO};">{val_str}</span></div>'
        )

    rsi   = conf.get("rsi")
    macd  = conf.get("macd_hist")
    atr   = conf.get("atr_pct")
    vol   = conf.get("vol_ratio")
    above = conf.get("live_above")
    cross = conf.get("live_cross")

    rows = "".join([
        _row("RSI(14)",        rsi,  rsi  is not None and 35 <= rsi <= 75),
        _row("MACD Histogram", macd, macd is not None and macd > 0),
        _row("ATR %",          atr,  atr  is not None and atr < 3.0,  "%"),
        _row("Volume Ratio",   vol,  vol  is not None and vol >= 1.0,  "×"),
        _row("EMA > SMA",      "YES" if above else "NO", above),
        _row("Live Crossover", "YES" if cross else "—",  cross if cross else None),
    ])

    score = sum([
        1 if (rsi  is not None and 45 <= rsi  <= 65) else 0,
        1 if (macd is not None and macd > 0)          else 0,
        1 if (atr  is not None and atr  < 3.0)        else 0,
        1 if (vol  is not None and vol  >= 1.0)        else 0,
        1 if above else 0,
        1 if cross else 0,
    ])
    score_color = _GREEN if score >= 5 else (_AMBER if score >= 3 else _RED)
    label       = "STRONG" if score >= 5 else ("MODERATE" if score >= 3 else "WEAK")

    st.markdown(f"""
    <div style="background:{_BG};border:1px solid {_BORDER};
    border-top:3px solid {score_color};border-radius:0 0 4px 4px;">
        <div style="padding:10px 14px;border-bottom:1px solid {_BORDER};
        display:flex;align-items:center;justify-content:space-between;background:{_PANEL};">
            <span style="font-family:{_ORBI};font-size:11px;font-weight:600;
            color:{_TEXT};">LIVE CONFLUENCE</span>
            <span style="font-family:{_ORBI};font-size:13px;font-weight:700;
            color:{score_color};">{score}/6 — {label}</span>
        </div>
        {rows}
    </div>
    """, unsafe_allow_html=True)


# ── heatmap ───────────────────────────────────────────────────────────────────

def _draw_heatmap(matrix: list[dict], metric: str, label: str) -> None:
    vals = [m[metric] for m in matrix if m["trades"] > 0]
    if not vals:
        st.markdown(f"<div style='color:{_DIM};font-size:12px;'>No trades generated — try disabling filters.</div>",
                    unsafe_allow_html=True)
        return

    vmin, vmax = min(vals), max(vals)
    span       = vmax - vmin or 1
    lookup     = {(m["ema"], m["sma"]): m for m in matrix}

    col_w    = 60
    th_style = (f"font-family:{_ORBI};font-size:10px;color:{_DIM};font-weight:600;"
                f"text-align:center;padding:6px 4px;background:{_PANEL};")
    header   = (f'<th style="{th_style};text-align:right;padding-right:10px;">'
                f'SMA ↓ / EMA →</th>')
    for ep in EMA_PERIODS:
        header += f'<th style="{th_style};width:{col_w}px;">E{ep}</th>'

    rows_html = ""
    for sp in SMA_PERIODS:
        rows_html += (
            f'<tr><td style="font-family:{_ORBI};font-size:10px;font-weight:600;'
            f'color:{_DIM};text-align:right;padding:4px 10px 4px 0;'
            f'background:{_PANEL};white-space:nowrap;">S{sp}</td>'
        )
        for ep in EMA_PERIODS:
            key = (ep, sp)
            if ep >= sp or key not in lookup:
                rows_html += (
                    f'<td style="width:{col_w}px;height:36px;background:#f6f8fa;'
                    f'text-align:center;vertical-align:middle;">'
                    f'<span style="font-size:9px;color:{_BORDER};">—</span></td>'
                )
                continue
            m  = lookup[key]
            v  = m[metric]
            t  = (v - vmin) / span
            bg, tc = _cell_color(t)

            if metric == "win_rate":
                vstr = f"{v:.0f}%"
            elif metric == "cum_return":
                vstr = f"{v:+.1f}%"
            elif metric in ("sharpe", "rr"):
                vstr = f"{v:.2f}"
            else:
                vstr = str(v)

            # tooltip via title attr
            tip = f"EMA({ep})×SMA({sp}) | Trades:{m['trades']} WR:{m['win_rate']:.0f}%"
            rows_html += (
                f'<td title="{tip}" style="width:{col_w}px;height:36px;background:{bg};'
                f'text-align:center;vertical-align:middle;'
                f'border:1px solid rgba(0,0,0,0.06);cursor:default;">'
                f'<span style="font-family:{_MONO};font-size:10px;'
                f'font-weight:600;color:{tc};">{vstr}</span></td>'
            )
        rows_html += "</tr>"

    # legend
    stops = [("#ffd7d9", _RED), ("#fff8c5", _AMBER),
             ("#f6f8fa", _DIM), ("#dafbe1", _GREEN), ("#acf2bd", _GREEN)]
    legend = "".join(
        f'<div style="flex:1;background:{bg};height:8px;border-radius:1px;"></div>'
        for bg, _ in stops
    )

    st.markdown(f"""
    <div>
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;">
            <span style="font-size:11px;color:{_DIM};font-family:{_MONO};">Low</span>
            <div style="display:flex;flex:1;gap:2px;max-width:200px;">{legend}</div>
            <span style="font-size:11px;color:{_DIM};font-family:{_MONO};">High</span>
            <span style="font-size:11px;color:{_DIM};margin-left:12px;font-family:{_MONO};">
                {vmin:.1f} – {vmax:.1f}</span>
        </div>
        <div style="overflow-x:auto;">
            <table style="border-collapse:collapse;table-layout:fixed;">
                <thead><tr>{header}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        <div style="font-size:10px;color:{_DIM};margin-top:8px;font-family:{_MONO};">
            Hover cells for trade details &nbsp;|&nbsp;
            Metric: <b style="color:{_BLUE};">{label}</b> &nbsp;|&nbsp;
            Cumulative return is <b>compounded</b>. Sharpe = mean/std (per-trade).
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── price chart ───────────────────────────────────────────────────────────────

def _best_pair_chart(df: pd.DataFrame, best: dict, inds: dict) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.50, 0.17, 0.17, 0.16],
    )

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing=dict(line=dict(color=_GREEN, width=1),
                        fillcolor="rgba(26,127,55,0.15)"),
        decreasing=dict(line=dict(color=_RED,   width=1),
                        fillcolor="rgba(207,34,46,0.15)"),
        name="Price",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=inds["best_ema"], mode="lines",
        name=f"EMA({best['ema']})",
        line=dict(color=_RED, width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=inds["best_sma"], mode="lines",
        name=f"SMA({best['sma']})",
        line=dict(color=_AMBER, width=1.5, dash="dot"),
    ), row=1, col=1)

    # Entry / exit markers
    for t in best.get("trade_log", []):
        ei, xi = t["entry_idx"], t["exit_idx"]
        ret    = t["return_pct"]
        fig.add_trace(go.Scatter(
            x=[df.index[ei]], y=[df["Low"].iloc[ei] * 0.994],
            mode="markers",
            marker=dict(symbol="triangle-up", size=9, color=_GREEN),
            showlegend=False, hovertext=f"Entry ₹{t['entry_price']:,.2f}",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[df.index[xi]], y=[df["High"].iloc[xi] * 1.006],
            mode="markers",
            marker=dict(symbol="triangle-down", size=9,
                        color=_GREEN if ret > 0 else _RED),
            showlegend=False,
            hovertext=f"Exit ₹{t['exit_price']:,.2f} ({ret:+.1f}%)",
        ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=inds["rsi"], mode="lines", name="RSI(14)",
        line=dict(color=_BLUE, width=1.2),
    ), row=2, col=1)
    for lvl, cl in [(70, "rgba(207,34,46,0.3)"), (30, "rgba(26,127,55,0.3)"),
                    (50, "rgba(110,119,129,0.4)")]:
        fig.add_hline(y=lvl, row=2, col=1,
                      line=dict(color=cl, width=1, dash="dot"))

    # MACD
    macd_h = inds["macd_hist"]
    c_macd = [_GREEN if v >= 0 else _RED for v in macd_h.fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=macd_h, marker_color=c_macd,
        name="MACD Hist", showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=inds["macd_line"], mode="lines",
        line=dict(color=_BLUE, width=1), name="MACD", showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=inds["macd_signal"], mode="lines",
        line=dict(color=_AMBER, width=1, dash="dot"), name="Signal",
        showlegend=False,
    ), row=3, col=1)

    # ATR%
    fig.add_trace(go.Scatter(
        x=df.index, y=inds["atr_pct"], mode="lines", name="ATR%",
        line=dict(color="#8250df", width=1.2),
    ), row=4, col=1)
    fig.add_hline(y=3.5, row=4, col=1,
                  line=dict(color="rgba(207,34,46,0.4)", width=1, dash="dot"),
                  annotation_text="3.5% caution",
                  annotation_font_size=8, annotation_font_color=_RED)

    cfg = _chart_defaults()
    fig.update_layout(
        **{k: v for k, v in cfg.items() if k not in ("xaxis", "yaxis")},
        height=640,
        legend=dict(
            bgcolor=_PANEL, bordercolor=_BORDER, borderwidth=1,
            font=dict(size=9), orientation="h", yanchor="bottom", y=1.01,
        ),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=_BORDER, tickfont=dict(size=9),
                         linecolor=_BORDER, showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor=_BORDER, tickfont=dict(size=9),
                         linecolor=_BORDER, showgrid=True, row=i, col=1)
    fig.update_yaxes(title_text="RSI",  title_font=dict(size=9, color=_DIM), row=2, col=1)
    fig.update_yaxes(title_text="MACD", title_font=dict(size=9, color=_DIM), row=3, col=1)
    fig.update_yaxes(title_text="ATR%", title_font=dict(size=9, color=_DIM), row=4, col=1)
    return fig


# ── trade log ─────────────────────────────────────────────────────────────────

def _trade_log_table(trades: list[dict]) -> None:
    if not trades:
        st.markdown(f"<div style='color:{_DIM};font-size:12px;'>No trades generated.</div>",
                    unsafe_allow_html=True)
        return

    hdr = (
        f'<tr style="background:{_PANEL};border-bottom:2px solid {_BORDER};">'
        + "".join(
            f'<th style="padding:7px 10px;font-size:10px;font-weight:600;'
            f'color:{_DIM};font-family:{_ORBI};text-align:{al};">{h}</th>'
            for h, al in [
                ("#", "center"), ("Entry Date", "left"), ("Exit Date", "left"),
                ("Entry ₹", "right"), ("Exit ₹", "right"),
                ("Return", "right"), ("Hold (d)", "right"), ("Confluence", "center"),
            ]
        ) + "</tr>"
    )
    rows = ""
    for i, t in enumerate(trades):
        ret   = t["return_pct"]
        color = _GREEN if ret >= 0 else _RED
        sign  = "▲" if ret >= 0 else "▼"
        bg    = "#f0fff4" if ret >= 0 else "#fff0f0"
        open_flag = " ⏳" if t.get("open_trade") else ""
        conf  = "★" * t.get("conf_score", 0)
        rows += (
            f'<tr style="background:{"#ffffff" if i % 2 == 0 else _PANEL};">'
            f'<td style="padding:6px 10px;font-size:10px;color:{_DIM};text-align:center;">{i+1}</td>'
            f'<td style="padding:6px 10px;font-size:10px;color:{_TEXT};">{t["entry_date"]}</td>'
            f'<td style="padding:6px 10px;font-size:10px;color:{_TEXT};">{t["exit_date"]}{open_flag}</td>'
            f'<td style="padding:6px 10px;font-size:10px;color:{_TEXT};text-align:right;">₹{t["entry_price"]:,.2f}</td>'
            f'<td style="padding:6px 10px;font-size:10px;color:{_TEXT};text-align:right;">₹{t["exit_price"]:,.2f}</td>'
            f'<td style="padding:6px 10px;font-size:11px;font-weight:600;'
            f'color:{color};text-align:right;background:{bg};">{sign}{abs(ret):.1f}%</td>'
            f'<td style="padding:6px 10px;font-size:10px;color:{_DIM};text-align:right;">{t["hold_days"]}</td>'
            f'<td style="padding:6px 10px;font-size:11px;color:{_AMBER};text-align:center;">{conf}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="overflow-x:auto;border:1px solid {_BORDER};border-radius:4px;">'
        f'<table style="border-collapse:collapse;width:100%;font-family:{_MONO};">'
        f'<thead>{hdr}</thead><tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )
    open_count = sum(1 for t in trades if t.get("open_trade"))
    if open_count:
        st.markdown(
            f'<div style="font-size:10px;color:{_AMBER};margin-top:4px;">'
            f'⏳ {open_count} trade(s) still open at end of data — closed at last bar price.</div>',
            unsafe_allow_html=True,
        )


# ── return distribution ───────────────────────────────────────────────────────

def _return_dist_chart(trades: list[dict]) -> go.Figure:
    rets   = [t["return_pct"] for t in trades]
    colors = [_GREEN if r >= 0 else _RED for r in rets]
    fig = go.Figure(go.Bar(
        x=list(range(1, len(rets) + 1)), y=rets,
        marker_color=colors,
        text=[f"{r:+.1f}%" for r in rets],
        textposition="outside",
        textfont=dict(size=8, color=_TEXT),
        width=0.6,
    ))
    fig.add_hline(y=0, line=dict(color=_DIM, width=1))
    cfg = _chart_defaults()
    fig.update_layout(
        **{k: v for k, v in cfg.items() if k not in ("xaxis", "yaxis")},
        height=190,
        xaxis=dict(title="Trade #", tickfont=dict(size=9), gridcolor=_BORDER,
                   showgrid=False, linecolor=_BORDER),
        yaxis=dict(ticksuffix="%", tickfont=dict(size=9), gridcolor=_BORDER,
                   zeroline=True, zerolinecolor=_DIM, linecolor=_BORDER),
        showlegend=False,
    )
    return fig


# ── suggestions panel ─────────────────────────────────────────────────────────

def _suggestions_panel(suggestions: list[str]) -> None:
    icons = {
        "RSI": ("📡", _BLUE), "MACD": ("📊", _AMBER),
        "ATR": ("📐", "#8250df"), "Volume": ("📦", _GREEN),
        "ADDITIONAL": ("💡", _AMBER),
    }
    items = ""
    for s in suggestions:
        key    = next((k for k in icons if k in s), None)
        ic, cl = icons.get(key, ("▸", _DIM))
        for word, wc in [
            ("OVERBOUGHT", _RED), ("OVERSOLD", _RED),
            ("POSITIVE", _GREEN), ("NEGATIVE", _RED),
            ("STRONG INSTITUTIONAL", _GREEN), ("WEAK VOLUME", _RED),
            ("HIGH VOLATILITY", _RED), ("NORMAL RANGE", _DIM),
            ("MOMENTUM ZONE", _GREEN),
        ]:
            s = s.replace(word, f'<b style="color:{wc};">{word}</b>')
        items += (
            f'<div style="display:flex;gap:10px;padding:9px 14px;'
            f'border-left:3px solid {cl};background:{_BG};'
            f'margin-bottom:4px;border-bottom:1px solid {_BORDER};">'
            f'<span style="color:{cl};min-width:18px;font-size:14px;">{ic}</span>'
            f'<span style="color:{_TEXT};font-family:{_MONO};font-size:11px;'
            f'line-height:1.6;">{s}</span></div>'
        )

    st.markdown(f"""
    <div style="background:{_BG};border:1px solid {_BORDER};
    border-top:3px solid {_AMBER};border-radius:0 0 4px 4px;">
        <div style="padding:10px 14px;border-bottom:1px solid {_BORDER};
        background:{_PANEL};">
            <span style="font-family:{_ORBI};font-size:11px;font-weight:600;
            color:{_AMBER};">CONFLUENCE ANALYSIS & NOISE-REDUCTION RECOMMENDATIONS</span>
        </div>
        {items}
    </div>
    """, unsafe_allow_html=True)

    # filter guide table
    st.markdown(f"""
    <div style="margin-top:14px;background:{_BG};border:1px solid {_BORDER};
    border-top:3px solid {_BLUE};border-radius:0 0 4px 4px;">
        <div style="padding:10px 14px;border-bottom:1px solid {_BORDER};
        background:{_PANEL};">
            <span style="font-family:{_ORBI};font-size:11px;font-weight:600;
            color:{_BLUE};">FILTER PARAMETER GUIDE</span>
        </div>
        <table style="width:100%;border-collapse:collapse;
        font-family:{_MONO};font-size:11px;">
            <thead>
                <tr style="background:{_PANEL};border-bottom:1px solid {_BORDER};">
                    <th style="padding:8px 12px;text-align:left;color:{_DIM};font-size:10px;">FILTER</th>
                    <th style="padding:8px 12px;text-align:left;color:{_DIM};font-size:10px;">PARAMETER</th>
                    <th style="padding:8px 12px;text-align:left;color:{_DIM};font-size:10px;">REMOVES</th>
                    <th style="padding:8px 12px;text-align:left;color:{_DIM};font-size:10px;">IDEAL ZONE</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid {_BORDER};">
                    <td style="padding:7px 12px;color:{_BLUE};font-weight:600;">RSI(14)</td>
                    <td style="padding:7px 12px;color:{_TEXT};">35 – 75</td>
                    <td style="padding:7px 12px;color:{_DIM};">Overbought entries + oversold dead bounces</td>
                    <td style="padding:7px 12px;color:{_GREEN};font-weight:600;">50–60, rising</td>
                </tr>
                <tr style="background:{_PANEL};border-bottom:1px solid {_BORDER};">
                    <td style="padding:7px 12px;color:{_BLUE};font-weight:600;">MACD Hist</td>
                    <td style="padding:7px 12px;color:{_TEXT};">&gt; 0</td>
                    <td style="padding:7px 12px;color:{_DIM};">Crossovers against underlying momentum</td>
                    <td style="padding:7px 12px;color:{_GREEN};font-weight:600;">Hist positive & expanding</td>
                </tr>
                <tr style="border-bottom:1px solid {_BORDER};">
                    <td style="padding:7px 12px;color:{_BLUE};font-weight:600;">ATR%</td>
                    <td style="padding:7px 12px;color:{_TEXT};">1.0 – 3.5%</td>
                    <td style="padding:7px 12px;color:{_DIM};">High-vol chop and stop-hunt crossovers</td>
                    <td style="padding:7px 12px;color:{_GREEN};font-weight:600;">1.5 – 2.5%</td>
                </tr>
                <tr style="background:{_PANEL};border-bottom:1px solid {_BORDER};">
                    <td style="padding:7px 12px;color:{_BLUE};font-weight:600;">Volume</td>
                    <td style="padding:7px 12px;color:{_TEXT};">&ge; 1.3× avg</td>
                    <td style="padding:7px 12px;color:{_DIM};">Low-conviction drift crossovers</td>
                    <td style="padding:7px 12px;color:{_GREEN};font-weight:600;">&ge; 1.5× on signal day</td>
                </tr>
                <tr>
                    <td style="padding:7px 12px;color:{_BLUE};font-weight:600;">2-Bar Confirm</td>
                    <td style="padding:7px 12px;color:{_TEXT};">2 closes above SMA</td>
                    <td style="padding:7px 12px;color:{_DIM};">Single-candle false crossovers</td>
                    <td style="padding:7px 12px;color:{_GREEN};font-weight:600;">Most useful in choppy markets</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ── main render ───────────────────────────────────────────────────────────────

def render_crossover_tab(df: pd.DataFrame, symbol: str = "") -> None:
    st.markdown(f"""
    <div style="font-family:{_ORBI};font-size:14px;font-weight:700;color:{_TEXT};
    margin-bottom:6px;">MODULE 17 — EMA × SMA CROSSOVER MATRIX</div>
    <div style="font-size:11px;color:{_DIM};margin-bottom:16px;font-family:{_MONO};">
    Compounded returns · Fixed Sharpe · Open trades captured · Symbol-aware cache
    </div>
    """, unsafe_allow_html=True)

    if df is None or df.empty:
        st.markdown(f"<div style='color:{_DIM};'>No data loaded. Run analysis first.</div>",
                    unsafe_allow_html=True)
        return

    # ── Controls ─────────────────────────────────────────────────────────────
    with st.expander("⚙  FILTER SETTINGS", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            use_rsi = st.checkbox("RSI(14) filter", value=True)
            rsi_lo  = st.slider("RSI lower", 30, 55, 45, 5)
            rsi_hi  = st.slider("RSI upper", 55, 80, 65, 5)
        with c2:
            use_macd = st.checkbox("MACD histogram filter", value=True)
        with c3:
            use_atr = st.checkbox("ATR% filter", value=True)
            atr_max = st.slider("ATR% max", 2.0, 8.0, 4.0, 0.5)
        with c4:
            use_vol = st.checkbox("Volume ratio filter", value=True)
            vol_min = st.slider("Volume min ratio", 0.5, 2.0, 1.0, 0.1)

        metric_choice = st.selectbox(
            "Heatmap metric",
            ["cum_return", "sharpe", "win_rate", "rr", "trades"],
            format_func=lambda x: {
                "cum_return": "Compounded Return %",
                "sharpe":     "Sharpe Ratio (per-trade)",
                "win_rate":   "Win Rate %",
                "rr":         "Risk / Reward",
                "trades":     "Number of Trades",
            }[x],
        )

    # FIX: cache key includes symbol so changing stock recomputes
    cache_key = (symbol, use_rsi, rsi_lo, rsi_hi, use_macd, use_atr, atr_max, use_vol, vol_min)
    if (
        "crossover_result" not in st.session_state
        or st.session_state.get("crossover_cache_key") != cache_key
    ):
        with st.spinner("COMPUTING CROSSOVER MATRIX..."):
            result = compute_crossover_matrix(
                df,
                use_rsi=use_rsi, rsi_lo=rsi_lo, rsi_hi=rsi_hi,
                use_macd=use_macd,
                use_atr=use_atr, atr_max_pct=atr_max,
                use_volume=use_vol, vol_min=vol_min,
            )
        st.session_state["crossover_result"]    = result
        st.session_state["crossover_cache_key"] = cache_key
    else:
        result = st.session_state["crossover_result"]

    if "error" in result:
        st.error(result["error"])
        return

    matrix  = result["matrix"]
    best    = result["best"]
    top5    = result["top5"]
    summary = result["summary"]
    inds    = result["indicators"]
    conf    = result["confluence"]
    suggs   = result["suggestions"]

    # ── Summary metrics ───────────────────────────────────────────────────────
    _section("MATRIX SUMMARY", _BLUE)
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    with mc1: _metric_card("BEST PAIR",    summary["best_pair"], _GREEN)
    with mc2: _metric_card("COMPOUNDED RETURN", f"{summary['best_return']:+.1f}%",
                            _GREEN if summary["best_return"] > 0 else _RED)
    with mc3: _metric_card("VALID PAIRS",  str(summary["valid_pairs"]),
                            _DIM, f"of {summary['total_pairs']} computed")
    with mc4: _metric_card("AVG WIN RATE", f"{summary['avg_win_rate']:.1f}%",
                            _GREEN if summary["avg_win_rate"] > 50 else _AMBER)
    with mc5: _metric_card("AVG SHARPE",   f"{summary['avg_sharpe']:.2f}",
                            _GREEN if summary["avg_sharpe"] > 0.5 else _AMBER)
    with mc6: _metric_card("AVG R:R",      f"{summary['avg_rr']:.2f}",
                            _GREEN if summary["avg_rr"] > 1.5 else _AMBER)

    _divider()

    # ── Heatmap + Confluence ──────────────────────────────────────────────────
    left, right = st.columns([3, 1])
    metric_label = {
        "cum_return": "Compounded Return %", "sharpe": "Sharpe Ratio",
        "win_rate": "Win Rate %", "rr": "Risk / Reward", "trades": "Trades",
    }[metric_choice]
    with left:
        _section(f"HEATMAP — {metric_label}", _BLUE)
        _draw_heatmap(matrix, metric_choice, metric_label)
    with right:
        _section("LIVE CONFLUENCE", _AMBER)
        _confluence_card(conf)

    _divider()

    # ── Top 5 ─────────────────────────────────────────────────────────────────
    _section("TOP 5 PAIRS BY COMPOUNDED RETURN", _GREEN)
    if not top5:
        st.markdown(f"<div style='color:{_DIM};'>No pairs with trades. Try relaxing filters.</div>",
                    unsafe_allow_html=True)
    else:
        rank_cols = st.columns(min(5, len(top5)))
        for col, pair in zip(rank_cols, top5):
            with col:
                rank = top5.index(pair) + 1
                bc   = [_GREEN, "#2da44e", "#3d8b3d", "#4a7c59", "#57724d"][rank - 1]
                st.markdown(f"""
                <div style="background:{_PANEL};border:1px solid {_BORDER};
                border-top:3px solid {bc};padding:12px;border-radius:0 0 4px 4px;">
                    <div style="font-size:10px;color:{_DIM};margin-bottom:4px;">RANK #{rank}</div>
                    <div style="font-family:{_ORBI};font-size:12px;font-weight:700;
                    color:{bc};margin-bottom:8px;">
                        EMA({pair['ema']}) × SMA({pair['sma']})
                    </div>
                    <div style="font-family:{_MONO};font-size:11px;color:{_TEXT};line-height:1.8;">
                        Return: <b style="color:{_GREEN};">{pair['cum_return']:+.1f}%</b><br>
                        Win: {pair['win_rate']:.0f}% &nbsp; Sharpe: {pair['sharpe']:.2f}<br>
                        R:R: {pair['rr']:.2f} &nbsp; Trades: {pair['trades']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    _divider()

    # ── Best pair chart ───────────────────────────────────────────────────────
    _section(f"BEST PAIR: EMA({best['ema']}) × SMA({best['sma']}) — PRICE + RSI + MACD + ATR%", _BLUE)
    st.plotly_chart(_best_pair_chart(df, best, inds), use_container_width=True)

    _divider()

    # ── Trade log ─────────────────────────────────────────────────────────────
    _section(f"TRADE LOG — EMA({best['ema']}) × SMA({best['sma']})", _AMBER)
    tl1, tl2, tl3, tl4 = st.columns(4)
    wins_n  = sum(1 for t in best["trade_log"] if t["return_pct"] > 0)
    loss_n  = len(best["trade_log"]) - wins_n
    avg_h   = int(np.mean([t["hold_days"] for t in best["trade_log"]])) if best["trade_log"] else 0
    filt_r  = best["filtered_out"] / (best["raw_signals"] or 1) * 100
    with tl1: _metric_card("TOTAL TRADES",    str(best["trades"]))
    with tl2: _metric_card("WINS / LOSSES",   f"{wins_n}W  {loss_n}L",
                            _GREEN if wins_n > loss_n else _RED)
    with tl3: _metric_card("AVG HOLD",        f"{avg_h} days")
    with tl4: _metric_card("SIGNALS FILTERED",
                            f"{best['filtered_out']}/{best['raw_signals']}",
                            _AMBER, f"{filt_r:.0f}% suppressed")

    if best["trade_log"]:
        st.plotly_chart(_return_dist_chart(best["trade_log"]), use_container_width=True)
        _trade_log_table(best["trade_log"])

    _divider()

    # ── Suggestions ───────────────────────────────────────────────────────────
    _section("CONFLUENCE ANALYSIS & NOISE REDUCTION", _AMBER)
    _suggestions_panel(suggs)
