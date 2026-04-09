"""
crossover_tab.py
────────────────────────────────────────────────────────────────────────────
MODULE 17 — EMA × SMA CROSSOVER MATRIX  (Streamlit render layer)
Drop-in tab renderer.  Call render_crossover_tab(df) from app.py.
────────────────────────────────────────────────────────────────────────────
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

# ── colour palette ────────────────────────────────────────────────────────────
_BG      = "#020c06"
_PANEL   = "#040f08"
_BORDER  = "#0d3318"
_GREEN   = "#00ff6a"
_GREEN2  = "#007733"
_RED     = "#ff3355"
_AMBER   = "#ffaa00"
_CYAN    = "#00ffcc"
_DIM     = "#3a6648"
_TEXT    = "#a0ffc0"
_MONO    = "Share Tech Mono"
_ORBI    = "Orbitron"


# ── tiny helpers ──────────────────────────────────────────────────────────────

def _cell_color(t: float) -> tuple[str, str]:
    """(background, text) from normalised 0-1 value."""
    if t < 0.20: return ("#330011", _RED)
    if t < 0.40: return ("#1a1000", _AMBER)
    if t < 0.60: return ("#0d1a0e", _TEXT)
    if t < 0.80: return ("#003311", _GREEN)
    return ("#004d1a", _GREEN)


def _metric_card(label: str, value: str, color: str = _GREEN, note: str = "") -> None:
    st.markdown(f"""
    <div style="background:{_PANEL};border:1px solid {_BORDER};
    border-top:2px solid {color};padding:10px 12px;text-align:center;">
        <div style="font-size:9px;color:{_DIM};letter-spacing:2px;margin-bottom:4px;">{label}</div>
        <div style="font-family:'{_ORBI}',monospace;font-size:18px;font-weight:900;
        color:{color};letter-spacing:2px;">{value}</div>
        {"" if not note else f'<div style="font-size:9px;color:{_DIM};margin-top:4px;">{note}</div>'}
    </div>""", unsafe_allow_html=True)


def _section(title: str, color: str = _CYAN) -> None:
    st.markdown(f"""
    <div style="font-family:'{_ORBI}',monospace;font-size:10px;color:{color};
    letter-spacing:3px;margin:16px 0 8px;">{title}</div>
    """, unsafe_allow_html=True)


def _divider() -> None:
    st.markdown("<hr style='border:none;border-top:1px solid #0d3318;margin:14px 0;'>",
                unsafe_allow_html=True)


def _chart_defaults() -> dict:
    return dict(
        paper_bgcolor=_BG, plot_bgcolor=_PANEL,
        font=dict(family=_MONO, color=_TEXT, size=10),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor=_BORDER, showgrid=True, tickfont=dict(size=9)),
        yaxis=dict(gridcolor=_BORDER, showgrid=True, tickfont=dict(size=9)),
    )


# ── confluence gauge card ─────────────────────────────────────────────────────

def _confluence_card(conf: dict) -> None:
    def _row(label, value, good_cond: bool | None, fmt=""):
        if value is None:
            val_str, color = "N/A", _DIM
        else:
            val_str = f"{value}{fmt}"
            if good_cond is None:
                color = _AMBER
            else:
                color = _GREEN if good_cond else _RED

        return (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:5px 10px;border-bottom:1px solid #0a1a0e;font-family:\'{_MONO}\',monospace;">'
            f'<span style="font-size:10px;color:{_DIM};">{label}</span>'
            f'<span style="font-size:11px;color:{color};font-weight:bold;">{val_str}</span>'
            f'</div>'
        )

    rsi   = conf.get("rsi")
    macd  = conf.get("macd_hist")
    atr   = conf.get("atr_pct")
    vol   = conf.get("vol_ratio")
    above = conf.get("live_above")
    cross = conf.get("live_cross")

    rows = "".join([
        _row("RSI(14)",        rsi,  rsi is not None and 45 <= rsi <= 65,  ""),
        _row("MACD Histogram", macd, macd is not None and macd > 0,        ""),
        _row("ATR %",          atr,  atr  is not None and atr < 3.0,       "%"),
        _row("Volume Ratio",   vol,  vol  is not None and vol >= 1.0,      "×"),
        _row("EMA > SMA",      "YES" if above else "NO",
             above, ""),
        _row("Live Cross",     "YES" if cross else "—",
             cross if cross else None, ""),
    ])

    # compute a simple confluence score 0-6
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
    <div style="background:{_PANEL};border:1px solid {_BORDER};
    border-top:3px solid {score_color};padding:0;">
        <div style="padding:10px 12px;border-bottom:1px solid {_BORDER};
        display:flex;align-items:center;justify-content:space-between;">
            <span style="font-family:'{_ORBI}',monospace;font-size:9px;
            color:{_DIM};letter-spacing:3px;">LIVE CONFLUENCE</span>
            <span style="font-family:'{_ORBI}',monospace;font-size:14px;
            font-weight:900;color:{score_color};">{score}/6 — {label}</span>
        </div>
        {rows}
    </div>
    """, unsafe_allow_html=True)


# ── heatmap ───────────────────────────────────────────────────────────────────

def _draw_heatmap(matrix: list[dict], metric: str, label: str) -> None:
    vals  = [m[metric] for m in matrix if m["trades"] > 0]
    if not vals:
        st.markdown(f"<div style='color:{_DIM};'>No trades — try disabling filters.</div>",
                    unsafe_allow_html=True)
        return

    vmin, vmax = min(vals), max(vals)
    span       = vmax - vmin or 1

    # Build lookup
    lookup = {(m["ema"], m["sma"]): m for m in matrix}

    # Build HTML table
    col_w = 58
    th_style = (
        f"font-family:'{_ORBI}',monospace;font-size:9px;color:{_DIM};"
        f"text-align:center;padding:4px 2px;letter-spacing:1px;"
    )
    header  = (
        f'<th style="{th_style};text-align:right;padding-right:8px;">'
        f'SMA↓ / EMA→</th>'
    )
    for ep in EMA_PERIODS:
        header += f'<th style="{th_style};width:{col_w}px;">E{ep}</th>'

    rows_html = ""
    for sp in SMA_PERIODS:
        rows_html += (
            f'<tr><td style="font-family:\'{_ORBI}\',monospace;font-size:9px;'
            f'color:{_DIM};text-align:right;padding:2px 8px 2px 0;'
            f'white-space:nowrap;">S{sp}</td>'
        )
        for ep in EMA_PERIODS:
            key = (ep, sp)
            if ep >= sp or key not in lookup:
                rows_html += (
                    f'<td style="width:{col_w}px;height:38px;background:{_PANEL};'
                    f'text-align:center;vertical-align:middle;">'
                    f'<span style="font-size:9px;color:{_DIM};">n/a</span></td>'
                )
                continue
            m   = lookup[key]
            v   = m[metric]
            t   = (v - vmin) / span
            bg, tc = _cell_color(t)
            # format value
            if metric in ("win_rate",):
                vstr = f"{v:.0f}%"
            elif metric in ("cum_return",):
                vstr = f"{v:+.1f}%"
            elif metric in ("sharpe", "rr"):
                vstr = f"{v:.2f}"
            else:
                vstr = str(v)

            rows_html += (
                f'<td style="width:{col_w}px;height:38px;background:{bg};'
                f'text-align:center;vertical-align:middle;cursor:default;'
                f'border:1px solid rgba(0,0,0,0.25);">'
                f'<span style="font-family:\'{_MONO}\',monospace;font-size:10px;'
                f'font-weight:bold;color:{tc};">{vstr}</span></td>'
            )
        rows_html += "</tr>"

    legend_stops = [
        ("#330011", _RED),   ("#1a1000", _AMBER),
        ("#0d1a0e", _TEXT),  ("#003311", _GREEN), ("#004d1a", _GREEN),
    ]
    legend_html = "".join(
        f'<div style="flex:1;background:{bg};height:8px;border-radius:1px;"></div>'
        for bg, _ in legend_stops
    )

    full_html = f"""
    <div style="margin-bottom:6px;">
        <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
            <span style="font-size:10px;color:{_DIM};">Low</span>
            <div style="display:flex;flex:1;gap:2px;max-width:180px;">{legend_html}</div>
            <span style="font-size:10px;color:{_DIM};">High</span>
            <span style="font-size:10px;color:{_DIM};margin-left:12px;">
                Range: {vmin:.1f} – {vmax:.1f}
            </span>
        </div>
        <div style="overflow-x:auto;">
            <table style="border-collapse:collapse;table-layout:fixed;">
                <thead><tr>{header}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        <div style="font-size:10px;color:{_DIM};margin-top:6px;">
            X-axis: EMA period &nbsp;|&nbsp; Y-axis: SMA period &nbsp;|&nbsp;
            Metric: <b style="color:{_CYAN};">{label}</b> &nbsp;|&nbsp;
            n/a = EMA ≥ SMA (invalid pair)
        </div>
    </div>
    """
    st.markdown(full_html, unsafe_allow_html=True)


# ── best-pair price chart ─────────────────────────────────────────────────────

def _best_pair_chart(df: pd.DataFrame, best: dict, inds: dict) -> go.Figure:
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.50, 0.17, 0.17, 0.16],
    )

    # Price + EMA + SMA
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing=dict(line=dict(color=_GREEN, width=1),
                        fillcolor="rgba(0,255,106,0.1)"),
        decreasing=dict(line=dict(color=_RED,   width=1),
                        fillcolor="rgba(255,51,85,0.1)"),
        name="Price",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=inds["best_ema"],
        mode="lines", name=f"EMA({best['ema']})",
        line=dict(color=_RED, width=1.2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=inds["best_sma"],
        mode="lines", name=f"SMA({best['sma']})",
        line=dict(color=_AMBER, width=1.2, dash="dot"),
    ), row=1, col=1)

    # Mark trade entries/exits on best pair
    for t in best.get("trade_log", []):
        ei  = t["entry_idx"]
        xi  = t["exit_idx"]
        ret = t["return_pct"]
        col = _GREEN if ret > 0 else _RED
        fig.add_trace(go.Scatter(
            x=[df.index[ei]], y=[df["Low"].iloc[ei] * 0.995],
            mode="markers", marker=dict(symbol="triangle-up", size=10, color=_GREEN),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[df.index[xi]], y=[df["High"].iloc[xi] * 1.005],
            mode="markers", marker=dict(symbol="triangle-down", size=10, color=_RED),
            showlegend=False,
        ), row=1, col=1)

    # RSI
    rsi = inds["rsi"]
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, mode="lines", name="RSI(14)",
        line=dict(color=_CYAN, width=1.2),
    ), row=2, col=1)
    for level, color in [(70, "rgba(255,51,85,0.3)"), (30, "rgba(0,255,106,0.3)"),
                         (50, "rgba(58,102,72,0.5)")]:
        fig.add_hline(y=level, row=2, col=1,
                      line=dict(color=color, width=1, dash="dot"))

    # MACD
    macd_h = inds["macd_hist"]
    colors_macd = [_GREEN if v >= 0 else _RED for v in macd_h.fillna(0)]
    fig.add_trace(go.Bar(
        x=df.index, y=macd_h, name="MACD Hist",
        marker_color=colors_macd, showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=inds["macd_line"],   mode="lines",
        line=dict(color=_CYAN, width=1), name="MACD", showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=inds["macd_signal"], mode="lines",
        line=dict(color=_AMBER, width=1, dash="dot"), name="Signal", showlegend=False,
    ), row=3, col=1)

    # ATR%
    fig.add_trace(go.Scatter(
        x=df.index, y=inds["atr_pct"], mode="lines", name="ATR%",
        line=dict(color="#7c3aed", width=1.2),
    ), row=4, col=1)
    fig.add_hline(y=3.5, row=4, col=1,
                  line=dict(color="rgba(255,51,85,0.35)", width=1, dash="dot"),
                  annotation_text="3.5% Caution",
                  annotation_font_size=8, annotation_font_color=_RED)

    cfg = _chart_defaults()
    fig.update_layout(
        **{k: v for k, v in cfg.items()
           if k not in ("xaxis", "yaxis")},
        height=620,
        legend=dict(
            bgcolor=_PANEL, bordercolor=_BORDER, borderwidth=1,
            font=dict(size=9), orientation="h", yanchor="bottom", y=1.01,
        ),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=_BORDER, tickfont=dict(size=9),
                         showgrid=True, row=i, col=1)
        fig.update_yaxes(gridcolor=_BORDER, tickfont=dict(size=9),
                         showgrid=True, row=i, col=1)
    fig.update_yaxes(title_text="RSI",  title_font=dict(size=9, color=_DIM),
                     row=2, col=1)
    fig.update_yaxes(title_text="MACD", title_font=dict(size=9, color=_DIM),
                     row=3, col=1)
    fig.update_yaxes(title_text="ATR%", title_font=dict(size=9, color=_DIM),
                     row=4, col=1)
    return fig


# ── trade log table ───────────────────────────────────────────────────────────

def _trade_log_table(trades: list[dict]) -> None:
    if not trades:
        st.markdown(f"<div style='color:{_DIM};font-size:11px;'>No trades generated.</div>",
                    unsafe_allow_html=True)
        return

    header = (
        f'<tr style="background:#061510;">'
        f'<th style="padding:5px 8px;font-size:9px;color:{_DIM};">  #</th>'
        f'<th style="padding:5px 8px;font-size:9px;color:{_DIM};">ENTRY DATE</th>'
        f'<th style="padding:5px 8px;font-size:9px;color:{_DIM};">EXIT DATE</th>'
        f'<th style="padding:5px 8px;font-size:9px;color:{_DIM};text-align:right;">ENTRY ₹</th>'
        f'<th style="padding:5px 8px;font-size:9px;color:{_DIM};text-align:right;">EXIT ₹</th>'
        f'<th style="padding:5px 8px;font-size:9px;color:{_DIM};text-align:right;">RETURN</th>'
        f'<th style="padding:5px 8px;font-size:9px;color:{_DIM};text-align:right;">HOLD (d)</th>'
        f'</tr>'
    )
    rows = ""
    for i, t in enumerate(trades):
        ret   = t["return_pct"]
        color = _GREEN if ret >= 0 else _RED
        sign  = "▲" if ret >= 0 else "▼"
        bg    = "#001a0a" if ret >= 0 else "#1a0008"
        rows += (
            f'<tr style="background:{"#040f08" if i % 2 == 0 else bg};">'
            f'<td style="padding:5px 8px;font-size:10px;color:{_DIM};">{i+1}</td>'
            f'<td style="padding:5px 8px;font-size:10px;color:{_TEXT};">{t["entry_date"]}</td>'
            f'<td style="padding:5px 8px;font-size:10px;color:{_TEXT};">{t["exit_date"]}</td>'
            f'<td style="padding:5px 8px;font-size:10px;color:{_TEXT};text-align:right;">₹{t["entry_price"]:,.2f}</td>'
            f'<td style="padding:5px 8px;font-size:10px;color:{_TEXT};text-align:right;">₹{t["exit_price"]:,.2f}</td>'
            f'<td style="padding:5px 8px;font-size:11px;font-weight:bold;'
            f'color:{color};text-align:right;">{sign}{abs(ret):.1f}%</td>'
            f'<td style="padding:5px 8px;font-size:10px;color:{_DIM};text-align:right;">{t["hold_days"]}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="overflow-x:auto;">'
        f'<table style="border-collapse:collapse;width:100%;font-family:\'{_MONO}\',monospace;">'
        f'<thead>{header}</thead><tbody>{rows}</tbody></table></div>',
        unsafe_allow_html=True,
    )


# ── suggestions panel ─────────────────────────────────────────────────────────

def _suggestions_panel(suggestions: list[str], conf: dict) -> None:
    icons = {
        "RSI":       ("📡", _CYAN),
        "MACD":      ("📊", _AMBER),
        "ATR":       ("📐", "#7c3aed"),
        "Volume":    ("📦", _GREEN),
        "NIFTY":     ("🏛", _DIM),
        "ADDITIONAL":("💡", _AMBER),
    }

    items_html = ""
    for s in suggestions:
        key    = next((k for k in icons if k in s), None)
        ic, cl = icons.get(key, ("▸", _DIM))
        # highlight strong words
        for word, wc in [
            ("OVERBOUGHT", _RED), ("OVERSOLD", _RED),
            ("POSITIVE", _GREEN), ("NEGATIVE", _RED),
            ("STRONG INSTITUTIONAL", _GREEN),
            ("WEAK VOLUME", _RED), ("HIGH VOLATILITY", _RED),
            ("NORMAL RANGE", _TEXT), ("MOMENTUM ZONE", _GREEN),
        ]:
            s = s.replace(word, f'<b style="color:{wc};">{word}</b>')

        items_html += (
            f'<div style="display:flex;gap:10px;padding:8px 12px;'
            f'border-left:3px solid {cl};background:#030b06;'
            f'margin-bottom:4px;font-family:\'{_MONO}\',monospace;font-size:11px;">'
            f'<span style="color:{cl};min-width:18px;">{ic}</span>'
            f'<span style="color:{_TEXT};line-height:1.6;">{s}</span>'
            f'</div>'
        )

    st.markdown(f"""
    <div style="background:{_PANEL};border:1px solid {_BORDER};
    border-top:2px solid {_AMBER};padding:14px;">
        <div style="font-family:'{_ORBI}',monospace;font-size:9px;color:{_AMBER};
        letter-spacing:3px;margin-bottom:12px;">
        ▶ CONFLUENCE ANALYSIS &amp; NOISE-REDUCTION RECOMMENDATIONS
        </div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)

    # parameter table
    st.markdown(f"""
    <div style="margin-top:14px;background:{_PANEL};border:1px solid {_BORDER};
    border-top:2px solid {_CYAN};padding:14px;">
        <div style="font-family:'{_ORBI}',monospace;font-size:9px;color:{_CYAN};
        letter-spacing:3px;margin-bottom:12px;">PARAMETER GUIDE — WHAT EACH FILTER REMOVES</div>
        <table style="width:100%;border-collapse:collapse;
        font-family:'{_MONO}',monospace;font-size:11px;">
            <thead>
                <tr style="background:#061510;border-bottom:1px solid {_BORDER};">
                    <th style="padding:6px 10px;text-align:left;color:{_DIM};font-size:9px;">FILTER</th>
                    <th style="padding:6px 10px;text-align:left;color:{_DIM};font-size:9px;">PARAMETER</th>
                    <th style="padding:6px 10px;text-align:left;color:{_DIM};font-size:9px;">REMOVES</th>
                    <th style="padding:6px 10px;text-align:left;color:{_DIM};font-size:9px;">IDEAL ZONE</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid #0a1a0e;">
                    <td style="padding:6px 10px;color:{_CYAN};">RSI(14)</td>
                    <td style="padding:6px 10px;color:{_TEXT};">45 – 65</td>
                    <td style="padding:6px 10px;color:{_DIM};">Overbought (late) + oversold (dead bounce)</td>
                    <td style="padding:6px 10px;color:{_GREEN};">50–60 rising</td>
                </tr>
                <tr style="border-bottom:1px solid #0a1a0e;background:#030b06;">
                    <td style="padding:6px 10px;color:{_CYAN};">MACD Hist</td>
                    <td style="padding:6px 10px;color:{_TEXT};">&gt; 0 (expanding)</td>
                    <td style="padding:6px 10px;color:{_DIM};">Crossovers against underlying momentum</td>
                    <td style="padding:6px 10px;color:{_GREEN};">Hist rising + positive</td>
                </tr>
                <tr style="border-bottom:1px solid #0a1a0e;">
                    <td style="padding:6px 10px;color:{_CYAN};">ATR%</td>
                    <td style="padding:6px 10px;color:{_TEXT};">1.0 – 3.5%</td>
                    <td style="padding:6px 10px;color:{_DIM};">High-vol chop + stop-hunt crossovers</td>
                    <td style="padding:6px 10px;color:{_GREEN};">1.5–2.5%</td>
                </tr>
                <tr style="border-bottom:1px solid #0a1a0e;background:#030b06;">
                    <td style="padding:6px 10px;color:{_CYAN};">Volume</td>
                    <td style="padding:6px 10px;color:{_TEXT};">&ge; 1.3× avg</td>
                    <td style="padding:6px 10px;color:{_DIM};">Low-conviction drift crossovers</td>
                    <td style="padding:6px 10px;color:{_GREEN};">&ge; 1.5× on signal day</td>
                </tr>
                <tr style="border-bottom:1px solid #0a1a0e;">
                    <td style="padding:6px 10px;color:{_CYAN};">NIFTY Context</td>
                    <td style="padding:6px 10px;color:{_TEXT};">NIFTY &gt; SMA(55)</td>
                    <td style="padding:6px 10px;color:{_DIM};">Long entries in market downtrend</td>
                    <td style="padding:6px 10px;color:{_GREEN};">Market in uptrend</td>
                </tr>
                <tr style="background:#030b06;">
                    <td style="padding:6px 10px;color:{_CYAN};">2-Bar Confirm</td>
                    <td style="padding:6px 10px;color:{_TEXT};">2 closes above SMA</td>
                    <td style="padding:6px 10px;color:{_DIM};">Single-candle false breakout crossovers</td>
                    <td style="padding:6px 10px;color:{_GREEN};">Most effective in choppy markets</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ── return distribution chart ─────────────────────────────────────────────────

def _return_dist_chart(trades: list[dict]) -> go.Figure:
    rets   = [t["return_pct"] for t in trades]
    colors = [_GREEN if r >= 0 else _RED for r in rets]

    fig = go.Figure()
    fig.add_trace(go.Bar(
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
        **{k: v for k, v in cfg.items()
           if k not in ("xaxis", "yaxis")},
        height=180,
        xaxis=dict(title="Trade #", tickfont=dict(size=9), gridcolor=_BORDER,
                   showgrid=False),
        yaxis=dict(ticksuffix="%", tickfont=dict(size=9), gridcolor=_BORDER,
                   zeroline=True, zerolinecolor=_DIM),
        showlegend=False,
    )
    return fig


# ── main render ───────────────────────────────────────────────────────────────

def render_crossover_tab(df: pd.DataFrame) -> None:
    """Call this from app.py inside the MODULE 17 tab."""

    st.markdown(f"""
    <div style="font-family:'{_ORBI}',monospace;font-size:13px;color:{_GREEN};
    letter-spacing:3px;margin-bottom:16px;">
    MODULE 17 &nbsp;|&nbsp; EMA × SMA CROSSOVER MATRIX &amp; CONFLUENCE ENGINE
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
            use_rsi  = st.checkbox("RSI(14) filter", value=True,
                                   help="Only enter when RSI is in 45–65")
            rsi_lo   = st.slider("RSI lower bound", 30, 55, 45, 5)
            rsi_hi   = st.slider("RSI upper bound", 55, 80, 65, 5)
        with c2:
            use_macd = st.checkbox("MACD histogram filter", value=True,
                                   help="Only enter when MACD hist is positive")
        with c3:
            use_atr  = st.checkbox("ATR% filter", value=True,
                                   help="Skip when ATR% > threshold")
            atr_max  = st.slider("ATR% max", 2.0, 8.0, 4.0, 0.5)
        with c4:
            use_vol  = st.checkbox("Volume ratio filter", value=True,
                                   help="Require volume ≥ threshold × avg")
            vol_min  = st.slider("Volume min ratio", 0.5, 2.0, 1.0, 0.1)

        metric_choice = st.selectbox(
            "Heatmap metric",
            ["cum_return", "sharpe", "win_rate", "rr", "trades"],
            format_func=lambda x: {
                "cum_return": "Cumulative Return %",
                "sharpe":     "Sharpe Ratio",
                "win_rate":   "Win Rate %",
                "rr":         "Risk / Reward",
                "trades":     "Number of Trades",
            }[x],
        )

    # ── Compute ───────────────────────────────────────────────────────────────
    cache_key = (
        use_rsi, rsi_lo, rsi_hi, use_macd, use_atr, atr_max, use_vol, vol_min
    )
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

    matrix      = result["matrix"]
    best        = result["best"]
    top5        = result["top5"]
    summary     = result["summary"]
    inds        = result["indicators"]
    conf        = result["confluence"]
    suggestions = result["suggestions"]

    # ── Summary metrics ───────────────────────────────────────────────────────
    _section("MATRIX SUMMARY", _GREEN)
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    with mc1: _metric_card("BEST PAIR",    summary["best_pair"],
                            _GREEN,        f"Top by {metric_choice}")
    with mc2: _metric_card("BEST RETURN",  f"{summary['best_return']:+.1f}%",
                            _GREEN if summary["best_return"] > 0 else _RED)
    with mc3: _metric_card("VALID PAIRS",  str(summary["valid_pairs"]),
                            _DIM,          f"of {summary['total_pairs']} computed")
    with mc4: _metric_card("AVG WIN RATE", f"{summary['avg_win_rate']:.1f}%",
                            _GREEN if summary["avg_win_rate"] > 50 else _AMBER)
    with mc5: _metric_card("AVG SHARPE",   f"{summary['avg_sharpe']:.2f}",
                            _GREEN if summary["avg_sharpe"] > 1 else _AMBER)
    with mc6: _metric_card("AVG R:R",      f"{summary['avg_rr']:.2f}",
                            _GREEN if summary["avg_rr"] > 1.5 else _AMBER)

    _divider()

    # ── Heatmap + Confluence side by side ─────────────────────────────────────
    left, right = st.columns([3, 1])
    with left:
        _section("EMA × SMA CORRELATION MATRIX", _CYAN)
        metric_label = {
            "cum_return": "Cumulative Return %",
            "sharpe":     "Sharpe Ratio",
            "win_rate":   "Win Rate %",
            "rr":         "Risk / Reward",
            "trades":     "Number of Trades",
        }[metric_choice]
        _draw_heatmap(matrix, metric_choice, metric_label)
    with right:
        _section("LIVE CONFLUENCE", _AMBER)
        _confluence_card(conf)

    _divider()

    # ── Top 5 pairs ───────────────────────────────────────────────────────────
    _section("TOP 5 PAIRS BY RETURN", _GREEN)
    rank_cols = st.columns(5)
    rank_colors = [_GREEN, "#3B6D11", "#3B6D11", "#639922", "#639922"]
    for col, pair, rc in zip(rank_cols, top5, rank_colors):
        with col:
            st.markdown(f"""
            <div style="background:{_PANEL};border:1px solid {_BORDER};
            border-top:3px solid {rc};padding:10px 12px;">
                <div style="font-size:9px;color:{_DIM};letter-spacing:1px;margin-bottom:4px;">
                    RANK #{top5.index(pair)+1}
                </div>
                <div style="font-family:'{_ORBI}',monospace;font-size:11px;
                font-weight:900;color:{rc};margin-bottom:6px;">
                    EMA({pair['ema']}) × SMA({pair['sma']})
                </div>
                <div style="font-family:'{_MONO}',monospace;font-size:10px;color:{_TEXT};">
                    Return: <b style="color:{_GREEN};">{pair['cum_return']:+.1f}%</b><br>
                    Win rate: {pair['win_rate']:.0f}%<br>
                    Sharpe: {pair['sharpe']:.2f}<br>
                    R:R: {pair['rr']:.2f}<br>
                    Trades: {pair['trades']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    _divider()

    # ── Best pair chart ───────────────────────────────────────────────────────
    _section(
        f"BEST PAIR: EMA({best['ema']}) × SMA({best['sma']}) — "
        f"PRICE + RSI + MACD + ATR%", _CYAN,
    )
    st.plotly_chart(
        _best_pair_chart(df, best, inds),
        use_container_width=True,
    )

    _divider()

    # ── Trade log ─────────────────────────────────────────────────────────────
    _section(f"TRADE LOG — EMA({best['ema']}) × SMA({best['sma']})", _AMBER)

    tl_c1, tl_c2, tl_c3, tl_c4 = st.columns(4)
    wins_count = sum(1 for t in best["trade_log"] if t["return_pct"] > 0)
    loss_count = len(best["trade_log"]) - wins_count
    avg_hold   = (
        int(np.mean([t["hold_days"] for t in best["trade_log"]]))
        if best["trade_log"] else 0
    )
    filt_rate  = (
        best["filtered_out"] / (best["raw_signals"] or 1) * 100
    )
    with tl_c1: _metric_card("TOTAL TRADES",   str(best["trades"]))
    with tl_c2: _metric_card("WINS / LOSSES",  f"{wins_count}W  {loss_count}L",
                              _GREEN if wins_count > loss_count else _RED)
    with tl_c3: _metric_card("AVG HOLD",       f"{avg_hold}d")
    with tl_c4: _metric_card("SIGNALS FILTERED",
                              f"{best['filtered_out']}/{best['raw_signals']}",
                              _AMBER,
                              f"{filt_rate:.0f}% removed by filters")

    if best["trade_log"]:
        st.plotly_chart(
            _return_dist_chart(best["trade_log"]),
            use_container_width=True,
        )
        _trade_log_table(best["trade_log"])

    _divider()

    # ── Suggestions ───────────────────────────────────────────────────────────
    _section("CONFLUENCE ANALYSIS & NOISE REDUCTION", _AMBER)
    _suggestions_panel(suggestions, conf)
