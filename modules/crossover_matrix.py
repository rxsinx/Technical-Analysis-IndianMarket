"""
crossover_matrix.py
────────────────────────────────────────────────────────────────────────────
MODULE 17 — EMA × SMA CROSSOVER MATRIX
Computes every valid EMA(fast) × SMA(slow) crossover pair for a given price
series, backtests cumulative return, and evaluates confluence filters
(RSI, MACD, ATR%) to rank signal quality and suppress fake crossovers.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


# ── Period sets ──────────────────────────────────────────────────────────────
EMA_PERIODS  = [5, 8, 10, 13, 20, 21, 34, 50]
SMA_PERIODS  = [10, 20, 30, 50, 55, 89, 100, 144, 200]


# ── Indicator helpers ────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9
          ) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast   = _ema(series, fast)
    ema_slow   = _ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def _vol_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["Volume"] / df["Volume"].rolling(period).mean()


# ── Core backtester ──────────────────────────────────────────────────────────

def _backtest_pair(
    df: pd.DataFrame,
    ema_period: int,
    sma_period: int,
    rsi_series: pd.Series,
    macd_hist:  pd.Series,
    atr_pct:    pd.Series,
    vol_ratio:  pd.Series,
    # confluence filters
    use_rsi:    bool = True,
    use_macd:   bool = True,
    use_atr:    bool = True,
    use_volume: bool = True,
    rsi_lo:     float = 45.0,
    rsi_hi:     float = 65.0,
    atr_max_pct: float = 4.0,
    vol_min:    float = 1.0,
) -> dict:
    """
    Run a single EMA(ema_period) × SMA(sma_period) crossover backtest.
    Returns a dict with performance statistics.
    """
    close   = df["Close"]
    ema_s   = _ema(close, ema_period)
    sma_s   = _sma(close, sma_period)

    # Need SMA to have enough warm-up
    valid_from = sma_period + 1

    trades          = []
    in_trade        = False
    entry_price     = np.nan
    entry_idx       = None
    filtered_out    = 0
    raw_signals     = 0

    for i in range(valid_from, len(close)):
        e_cur, e_prv = ema_s.iloc[i], ema_s.iloc[i - 1]
        s_cur, s_prv = sma_s.iloc[i], sma_s.iloc[i - 1]

        golden = (e_cur > s_cur) and (e_prv <= s_prv)
        death  = (e_cur < s_cur) and (e_prv >= s_prv)

        if golden and not in_trade:
            raw_signals += 1
            # ── confluence checks ──────────────────────────────────────
            ok = True
            rsi_v   = rsi_series.iloc[i]
            macd_v  = macd_hist.iloc[i]
            atr_v   = atr_pct.iloc[i]
            vol_v   = vol_ratio.iloc[i]

            if use_rsi   and pd.notna(rsi_v)  and not (rsi_lo <= rsi_v <= rsi_hi):
                ok = False
            if use_macd  and pd.notna(macd_v) and macd_v < 0:
                ok = False
            if use_atr   and pd.notna(atr_v)  and atr_v > atr_max_pct:
                ok = False
            if use_volume and pd.notna(vol_v) and vol_v < vol_min:
                ok = False

            if ok:
                in_trade    = True
                entry_price = close.iloc[i]
                entry_idx   = i
                # score confluences
                score = 0
                if pd.notna(rsi_v)  and 50 <= rsi_v <= 65:  score += 1
                if pd.notna(macd_v) and macd_v > 0:          score += 1
                if pd.notna(vol_v)  and vol_v >= 1.3:        score += 1
                if pd.notna(atr_v)  and atr_v < 2.0:         score += 1
            else:
                filtered_out += 1

        elif death and in_trade:
            exit_price = close.iloc[i]
            ret        = (exit_price - entry_price) / entry_price * 100
            hold_days  = i - entry_idx
            trades.append({
                "entry_idx":   entry_idx,
                "exit_idx":    i,
                "entry_price": round(entry_price, 2),
                "exit_price":  round(exit_price, 2),
                "return_pct":  round(ret, 2),
                "hold_days":   hold_days,
                "entry_date":  df.index[entry_idx].strftime("%d-%b-%y"),
                "exit_date":   df.index[i].strftime("%d-%b-%y"),
            })
            in_trade    = False
            entry_price = np.nan
            entry_idx   = None

    # ── Statistics ───────────────────────────────────────────────────────────
    if not trades:
        return {
            "ema": ema_period, "sma": sma_period,
            "cum_return": 0.0, "sharpe": 0.0, "win_rate": 0.0,
            "trades": 0, "rr": 0.0, "avg_hold": 0,
            "raw_signals": raw_signals, "filtered_out": filtered_out,
            "trade_log": [],
        }

    rets       = [t["return_pct"] for t in trades]
    wins       = [r for r in rets if r > 0]
    losses     = [r for r in rets if r <= 0]
    cum_ret    = sum(rets)
    win_rate   = len(wins) / len(rets) * 100
    avg_win    = np.mean(wins)   if wins   else 0
    avg_loss   = abs(np.mean(losses)) if losses else 0.01
    rr         = avg_win / avg_loss
    avg_hold   = np.mean([t["hold_days"] for t in trades])
    mean_r     = np.mean(rets)
    std_r      = np.std(rets) if len(rets) > 1 else 0.01
    sharpe     = (mean_r / std_r) * np.sqrt(252 / 30) if std_r > 0 else 0

    return {
        "ema":          ema_period,
        "sma":          sma_period,
        "cum_return":   round(cum_ret, 1),
        "sharpe":       round(sharpe, 2),
        "win_rate":     round(win_rate, 1),
        "trades":       len(trades),
        "rr":           round(rr, 2),
        "avg_hold":     round(avg_hold),
        "raw_signals":  raw_signals,
        "filtered_out": filtered_out,
        "trade_log":    trades,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def compute_crossover_matrix(
    df: pd.DataFrame,
    use_rsi:    bool = True,
    use_macd:   bool = True,
    use_atr:    bool = True,
    use_volume: bool = True,
    rsi_lo:     float = 45.0,
    rsi_hi:     float = 65.0,
    atr_max_pct: float = 4.0,
    vol_min:    float = 1.0,
) -> dict:
    """
    Compute the full EMA × SMA crossover matrix for a given OHLCV DataFrame.

    Returns
    -------
    {
        "matrix":       list[dict],        # one dict per valid (ema < sma) pair
        "best":         dict,              # best by cum_return
        "top5":         list[dict],        # top 5 pairs
        "summary":      dict,              # aggregate stats
        "indicators":   dict,              # pre-computed indicator series for chart
        "confluence":   dict,              # current live confluence status
        "suggestions":  list[str],         # noise-reduction tips
    }
    """
    if df is None or df.empty or len(df) < 210:
        return {"error": "Insufficient data (need ≥ 210 bars)"}

    close = df["Close"]

    # Pre-compute shared indicators once
    rsi_s          = _rsi(close, 14)
    macd_l, macd_sig, macd_hist = _macd(close)
    atr_s          = _atr(df, 14)
    atr_pct_s      = (atr_s / close) * 100          # ATR as % of price
    vol_ratio_s    = _vol_ratio(df, 20)

    matrix = []
    for sma_p in SMA_PERIODS:
        for ema_p in EMA_PERIODS:
            if ema_p >= sma_p:
                continue
            result = _backtest_pair(
                df, ema_p, sma_p,
                rsi_s, macd_hist, atr_pct_s, vol_ratio_s,
                use_rsi=use_rsi, use_macd=use_macd,
                use_atr=use_atr, use_volume=use_volume,
                rsi_lo=rsi_lo, rsi_hi=rsi_hi,
                atr_max_pct=atr_max_pct, vol_min=vol_min,
            )
            matrix.append(result)

    if not matrix:
        return {"error": "No valid pairs computed"}

    matrix.sort(key=lambda x: x["cum_return"], reverse=True)
    best   = matrix[0]
    top5   = matrix[:5]

    valid  = [m for m in matrix if m["trades"] > 0]
    avg_wr = np.mean([m["win_rate"]   for m in valid]) if valid else 0
    avg_sh = np.mean([m["sharpe"]     for m in valid]) if valid else 0
    avg_rr = np.mean([m["rr"]         for m in valid]) if valid else 0

    # ── Live confluence status (last bar) ────────────────────────────────────
    last_rsi  = round(rsi_s.iloc[-1],   1) if pd.notna(rsi_s.iloc[-1])   else None
    last_macd = round(macd_hist.iloc[-1], 3) if pd.notna(macd_hist.iloc[-1]) else None
    last_atr  = round(atr_pct_s.iloc[-1], 2) if pd.notna(atr_pct_s.iloc[-1]) else None
    last_vol  = round(vol_ratio_s.iloc[-1], 2) if pd.notna(vol_ratio_s.iloc[-1]) else None
    last_macd_line = round(macd_l.iloc[-1], 2) if pd.notna(macd_l.iloc[-1]) else None
    last_macd_sig  = round(macd_sig.iloc[-1], 2) if pd.notna(macd_sig.iloc[-1]) else None

    # Best pair live crossover status
    best_ema_s = _ema(close, best["ema"])
    best_sma_s = _sma(close, best["sma"])
    live_above = bool(best_ema_s.iloc[-1] > best_sma_s.iloc[-1])
    live_cross = bool(
        best_ema_s.iloc[-1] > best_sma_s.iloc[-1] and
        best_ema_s.iloc[-2] <= best_sma_s.iloc[-2]
    )

    # ── Noise-reduction suggestions ──────────────────────────────────────────
    suggestions = []
    if last_rsi is not None:
        if last_rsi > 70:
            suggestions.append(
                f"RSI({last_rsi}) is OVERBOUGHT — avoid new long entries; "
                "wait for pullback to 45–60 zone before acting on crossover."
            )
        elif last_rsi < 40:
            suggestions.append(
                f"RSI({last_rsi}) is OVERSOLD — crossover may be a dead-cat bounce; "
                "require 2 consecutive closes above SMA for confirmation."
            )
        else:
            suggestions.append(
                f"RSI({last_rsi}) is in MOMENTUM ZONE (40–70) — crossover signals "
                "carry higher conviction here."
            )

    if last_macd is not None:
        if last_macd > 0:
            suggestions.append(
                f"MACD histogram is POSITIVE ({last_macd}) — momentum is aligned; "
                "adds confluence to any golden cross signal."
            )
        else:
            suggestions.append(
                f"MACD histogram is NEGATIVE ({last_macd}) — bearish momentum "
                "undercurrent; treat EMA crossover with caution, require volume surge."
            )

    if last_atr is not None:
        if last_atr > 3.5:
            suggestions.append(
                f"ATR% = {last_atr}% — HIGH VOLATILITY. Widen stop-loss to 2× ATR "
                "below entry; reduce position size by 30–50% to normalise risk."
            )
        elif last_atr < 1.0:
            suggestions.append(
                f"ATR% = {last_atr}% — LOW VOLATILITY compression. A breakout may be "
                "imminent; watch for volume expansion on the crossover day."
            )
        else:
            suggestions.append(
                f"ATR% = {last_atr}% — NORMAL RANGE. Standard 1.5× ATR stop-loss "
                "is appropriate for entries off this crossover."
            )

    if last_vol is not None:
        if last_vol >= 1.5:
            suggestions.append(
                f"Volume ratio = {last_vol}× avg — STRONG INSTITUTIONAL PARTICIPATION. "
                "High-conviction signal; full position size justified."
            )
        elif last_vol < 0.8:
            suggestions.append(
                f"Volume ratio = {last_vol}× avg — WEAK VOLUME. Crossover lacks "
                "institutional support; reduce size or wait for volume expansion."
            )
        else:
            suggestions.append(
                f"Volume ratio = {last_vol}× avg — MODERATE. Acceptable; "
                "prefer ≥ 1.3× for highest-conviction entries."
            )

    suggestions.append(
        "ADDITIONAL FILTERS TO CONSIDER: "
        "(1) NIFTY 50 must be above its own SMA(55) — avoid longs in market downtrend. "
        "(2) 2-bar confirmation: wait for 2nd consecutive close with EMA above SMA. "
        "(3) Sector momentum: ensure sector index is not in a distribution phase. "
        "(4) Avoid entries within 3 days of results/event — news overrides technicals."
    )

    return {
        "matrix":     matrix,
        "best":       best,
        "top5":       top5,
        "summary": {
            "total_pairs":   len(matrix),
            "valid_pairs":   len(valid),
            "avg_win_rate":  round(avg_wr, 1),
            "avg_sharpe":    round(avg_sh, 2),
            "avg_rr":        round(avg_rr, 2),
            "best_return":   best["cum_return"],
            "best_pair":     f"EMA({best['ema']}) × SMA({best['sma']})",
        },
        "indicators": {
            "rsi":             rsi_s,
            "macd_line":       macd_l,
            "macd_signal":     macd_sig,
            "macd_hist":       macd_hist,
            "atr_pct":         atr_pct_s,
            "vol_ratio":       vol_ratio_s,
            "best_ema":        best_ema_s,
            "best_sma":        best_sma_s,
        },
        "confluence": {
            "rsi":        last_rsi,
            "macd_hist":  last_macd,
            "macd_line":  last_macd_line,
            "macd_sig":   last_macd_sig,
            "atr_pct":    last_atr,
            "vol_ratio":  last_vol,
            "live_above": live_above,
            "live_cross": live_cross,
        },
        "suggestions": suggestions,
    }
