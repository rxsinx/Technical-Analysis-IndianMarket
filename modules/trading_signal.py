import pandas as pd
import numpy as np


def compute_trading_signal(df: pd.DataFrame, indicators: dict) -> dict:
    """
    Compute a composite trading signal score (-100 to +100).
    Positive = Bullish, Negative = Bearish.
    Returns score, signal label, and detailed signal list.
    """
    if len(df) < 30:
        return _empty()

    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    vol    = df["Volume"].values
    last   = close[-1]

    score   = 0.0
    signals = []

    # ── 1. Price vs SMA 20 ──────────────────────────────────────────────────
    sma20 = pd.Series(close).rolling(20).mean().iloc[-1]
    if last > sma20:
        score += 10
        signals.append({"text": f"Price above 20 SMA (Short-term bullish)", "type": "bull"})
    else:
        score -= 10
        signals.append({"text": f"Price below 20 SMA (Short-term bearish)", "type": "bear"})

    # ── 2. Price vs SMA 50 ──────────────────────────────────────────────────
    sma50 = pd.Series(close).rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    if last > sma50:
        score += 10
        signals.append({"text": f"Price above 50 SMA (Medium-term bullish)", "type": "bull"})
    else:
        score -= 10
        signals.append({"text": f"Price below 50 SMA (Medium-term bearish)", "type": "bear"})

    # ── 3. Price vs EMA 200 ──────────────────────────────────────────────────
    ema200 = indicators.get("ema200")
    if ema200 is not None and not (hasattr(ema200, '__len__') and len(ema200) == 0):
        e200 = float(ema200.iloc[-1]) if hasattr(ema200, 'iloc') else float(ema200[-1])
        if last > e200:
            score += 15
            signals.append({"text": f"Price above EMA 200 (Long-term bullish)", "type": "bull"})
        else:
            score -= 15
            signals.append({"text": f"Price below EMA 200 (Long-term bearish)", "type": "bear"})

    # ── 4. MACD ──────────────────────────────────────────────────────────────
    macd      = indicators.get("macd")
    macd_sig  = indicators.get("macd_signal")
    if macd is not None and macd_sig is not None:
        m  = float(macd.iloc[-1]) if hasattr(macd, 'iloc') else float(macd[-1])
        ms = float(macd_sig.iloc[-1]) if hasattr(macd_sig, 'iloc') else float(macd_sig[-1])
        if m > ms:
            score += 12
            signals.append({"text": "MACD bullish (Above signal line)", "type": "bull"})
        else:
            score -= 12
            signals.append({"text": "MACD bearish (Below signal line)", "type": "bear"})

    # ── 5. RSI ───────────────────────────────────────────────────────────────
    rsi_series = indicators.get("rsi")
    if rsi_series is not None:
        rsi_val = float(rsi_series.iloc[-1]) if hasattr(rsi_series, 'iloc') else float(rsi_series[-1])
        if rsi_val > 70:
            score -= 8
            signals.append({"text": f"RSI Overbought ({rsi_val:.1f}) – caution", "type": "warn"})
        elif rsi_val < 30:
            score += 8
            signals.append({"text": f"RSI Oversold ({rsi_val:.1f}) – potential reversal", "type": "bull"})
        elif 40 <= rsi_val <= 60:
            score += 3
            signals.append({"text": f"RSI Neutral ({rsi_val:.1f})", "type": "neutral"})
        elif rsi_val > 50:
            score += 6
            signals.append({"text": f"RSI Bullish zone ({rsi_val:.1f})", "type": "bull"})
        else:
            score -= 6
            signals.append({"text": f"RSI Bearish zone ({rsi_val:.1f})", "type": "bear"})

    # ── 6. Volume ─────────────────────────────────────────────────────────────
    avg_vol = np.mean(vol[-20:]) if len(vol) >= 20 else np.mean(vol)
    today_vol = vol[-1]
    if today_vol > avg_vol * 1.5:
        score += 8
        signals.append({"text": f"High volume ({today_vol/1e6:.2f}M vs avg {avg_vol/1e6:.2f}M)", "type": "bull"})
    elif today_vol > avg_vol * 0.8:
        signals.append({"text": "Average volume", "type": "neutral"})
    else:
        score -= 5
        signals.append({"text": f"Low volume ({today_vol/1e6:.2f}M vs avg {avg_vol/1e6:.2f}M)", "type": "warn"})

    # ── 7. OBV trend ─────────────────────────────────────────────────────────
    obv = indicators.get("obv")
    if obv is not None and len(obv) >= 10:
        obv_arr = obv.values if hasattr(obv, 'values') else obv
        obv_slope = np.polyfit(np.arange(10), obv_arr[-10:], 1)[0]
        if obv_slope > 0:
            score += 8
            signals.append({"text": "OBV trending up (Accumulation)", "type": "bull"})
        else:
            score -= 8
            signals.append({"text": "OBV trending down (Distribution)", "type": "bear"})

    # ── 8. ATR / Volatility ───────────────────────────────────────────────────
    atr_series = indicators.get("atr")
    if atr_series is not None:
        atr_val = float(atr_series.iloc[-1]) if hasattr(atr_series, 'iloc') else float(atr_series[-1])
        atr_pct = atr_val / last * 100
        if atr_pct > 3:
            signals.append({"text": f"High volatility (ATR: {atr_pct:.1f}%)", "type": "warn"})
        elif atr_pct < 1:
            signals.append({"text": f"Low volatility (ATR: {atr_pct:.1f}%) – possible squeeze", "type": "warn"})
        else:
            signals.append({"text": f"Normal volatility (ATR: {atr_pct:.1f}%)", "type": "neutral"})

    # ── 9. Bollinger Bands position ───────────────────────────────────────────
    bb_upper = indicators.get("bb_upper")
    bb_lower = indicators.get("bb_lower")
    bb_mid   = indicators.get("bb_mid")
    if bb_upper is not None and bb_lower is not None:
        bbu = float(bb_upper.iloc[-1]) if hasattr(bb_upper, 'iloc') else float(bb_upper[-1])
        bbl = float(bb_lower.iloc[-1]) if hasattr(bb_lower, 'iloc') else float(bb_lower[-1])
        bbm = float(bb_mid.iloc[-1]) if (bb_mid is not None and hasattr(bb_mid, 'iloc')) else (bbu + bbl) / 2
        bb_pos = (last - bbl) / (bbu - bbl) * 100 if (bbu - bbl) > 0 else 50
        if last >= bbu:
            score -= 5
            signals.append({"text": f"Price at BB Upper – overbought risk", "type": "warn"})
        elif last <= bbl:
            score += 5
            signals.append({"text": f"Price at BB Lower – oversold bounce potential", "type": "bull"})
        elif last > bbm:
            score += 4
            signals.append({"text": f"Price above BB midline ({bb_pos:.0f}% of band)", "type": "bull"})
        else:
            score -= 4
            signals.append({"text": f"Price below BB midline ({bb_pos:.0f}% of band)", "type": "bear"})

    # ── 10. EMA stack (trend alignment) ──────────────────────────────────────
    ema20  = indicators.get("ema20")
    ema50  = indicators.get("ema50")
    if ema20 is not None and ema50 is not None and ema200 is not None:
        e20 = float(ema20.iloc[-1]) if hasattr(ema20, 'iloc') else float(ema20[-1])
        e50 = float(ema50.iloc[-1]) if hasattr(ema50, 'iloc') else float(ema50[-1])
        e200_v = float(ema200.iloc[-1]) if hasattr(ema200, 'iloc') else float(ema200[-1])
        if e20 > e50 > e200_v:
            score += 12
            signals.append({"text": "EMA stack bullish (20>50>200)", "type": "bull"})
        elif e20 < e50 < e200_v:
            score -= 12
            signals.append({"text": "EMA stack bearish (20<50<200)", "type": "bear"})
        else:
            signals.append({"text": "EMA stack mixed – no clear trend", "type": "neutral"})

    # Clamp score
    score = max(-100, min(100, round(score, 1)))

    # Signal label
    if score >= 60:
        label = "STRONG BUY"
        color = "#00ff6a"
    elif score >= 25:
        label = "BUY"
        color = "#00cc55"
    elif score >= 10:
        label = "WEAK BUY"
        color = "#66ffaa"
    elif score > -10:
        label = "NEUTRAL"
        color = "#ffaa00"
    elif score > -25:
        label = "WEAK SELL"
        color = "#ff8866"
    elif score > -60:
        label = "SELL"
        color = "#ff3355"
    else:
        label = "STRONG SELL"
        color = "#cc0033"

    bull_count    = sum(1 for s in signals if s["type"] == "bull")
    bear_count    = sum(1 for s in signals if s["type"] == "bear")
    neutral_count = sum(1 for s in signals if s["type"] in ("neutral", "warn"))

    return {
        "score":         score,
        "label":         label,
        "color":         color,
        "signals":       signals,
        "bull_count":    bull_count,
        "bear_count":    bear_count,
        "neutral_count": neutral_count,
        "total":         len(signals),
    }


def _empty():
    return {
        "score": 0, "label": "N/A", "color": "#ffaa00",
        "signals": [], "bull_count": 0, "bear_count": 0,
        "neutral_count": 0, "total": 0,
    }
