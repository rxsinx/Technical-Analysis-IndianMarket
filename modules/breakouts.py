import pandas as pd
import numpy as np


def detect_breakouts(df: pd.DataFrame, sr: dict) -> dict:
    """Detect breakouts and breakdowns."""
    if len(df) < 10:
        return {}

    close  = df["Close"]
    volume = df["Volume"]
    last   = close.iloc[-1]
    prev   = close.iloc[-2]
    atr    = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    avg_vol= volume.rolling(20).mean().iloc[-1]
    last_vol = volume.iloc[-1]

    levels = sr.get("levels", [])
    r_levels = sorted([l["price"] for l in levels if l["type"] == "R"])
    s_levels = sorted([l["price"] for l in levels if l["type"] == "S"], reverse=True)

    nearest_r = r_levels[0] if r_levels else last * 1.03
    nearest_s = s_levels[0] if s_levels else last * 0.97

    status    = "RANGING"
    bk_type   = "N/A"
    level     = nearest_r
    retest    = "N/A"
    confirmed = "NO"

    # Breakout above resistance
    if last > nearest_r and prev <= nearest_r:
        status  = "BREAKOUT"
        bk_type = "Resistance Breakout"
        level   = nearest_r
        confirmed = "YES" if last_vol > avg_vol * 1.2 else "WEAK (low vol)"
        retest  = "Watch for retest"

    # Breakdown below support
    elif last < nearest_s and prev >= nearest_s:
        status  = "BREAKDOWN"
        bk_type = "Support Breakdown"
        level   = nearest_s
        confirmed = "YES" if last_vol > avg_vol * 1.2 else "WEAK (low vol)"
        retest  = "Watch for retest"

    # Near resistance (potential breakout)
    elif abs(last - nearest_r) / last < 0.01:
        status  = "AT RESISTANCE"
        bk_type = "Potential Breakout"
        level   = nearest_r
        retest  = "Watching"

    # Near support (potential breakdown)
    elif abs(last - nearest_s) / last < 0.01:
        status  = "AT SUPPORT"
        bk_type = "Potential Breakdown"
        level   = nearest_s
        retest  = "Watching"

    # Retest after prior breakout
    else:
        # Check if price is pulling back to a prior breakout level
        for rl in r_levels[1:3]:
            if abs(last - rl) / last < 0.015 and last > rl:
                status  = "RETEST"
                bk_type = "Retest of Breakout Level"
                level   = rl
                retest  = "YES – Potential entry"
                break

    return {
        "status":    status,
        "type":      bk_type,
        "level":     round(level, 2),
        "retest":    retest,
        "confirmed": confirmed,
    }
