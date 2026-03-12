import pandas as pd
import numpy as np
from modules.indicators import compute_indicators


def analyze_trend(df: pd.DataFrame, indicators: dict) -> dict:
    """Analyze trend using EMA, ADX, and price action."""
    close  = df["Close"]
    last   = close.iloc[-1]

    ema20  = indicators["ema20"].iloc[-1]
    ema50  = indicators["ema50"].iloc[-1]
    ema200 = indicators["ema200"].iloc[-1]
    adx    = indicators["adx"].iloc[-1]
    rsi    = indicators["rsi"].iloc[-1]

    # Daily trend
    if last > ema20 > ema50 > ema200:
        daily = "UPTREND"
    elif last < ema20 < ema50 < ema200:
        daily = "DOWNTREND"
    elif last > ema200:
        daily = "UPTREND (choppy)"
    elif last < ema200:
        daily = "DOWNTREND (choppy)"
    else:
        daily = "SIDEWAYS"

    # Weekly trend via 10-bar EMA proxy
    weekly_close = close.resample("W").last().dropna() if hasattr(close.index, "freq") else close
    w_ema20 = weekly_close.ewm(span=20, adjust=False).mean()
    weekly = "UPTREND" if weekly_close.iloc[-1] > w_ema20.iloc[-1] else "DOWNTREND"

    # ADX strength
    if adx >= 40:
        direction = "STRONG TREND"
    elif adx >= 25:
        direction = "TRENDING"
    elif adx >= 20:
        direction = "WEAK TREND"
    else:
        direction = "RANGING / NO TREND"

    return {
        "daily":     daily,
        "weekly":    weekly,
        "adx":       f"{adx:.1f}",
        "direction": direction,
        "rsi":       f"{rsi:.1f}",
        "ema20_val": round(ema20, 2),
        "ema50_val": round(ema50, 2),
        "ema200_val":round(ema200, 2),
    }
