import pandas as pd
import numpy as np


def detect_chart_patterns(df: pd.DataFrame) -> dict:
    """Detect common chart patterns: triangles, flags, rectangles."""
    if len(df) < 30:
        return {"pattern": "None", "completion": "N/A", "target": 0, "invalidation": 0}

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    last   = close.iloc[-1]
    window = min(60, len(df))
    dff    = df.tail(window)

    atr    = (dff["High"] - dff["Low"]).rolling(14).mean().iloc[-1]

    # Rolling highs/lows over 20 bars
    roll_high = dff["High"].rolling(10).max()
    roll_low  = dff["Low"].rolling(10).min()

    # Trend of recent highs and lows
    recent_highs = roll_high.dropna().values[-10:]
    recent_lows  = roll_low.dropna().values[-10:]

    def slope(arr):
        x = np.arange(len(arr))
        return np.polyfit(x, arr, 1)[0] if len(arr) > 1 else 0

    sh = slope(recent_highs)
    sl = slope(recent_lows)

    pattern      = "None"
    completion   = "N/A"
    target       = last
    invalidation = last

    # Ascending Triangle: flat highs, rising lows
    if abs(sh) < atr * 0.05 and sl > atr * 0.01:
        pattern    = "Ascending Triangle"
        completion = f"{min(95, int(60 + sl/atr*100))}%"
        target     = last + (recent_highs[-1] - recent_lows[0])
        invalidation = recent_lows[-1] - atr

    # Descending Triangle: falling highs, flat lows
    elif sh < -atr * 0.01 and abs(sl) < atr * 0.05:
        pattern    = "Descending Triangle"
        completion = f"{min(95, int(60 + abs(sh)/atr*100))}%"
        target     = last - (recent_highs[0] - recent_lows[-1])
        invalidation = recent_highs[-1] + atr

    # Symmetrical Triangle: converging
    elif sh < -atr * 0.005 and sl > atr * 0.005:
        pattern    = "Symmetrical Triangle"
        completion = "65%"
        target     = last + (recent_highs[0] - recent_lows[0]) * 0.7
        invalidation = recent_lows[-1] - atr

    # Rectangle / Range
    elif abs(sh) < atr * 0.02 and abs(sl) < atr * 0.02:
        band = recent_highs[-1] - recent_lows[-1]
        if band < atr * 4:
            pattern    = "Rectangle / Range"
            completion = "N/A – Waiting breakout"
            target     = recent_highs[-1] + band
            invalidation = recent_lows[-1] - atr * 0.5

    # Bull Flag: strong uptrend then tight pullback
    if pattern == "None":
        prior_move = close.iloc[-20] - close.iloc[-40] if len(df) >= 40 else 0
        recent_move= close.iloc[-1]  - close.iloc[-20]
        if prior_move > atr * 5 and abs(recent_move) < prior_move * 0.4:
            pattern    = "Bull Flag"
            completion = "Flag formed – watch breakout"
            target     = last + prior_move
            invalidation = dff["Low"].tail(10).min() - atr * 0.5

    return {
        "pattern":      pattern,
        "completion":   completion,
        "target":       round(target, 2),
        "invalidation": round(invalidation, 2),
    }
