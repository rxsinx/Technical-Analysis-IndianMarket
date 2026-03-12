import pandas as pd
import numpy as np


def detect_candlestick_patterns(df: pd.DataFrame) -> dict:
    """Detect common candlestick patterns."""
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    results = []

    for i in range(2, len(df)):
        body      = abs(c.iloc[i] - o.iloc[i])
        full_range= h.iloc[i] - l.iloc[i]
        upper_wick= h.iloc[i] - max(c.iloc[i], o.iloc[i])
        lower_wick= min(c.iloc[i], o.iloc[i]) - l.iloc[i]
        prev_body = abs(c.iloc[i-1] - o.iloc[i-1])
        prev_bull = c.iloc[i-1] > o.iloc[i-1]
        curr_bull = c.iloc[i] > o.iloc[i]

        patterns_found = []

        # Bullish Engulfing
        if not prev_bull and curr_bull and c.iloc[i] > o.iloc[i-1] and o.iloc[i] < c.iloc[i-1]:
            patterns_found.append(("Bullish Engulfing", "BULLISH"))

        # Bearish Engulfing
        if prev_bull and not curr_bull and c.iloc[i] < o.iloc[i-1] and o.iloc[i] > c.iloc[i-1]:
            patterns_found.append(("Bearish Engulfing", "BEARISH"))

        # Hammer (Pin Bar Bullish)
        if full_range > 0 and lower_wick >= 2 * body and upper_wick <= 0.3 * body:
            patterns_found.append(("Hammer / Pin Bar", "BULLISH"))

        # Shooting Star (Pin Bar Bearish)
        if full_range > 0 and upper_wick >= 2 * body and lower_wick <= 0.3 * body:
            patterns_found.append(("Shooting Star / Pin Bar", "BEARISH"))

        # Doji
        if full_range > 0 and body / full_range < 0.1:
            patterns_found.append(("Doji", "NEUTRAL"))

        # Inside Bar
        if h.iloc[i] <= h.iloc[i-1] and l.iloc[i] >= l.iloc[i-1]:
            patterns_found.append(("Inside Bar", "NEUTRAL"))

        # Marubozu (strong momentum)
        if full_range > 0 and body / full_range > 0.9:
            sig = "BULLISH" if curr_bull else "BEARISH"
            patterns_found.append(("Marubozu", sig))

        for pat, sig in patterns_found:
            results.append({
                "date":    df.index[i].strftime("%Y-%m-%d"),
                "pattern": pat,
                "signal":  sig,
                "close":   round(c.iloc[i], 2),
            })

    # Summary from last bar
    last_idx = len(df) - 1
    last_result = next((r for r in reversed(results) if r["date"] == df.index[last_idx].strftime("%Y-%m-%d")), None)
    if not last_result and results:
        last_result = results[-1]

    # Engulfing & pin bar summaries
    last_engulfing = next((r["pattern"] for r in reversed(results) if "Engulfing" in r["pattern"]), "None")
    last_pinbar    = next((r["pattern"] for r in reversed(results) if "Pin Bar" in r["pattern"]), "None")
    last_inside    = next(("YES" for r in reversed(results[-3:]) if "Inside Bar" in r["pattern"]), "No")

    return {
        "last_pattern": last_result["pattern"] if last_result else "None",
        "signal":       last_result["signal"]  if last_result else "Neutral",
        "engulfing":    last_engulfing,
        "pin_bar":      last_pinbar,
        "inside_bar":   last_inside,
        "history":      results[-20:],  # last 20 signals for table
    }
