import pandas as pd
import numpy as np


def analyze_volume(df: pd.DataFrame) -> dict:
    """Volume analysis – trend, climax, dry pullback."""
    vol      = df["Volume"]
    close    = df["Close"]
    last_vol = vol.iloc[-1]
    avg_vol  = vol.rolling(20).mean().iloc[-1]
    ratio    = last_vol / avg_vol if avg_vol > 0 else 1.0

    # Format helper
    def fmt(v):
        if v >= 1e9: return f"{v/1e9:.1f}B"
        if v >= 1e6: return f"{v/1e6:.1f}M"
        if v >= 1e3: return f"{v/1e3:.0f}K"
        return str(int(v))

    # Signal
    if ratio >= 2.0:
        signal   = "CLIMAX VOLUME"
        climax   = "YES"
    elif ratio >= 1.5:
        signal   = "HIGH VOLUME BREAKOUT"
        climax   = "NO"
    elif ratio >= 1.1:
        signal   = "ABOVE AVERAGE"
        climax   = "NO"
    elif ratio >= 0.7:
        signal   = "AVERAGE / DRY PULLBACK"
        climax   = "NO"
    else:
        signal   = "WEAK PARTICIPATION"
        climax   = "NO"

    # Trend confirmation
    price_up  = close.iloc[-1] > close.iloc[-2]
    confirmed = (price_up and ratio >= 1.1) or (not price_up and ratio >= 1.1)

    # Volume trend (5-bar)
    vol_trend = "RISING" if vol.iloc[-5:].is_monotonic_increasing else (
                "FALLING" if vol.iloc[-5:].is_monotonic_decreasing else "MIXED")

    return {
        "today":     fmt(last_vol),
        "avg":       fmt(avg_vol),
        "ratio":     f"{ratio:.2f}x",
        "ratio_raw": ratio,
        "signal":    signal,
        "climax":    climax,
        "vol_trend": vol_trend,
        "confirmed": confirmed,
    }
