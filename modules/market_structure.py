import pandas as pd
import numpy as np


def find_swing_points(df: pd.DataFrame, window: int = 5):
    """Identify swing highs and lows using a rolling window."""
    highs = df["High"]
    lows  = df["Low"]
    n     = len(df)

    swing_high_idx = []
    swing_low_idx  = []

    for i in range(window, n - window):
        if highs.iloc[i] == highs.iloc[i - window: i + window + 1].max():
            swing_high_idx.append(i)
        if lows.iloc[i] == lows.iloc[i - window: i + window + 1].min():
            swing_low_idx.append(i)

    return swing_high_idx, swing_low_idx


def analyze_market_structure(df: pd.DataFrame) -> dict:
    """Full market structure analysis."""
    if len(df) < 20:
        return {}

    swing_high_idx, swing_low_idx = find_swing_points(df)

    sh_prices = [df["High"].iloc[i]  for i in swing_high_idx[-6:]]
    sl_prices = [df["Low"].iloc[i]   for i in swing_low_idx[-6:]]

    # Determine HH/HL or LH/LL
    def classify(prices):
        if len(prices) < 2:
            return "INSUFFICIENT DATA"
        trend = []
        for i in range(1, len(prices)):
            trend.append("UP" if prices[i] > prices[i-1] else "DOWN")
        ups   = trend.count("UP")
        downs = trend.count("DOWN")
        if ups > downs:
            return "HIGHER"
        elif downs > ups:
            return "LOWER"
        return "MIXED"

    highs_structure = classify(sh_prices)
    lows_structure  = classify(sl_prices)

    if highs_structure == "HIGHER" and lows_structure == "HIGHER":
        structure = "UPTREND (HH/HL)"
        last_break = "BULLISH CONTINUATION"
    elif highs_structure == "LOWER" and lows_structure == "LOWER":
        structure = "DOWNTREND (LH/LL)"
        last_break = "BEARISH CONTINUATION"
    elif highs_structure == "HIGHER" and lows_structure == "LOWER":
        structure = "EXPANSION / VOLATILE"
        last_break = "VOLATILE"
    else:
        structure = "CONSOLIDATION / RANGE"
        last_break = "RANGE BOUND"

    # Support / Resistance from swings
    close = df["Close"].iloc[-1]
    all_r  = sorted([p for p in sh_prices if p > close], key=lambda x: abs(x - close))
    all_s  = sorted([p for p in sl_prices if p < close], key=lambda x: abs(x - close))

    major_r       = max(sh_prices) if sh_prices else close * 1.05
    major_s       = min(sl_prices) if sl_prices else close * 0.95
    nearest_r     = all_r[0]  if all_r  else close * 1.03
    nearest_s     = all_s[0]  if all_s  else close * 0.97

    # Demand/Supply (simplified)
    demand_low    = major_s * 0.995
    demand_high   = major_s * 1.005
    supply_low    = major_r * 0.995
    supply_high   = major_r * 1.005

    return {
        "swing_high_idx":    swing_high_idx[-8:],
        "swing_low_idx":     swing_low_idx[-8:],
        "swing_highs":       f"{len(sh_prices)} points | {highs_structure}",
        "swing_lows":        f"{len(sl_prices)} points | {lows_structure}",
        "structure":         structure,
        "last_break":        last_break,
        "major_resistance":  round(major_r, 2),
        "major_support":     round(major_s, 2),
        "nearest_resistance":round(nearest_r, 2),
        "nearest_support":   round(nearest_s, 2),
        "demand_low":        round(demand_low, 2),
        "demand_high":       round(demand_high, 2),
        "supply_low":        round(supply_low, 2),
        "supply_high":       round(supply_high, 2),
        "zone_strength":     "STRONG" if len(sh_prices) >= 3 else "MODERATE",
        "zone_tested":       f"{min(len(sh_prices), 3)}x tested",
    }
