import pandas as pd
import numpy as np


def generate_trade_plan(df: pd.DataFrame, ms: dict, sr: dict, tr: dict) -> dict:
    """Generate entry, SL, and target levels."""
    if len(df) < 20:
        return {}

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    last   = close.iloc[-1]

    atr = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]

    trend      = tr.get("daily", "NEUTRAL")
    nearest_r  = ms.get("nearest_resistance", last * 1.03)
    nearest_s  = ms.get("nearest_support",    last * 0.97)
    demand_low = ms.get("demand_low",  last * 0.95)
    supply_high= ms.get("supply_high", last * 1.05)

    rr_ratio   = 2.0   # default

    if "UPTREND" in trend:
        # Long setup
        entry     = round(last, 2)
        stop_loss = round(max(nearest_s, last - 1.5 * atr), 2)
        risk_pts  = entry - stop_loss
        target1   = round(entry + risk_pts * 1.5, 2)
        target2   = round(entry + risk_pts * 2.5, 2)
        direction = "LONG"
        if risk_pts > 0:
            rr_ratio = round(risk_pts * 2.5 / risk_pts, 1)

    elif "DOWNTREND" in trend:
        # Short setup
        entry     = round(last, 2)
        stop_loss = round(min(nearest_r, last + 1.5 * atr), 2)
        risk_pts  = stop_loss - entry
        target1   = round(entry - risk_pts * 1.5, 2)
        target2   = round(entry - risk_pts * 2.5, 2)
        direction = "SHORT"
        if risk_pts > 0:
            rr_ratio = round(risk_pts * 2.5 / risk_pts, 1)
    else:
        # Range – trade to opposite side
        entry     = round(last, 2)
        stop_loss = round(last - atr, 2)
        target1   = round(nearest_r * 0.995, 2)
        target2   = round(supply_high * 0.99, 2)
        direction = "NEUTRAL"
        rr_ratio  = round((target1 - entry) / (entry - stop_loss), 1) if entry > stop_loss else 1.0

    return {
        "entry":     entry,
        "stop_loss": stop_loss,
        "target1":   target1,
        "target2":   target2,
        "rr_ratio":  rr_ratio,
        "direction": direction,
        "atr":       round(atr, 2),
    }
