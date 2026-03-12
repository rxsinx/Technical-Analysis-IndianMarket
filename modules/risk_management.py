import pandas as pd
import numpy as np


def calculate_risk(df: pd.DataFrame, tp: dict, capital: float, risk_pct: float) -> dict:
    """Calculate position sizing and risk metrics."""
    if not tp or df.empty:
        return {}

    entry     = tp.get("entry", df["Close"].iloc[-1])
    stop_loss = tp.get("stop_loss", entry * 0.97)
    target1   = tp.get("target1",  entry * 1.03)
    target2   = tp.get("target2",  entry * 1.06)
    direction = tp.get("direction", "LONG")

    risk_amount = capital * (risk_pct / 100)

    risk_per_share = abs(entry - stop_loss)
    if risk_per_share == 0:
        risk_per_share = entry * 0.01

    position_size = int(risk_amount / risk_per_share)
    notional      = position_size * entry

    # Reward
    reward1 = abs(target1 - entry) * position_size
    reward2 = abs(target2 - entry) * position_size
    rr1     = round(reward1 / risk_amount, 2) if risk_amount > 0 else 0
    rr2     = round(reward2 / risk_amount, 2) if risk_amount > 0 else 0

    # Max position size as % of capital
    allocation_pct = round(notional / capital * 100, 1) if capital > 0 else 0

    return {
        "risk_amount":    round(risk_amount, 2),
        "position_size":  position_size,
        "notional":       round(notional, 2),
        "allocation_pct": allocation_pct,
        "risk_per_share": round(risk_per_share, 2),
        "reward1":        round(reward1, 2),
        "reward2":        round(reward2, 2),
        "rr1":            rr1,
        "rr2":            rr2,
        "direction":      direction,
    }
