import pandas as pd
import numpy as np


def find_demand_supply_zones(df: pd.DataFrame) -> dict:
    """Identify demand and supply zones from sharp moves."""
    zones = []
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    last  = close.iloc[-1]

    atr = (high - low).rolling(14).mean().iloc[-1]
    threshold = atr * 1.5

    for i in range(2, len(df) - 1):
        body = abs(close.iloc[i] - df["Open"].iloc[i])
        move = abs(close.iloc[i] - close.iloc[i-1])

        if move >= threshold and body >= atr * 0.7:
            if close.iloc[i] > df["Open"].iloc[i]:
                # Bullish impulse → demand zone at base
                z_low  = low.iloc[i-1]
                z_high = df["Open"].iloc[i]
                zones.append({
                    "type":  "demand",
                    "low":   round(z_low, 2),
                    "high":  round(z_high, 2),
                    "index": i,
                    "fresh": z_high > last,
                })
            else:
                # Bearish impulse → supply zone at base
                z_low  = df["Open"].iloc[i]
                z_high = high.iloc[i-1]
                zones.append({
                    "type":  "supply",
                    "low":   round(z_low, 2),
                    "high":  round(z_high, 2),
                    "index": i,
                    "fresh": z_low < last,
                })

    # Keep latest zones, filter active ones
    demand_zones = [z for z in zones if z["type"] == "demand"][-3:]
    supply_zones = [z for z in zones if z["type"] == "supply"][-3:]
    all_zones    = demand_zones + supply_zones

    top_demand = demand_zones[-1] if demand_zones else {"low": last*0.95, "high": last*0.96}
    top_supply = supply_zones[-1] if supply_zones else {"low": last*1.04, "high": last*1.05}

    return {
        "zones": all_zones,
        "summary": {
            "demand_low":   top_demand["low"],
            "demand_high":  top_demand["high"],
            "supply_low":   top_supply["low"],
            "supply_high":  top_supply["high"],
            "zone_strength":"STRONG" if len(all_zones) > 3 else "MODERATE",
            "zone_tested":  f"{len(demand_zones)} demand | {len(supply_zones)} supply",
        }
    }
