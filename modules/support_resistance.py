import pandas as pd
import numpy as np


def find_support_resistance(df: pd.DataFrame, n_levels: int = 10) -> dict:
    """Find key S/R levels using pivot points and price clusters."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    last  = close.iloc[-1]

    # Classic pivot (last complete bar)
    ph = high.iloc[-2]
    pl = low.iloc[-2]
    pc = close.iloc[-2]
    pivot = (ph + pl + pc) / 3
    r1 = 2 * pivot - pl
    r2 = pivot + (ph - pl)
    r3 = ph + 2 * (pivot - pl)
    s1 = 2 * pivot - ph
    s2 = pivot - (ph - pl)
    s3 = pl - 2 * (ph - pivot)

    # Swing-based levels (last 100 bars)
    window = min(100, len(df))
    dff    = df.tail(window)
    price_range  = dff["High"].max() - dff["Low"].min()
    tolerance    = price_range * 0.005

    candidates = list(dff["High"]) + list(dff["Low"])
    levels_raw  = []
    for price in candidates:
        near = [p for p in candidates if abs(p - price) <= tolerance]
        if len(near) >= 2:
            levels_raw.append(np.mean(near))

    # Deduplicate
    levels_raw = sorted(set(round(l, 2) for l in levels_raw))
    merged = []
    for l in levels_raw:
        if not merged or abs(l - merged[-1]) > tolerance * 2:
            merged.append(l)

    # Tag as S or R
    tagged = []
    for price in merged:
        t = "R" if price > last else "S"
        tagged.append({"price": price, "type": t})

    # Add pivots
    pivot_levels = [
        {"price": round(r3,2), "type":"R", "label":"R3"},
        {"price": round(r2,2), "type":"R", "label":"R2"},
        {"price": round(r1,2), "type":"R", "label":"R1"},
        {"price": round(pivot,2), "type":"P", "label":"Pivot"},
        {"price": round(s1,2), "type":"S", "label":"S1"},
        {"price": round(s2,2), "type":"S", "label":"S2"},
        {"price": round(s3,2), "type":"S", "label":"S3"},
    ]

    all_levels = sorted(tagged + pivot_levels, key=lambda x: abs(x["price"] - last))[:n_levels]

    nearest_r = next((l["price"] for l in all_levels if l["type"] == "R"), last * 1.03)
    nearest_s = next((l["price"] for l in all_levels if l["type"] == "S"), last * 0.97)

    return {
        "levels": all_levels,
        "summary": {
            "nearest_resistance": round(nearest_r, 2),
            "nearest_support":    round(nearest_s, 2),
            "pivot":              round(pivot, 2),
            "r1": round(r1,2), "r2": round(r2,2),
            "s1": round(s1,2), "s2": round(s2,2),
        }
    }
