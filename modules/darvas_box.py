import pandas as pd
import numpy as np


def detect_darvas_boxes(df: pd.DataFrame) -> dict:
    """
    Advanced Darvas Box detection.
    Rules:
      - Box Top: a high that is NOT exceeded for 3 consecutive bars
      - Box Bottom: a low that holds for 3 consecutive bars after top is set
      - Breakout: close > box_top
      - Entry: breakout close
      - Stop Loss: box_bottom
      - Target: entry + (box_top - box_bottom)  [1:1 projection]
    Returns current/last box + history of recent boxes.
    """
    if len(df) < 20:
        return _empty()

    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    dates  = df.index

    boxes = []
    i = 3
    while i < len(highs) - 3:
        # Candidate top: high[i] > all 3 bars before AND all 3 bars after
        if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i-3]
                and highs[i] >= highs[i+1] and highs[i] >= highs[i+2] and highs[i] >= highs[i+3]):
            box_top = highs[i]
            top_date = dates[i]

            # Look forward for box bottom (low that holds)
            j = i + 1
            box_bottom = None
            while j < min(i + 30, len(lows) - 3):
                if (lows[j] < lows[j-1] and lows[j] <= lows[j+1] and lows[j] <= lows[j+2]):
                    box_bottom = lows[j]
                    bottom_date = dates[j]
                    # Check box not too wide (max 20% of top)
                    if (box_top - box_bottom) / box_top < 0.20:
                        # Look for breakout
                        breakout_date = None
                        breakout_price = None
                        for k in range(j + 1, min(j + 40, len(closes))):
                            if closes[k] > box_top:
                                breakout_date  = dates[k]
                                breakout_price = closes[k]
                                break
                        width  = round(box_top - box_bottom, 2)
                        target = round(box_top + width, 2)
                        boxes.append({
                            "box_top":        round(box_top, 2),
                            "box_bottom":     round(box_bottom, 2),
                            "width":          width,
                            "width_pct":      round(width / box_top * 100, 2),
                            "top_date":       str(top_date)[:10],
                            "bottom_date":    str(bottom_date)[:10],
                            "breakout":       breakout_price is not None,
                            "breakout_price": round(breakout_price, 2) if breakout_price else None,
                            "breakout_date":  str(breakout_date)[:10] if breakout_date else None,
                            "entry":          round(box_top * 1.001, 2),   # 0.1% above box top
                            "stop_loss":      round(box_bottom * 0.999, 2),
                            "target":         target,
                            "rr":             round(width / (box_top - box_bottom * 0.999), 2) if width > 0 else 0,
                        })
                        i = j + 1
                        break
                j += 1
            else:
                i += 1
                continue
        else:
            i += 1

    if not boxes:
        return _empty()

    # Current box = last detected
    current = boxes[-1]
    last_close = closes[-1]

    # Status
    if current["breakout"]:
        status = "BREAKOUT ✓"
    elif last_close > current["box_top"]:
        status = "BREAKING OUT"
    elif last_close < current["box_bottom"]:
        status = "BROKEN DOWN"
    else:
        status = "INSIDE BOX"

    return {
        "box_top":        current["box_top"],
        "box_bottom":     current["box_bottom"],
        "width":          current["width"],
        "width_pct":      current["width_pct"],
        "entry":          current["entry"],
        "stop_loss":      current["stop_loss"],
        "target":         current["target"],
        "rr":             current["rr"],
        "status":         status,
        "breakout":       current["breakout"],
        "breakout_price": current["breakout_price"],
        "breakout_date":  current["breakout_date"],
        "top_date":       current["top_date"],
        "history":        boxes[-5:],   # last 5 boxes
        "total_boxes":    len(boxes),
    }


def _empty():
    return {
        "box_top": 0, "box_bottom": 0, "width": 0, "width_pct": 0,
        "entry": 0, "stop_loss": 0, "target": 0, "rr": 0,
        "status": "No Box Found", "breakout": False,
        "breakout_price": None, "breakout_date": None,
        "top_date": None, "history": [], "total_boxes": 0,
    }
