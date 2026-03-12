import pandas as pd
import numpy as np
import yfinance as yf


def _analyze_single_tf(df: pd.DataFrame) -> dict:
    """Analyze a single timeframe DataFrame."""
    if df is None or df.empty or len(df) < 10:
        return {"bias": "N/A", "trend": "N/A", "ema200_pos": "N/A", "rsi": "N/A", "volume": "N/A", "note": "Insufficient data"}

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    last   = close.iloc[-1]

    ema20  = close.ewm(span=20,  adjust=False).mean().iloc[-1]
    ema50  = close.ewm(span=50,  adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=min(200, len(df)-1), adjust=False).mean().iloc[-1]

    delta  = close.diff()
    gain   = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss   = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi    = (100 - 100 / (1 + rs)).iloc[-1]

    avg_vol  = volume.rolling(20).mean().iloc[-1]
    last_vol = volume.iloc[-1]
    vol_str  = "HIGH" if last_vol > avg_vol * 1.2 else ("LOW" if last_vol < avg_vol * 0.7 else "AVG")

    if last > ema20 > ema50:
        trend   = "UPTREND"
        bias    = "BULLISH"
    elif last < ema20 < ema50:
        trend   = "DOWNTREND"
        bias    = "BEARISH"
    else:
        trend   = "MIXED"
        bias    = "NEUTRAL"

    ema200_pos = "ABOVE EMA200" if last > ema200 else "BELOW EMA200"
    note = f"RSI {rsi:.0f} | Vol {vol_str}"

    return {
        "bias":       bias,
        "trend":      trend,
        "ema200_pos": ema200_pos,
        "rsi":        f"{rsi:.1f}",
        "volume":     vol_str,
        "note":       note,
    }


def multi_timeframe_analysis(symbol: str, period: str) -> dict:
    """Fetch and analyze monthly, weekly, daily, 4H."""
    tf_map = {
        "monthly": ("1mo", "2y"),
        "weekly":  ("1wk", "2y"),
        "daily":   ("1d",  period),
        "h4":      ("1h",  "60d"),
    }

    results = {}
    for key, (interval, per) in tf_map.items():
        try:
            ticker = yf.Ticker(symbol)
            df     = ticker.history(period=per, interval=interval, auto_adjust=True)
            if not df.empty:
                df = df[["Open","High","Low","Close","Volume"]].dropna()
            results[key] = _analyze_single_tf(df)
        except Exception:
            results[key] = {"bias": "ERROR", "trend": "N/A", "ema200_pos": "N/A", "rsi": "N/A", "volume": "N/A", "note": "Fetch error"}

    # Alignment
    biases = [results[k]["bias"] for k in ["monthly", "weekly", "daily"]]
    bull_count = biases.count("BULLISH")
    bear_count = biases.count("BEARISH")

    if bull_count >= 3:
        alignment = "FULLY BULLISH"
    elif bull_count == 2:
        alignment = "BULLISH"
    elif bear_count >= 3:
        alignment = "FULLY BEARISH"
    elif bear_count == 2:
        alignment = "BEARISH"
    else:
        alignment = "MIXED / NO EDGE"

    results["alignment"] = alignment
    return results
