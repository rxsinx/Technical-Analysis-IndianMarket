import pandas as pd
import numpy as np


def compute_indicators(df: pd.DataFrame) -> dict:
    """Compute all technical indicators."""
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]
    ind   = {}

    # ── EMAs ──────────────────────────────────────────────────
    ind["ema20"]  = close.ewm(span=20,  adjust=False).mean()
    ind["ema50"]  = close.ewm(span=50,  adjust=False).mean()
    ind["ema200"] = close.ewm(span=200, adjust=False).mean()

    # ── Bollinger Bands (20, 2) ────────────────────────────────
    bb_mid          = close.rolling(20).mean()
    bb_std          = close.rolling(20).std()
    ind["bb_upper"] = bb_mid + 2 * bb_std
    ind["bb_lower"] = bb_mid - 2 * bb_std
    ind["bb_mid"]   = bb_mid

    # ── RSI (14) ───────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs        = avg_gain / avg_loss.replace(0, np.nan)
    ind["rsi"] = 100 - (100 / (1 + rs))

    # ── MACD (12, 26, 9) ──────────────────────────────────────
    ema12             = close.ewm(span=12, adjust=False).mean()
    ema26             = close.ewm(span=26, adjust=False).mean()
    ind["macd"]       = ema12 - ema26
    ind["macd_signal"]= ind["macd"].ewm(span=9, adjust=False).mean()
    ind["macd_hist"]  = ind["macd"] - ind["macd_signal"]

    # ── ATR (14) ───────────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    ind["atr"] = tr.ewm(com=13, adjust=False).mean()

    # ── Stochastic (14, 3) ────────────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    k = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    ind["stoch_k"] = k
    ind["stoch_d"] = k.rolling(3).mean()

    # ── ADX (14) ──────────────────────────────────────────────
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm]  = 0
    minus_dm[minus_dm < plus_dm] = 0
    atr14     = tr.rolling(14).mean()
    plus_di   = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    minus_di  = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    ind["adx"]       = dx.rolling(14).mean()
    ind["plus_di"]   = plus_di
    ind["minus_di"]  = minus_di

    # ── VWAP ──────────────────────────────────────────────────
    tp_vwap     = (high + low + close) / 3
    cum_vol     = vol.cumsum()
    cum_tp_vol  = (tp_vwap * vol).cumsum()
    ind["vwap"] = cum_tp_vol / cum_vol.replace(0, np.nan)

    # ── OBV ───────────────────────────────────────────────────
    direction       = np.sign(close.diff())
    ind["obv"]      = (vol * direction).fillna(0).cumsum()

    return ind
