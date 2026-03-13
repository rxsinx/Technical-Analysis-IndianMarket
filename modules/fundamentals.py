"""
Fundamental Analysis Module
Fetches and computes MarketSmith-style metrics using yfinance.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf


@st.cache_data(ttl=600)
def get_fundamentals(symbol: str, _df_price: pd.DataFrame) -> dict:
    return _get_fundamentals_impl(symbol, _df_price)


def _get_fundamentals_impl(symbol: str, df_price: pd.DataFrame) -> dict:
    """
    Fetch comprehensive fundamental data for a symbol.
    Returns a dict with all sections needed for the Fundamentals panel.
    """
    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info or {}

        # ── Ratings & Scores ─────────────────────────────────────────────────
        scores = _compute_scores(df_price, info)

        # ── Key Ratios ───────────────────────────────────────────────────────
        ratios = _key_ratios(info)

        # ── Quarterly Earnings ───────────────────────────────────────────────
        quarterly = _quarterly_earnings(ticker, info)

        # ── Annual EPS + Price ───────────────────────────────────────────────
        annual = _annual_eps_price(ticker, df_price, info)

        # ── Income / Balance summary ─────────────────────────────────────────
        financials = _financial_summary(ticker, info)

        # ── CAN SLIM Checklist ───────────────────────────────────────────────
        canslim = _canslim_check(scores, ratios, quarterly, annual)

        return {
            "scores":     scores,
            "ratios":     ratios,
            "quarterly":  quarterly,
            "annual":     annual,
            "financials": financials,
            "canslim":    canslim,
            "info":       info,
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
def _compute_scores(df: pd.DataFrame, info: dict) -> dict:
    """Compute proprietary-style scores (0–99) similar to MarketSmith."""
    close = df["Close"]
    vol   = df["Volume"]
    scores = {}

    # EPS Rating (0–99): based on EPS growth vs universe proxy
    eps_ttm  = info.get("trailingEps", 0) or 0
    eps_fwd  = info.get("forwardEps",  0) or 0
    eps_growth = ((eps_fwd - eps_ttm) / abs(eps_ttm) * 100) if eps_ttm != 0 else 0
    scores["eps_rating"] = min(99, max(1, int(50 + eps_growth * 0.4)))

    # Price Strength Rating (RS): 52-week price performance rank (proxy)
    ret_1y = ((close.iloc[-1] / close.iloc[0]) - 1) * 100 if len(close) >= 252 else 0
    scores["rs_rating"] = min(99, max(1, int(50 + ret_1y * 0.3)))

    # Accumulation/Distribution (A–E)
    # Proxy: ratio of up-vol to down-vol over last 50 days
    last50 = df.tail(50)
    up_vol   = last50.loc[last50["Close"] >= last50["Open"], "Volume"].sum()
    down_vol = last50.loc[last50["Close"] <  last50["Open"], "Volume"].sum()
    ud_ratio = up_vol / down_vol if down_vol > 0 else 1.0
    if ud_ratio >= 1.5:   acc_dis = "A"
    elif ud_ratio >= 1.1: acc_dis = "B"
    elif ud_ratio >= 0.9: acc_dis = "C"
    elif ud_ratio >= 0.7: acc_dis = "D"
    else:                  acc_dis = "E"
    scores["acc_dis"]  = acc_dis
    scores["ud_ratio"] = round(ud_ratio, 2)

    # Group Rank (1–197 proxy): use sector momentum
    scores["group_rank"] = 42   # placeholder — requires universe data

    # Master Score (composite)
    master = int((scores["eps_rating"] + scores["rs_rating"]) / 2)
    scores["master"] = master

    # EPS Growth Rate (TTM vs prior year)
    scores["eps_growth_rate"] = round(eps_growth, 1)

    # RS 4-week, 13-week, 26-week, 52-week returns
    def pct(n):
        if len(close) > n:
            return round(((close.iloc[-1] / close.iloc[-n]) - 1) * 100, 1)
        return None

    scores["ret_4w"]  = pct(20)
    scores["ret_13w"] = pct(65)
    scores["ret_26w"] = pct(130)
    scores["ret_52w"] = pct(252)

    return scores


def _key_ratios(info: dict) -> dict:
    def g(k, default="N/A"):
        v = info.get(k)
        return v if v is not None else default

    def fmt_cr(v):
        """Format large numbers as Cr (crores) for Indian context."""
        if isinstance(v, (int, float)) and v > 0:
            cr = v / 1e7
            if cr >= 1000:
                return f"₹{cr/100:.1f}K Cr"
            return f"₹{cr:.0f} Cr"
        return "N/A"

    mkt_cap   = g("marketCap", 0)
    revenue   = g("totalRevenue", 0)
    net_inc   = g("netIncomeToCommon", 0)
    total_dbt = g("totalDebt", 0)
    equity    = g("bookValue", 0)

    pe    = g("trailingPE")
    fwd_pe= g("forwardPE")
    pb    = g("priceToBook")
    ps    = g("priceToSalesTrailing12Months")
    roe   = g("returnOnEquity")
    roa   = g("returnOnAssets")
    npm   = g("profitMargins")
    gpm   = g("grossMargins")
    opm   = g("operatingMargins")
    debt_eq = g("debtToEquity")
    curr_r  = g("currentRatio")
    quick_r = g("quickRatio")
    beta    = g("beta")
    div_yld = g("dividendYield")
    payout  = g("payoutRatio")
    shares  = g("sharesOutstanding", 0)
    float_s = g("floatShares", 0)
    inst_own= g("institutionPercentHeld")
    prom_own= g("heldPercentInsiders")
    eps_ttm = g("trailingEps")
    eps_fwd = g("forwardEps")
    bv      = g("bookValue")
    cf      = g("operatingCashflow", 0)
    fcf     = g("freeCashflow", 0)
    ev      = g("enterpriseValue", 0)
    ev_ebit = g("enterpriseToEbitda")
    rev_gr  = g("revenueGrowth")
    earn_gr = g("earningsGrowth")

    return {
        "market_cap":    fmt_cr(mkt_cap),
        "revenue":       fmt_cr(revenue),
        "net_income":    fmt_cr(net_inc),
        "pe":            round(pe, 1)     if isinstance(pe, float)     else pe,
        "fwd_pe":        round(fwd_pe, 1) if isinstance(fwd_pe, float) else fwd_pe,
        "pb":            round(pb, 2)     if isinstance(pb, float)     else pb,
        "ps":            round(ps, 2)     if isinstance(ps, float)     else ps,
        "roe":           f"{roe*100:.1f}%" if isinstance(roe, float)   else "N/A",
        "roa":           f"{roa*100:.1f}%" if isinstance(roa, float)   else "N/A",
        "net_margin":    f"{npm*100:.1f}%" if isinstance(npm, float)   else "N/A",
        "gross_margin":  f"{gpm*100:.1f}%" if isinstance(gpm, float)   else "N/A",
        "op_margin":     f"{opm*100:.1f}%" if isinstance(opm, float)   else "N/A",
        "debt_equity":   round(debt_eq/100, 2) if isinstance(debt_eq, float) else "N/A",
        "current_ratio": round(curr_r, 2) if isinstance(curr_r, float) else "N/A",
        "quick_ratio":   round(quick_r, 2) if isinstance(quick_r, float) else "N/A",
        "beta":          round(beta, 2)   if isinstance(beta, float)   else "N/A",
        "div_yield":     f"{div_yld*100:.2f}%" if isinstance(div_yld, float) else "0%",
        "payout_ratio":  f"{payout*100:.1f}%" if isinstance(payout, float)  else "N/A",
        "shares_out":    f"{shares/1e7:.2f} Cr" if isinstance(shares, (int,float)) and shares > 0 else "N/A",
        "float_shares":  f"{float_s/1e7:.2f} Cr" if isinstance(float_s, (int,float)) and float_s > 0 else "N/A",
        "inst_hold":     f"{inst_own*100:.1f}%" if isinstance(inst_own, float) else "N/A",
        "prom_hold":     f"{prom_own*100:.1f}%" if isinstance(prom_own, float) else "N/A",
        "eps_ttm":       round(eps_ttm, 2) if isinstance(eps_ttm, float) else "N/A",
        "eps_fwd":       round(eps_fwd, 2) if isinstance(eps_fwd, float) else "N/A",
        "book_value":    round(bv, 2)     if isinstance(bv, float)     else "N/A",
        "op_cashflow":   fmt_cr(cf),
        "free_cashflow": fmt_cr(fcf),
        "ev":            fmt_cr(ev),
        "ev_ebitda":     round(ev_ebit, 1) if isinstance(ev_ebit, float) else "N/A",
        "rev_growth":    f"{rev_gr*100:.1f}%" if isinstance(rev_gr, float) else "N/A",
        "earn_growth":   f"{earn_gr*100:.1f}%" if isinstance(earn_gr, float) else "N/A",
    }


def _quarterly_earnings(ticker, info: dict) -> list:
    """Get last 8 quarters of EPS + Revenue."""
    rows = []
    try:
        qf = ticker.quarterly_financials
        qi = ticker.quarterly_income_stmt
        if qf is None or qf.empty:
            qf = qi
        if qf is None or qf.empty:
            return []

        # Revenue row
        rev_row = None
        for k in ["Total Revenue", "Revenue"]:
            if k in qf.index:
                rev_row = qf.loc[k]
                break

        # Net income / EPS proxy
        ni_row = None
        for k in ["Net Income", "Net Income Common Stockholders"]:
            if k in qf.index:
                ni_row = qf.loc[k]
                break

        shares = info.get("sharesOutstanding", 1) or 1
        cols   = qf.columns[:8]   # most recent 8 quarters

        for i, col in enumerate(cols):
            date_str = str(col)[:7]
            rev  = rev_row[col] if rev_row is not None else None
            ni   = ni_row[col]  if ni_row is not None  else None
            eps  = round(ni / shares, 2) if isinstance(ni, (int, float)) and shares > 0 else None

            # YoY % change
            eps_chg = None
            rev_chg = None
            if i + 4 < len(cols):
                prev_col = cols[i + 4]
                if ni_row is not None and isinstance(ni_row.get(prev_col), (int, float)) and ni_row[prev_col] != 0:
                    eps_chg = round((ni - ni_row[prev_col]) / abs(ni_row[prev_col]) * 100, 1) if isinstance(ni, (int, float)) else None
                if rev_row is not None and isinstance(rev_row.get(prev_col), (int, float)) and rev_row[prev_col] != 0:
                    rev_chg = round((rev - rev_row[prev_col]) / abs(rev_row[prev_col]) * 100, 1) if isinstance(rev, (int, float)) else None

            rows.append({
                "date":    date_str,
                "eps":     eps,
                "eps_chg": eps_chg,
                "rev_cr":  round(rev / 1e7, 1) if isinstance(rev, (int, float)) else None,
                "rev_chg": rev_chg,
            })
    except Exception:
        pass
    return rows[:8]


def _annual_eps_price(ticker, df: pd.DataFrame, info: dict) -> list:
    """Annual EPS with Hi/Lo price for last 5 years."""
    rows = []
    try:
        ann = ticker.income_stmt
        if ann is None or ann.empty:
            ann = ticker.financials
        if ann is None or ann.empty:
            return []

        shares = info.get("sharesOutstanding", 1) or 1
        ni_row = None
        for k in ["Net Income", "Net Income Common Stockholders"]:
            if k in ann.index:
                ni_row = ann.loc[k]
                break

        if ni_row is None:
            return []

        for col in ann.columns[:5]:
            year = str(col)[:4]
            ni   = ni_row[col]
            eps  = round(ni / shares, 2) if isinstance(ni, (int, float)) and shares > 0 else None

            # Price range for that year from df
            try:
                yr_df = df[df.index.year == int(year)]
                hi = round(yr_df["High"].max(), 2) if not yr_df.empty else None
                lo = round(yr_df["Low"].min(),  2) if not yr_df.empty else None
            except Exception:
                hi = lo = None

            rows.append({"year": year, "eps": eps, "hi": hi, "lo": lo})
    except Exception:
        pass
    return rows[:5]


def _financial_summary(ticker, info: dict) -> dict:
    """P&L + Balance Sheet summary."""
    try:
        ann = ticker.income_stmt
        if ann is None or ann.empty:
            ann = ticker.financials

        bs  = ticker.balance_sheet
        # cash_flow is the current yfinance attribute; cashflow is deprecated
        try:
            cf = ticker.cash_flow
        except Exception:
            try:
                cf = ticker.cashflow
            except Exception:
                cf = None

        def get_row(df, keys):
            if df is None or df.empty:
                return None
            for k in keys:
                if k in df.index:
                    return df.loc[k]
            return None

        def series_to_list(s, n=4):
            if s is None:
                return [None] * n
            vals = []
            for col in list(s.index)[:n]:
                v = s[col]
                vals.append(round(v / 1e7, 1) if isinstance(v, (int, float)) else None)
            while len(vals) < n:
                vals.append(None)
            return vals

        years = [str(c)[:4] for c in (ann.columns[:4] if ann is not None and not ann.empty else [])]

        rev   = series_to_list(get_row(ann, ["Total Revenue", "Revenue"]))
        ebit  = series_to_list(get_row(ann, ["EBIT", "Operating Income"]))
        ni    = series_to_list(get_row(ann, ["Net Income", "Net Income Common Stockholders"]))
        ocf   = series_to_list(get_row(cf,  ["Operating Cash Flow", "Total Cash From Operating Activities"]))
        fcf_s = series_to_list(get_row(cf,  ["Free Cash Flow"]))
        total_assets = series_to_list(get_row(bs, ["Total Assets"]))
        total_debt   = series_to_list(get_row(bs, ["Total Debt", "Long Term Debt"]))
        equity_s     = series_to_list(get_row(bs, ["Stockholders Equity", "Total Stockholder Equity",
                                                     "Common Stock Equity"]))

        return {
            "years":        years,
            "revenue":      rev,
            "ebit":         ebit,
            "net_income":   ni,
            "op_cashflow":  ocf,
            "free_cashflow":fcf_s,
            "total_assets": total_assets,
            "total_debt":   total_debt,
            "equity":       equity_s,
        }
    except Exception as e:
        return {"error": str(e)}


def _canslim_check(scores: dict, ratios: dict, quarterly: list, annual: list) -> list:
    """CAN SLIM style checklist items."""
    checks = []

    def add(label, value, condition, note=""):
        checks.append({
            "label":     label,
            "value":     str(value),
            "pass":      condition,
            "note":      note,
        })

    # C – Current Quarterly EPS growth ≥ 25%
    qeps_chg = quarterly[0]["eps_chg"] if quarterly and quarterly[0]["eps_chg"] is not None else None
    add("C – Current Qtr EPS Growth",
        f"{qeps_chg:+.1f}%" if qeps_chg is not None else "N/A",
        qeps_chg is not None and qeps_chg >= 25,
        "Target: ≥25% YoY")

    # A – Annual EPS growth ≥ 25% for 3 years
    ann_eps = [r["eps"] for r in annual if r["eps"] is not None]
    ann_ok  = len(ann_eps) >= 2 and all(ann_eps[i] > ann_eps[i+1] for i in range(min(2, len(ann_eps)-1)))
    add("A – Annual EPS Growth (3yr)", "Rising" if ann_ok else "Check",
        ann_ok, "3 years of rising EPS")

    # N – New product/service/mgmt — qualitative, flag as neutral
    add("N – New Catalyst", "Monitor", None, "Check for new products/mgmt")

    # S – Supply & Demand (volume)
    ud = scores.get("ud_ratio", 1.0)
    add("S – Supply/Demand (U/D Ratio)", f"{ud:.2f}",
        ud >= 1.1, "Target: >1.0 (more up-vol)")

    # L – Leader or Laggard (RS Rating)
    rs = scores.get("rs_rating", 0)
    add("L – Price Strength (RS Rating)", str(rs),
        rs >= 80, "Target: ≥80 (top 20%)")

    # I – Institutional Sponsorship
    inst = ratios.get("inst_hold", "N/A")
    inst_val = float(inst.replace("%","")) if isinstance(inst, str) and "%" in inst else 0
    add("I – Institutional Holding", inst,
        inst_val >= 30, "Target: ≥30%")

    # M – Market Direction (EPS rating proxy)
    eps_r = scores.get("eps_rating", 0)
    add("M – EPS Rating", str(eps_r),
        eps_r >= 80, "Target: ≥80")

    # Extra: ROE
    roe = ratios.get("roe", "N/A")
    roe_val = float(roe.replace("%","")) if isinstance(roe, str) and "%" in roe else 0
    add("ROE Quality", roe,
        roe_val >= 17, "Target: ≥17%")

    # Extra: Debt/Equity
    de = ratios.get("debt_equity", "N/A")
    try:
        de_val = float(de)
        add("Debt / Equity", str(de_val),
            de_val < 1.0, "Target: <1.0 (low leverage)")
    except Exception:
        add("Debt / Equity", str(de), None, "N/A")

    return checks
