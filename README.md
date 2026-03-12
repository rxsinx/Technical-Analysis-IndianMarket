# 📈 TA Terminal — Technical Analysis System

A professional-grade **Technical Analysis Terminal** built with Streamlit.  
Features all 16 TA modules — from chart setup to final trade result.

![Python](https://img.shields.io/badge/Python-3.9%2B-green?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🚀 Deploy on Streamlit Cloud (GitHub)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: TA Terminal"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ta-terminal.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **New app**
3. Connect your GitHub repository
4. Set **Main file path** to: `app.py`
5. Click **Deploy**

✅ Your app will be live at `https://YOUR_USERNAME-ta-terminal.streamlit.app`

---

## 💻 Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/ta-terminal.git
cd ta-terminal

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

---

## 📦 Project Structure

```
ta-terminal/
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Theme config
└── modules/
    ├── __init__.py
    ├── indicators.py          # EMA, RSI, MACD, ATR, Bollinger, ADX
    ├── market_structure.py    # HH/HL, LH/LL, swing points
    ├── trend_analysis.py      # Trend direction, strength
    ├── support_resistance.py  # S/R levels, pivot points
    ├── demand_supply.py       # Institutional zones
    ├── candlestick_patterns.py# Engulfing, pin bar, inside bar
    ├── breakouts.py           # Breakout / breakdown detection
    ├── volume_analysis.py     # Volume confirmation, climax
    ├── chart_patterns.py      # Triangles, flags, rectangles
    ├── multi_timeframe.py     # Monthly / Weekly / Daily / 4H
    ├── trade_planning.py      # Entry, SL, Target, R:R
    ├── risk_management.py     # Position sizing, risk $
    └── utils.py               # Formatting helpers
```

---

## 📊 16 TA Modules

| # | Module | Description |
|---|--------|-------------|
| 01 | Chart Setup | Multi-timeframe chart configuration |
| 02 | Market Structure | HH/HL, LH/LL swing analysis |
| 03 | Trend Analysis | EMA stack, ADX, direction |
| 04 | Support & Resistance | Key levels, pivot points |
| 05 | Demand & Supply | Institutional order zones |
| 06 | Candlestick Behaviour | Engulfing, Pin Bar, Inside Bar |
| 07 | Breakouts & Breakdowns | Level breaks with confirmation |
| 08 | Volume Analysis | Climax, dry pullback, participation |
| 09 | Chart Patterns | Triangles, flags, rectangles |
| 10 | Multi-Timeframe | Monthly → 4H alignment |
| 11 | Trade Planning | Entry zone, SL, T1, T2, R:R |
| 12 | Risk Management | Position sizing, risk $ |
| 13 | Trade Execution | Checklist & discipline |
| 14 | Common Traps | Fake breakout, late entry |
| 15 | Post-Trade Review | Journal checklist |
| 16 | Final Result | Overall bias & outcome |

---

## ⚠️ Disclaimer

This tool is for **educational purposes only**.  
It is **not financial advice**. Always do your own research and manage your risk.

---

## 📄 License

MIT License — free to use, modify, and distribute.
