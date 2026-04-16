# ◉ Signal — Professional Trading Analytics (Pro Edition)

A Streamlit trading dashboard built from a professional day trader's perspective, combining ML predictions with float-weighted breakout scoring, intraday VWAP analysis, and news catalyst detection.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Features

### Daily Breakout Scanner (0–100 base + 0–50 pro = 0–150 total)

**Base Score (technical, 100 pts):**
- Volume confirmation (25) — RVol 5d/20d
- Compression / TTM Squeeze (20) — Bollinger inside Keltner
- Momentum alignment (15) — RSI + ROC + MACD
- Accumulation evidence (15) — A/D line + MFI + OBV
- Proximity to 52w high (10)
- Ensemble ML conviction (15)

**Pro Layer (trader data, 50 pts):**
- **Float Tier (15)** — Micro/Low/Medium/Large. Lower float = bigger breakouts.
- **Float Turnover (10)** — Daily volume as % of float. 50%+ = institutional rotation.
- **Short Squeeze Potential (10)** — % short of float. 20%+ = squeeze setup.
- **News Catalysts (10)** — Sentiment-classified high-importance news from past 3 days.
- **Intraday Structure (5)** — VWAP position + opening range break + new day highs.

### Intraday Scanner (5-minute bars)
Real-time session analysis with VWAP line, opening range breakouts, and cumulative volume vs. average. Sorted by intraday % gain.

### Options Plays
8 strategies auto-selected based on direction + volatility regime. Full Greeks, 90% confidence intervals via log-normal projection, Black-Scholes pricing.

### News Feed
Finnhub-powered news with keyword-based sentiment classification (bullish/bearish) and importance tagging (FDA, earnings, M&A, etc.).

---

## Setup

### 1. Free Finnhub API Key (for news + company profile)

1. Sign up at **[finnhub.io](https://finnhub.io)** — free tier is 60 req/min, no expiry
2. Copy your API key

### 2. Add Key to Streamlit Cloud

In your deployed app: **Settings → Secrets** → add:
```
FINNHUB_API_KEY = "your_key_here"
```

### 3. Or Set Locally

```bash
export FINNHUB_API_KEY=your_key_here
streamlit run app.py
```

The app works without the key — news and company profile features will just be disabled (score tops out at 135/150 instead of 150).

---

## Refresh Intervals

Configurable from the sidebar: 30s, 1m, 2m, 5m, or 15m.
Cache TTLs auto-adjust to match so data is always fresh on each tick.

---

## Architecture

```
signal/
├── app.py                    # Main Streamlit app (6 tabs)
├── models/
│   ├── predictor.py          # Ensemble ML (GBT + RF) + base breakout scoring
│   ├── pro_scorer.py         # Pro breakout enhancement (float/short/news/intraday)
│   └── options_engine.py     # Options plays w/ Black-Scholes + 90% CI
├── utils/
│   ├── data.py               # Batch yfinance download + caching
│   ├── features.py           # 44 technical indicators
│   ├── catalysts.py          # Finnhub news + company profile
│   └── intraday.py           # 5-min bars + VWAP + ORB detection
├── .streamlit/config.toml
├── requirements.txt
├── runtime.txt               # python-3.12.8
├── packages.txt              # build-essential, python3-dev
└── README.md
```

---

## Deployment to Streamlit Cloud

```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

Then go to **share.streamlit.io** → New App → select your repo → `app.py` → Deploy.
Don't forget to add `FINNHUB_API_KEY` in Settings → Secrets.

---

## Disclaimers

This is an **educational tool**. Not financial advice. All predictions are probabilistic. Breakout scores reflect historical patterns, not guaranteed outcomes. Options pricing is estimated from Black-Scholes — actual market prices differ. Always do your own research and consult a licensed financial advisor.
