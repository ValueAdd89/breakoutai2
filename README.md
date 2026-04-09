# ◉ Signal — Predictive Stock Analytics

A real-time stock signal prediction dashboard powered by ensemble ML models trained on 30+ technical indicators from live market data.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Architecture

```
signal/
├── app.py                  # Main Streamlit application (4 tabs)
├── models/
│   ├── predictor.py        # Ensemble ML prediction engine (GBT + RF)
│   └── options_engine.py   # Options play generator with 90% CI, B-S pricing
├── utils/
│   ├── data.py             # Market data fetching with TTL caching
│   └── features.py         # 30+ technical indicator feature pipeline
├── .streamlit/
│   └── config.toml         # Apple-inspired dark theme config
├── .github/
│   └── workflows/ci.yml    # GitHub Actions CI pipeline
├── requirements.txt
└── README.md
```

## How It Works

### Prediction Pipeline

1. **Data Ingestion** — Live OHLCV data via `yfinance` with 5-minute TTL caching
2. **Feature Engineering** — 30+ technical indicators computed via `ta`:
   - Trend: SMA(10/20/50), EMA(12/26), MACD, ADX
   - Momentum: RSI(7/14), Stochastic, Williams %R, ROC
   - Volatility: Bollinger Bands, ATR
   - Volume: OBV, MFI, relative volume
   - Price action: returns, volatility, candle ratios
3. **Ensemble Model** — Weighted combination of:
   - Gradient Boosted Trees (60% weight) — captures non-linear patterns
   - Random Forest (40% weight) — reduces overfitting, adds diversity
4. **Walk-Forward Validation** — 80/20 time-series split prevents lookahead bias
5. **Signal Detection** — Rule-based pattern recognition on latest indicators

### Options Play Engine

The options engine generates actionable trade recommendations with:

- **90% Confidence Interval** — Log-normal price projection using blended 20/60-day realized volatility with model-adjusted drift
- **Black-Scholes Pricing** — Analytical option pricing and full Greeks (Δ, Γ, Θ, ν)
- **Strategy Selection** — Automatically selects the optimal strategy based on signal direction + volatility regime:
  - Bullish + Low Vol → Long Call
  - Bullish + High Vol → Bull Call Spread
  - Bearish + Low Vol → Long Put
  - Bearish + High Vol → Bear Put Spread
  - High Vol + Range-bound → Iron Condor
  - Squeeze detected → Long Straddle
  - Bullish credit → Short Put Spread
  - Bearish credit → Short Call Spread
- **Entry/Exit Rules** — Specific timing guidance for entry, profit targets, stop losses
- **Risk/Reward Analysis** — Max loss, max gain, break-even, probability of profit
- **Reasoning Chain** — Every recommendation includes a full breakdown of the driving factors with color-coded impact levels
- **Risk Factors** — Explicit warnings for each play

### Performance Optimizations

| Optimization | Impact |
|---|---|
| `@st.cache_data(ttl=300)` on data fetching | Eliminates redundant API calls during 5-min refresh |
| `@st.cache_data(ttl=300)` on full prediction pipeline | Re-uses model outputs between tab switches |
| Session state signal storage | Persists predictions across Streamlit reruns |
| `streamlit-autorefresh` | Client-side refresh, no manual polling |
| Lightweight GBT (100 trees, depth 4) | Sub-second inference per ticker |
| Batch data fetching | Single loop with per-ticker caching |

---

## Deploy to Streamlit Community Cloud (Free)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit — Signal predictive analytics"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/stock-signal.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your `stock-signal` repository
4. Set:
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy**

Your app will be live at `https://YOUR_USERNAME-stock-signal.streamlit.app`

---

## Alternative Deployment Options

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t signal .
docker run -p 8501:8501 signal
```

### Railway / Render / Fly.io

All three support `requirements.txt` + a start command:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Set this as your **Start Command** in the platform dashboard.

### AWS / GCP / Azure

Use the Docker image above with any container service:
- AWS: ECS Fargate or App Runner
- GCP: Cloud Run
- Azure: Container Apps

---

## Local Development

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/stock-signal.git
cd stock-signal

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

Open `http://localhost:8501`

---

## Disclaimer

Signal is for **educational and research purposes only**. It is not financial advice.
Predictive models are probabilistic and carry inherent uncertainty. Past signals do
not guarantee future performance. Always consult a licensed financial advisor before
making investment decisions.
