# FX Exchange Rate Modelling

This project provides ready-to-run utilities for exchange rate forecasting and volatility/correlation modelling across both major and exotic FX pairs.

Key features
- ARIMA forecasting for short-term movements
- Factor models using interest-rate differentials and other fundamentals
- PPP long-run equilibrium with cointegration and error-correction
- Regime-switching models for trending vs mean-reverting periods
- GARCH/EGARCH volatility models
- Optional Bayesian stochastic volatility (PyMC) or EGARCH fallback
- Realized volatility estimation from intraday data
- DCC correlation forecasting for multi-currency portfolios

Data sources supported
- Yahoo Finance (via `yfinance`)
- Alpha Vantage (requires API key)
- Nasdaq Data Link (requires API key)
- IMF/IFS (via `pandasdmx`, best-effort)

## Quickstart

1) Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) (Optional) Configure API keys by copying `.env.example` to `.env` and updating values.

3) Run a quick demo
```bash
python main.py demo --pair EURUSD=X --start 2018-01-01
```

4) Examples
- ARIMA forecast
```bash
python main.py arima --pair EURUSD=X --start 2018-01-01 --horizon 10
```
- GARCH volatility
```bash
python main.py garch --pair EURUSD=X --start 2018-01-01 --dist studentst
```
- Regime switching
```bash
python main.py regime --pair EURUSD=X --start 2018-01-01
```
- DCC correlation (EURUSD, GBPUSD)
```bash
python main.py dcc --pairs EURUSD=X GBPUSD=X --start 2018-01-01
```

- PPP (requires price indices time series csvs)
```bash
python main.py ppp --fx EURUSD=X --domestic_cpi data/CPI_EA.csv --foreign_cpi data/CPI_US.csv --start 2000-01-01
```

- Factor model (requires fundamentals CSV)
```bash
python main.py factor --pair EURUSD=X --fundamentals data/factors.csv --start 2018-01-01
```

## Notes
- Yahoo Finance FX tickers typically use the suffix `=X` (e.g., `EURUSD=X`, `USDJPY=X`).
- Alpha Vantage & Nasdaq Data Link require API keys. Place them in `.env` as `ALPHAVANTAGE_API_KEY` and `NASDAQ_DATA_LINK_API_KEY`.
- For realized volatility, provide a CSV with intraday timestamps and prices. See `scripts/rv_example.csv` for a template.

## Disclaimer
This software is for research and educational purposes. It is not financial advice. Use at your own risk.