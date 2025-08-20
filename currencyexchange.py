import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import coint
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import argparse

def parse_pairs(pairs_str):
    if not pairs_str:
        return None
    return [p.strip() for p in pairs_str.split(",")]

def load_currency_data(csv_file, date_col='Date', pairs=None):
    df = pd.read_csv(csv_file)
    
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not found in CSV")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.set_index(date_col)
    
    if pairs is None:
        pairs = df.select_dtypes(include=[np.number]).columns.tolist()

    clean_map = {}
    for col in pairs:
        if "EUR" in col:
            clean_map[col] = "EUR"
        elif "GBP" in col:
            clean_map[col] = "GBP"
        elif "USD" in col:
            clean_map[col] = "USD"
        else:
            clean_map[col] = col
    df = df.rename(columns=clean_map)
    pairs = list(clean_map.values())
    
    rates = df[pairs].copy()
    return rates, pairs

# ------------------------------
# Modeling functions
# ------------------------------

def forecast_arima(series, horizon=10):
    series = series.dropna()
    if len(series) < 10:
        return pd.Series([np.nan]*horizon, index=pd.date_range(series.index[-1], periods=horizon+1, freq='D')[1:])
    model = ARIMA(series, order=(1,0,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=horizon)
    return forecast

def factor_model_forecast(series, factors=None, horizon=10):
    series = series.dropna()
    last_value = series.iloc[-1] if len(series) > 0 else np.nan
    return pd.Series([last_value]*horizon, index=pd.date_range(series.index[-1], periods=horizon+1, freq='D')[1:])

def ppp_cointegration(series1, series2):
    series1 = series1.dropna()
    series2 = series2.dropna()
    min_len = min(len(series1), len(series2))
    if min_len < 10:
        return None
    score, pvalue, _ = coint(series1[-min_len:], series2[-min_len:])
    return pvalue

def regime_switching(series, k_regimes=2):
    series = series.dropna()
    if len(series) < 20:
        return None
    try:
        mod = MarkovRegression(series, k_regimes=k_regimes, trend='c', switching_variance=True)
        res = mod.fit(em_iter=10, search_reps=5, disp=False)
        return res
    except:
        return None

def garch_volatility(series):
    series = series.dropna()
    if len(series) < 10:
        return None
    am = arch_model(series, vol='Garch', p=1, q=1)
    res = am.fit(disp='off')
    return res

def stochastic_volatility(series):
    return garch_volatility(series)

def realized_volatility(series):
    return series.rolling(5).std()

def correlation_forecast(rates):
    return rates.corr()

# ------------------------------
# Plotting functions
# ------------------------------

def plot_forecast(series, forecast, title, filepath):
    plt.figure(figsize=(10,5))
    plt.plot(series, label='Historical')
    plt.plot(forecast, label='Forecast', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_volatility(series, vol_series, title, filepath):
    plt.figure(figsize=(10,5))
    plt.plot(series, label='Exchange Rate')
    plt.plot(vol_series, label='Volatility', linestyle='--')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# ------------------------------
# Main pipeline
# ------------------------------

def run_pipeline(args):
    print(f"Loading data from {args.csv} ...")
    rates, pairs = load_currency_data(args.csv, args.date_col, parse_pairs(args.pairs) if args.pairs else None)
    print(f"Detected pairs: {pairs}")
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # ARIMA forecasts
    print("Running ARIMA forecasts...")
    for pair in pairs:
        forecast = forecast_arima(rates[pair], horizon=args.horizon)
        forecast.to_csv(os.path.join(args.outdir, f"{pair}_arima_forecast.csv"))
        plot_forecast(rates[pair], forecast, f"{pair} ARIMA Forecast", os.path.join(args.outdir, f"{pair}_arima_forecast.png"))
        print(f"{pair} ARIMA forecast & plot saved.")
    
    # Factor model forecasts
    print("Running Factor Model forecasts (placeholder)...")
    for pair in pairs:
        forecast = factor_model_forecast(rates[pair], horizon=args.horizon)
        forecast.to_csv(os.path.join(args.outdir, f"{pair}_factor_forecast.csv"))
        plot_forecast(rates[pair], forecast, f"{pair} Factor Forecast", os.path.join(args.outdir, f"{pair}_factor_forecast.png"))
        print(f"{pair} Factor forecast & plot saved.")
    
    # PPP cointegration
    print("Running PPP cointegration tests...")
    for i in range(len(pairs)):
        for j in range(i+1, len(pairs)):
            pvalue = ppp_cointegration(rates[pairs[i]], rates[pairs[j]])
            if pvalue is not None:
                print(f"PPP cointegration p-value {pairs[i]}-{pairs[j]}: {pvalue:.4f}")
    
    # Regime switching
    print("Running Regime-Switching models...")
    for pair in pairs:
        res = regime_switching(rates[pair])
        if res is not None:
            summary_file = os.path.join(args.outdir, f"{pair}_regime_summary.txt")
            with open(summary_file, "w") as f:
                f.write(res.summary().as_text())
            print(f"{pair} Regime-switching summary saved.")
    
    # Volatility
    print("Running GARCH and stochastic volatility...")
    for pair in pairs:
        garch_res = garch_volatility(rates[pair])
        if garch_res is not None:
            garch_file = os.path.join(args.outdir, f"{pair}_garch_summary.txt")
            with open(garch_file, "w") as f:
                f.write(garch_res.summary().as_text())
            print(f"{pair} GARCH summary saved.")
            plot_volatility(rates[pair], garch_res.conditional_volatility, f"{pair} GARCH Volatility", os.path.join(args.outdir, f"{pair}_garch_volatility.png"))
        
        sv_res = stochastic_volatility(rates[pair])
    
    # Correlation
    print("Calculating correlation matrix...")
    corr = correlation_forecast(rates)
    corr.to_csv(os.path.join(args.outdir, "correlation_matrix.csv"))
    print("Correlation matrix saved.")
    
    print("Pipeline finished.")

# ------------------------------
# Argument parser
# ------------------------------

def build_argparser():
    parser = argparse.ArgumentParser(description="Currency Exchange Rate Modeling")
    parser.add_argument("--csv", type=str, required=True, help="CSV file with currency data")
    parser.add_argument("--date_col", type=str, default="Date", help="Column name for dates")
    parser.add_argument("--pairs", type=str, help="Comma-separated list of currency columns")
    parser.add_argument("--horizon", type=int, default=10, help="Forecast horizon")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory")
    return parser

if __name__ == '__main__':
    parser = build_argparser()
    
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--csv', 'currency.csv',
            '--date_col', 'Date',
            '--horizon', '10',
            '--outdir', 'outputs'
        ])
    
    args = parser.parse_args()
    run_pipeline(args)

