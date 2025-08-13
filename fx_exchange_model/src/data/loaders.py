import os
from typing import Optional

import numpy as np
import pandas as pd


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_from_yahoo(pair: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for Yahoo source. Install with `pip install yfinance`. ") from exc

    data = yf.download(pair, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise RuntimeError(f"No data returned by Yahoo for {pair}")
    df = data[["Adj Close" if "Adj Close" in data.columns else "Close"]].copy()
    df.columns = ["price"]
    return _ensure_datetime_index(df).dropna()


def load_from_alphavantage(pair: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY in environment to use Alpha Vantage source")

    try:
        from alpha_vantage.foreignexchange import ForeignExchange
    except ImportError as exc:
        raise RuntimeError("alpha_vantage package required. Install with `pip install alpha_vantage`. ") from exc

    # Alpha Vantage expects from_symbol/to_symbol
    if len(pair) == 8 and pair.endswith("=X"):
        base = pair[:3]
        quote = pair[3:6]
    elif len(pair) == 6:
        base = pair[:3]
        quote = pair[3:6]
    else:
        raise RuntimeError("Provide pair as 6 letters (e.g., EURUSD) or Yahoo format EURUSD=X")

    fe = ForeignExchange(key=key, output_format='pandas')
    ts, _meta = fe.get_currency_exchange_daily(from_symbol=base, to_symbol=quote, outputsize='full')
    ts.index = pd.to_datetime(ts.index)
    ts.sort_index(inplace=True)
    if start:
        ts = ts.loc[ts.index >= pd.to_datetime(start)]
    if end:
        ts = ts.loc[ts.index <= pd.to_datetime(end)]
    price = ts["4. close"].rename("price").to_frame()
    return price


def load_from_nasdaq(pair: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    key = os.getenv("NASDAQ_DATA_LINK_API_KEY")
    if not key:
        raise RuntimeError("Set NASDAQ_DATA_LINK_API_KEY in environment to use Nasdaq Data Link source")

    try:
        import nasdaqdatalink as ndl
    except ImportError as exc:
        raise RuntimeError("nasdaq-data-link package required. Install with `pip install nasdaq-data-link`. ") from exc

    ndl.ApiConfig.api_key = key
    # Users must specify the dataset code themselves if not using Yahoo format
    # For convenience, map Yahoo-like pairs to CURRFX datasets when possible (not exhaustive)
    dataset = None
    if pair.endswith("=X"):
        base = pair[:3]
        quote = pair[3:6]
        dataset = f"CURRFX/{base}{quote}"
    if dataset is None:
        dataset = pair  # allow passing explicit dataset code

    df = ndl.get(dataset)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    # Try common price columns
    for col in ["Rate", "Value", "Settle", "Close"]:
        if col in df.columns:
            price = df[[col]].rename(columns={col: "price"})
            break
    else:
        raise RuntimeError("Could not infer price column from Nasdaq data")

    if start:
        price = price.loc[price.index >= pd.to_datetime(start)]
    if end:
        price = price.loc[price.index <= pd.to_datetime(end)]
    return price


def load_fx_timeseries(pair: str, start: Optional[str] = None, end: Optional[str] = None, source: str = "yahoo") -> pd.DataFrame:
    source = source.lower()
    if source == "yahoo":
        return load_from_yahoo(pair, start, end)
    if source == "alphavantage":
        return load_from_alphavantage(pair, start, end)
    if source == "nasdaq":
        return load_from_nasdaq(pair, start, end)
    raise ValueError("source must be one of: yahoo, alphavantage, nasdaq")