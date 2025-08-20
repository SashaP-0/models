import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.data.loaders import load_fx_timeseries
from src.models.arima import fit_arima_forecast
from src.models.volatility import fit_garch_volatility, fit_stochastic_volatility, compute_realized_volatility
from src.models.regime_switching import fit_markov_switching
from src.models.correlation import fit_dcc_correlation
from src.models.factor import fit_factor_model
from src.models.ppp import estimate_ppp_ecm


def _parse_date(date_str: str) -> str:
    if date_str is None:
        return None
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise SystemExit("Dates must be in YYYY-MM-DD format")


def cmd_demo(args: argparse.Namespace) -> None:
    pair = args.pair
    start = args.start
    end = args.end

    df = load_fx_timeseries(pair, start=start, end=end, source=args.source)
    print(df.tail())

    print("\n=== ARIMA Forecast ===")
    arima_res = fit_arima_forecast(df["price"], horizon=5)
    print(arima_res["summary"])  # type: ignore
    print("Forecast:")
    print(arima_res["forecast"])  # type: ignore

    print("\n=== GARCH Volatility ===")
    ret = np.log(df["price"]).diff().dropna() * 100
    garch_res = fit_garch_volatility(ret)
    print(garch_res["summary"])  # type: ignore

    if len(df) > 500:
        print("\n=== Regime Switching (2-state) ===")
        regime_res = fit_markov_switching(ret)
        print(regime_res["summary"])  # type: ignore

    if args.dcc:
        print("\n=== DCC Correlation (EURUSD & GBPUSD) ===")
        df2 = load_fx_timeseries("GBPUSD=X", start=start, end=end, source=args.source)
        ret1 = np.log(df["price"]).diff().dropna() * 100
        ret2 = np.log(df2["price"]).diff().dropna() * 100
        aligned = pd.concat([ret1.rename("EURUSD"), ret2.rename("GBPUSD")], axis=1).dropna()
        dcc_res = fit_dcc_correlation(aligned)
        print(dcc_res["summary"])  # type: ignore


def cmd_arima(args: argparse.Namespace) -> None:
    df = load_fx_timeseries(args.pair, start=args.start, end=args.end, source=args.source)
    res = fit_arima_forecast(df["price"], horizon=args.horizon)
    print(res["summary"])  # type: ignore
    print("Forecast:")
    print(res["forecast"])  # type: ignore


def cmd_garch(args: argparse.Namespace) -> None:
    df = load_fx_timeseries(args.pair, start=args.start, end=args.end, source=args.source)
    ret = np.log(df["price"]).diff().dropna() * 100
    res = fit_garch_volatility(ret, dist=args.dist)
    print(res["summary"])  # type: ignore


def cmd_sv(args: argparse.Namespace) -> None:
    df = load_fx_timeseries(args.pair, start=args.start, end=args.end, source=args.source)
    ret = np.log(df["price"]).diff().dropna() * 100
    res = fit_stochastic_volatility(ret)
    print(res["summary"])  # type: ignore


def cmd_regime(args: argparse.Namespace) -> None:
    df = load_fx_timeseries(args.pair, start=args.start, end=args.end, source=args.source)
    ret = np.log(df["price"]).diff().dropna() * 100
    res = fit_markov_switching(ret)
    print(res["summary"])  # type: ignore


def cmd_dcc(args: argparse.Namespace) -> None:
    frames = []
    for p in args.pairs:
        df = load_fx_timeseries(p, start=args.start, end=args.end, source=args.source)
        ret = np.log(df["price"]).diff().dropna() * 100
        frames.append(ret.rename(p))
    panel = pd.concat(frames, axis=1).dropna()
    res = fit_dcc_correlation(panel)
    print(res["summary"])  # type: ignore


def cmd_ppp(args: argparse.Namespace) -> None:
    fx = load_fx_timeseries(args.fx, start=args.start, end=args.end, source=args.source)["price"]
    cpi_d = pd.read_csv(args.domestic_cpi, parse_dates=[0], index_col=0).squeeze("columns")
    cpi_f = pd.read_csv(args.foreign_cpi, parse_dates=[0], index_col=0).squeeze("columns")
    res = estimate_ppp_ecm(fx, cpi_domestic=cpi_d, cpi_foreign=cpi_f)
    print(res["summary"])  # type: ignore


def cmd_factor(args: argparse.Namespace) -> None:
    df = load_fx_timeseries(args.pair, start=args.start, end=args.end, source=args.source)
    returns = np.log(df["price"]).diff().rename("ret").dropna()
    fundamentals = pd.read_csv(args.fundamentals, parse_dates=[0], index_col=0)
    aligned = pd.concat([returns, fundamentals], axis=1).dropna()
    res = fit_factor_model(aligned["ret"], aligned.drop(columns=["ret"]))
    print(res["summary"])  # type: ignore


def cmd_rv(args: argparse.Namespace) -> None:
    data = pd.read_csv(args.intraday_csv, parse_dates=[0])
    if set(["timestamp", "price"]).issubset(data.columns):
        data = data[["timestamp", "price"]]
    else:
        data.columns = ["timestamp", "price"]
    rv = compute_realized_volatility(data, sampling_minutes=args.sampling)
    print(rv.tail())


def main():
    parser = argparse.ArgumentParser(description="FX Modelling Toolkit")
    sub = parser.add_subparsers(dest="cmd")

    # Demo
    p = sub.add_parser("demo")
    p.add_argument("--pair", default="EURUSD=X")
    p.add_argument("--source", default="yahoo", choices=["yahoo", "alphavantage", "nasdaq"]) 
    p.add_argument("--start", type=_parse_date, default="2018-01-01")
    p.add_argument("--end", type=_parse_date, default=None)
    p.add_argument("--dcc", action="store_true")
    p.set_defaults(func=cmd_demo)

    # ARIMA
    p = sub.add_parser("arima")
    p.add_argument("--pair", required=True)
    p.add_argument("--source", default="yahoo", choices=["yahoo", "alphavantage", "nasdaq"]) 
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, default=None)
    p.add_argument("--horizon", type=int, default=10)
    p.set_defaults(func=cmd_arima)

    # GARCH
    p = sub.add_parser("garch")
    p.add_argument("--pair", required=True)
    p.add_argument("--source", default="yahoo", choices=["yahoo", "alphavantage", "nasdaq"]) 
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, default=None)
    p.add_argument("--dist", default="normal", choices=["normal", "studentst", "skewt"]) 
    p.set_defaults(func=cmd_garch)

    # Stochastic volatility
    p = sub.add_parser("sv")
    p.add_argument("--pair", required=True)
    p.add_argument("--source", default="yahoo", choices=["yahoo", "alphavantage", "nasdaq"]) 
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, default=None)
    p.set_defaults(func=cmd_sv)

    # Regime switching
    p = sub.add_parser("regime")
    p.add_argument("--pair", required=True)
    p.add_argument("--source", default="yahoo", choices=["yahoo", "alphavantage", "nasdaq"]) 
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, default=None)
    p.set_defaults(func=cmd_regime)

    # DCC correlation
    p = sub.add_parser("dcc")
    p.add_argument("--pairs", nargs='+', required=True)
    p.add_argument("--source", default="yahoo", choices=["yahoo", "alphavantage", "nasdaq"]) 
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, default=None)
    p.set_defaults(func=cmd_dcc)

    # PPP
    p = sub.add_parser("ppp")
    p.add_argument("--fx", required=True)
    p.add_argument("--domestic_cpi", required=True)
    p.add_argument("--foreign_cpi", required=True)
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, default=None)
    p.set_defaults(func=cmd_ppp)

    # Factor model
    p = sub.add_parser("factor")
    p.add_argument("--pair", required=True)
    p.add_argument("--fundamentals", required=True)
    p.add_argument("--source", default="yahoo", choices=["yahoo", "alphavantage", "nasdaq"]) 
    p.add_argument("--start", type=_parse_date, required=True)
    p.add_argument("--end", type=_parse_date, default=None)
    p.set_defaults(func=cmd_factor)

    # Realized volatility
    p = sub.add_parser("rv")
    p.add_argument("--intraday_csv", required=True)
    p.add_argument("--sampling", type=int, default=5)
    p.set_defaults(func=cmd_rv)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()