#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL


@dataclass
class ModelSpec:
    order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    aic: float


def read_stock_csv(
    csv_path: str,
    date_column: str = "Date",
    target_column: str = "Close",
    fill_missing_business_days: bool = True,
) -> pd.Series:
    """Load the stock CSV and return a pandas Series indexed by DatetimeIndex for the target column."""
    if not os.path.exists(csv_path):
        alt_path = os.path.join(os.getcwd(), os.path.basename(csv_path))
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found. Available columns: {list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available columns: {list(df.columns)}")

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)
    series = pd.to_numeric(df[target_column], errors="coerce")
    series.index = df[date_column]
    series = series.dropna()

    if fill_missing_business_days:
        full_index = pd.bdate_range(start=series.index.min(), end=series.index.max())
        series = series.reindex(full_index).ffill()

    series.name = target_column
    return series


def infer_seasonal_period(series: pd.Series, user_period: Optional[int]) -> int:
    if user_period is not None and user_period > 0:
        return user_period

    # For daily stock data, a 5-day (business week) seasonal period is a practical default
    # If the series is long enough, prefer 5; otherwise, fall back to 7
    if len(series) >= 100:
        return 5
    return 7


def time_train_test_split(series: pd.Series, test_size: float) -> Tuple[pd.Series, pd.Series]:
    if not (0.05 <= test_size <= 0.5):
        raise ValueError("test_size should be between 0.05 and 0.5 for sensible evaluation")
    split_index = int(len(series) * (1 - test_size))
    train, test = series.iloc[:split_index], series.iloc[split_index:]
    return train, test


def candidate_orders() -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]

    candidates: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            candidates.append(((p, d, q), (P, D, Q)))
    return candidates


def select_sarimax_model(
    train: pd.Series, seasonal_period: int, max_candidates: int = 40
) -> ModelSpec:
    candidates = candidate_orders()
    # Try a subset first for speed
    prioritized = [
        ((1, 1, 1), (1, 0, 1)),
        ((2, 1, 2), (0, 1, 1)),
        ((0, 1, 1), (1, 1, 0)),
        ((1, 0, 1), (1, 1, 1)),
        ((2, 0, 2), (0, 1, 1)),
        ((1, 1, 0), (0, 1, 1)),
        ((0, 1, 2), (1, 0, 1)),
        ((1, 1, 2), (1, 1, 0)),
        ((2, 1, 1), (1, 0, 0)),
        ((0, 1, 1), (0, 1, 1)),
    ]
    tried: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
    best_spec: Optional[ModelSpec] = None

    def try_fit(order: Tuple[int, int, int], seasonal: Tuple[int, int, int]) -> Optional[ModelSpec]:
        try:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=(seasonal[0], seasonal[1], seasonal[2], seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False,
                trend="c",
            )
            result = model.fit(disp=False)
            return ModelSpec(order=order, seasonal_order=(seasonal[0], seasonal[1], seasonal[2], seasonal_period), aic=result.aic)
        except Exception:
            return None

    # Prioritized list first
    for order, seas in prioritized:
        spec = try_fit(order, seas)
        tried.append((order, seas))
        if spec is not None and (best_spec is None or spec.aic < best_spec.aic):
            best_spec = spec
        if len(tried) >= max_candidates:
            break

    # Fill remaining with generic candidates
    for order, seas in candidates:
        if len(tried) >= max_candidates:
            break
        if (order, seas) in tried:
            continue
        spec = try_fit(order, seas)
        tried.append((order, seas))
        if spec is not None and (best_spec is None or spec.aic < best_spec.aic):
            best_spec = spec

    if best_spec is None:
        # Fallback to a simple specification
        best_spec = ModelSpec(order=(1, 1, 1), seasonal_order=(0, 1, 1, seasonal_period), aic=float("inf"))
    return best_spec


def fit_sarimax(series: pd.Series, order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int]):
    model = SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        trend="c",
    )
    return model.fit(disp=False)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float, float]:
    residuals = y_true - y_pred
    mae = float(residuals.abs().mean())
    rmse = float((residuals.pow(2).mean()) ** 0.5)
    eps = 1e-8
    mape = float((residuals.abs() / (y_true.abs() + eps)).mean() * 100.0)
    return mae, rmse, mape


def ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def plot_all(
    output_dir: str,
    series: pd.Series,
    train: pd.Series,
    test: pd.Series,
    predictions: pd.Series,
    residuals: pd.Series,
):
    sns.set(style="whitegrid")

    # 1) Time series line plot with forecast
    plt.figure(figsize=(12, 5))
    plt.plot(series.index, series.values, label="Actual (All)", color="#1f77b4", alpha=0.6)
    plt.plot(test.index, predictions.values, label="Forecast (Test)", color="#d62728")
    plt.axvline(train.index[-1], color="gray", linestyle="--", alpha=0.8, label="Train/Test Split")
    plt.title("Time Series and Forecast")
    plt.xlabel("Date")
    plt.ylabel(series.name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_series_forecast.png"), dpi=150)

    # 2) Predictions scatter (y_true vs y_pred)
    plt.figure(figsize=(6, 6))
    plt.scatter(test, predictions, alpha=0.7, color="#2ca02c")
    min_val = float(min(test.min(), predictions.min()))
    max_val = float(max(test.max(), predictions.max()))
    plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1)
    plt.title("Predictions vs Actuals (Test)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions_scatter.png"), dpi=150)

    # 3) Residuals scatter over time
    plt.figure(figsize=(12, 4))
    plt.scatter(residuals.index, residuals, alpha=0.7, color="#ff7f0e", s=12)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.title("Residuals Over Time (Test)")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_scatter.png"), dpi=150)

    # 4) Residuals boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=residuals, color="#9467bd")
    plt.title("Residuals Boxplot (Test)")
    plt.xlabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_boxplot.png"), dpi=150)

    # 5) Residuals histogram
    plt.figure(figsize=(8, 4))
    sns.histplot(residuals, bins=30, kde=True, color="#8c564b")
    plt.title("Residuals Histogram (Test)")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals_histogram.png"), dpi=150)

    # Optionally show plots when run interactively
    try:
        plt.show()
    except Exception:
        pass


def seasonal_decomposition_plot(series: pd.Series, seasonal_period: int, output_dir: str):
    try:
        stl = STL(series, period=seasonal_period, robust=True)
        res = stl.fit()
        fig = res.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle(f"STL Decomposition (period={seasonal_period})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "stl_decomposition.png"), dpi=150)
    except Exception:
        # Decomposition is optional; ignore failures silently
        pass


def main():
    parser = argparse.ArgumentParser(description="Train a SARIMAX time series model on stock Close prices and create diagnostic plots.")
    parser.add_argument("--csv", type=str, default="/workspace/Stock.csv", help="Path to Stock CSV (default: /workspace/Stock.csv)")
    parser.add_argument("--date-col", type=str, default="Date", help="Name of the date column (default: Date)")
    parser.add_argument("--target", type=str, default="Close", help="Target column to model (default: Close)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for test set (default: 0.2)")
    parser.add_argument("--seasonal-period", type=int, default=0, help="Seasonal period; if 0, infer a reasonable default (default: 0)")
    parser.add_argument("--no-fill-business-days", action="store_true", help="Disable filling missing business days with forward fill")
    parser.add_argument("--output-dir", type=str, default="/workspace/figures", help="Directory to save plots and outputs")

    args = parser.parse_args()

    series = read_stock_csv(
        csv_path=args.csv,
        date_column=args.date_col,
        target_column=args.target,
        fill_missing_business_days=(not args.no_fill_business_days),
    )

    seasonal_period = infer_seasonal_period(series, args.seasonal_period if args.seasonal_period > 0 else None)

    # STL decomposition for temporal/seasonal insight
    output_dir = ensure_output_dir(args.output_dir)
    seasonal_decomposition_plot(series, seasonal_period, output_dir)

    # Train/test split
    train, test = time_train_test_split(series, args.test_size)

    # Model selection on train
    best_spec = select_sarimax_model(train, seasonal_period=seasonal_period)

    # Fit on train, forecast test horizon
    model_fit = fit_sarimax(train, order=best_spec.order, seasonal_order=best_spec.seasonal_order)
    forecast = model_fit.forecast(steps=len(test))
    forecast.index = test.index

    # Metrics
    mae, rmse, mape = compute_metrics(test, forecast)

    # Residuals
    residuals = (test - forecast).rename("Residuals")

    # Save predictions CSV
    preds_df = pd.DataFrame({
        "date": test.index,
        "actual": test.values,
        "predicted": forecast.values,
        "residual": residuals.values,
    })
    preds_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # Plots
    plot_all(output_dir, series, train, test, forecast, residuals)

    # Print summary info
    print("\n=== Model & Evaluation ===")
    print(f"Best order: {best_spec.order}")
    print(f"Best seasonal_order: {best_spec.seasonal_order}")
    print(f"AIC (selection): {best_spec.aic:.2f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Seasonal period used: {seasonal_period}")
    print(f"Plots and predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()