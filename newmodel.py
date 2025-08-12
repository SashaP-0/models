import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np

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
    fill_missing_business_days: bool = False,
) -> pd.Series:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path, thousands=",")
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found. Available: {list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found. Available: {list(df.columns)}")

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column])

    df[target_column] = (
        df[target_column]
        .astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
    )
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

    df = df.sort_values(by=date_column)
    series = df.set_index(df[date_column])[target_column].dropna()

    if fill_missing_business_days and len(series) > 0:
        full_index = pd.bdate_range(start=series.index.min(), end=series.index.max())
        series = series.reindex(full_index).ffill()

    series.name = target_column
    print(f"Rows read: {len(df)} | Non-null {target_column}: {series.notna().sum()} | "
          f"Date range: {series.index.min()} to {series.index.max()}")
    return series


def interpret_seasonal_period(user_period: int) -> Optional[int]:
    if user_period is None:
        return 5
    if user_period <= 1:
        return None
    return int(user_period)


def time_train_test_split(series: pd.Series, test_size: float) -> Tuple[pd.Series, pd.Series]:
    if not (0.01 <= test_size <= 0.9):
        raise ValueError("test_size should be between 0.01 and 0.9")
    if len(series) < 2:
        return series, series.iloc[0:0]
    raw_split = int(len(series) * (1 - test_size))
    split_index = max(1, min(len(series) - 1, raw_split))
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
    train: pd.Series,
    seasonal_period: Optional[int],
    max_candidates: int = 40
) -> ModelSpec:
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

    def try_fit(order: Tuple[int, int, int], seas: Tuple[int, int, int]) -> Optional[ModelSpec]:
        try:
            if seasonal_period is None:
                seasonal_order = (0, 0, 0, 0)
            else:
                seasonal_order = (seas[0], seas[1], seas[2], seasonal_period)
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                trend="c",
            )
            res = model.fit(disp=False)
            return ModelSpec(order=order, seasonal_order=seasonal_order, aic=res.aic)
        except Exception:
            return None

    for order, seas in prioritized:
        spec = try_fit(order, seas)
        tried.append((order, seas))
        if spec is not None and (best_spec is None or spec.aic < best_spec.aic):
            best_spec = spec
        if len(tried) >= max_candidates:
            break

    for order, seas in candidate_orders():
        if len(tried) >= max_candidates:
            break
        if (order, seas) in tried:
            continue
        spec = try_fit(order, seas)
        tried.append((order, seas))
        if spec is not None and (best_spec is None or spec.aic < best_spec.aic):
            best_spec = spec

    if best_spec is None:
        best_spec = ModelSpec(order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), aic=np.inf)
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan"), float("nan"), float("nan")
    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    eps = 1e-8
    mape = float(np.mean(np.abs(residuals) / (np.abs(y_true) + eps)) * 100.0)
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

    preds_aligned = predictions.copy()
    if not preds_aligned.index.equals(test.index):
        preds_aligned = preds_aligned.reindex(test.index)
    mask = (~test.isna()) & (~preds_aligned.isna())
    test_clean = test[mask]
    preds_clean = preds_aligned[mask]
    residuals_clean = residuals.dropna()

    plt.figure(figsize=(12, 5))
    if len(series) > 0:
        plt.plot(series.index, series.values, label="Actual (All)", color="#1f77b4", alpha=0.6)
    if len(preds_clean) > 0:
        plt.plot(preds_clean.index, preds_clean.values, label="Forecast (Test)", color="#d62728")
    if len(train) > 0 and len(test) > 0:
        plt.axvline(train.index[-1], color="gray", linestyle="--", alpha=0.8, label="Train/Test Split")
    plt.title("Time Series and Forecast")
    plt.xlabel("Date")
    plt.ylabel(series.name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "time_series_forecast.png"), dpi=150)

    if len(test_clean) > 0 and len(preds_clean) > 0:
        plt.figure(figsize=(6, 6))
        plt.scatter(test_clean.values, preds_clean.values, alpha=0.7, color="#2ca02c")
        min_val = float(np.min([test_clean.values.min(), preds_clean.values.min()]))
        max_val = float(np.max([test_clean.values.max(), preds_clean.values.max()]))
        plt.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1)
        plt.title("Predictions vs Actuals (Test)")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "predictions_scatter.png"), dpi=150)

    if len(residuals_clean) > 0:
        plt.figure(figsize=(12, 4))
        plt.scatter(residuals_clean.index, residuals_clean.values, alpha=0.7, color="#ff7f0e", s=12)
        plt.axhline(0.0, color="black", linewidth=1)
        plt.title("Residuals Over Time (Test)")
        plt.xlabel("Date")
        plt.ylabel("Residual")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "residuals_scatter.png"), dpi=150)

    if len(residuals_clean) > 0:
        plt.figure(figsize=(6, 4))
        plt.boxplot(residuals_clean.values, vert=False)
        plt.title("Residuals Boxplot (Test)")
        plt.xlabel("Residual")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "residuals_boxplot.png"), dpi=150)

    if len(residuals_clean) > 0:
        plt.figure(figsize=(8, 4))
        sns.histplot(residuals_clean.values, bins=30, kde=True, color="#8c564b")
        plt.title("Residuals Histogram (Test)")
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "residuals_histogram.png"), dpi=150)

    try:
        plt.show()
    except Exception:
        pass


def seasonal_decomposition_plot(series: pd.Series, seasonal_period: Optional[int], output_dir: str):
    try:
        if seasonal_period is None:
            return
        if len(series) < 2 * seasonal_period:
            return
        stl = STL(series, period=seasonal_period, robust=True)
        res = stl.fit()
        fig = res.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle(f"STL Decomposition (period={seasonal_period})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "stl_decomposition.png"), dpi=150)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Train a SARIMAX time series model on stock Close prices and create diagnostic plots."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to Stock CSV (e.g., ./Stock.csv)")
    parser.add_argument("--date-col", type=str, default="Date", help="Name of the date column (default: Date)")
    parser.add_argument("--target", type=str, default="Close", help="Target column to model (default: Close)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for test set (default: 0.2)")
    parser.add_argument(
        "--seasonal-period",
        type=int,
        default=5,
        help="Seasonal period (>1 for seasonal). Use 0 or 1 for non-seasonal (default: 5).",
    )
    parser.add_argument(
        "--fill-business-days",
        action="store_true",
        help="Fill missing business days via forward fill before modeling (off by default).",
    )
    parser.add_argument("--output-dir", type=str, default="./figures", help="Directory to save plots and outputs")

    args = parser.parse_args()

    series = read_stock_csv(
        csv_path=args.csv,
        date_column=args.date_col,
        target_column=args.target,
        fill_missing_business_days=args.fill_business_days,
    )

    seasonal_period = interpret_seasonal_period(args.seasonal_period)
    output_dir = ensure_output_dir(args.output_dir)
    seasonal_decomposition_plot(series, seasonal_period, output_dir)

    train, test = time_train_test_split(series, args.test_size)
    best_spec = select_sarimax_model(train, seasonal_period=seasonal_period)
    model_fit = fit_sarimax(train, order=best_spec.order, seasonal_order=best_spec.seasonal_order)

    steps = len(test)
    forecast = model_fit.forecast(steps=steps) if steps > 0 else pd.Series([], dtype=float)
    if steps > 0:
        forecast.index = test.index

    mae, rmse, mape = compute_metrics(test.values, forecast.values if steps > 0 else np.array([]))
    residuals = (test - forecast).rename("Residuals") if steps > 0 else pd.Series([], dtype=float, name="Residuals")

    preds_df = pd.DataFrame({
        "date": test.index if steps > 0 else pd.Index([]),
        "actual": test.values if steps > 0 else [],
        "predicted": forecast.values if steps > 0 else [],
        "residual": residuals.values if steps > 0 else [],
    })
    preds_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    plot_all(output_dir, series, train, test, forecast, residuals)

    print("\n=== Model & Evaluation ===")
    print(f"Best order: {best_spec.order}")
    print(f"Best seasonal_order: {best_spec.seasonal_order}")
    print(f"AIC (selection): {best_spec.aic:.2f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAPE: {mape:.2f}%")
    print(f"Seasonal period used: {seasonal_period if seasonal_period is not None else 'non-seasonal'}")
    print(f"Plots and predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()