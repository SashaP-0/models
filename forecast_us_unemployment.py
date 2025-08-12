import os
import sys
import argparse
from typing import Final, List

import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
from dotenv import load_dotenv

import evoml_client as ec
from evoml_client.trial_conf_models import BudgetMode, SplitMethodOptions


DEFAULT_SYMBOLS: Final[List[str]] = ["UNRATE", "USSLIND", "GDP"]


# Load environment variables from a .env file if present
load_dotenv()


def get_env_or_arg(name: str, arg_value: str | None, required: bool = True) -> str:
    value = arg_value if arg_value else os.environ.get(name, "").strip()
    if required and not value:
        raise ValueError(f"Missing required value for {name}. Provide via CLI flag or environment variable.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="US Unemployment forecasting with evoML and FRED")
    parser.add_argument("--api-url", default=os.environ.get("EVOML_API_URL", "https://evoml.ai"), help="evoML base URL")
    parser.add_argument("--username", default=os.environ.get("EVOML_USERNAME", ""), help="evoML username")
    parser.add_argument("--password", default=os.environ.get("EVOML_PASSWORD", ""), help="evoML password")
    parser.add_argument("--fred-api-key", default=os.environ.get("FRED_API_KEY", ""), help="FRED API key")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="FRED symbols to download")
    parser.add_argument("--dataset-name", default="Economic Indicators", help="Dataset name in evoML")
    parser.add_argument("--train-percentage", type=float, default=0.8, help="Train percentage for timeseries split")
    parser.add_argument("--window-size", type=int, default=6, help="Time series window size (months)")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon (months)")
    parser.add_argument("--timeout", type=int, default=900, help="Trial run timeout in seconds")
    parser.add_argument("--output-plot", default="forecast_plot.png", help="Path to save the prediction plot")
    return parser.parse_args()


def init_clients(api_url: str, username: str, password: str, fred_api_key: str) -> Fred:
    ec.init(base_url=api_url, username=username, password=password)
    fred = Fred(api_key=fred_api_key)
    return fred


def fetch_and_prepare_data(fred: Fred, symbols: List[str]) -> pd.DataFrame:
    series_map = {symbol: fred.get_series(symbol) for symbol in symbols}
    df = pd.concat(series_map.values(), axis=1)
    df.columns = symbols
    df = df.resample("M").mean()
    df = df.loc[pd.Timestamp("1950-01-01"):]
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)
    return df


def upload_dataset(df: pd.DataFrame, name: str, api_url: str) -> ec.Dataset:
    dataset = ec.Dataset.from_pandas(df, name=name)
    dataset.put()
    dataset.wait()
    print(f"Dataset URL: {api_url}/platform/datasets/view/{dataset.dataset_id}")
    return dataset


def build_and_run_trial(dataset: ec.Dataset, train_percentage: float, window_size: int, horizon: int, timeout: int) -> ec.Trial:
    config = ec.TrialConfig.with_models(
        models=[
            "linear_regressor",
            "elastic_net_regressor",
            "ridge_regressor",
            "bayesian_ridge_regressor",
        ],
        task=ec.MlTask.regression,
        budget_mode=BudgetMode.fast,
        loss_funcs=["Mean Absolute Error"],
        dataset_id=dataset.dataset_id,
        is_timeseries=True,
    )
    config.options.timeSeriesWindowSize = window_size
    config.options.timeSeriesHorizon = horizon
    config.options.splittingMethodOptions = SplitMethodOptions(
        method="percentage", trainPercentage=train_percentage
    )
    config.options.enableBudgetTuning = False

    trial, _ = ec.Trial.from_dataset_id(
        dataset.dataset_id,
        target_col="UNRATE",
        trial_name="Forecast - Unemployment Rate",
        config=config,
    )
    trial.run(timeout=timeout)
    return trial


def retrieve_pipeline_and_predict(trial: ec.Trial, dataset: ec.Dataset) -> pd.DataFrame:
    pipeline = trial.get_best()
    data_online = pipeline.predict_online(dataset=dataset)
    # Attach model and pipeline to returned frame for downstream inspection if needed
    data_online.attrs["pipeline"] = pipeline
    data_online.attrs["model_rep"] = pipeline.model_rep
    return data_online


def print_mae_from_pipeline(pipeline: ec.Pipeline) -> None:
    metrics = pipeline.model_rep.metrics
    mae_train = metrics["regression-mae"]["train"]["average"]
    mae_validation = metrics["regression-mae"]["validation"]["average"]
    mae_test = metrics["regression-mae"]["test"]["average"]
    print(f"Mean Absolute Error (MAE) on Train Set: {mae_train}")
    print(f"Mean Absolute Error (MAE) on Validation Set: {mae_validation}")
    print(f"Mean Absolute Error (MAE) on Test Set: {mae_test}")


def plot_predictions(data_online: pd.DataFrame, train_percentage: float, output_path: str) -> None:
    date_col = "index" if "index" in data_online.columns else "Date"

    plt.figure(figsize=(15, 6))
    plt.plot(data_online[date_col], data_online["UNRATE"], label="Actual")
    plt.plot(data_online[date_col], data_online["Prediction"], label="Predicted", linestyle="--")

    train_end_index = int(len(data_online) * train_percentage)
    test_start_date = data_online[date_col].iloc[train_end_index + 1]

    plt.axvspan(test_start_date, data_online[date_col].max(), color="gray", alpha=0.3, label="Test Range")
    plt.xlim(data_online[date_col].min(), data_online[date_col].max())

    plt.xlabel("Date")
    plt.ylabel("UNRATE")
    plt.title("Actual vs Predicted UNRATE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_args()

    api_url = get_env_or_arg("EVOML_API_URL", args.api_url)
    username = get_env_or_arg("EVOML_USERNAME", args.username)
    password = get_env_or_arg("EVOML_PASSWORD", args.password)
    fred_api_key = get_env_or_arg("FRED_API_KEY", args.fred_api_key)

    fred = init_clients(api_url=api_url, username=username, password=password, fred_api_key=fred_api_key)

    print("Downloading FRED data...", flush=True)
    df = fetch_and_prepare_data(fred, args.symbols)

    print("Uploading dataset to evoML...", flush=True)
    dataset = upload_dataset(df, name=args.dataset_name, api_url=api_url)

    print("Running evoML trial...", flush=True)
    trial = build_and_run_trial(
        dataset=dataset,
        train_percentage=args.train_percentage,
        window_size=args.window_size,
        horizon=args.horizon,
        timeout=args.timeout,
    )

    print("Retrieving best pipeline and performing online prediction...", flush=True)
    data_online = retrieve_pipeline_and_predict(trial, dataset)

    pipeline = data_online.attrs["pipeline"]
    print_mae_from_pipeline(pipeline)

    print("Plotting predictions...", flush=True)
    plot_predictions(data_online, train_percentage=args.train_percentage, output_path=args.output_plot)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)