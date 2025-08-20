import pandas as pd
import matplotlib.pyplot as plt
from typing import Final
from fredapi import Fred

import evoml_client as ec
from evoml_client.trial_conf_models import BudgetMode, SplitMethodOptions


# Set credentials and API key
API_URL: Final[str] = "https://evoml.ai"
EVOML_USERNAME: Final[str] = input("enter your evoml username: ")
EVOML_PASSWORD: Final[str] = input("enter your evoml password: ")
FRED_API_KEY: Final[str] = "eab5d746893379474ba9b89fb825331c"

# Connect to the evoML platform
ec.init(base_url=API_URL, username=EVOML_USERNAME, password=EVOML_PASSWORD)

# Initialize the FRED API client
fred = Fred(api_key=FRED_API_KEY)


# Download data from FRED
symbols = ['UNRATE', 'USSLIND', 'GDP']
data_dict = {symbol: fred.get_series(symbol) for symbol in symbols}

# Join datasets
data_fred = pd.concat(data_dict.values(), axis=1)
data_fred.columns = symbols

# Resample to monthly frequency
data_fred = data_fred.resample('M').mean()

# Use data since 1950
data_fred = data_fred.loc[pd.Timestamp('1950-01-01'):]

# Reset the index as a column
data_fred.reset_index(inplace=True)

# Upload and wait for completion of task
dataset = ec.Dataset.from_pandas(data_fred, name="Economic Indicators")
dataset.put()
dataset.wait()
print(f"Dataset URL: {API_URL}/platform/datasets/view/{dataset.dataset_id}")




train_percentage = 0.8
config = ec.TrialConfig.with_models(
    models=["linear_regressor", "elastic_net_regressor", "ridge_regressor", "bayesian_ridge_regressor"],
    task=ec.MlTask.regression,
    budget_mode=BudgetMode.fast,
    loss_funcs=["Mean Absolute Error"],
    dataset_id=dataset.dataset_id,
    is_timeseries=True,
)
config.options.timeSeriesWindowSize = 6
config.options.timeSeriesHorizon = 12
config.options.splittingMethodOptions = SplitMethodOptions(method="percentage", trainPercentage=train_percentage)
config.options.enableBudgetTuning = False

trial, _ = ec.Trial.from_dataset_id(
    dataset.dataset_id,
    target_col="UNRATE",
    trial_name="Forecast - Unemployment Rate",
    config=config,
)

trial.run(timeout=900)



# Retrieve the top ranked pipeline on the objective function
pipeline = trial.get_best()

# Online prediction
data_online = pipeline.predict_online(dataset=dataset)


# Retrieve the mean absolute error (MAE) for train, validation, and test sets found during the build-model
mae_train = pipeline.model_rep.metrics["regression-mae"]["train"]["average"]
mae_validation = pipeline.model_rep.metrics["regression-mae"]["validation"]["average"]
mae_test = pipeline.model_rep.metrics["regression-mae"]["test"]["average"]
print(f"Mean Absolute Error (MAE) on Train Set: {mae_train}")
print(f"Mean Absolute Error (MAE) on Validation Set: {mae_validation}")
print(f"Mean Absolute Error (MAE) on Test Set: {mae_test}")



plt.figure(figsize=(15, 6))
plt.plot(data_online["index"], data_online["UNRATE"], label="Actual")
plt.plot(data_online["index"], data_online["Prediction"], label="Predicted", linestyle='--')

# Calculate the index where the test range starts
train_end_index = int(len(data_online) * train_percentage)
test_start_date = data_online["index"].iloc[train_end_index + 1]

# Add shaded block for the test range
plt.axvspan(test_start_date, data_online["index"].max(), color='gray', alpha=0.3, label='Test Range')
plt.xlim(data_online["index"].min(), data_online["index"].max())

plt.xlabel("Date")
plt.ylabel("UNRATE")
plt.title("Actual vs Predicted UNRATE")
plt.legend()
plt.show()