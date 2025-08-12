import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

file_path = "Stock.csv"
df = pd.read_csv(file_path)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

series = df['Close']

model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit()

forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)

future_dates = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='B')  # business days
forecast_series = pd.Series(forecast.values, index=future_dates)

residuals = model_fit.resid

plt.figure(figsize=(12, 5))
plt.plot(series, label='Close Price', color='navy', linewidth=2)
plt.title("Microsoft Stock Closing Price Over Time", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.figure(figsize=(8, 5))
plt.scatter(range(len(series)), series, label='Actual', s=12, color='blue', alpha=0.6)
plt.scatter(range(len(series), len(series) + forecast_steps),
            forecast, label='Forecast', s=12, color='red', alpha=0.8)
plt.title("Predictions Scatter Plot", fontsize=16)
plt.xlabel("Time Index")
plt.ylabel("Close Price ($)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.figure(figsize=(8, 5))
plt.scatter(series.index, residuals, s=12, alpha=0.6, color='purple')
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
plt.title("Residuals Scatter Plot", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)

plt.figure(figsize=(6, 5))
sns.boxplot(y=residuals, color='orange')
plt.title("Residuals Boxplot", fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)

plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=20, kde=True, color='green')
plt.title("Residuals Histogram", fontsize=16)
plt.xlabel("Residual Value")
plt.grid(True, linestyle='--', alpha=0.7)

plt.figure(figsize=(12, 6))
plt.plot(series, label='Historical', color='blue', linewidth=2)
plt.axvspan(series.index[-1], forecast_series.index[-1], color='grey', alpha=0.2) 
plt.plot(forecast_series, label='Forecast', color='red', linewidth=2, linestyle='--')  
plt.title("Microsoft Stock Price Forecast", fontsize=18)
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.figure(figsize=(12, 6))
plt.plot(series, label='Actual', color='blue', linewidth=2)
plt.plot(model_fit.fittedvalues, label='Model Fitted', color='red', linewidth=2, linestyle='--')
plt.title("Model Performance: Actual vs Fitted Values", fontsize=18)
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.show()

print(model_fit.summary())
