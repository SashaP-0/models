# stock_forecast.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings("ignore")

# ====== Load Data ======
file_path = "Stock.csv"  # Make sure Stock.csv is in the same directory
df = pd.read_csv(file_path)

# Ensure correct datetime format
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Sort by date (just in case)
df.sort_index(inplace=True)

# Target variable: Close price
series = df['Close']

# ====== Plot Time Series ======
plt.figure(figsize=(12, 5))
plt.plot(series, label='Close Price')
plt.title("Microsoft Stock Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# ====== Fit ARIMA Model ======
# (p,d,q) parameters are chosen simply here; can be tuned
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit()

# ====== Forecast ======
forecast_steps = 30  # Number of days to forecast
forecast = model_fit.forecast(steps=forecast_steps)

# ====== Predictions Scatter Graph ======
plt.figure(figsize=(8, 5))
plt.scatter(range(len(series)), series, label='Actual', s=10)
plt.scatter(range(len(series), len(series) + forecast_steps),
            forecast, label='Forecast', s=10, color='red')
plt.title("Predictions Scatter Plot")
plt.xlabel("Time Index")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# ====== Residuals ======
residuals = model_fit.resid

# Scatter plot of residuals
plt.figure(figsize=(8, 5))
plt.scatter(series.index, residuals, s=10)
plt.axhline(y=0, color='red', linestyle='--')
plt.title("Residuals Scatter Plot")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.show()

# Boxplot of residuals
plt.figure(figsize=(6, 5))
sns.boxplot(y=residuals)
plt.title("Residuals Boxplot")
plt.show()

# Histogram of residuals
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=20, kde=True)
plt.title("Residuals Histogram")
plt.xlabel("Residual Value")
plt.show()

print(model_fit.summary())
