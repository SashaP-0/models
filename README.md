# Microsoft Stock Time Series Modeling (Kaggle)

This project includes a Jupyter notebook to analyze and forecast Microsoft (MSFT) stock prices using ARIMA and simple ML baselines.

## Setup

1. Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. (Optional) Configure Kaggle API for automated data download:
   - Get your API token from Kaggle and place `kaggle.json` in `~/.kaggle/` with permissions `600`.
   - Set the dataset slug inside the notebook (e.g., `username/dataset-name`).

3. Alternatively, manually place the CSV from Kaggle in `/workspace/data/msft/`.

## Run

Launch Jupyter and open `msft_time_series.ipynb`:

```bash
python -m ipykernel install --user --name msft-ts
jupyter lab
```

Inside the notebook, run cells top-to-bottom. It will:
- Load MSFT data from Kaggle or local CSV
- Explore and clean the time series
- Train ARIMA (via auto_arima) and simple baselines
- Evaluate with RMSE/MAPE and forecast future values