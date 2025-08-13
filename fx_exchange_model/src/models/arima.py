from typing import Dict, Optional

import numpy as np
import pandas as pd


def _auto_arima_order(y: pd.Series) -> tuple:
    try:
        import pmdarima as pm
        model = pm.auto_arima(y, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore')
        order = model.order  # type: ignore
        return order
    except Exception:
        return (1, 0, 1)


def fit_arima_forecast(price: pd.Series, horizon: int = 10, order: Optional[tuple] = None) -> Dict:
    from statsmodels.tsa.arima.model import ARIMA

    y = price.asfreq('B').interpolate().dropna()
    if order is None:
        order = _auto_arima_order(y)

    model = ARIMA(y, order=order)
    res = model.fit()
    fc = res.get_forecast(steps=horizon)
    forecast = fc.predicted_mean
    conf = fc.conf_int()

    out = {
        "order": order,
        "forecast": pd.DataFrame({
            "mean": forecast,
            "lower": conf.iloc[:, 0],
            "upper": conf.iloc[:, 1],
        }),
        "summary": res.summary().as_text(),
    }
    return out