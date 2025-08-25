from typing import Dict

import pandas as pd


def fit_factor_model(returns: pd.Series, factors: pd.DataFrame) -> Dict:
    import statsmodels.api as sm

    y = returns.dropna()
    X = factors.loc[y.index].dropna()
    aligned = y.to_frame("y").join(X, how='inner')

    X = sm.add_constant(aligned.drop(columns=["y"]))
    y = aligned["y"]

    model = sm.OLS(y, X)
    res = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    return {
        "summary": res.summary().as_text(),
        "params": res.params,
    }