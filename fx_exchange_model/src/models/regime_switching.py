from typing import Dict

import pandas as pd


def fit_markov_switching(returns_pct: pd.Series, k_regimes: int = 2) -> Dict:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    y = returns_pct.dropna()
    model = MarkovRegression(y, k_regimes=k_regimes, trend='c', switching_variance=True)
    res = model.fit(disp=False)

    out = {
        "summary": res.summary().as_text(),
        "smoothed_marginal_probabilities": res.smoothed_marginal_probabilities,
        "regime_means": res.params[[name for name in res.params.index if 'intercept' in name]],
    }
    return out