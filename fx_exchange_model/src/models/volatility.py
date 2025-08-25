from typing import Dict, Optional

import numpy as np
import pandas as pd


def fit_garch_volatility(returns_pct: pd.Series, dist: str = "normal") -> Dict:
    try:
        from arch import arch_model
    except ImportError as exc:
        raise RuntimeError("arch package required. Install with `pip install arch`. ") from exc

    y = returns_pct.dropna()
    am = arch_model(y, mean='Constant', vol='GARCH', p=1, q=1, dist=dist)
    res = am.fit(disp='off')

    out = {
        "summary": res.summary().as_text(),
        "conditional_vol": res.conditional_volatility.rename("sigma"),
        "params": res.params,
    }
    return out


def fit_stochastic_volatility(returns_pct: pd.Series) -> Dict:
    y = returns_pct.dropna()
    # Attempt Bayesian SV via PyMC; fallback to EGARCH if unavailable
    try:
        import pymc as pm
        import aesara.tensor as at
        import arviz as az
    except Exception:
        # Fallback: EGARCH as proxy for time-varying volatility
        try:
            from arch import arch_model
            am = arch_model(y, mean='Constant', vol='EGARCH', p=1, o=1, q=1, dist='studentst')
            res = am.fit(disp='off')
            return {
                "summary": "EGARCH(1,1) fitted as fallback for stochastic volatility.\n" + res.summary().as_text(),
                "conditional_vol": res.conditional_volatility.rename("sigma"),
                "params": res.params,
            }
        except Exception as exc:
            raise RuntimeError("Neither PyMC nor arch EGARCH available for SV modelling") from exc

    # Simple log-variance AR(1) SV model
    y_scaled = (y - y.mean()) / y.std()
    T = y_scaled.shape[0]
    with pm.Model() as model:
        sigma_eta = pm.Exponential("sigma_eta", 1.0)
        phi = pm.Normal("phi", mu=0.95, sigma=0.1)
        mu = pm.Normal("mu", mu=-1.0, sigma=1.0)

        h = pm.AR("h", rho=[phi], constant=True, sigma=sigma_eta, init=pm.draw(pm.Normal.dist(mu, 1.0), draws=1)[0],
                  constant_offset=mu, shape=T)
        # Observation variance is exp(h)
        eps = pm.Normal("eps", mu=0, sigma=pm.math.exp(0.5 * h), observed=y_scaled.values)

        idata = pm.sample(1000, tune=1000, target_accept=0.9, chains=2, cores=1, progressbar=False)

    post = az.summary(idata, var_names=["mu", "phi", "sigma_eta"]).to_string()
    # Use posterior mean h as log-variance
    h_mean = idata.posterior["h"].mean(dim=("chain", "draw")).to_numpy()
    sigma = np.exp(0.5 * h_mean)

    return {
        "summary": "PyMC Stochastic Volatility model fitted.\n" + post,
        "conditional_vol": pd.Series(sigma, index=y.index, name="sigma"),
    }


def compute_realized_volatility(intraday_price_df: pd.DataFrame, sampling_minutes: int = 5) -> pd.Series:
    df = intraday_price_df.copy()
    if "timestamp" not in df.columns or "price" not in df.columns:
        raise ValueError("DataFrame must have columns: timestamp, price")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()

    # Sample to fixed interval
    df = df.resample(f"{sampling_minutes}T").last().dropna()
    ret = np.log(df["price"]).diff()
    # Realized variance per day: sum of intraday squared returns
    rv = (ret.pow(2).groupby(pd.Grouper(freq='1D')).sum()).dropna()
    # Annualize assuming 252 trading days
    realized_vol = np.sqrt(rv * 252)
    realized_vol.name = "realized_vol"
    return realized_vol