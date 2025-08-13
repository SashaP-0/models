from typing import Dict

import numpy as np
import pandas as pd


def estimate_ppp_ecm(fx_rate: pd.Series, cpi_domestic: pd.Series, cpi_foreign: pd.Series) -> Dict:
    """
    Estimate long-run PPP via cointegration and an error-correction model.

    fx_rate: price of foreign in domestic terms (e.g., USD per EUR)
    cpi_domestic: CPI index for domestic country
    cpi_foreign: CPI index for foreign country
    """
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint

    df = pd.concat([
        fx_rate.rename('s'),
        cpi_domestic.rename('p_d'),
        cpi_foreign.rename('p_f')
    ], axis=1).dropna()

    # Theory: s_t ~ p_f - p_d (in logs)
    df_log = np.log(df)
    z = df_log["p_f"] - df_log["p_d"]
    s = df_log["s"]

    cstat, pvalue, _ = coint(s, z)

    # Long-run relationship: s_t = alpha + beta * (p_f - p_d) + u_t
    X = sm.add_constant(z)
    long_run = sm.OLS(s, X).fit()
    u = long_run.resid

    # Error-correction model on returns
    ds = s.diff().dropna()
    u_lag = u.shift(1).loc[ds.index]
    X_ecm = sm.add_constant(pd.concat([u_lag], axis=1).dropna())
    y_ecm = ds.loc[X_ecm.index]
    ecm = sm.OLS(y_ecm, X_ecm).fit()

    return {
        "summary": (
            "Cointegration p-value: %.4f\n\nLong-run relation:\n%s\n\nECM (speed of adjustment on lagged disequilibrium):\n%s"
            % (pvalue, long_run.summary().as_text(), ecm.summary().as_text())
        ),
        "cointegration_pvalue": pvalue,
        "long_run_params": long_run.params,
        "ecm_params": ecm.params,
    }