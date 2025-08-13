from typing import Dict, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import inv, det


def _fit_univariate_garch(returns: pd.Series) -> pd.Series:
    try:
        from arch import arch_model
    except ImportError as exc:
        raise RuntimeError("arch package required. Install with `pip install arch`. ") from exc

    y = returns.dropna()
    am = arch_model(y, mean='Constant', vol='GARCH', p=1, q=1, dist='studentst')
    res = am.fit(disp='off')
    sigma = res.conditional_volatility
    sigma.name = "sigma"
    return sigma


def _dcc_loglik(params: np.ndarray, z: np.ndarray, S: np.ndarray) -> float:
    a, b = params
    if a < 0 or b < 0 or (a + b) >= 0.999:
        return 1e12
    T, N = z.shape
    Q = S.copy()
    ll = 0.0
    for t in range(T):
        if t > 0:
            outer = np.outer(z[t - 1], z[t - 1])
            Q = (1 - a - b) * S + a * outer + b * Q
        d = np.sqrt(np.diag(Q))
        Dinv = np.diag(1.0 / d)
        R = Dinv @ Q @ Dinv
        try:
            Ri = inv(R)
            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0:
                return 1e11
            ll_t = -0.5 * (logdet + z[t] @ Ri @ z[t])
        except np.linalg.LinAlgError:
            return 1e10
        ll += ll_t
    return -ll


def fit_dcc_correlation(returns_panel_pct: pd.DataFrame) -> Dict:
    from scipy.optimize import minimize

    data = returns_panel_pct.dropna().astype(float)
    data = data.loc[~(data.var(axis=1) == 0)]
    if data.shape[1] < 2:
        raise ValueError("Need at least two return series for DCC")

    # Step 1: Univariate GARCH(1,1) per series to standardize residuals
    cond_vol = {}
    for col in data.columns:
        sigma = _fit_univariate_garch(data[col])
        cond_vol[col] = sigma
    vol_df = pd.DataFrame(cond_vol).loc[data.index].dropna()
    aligned = data.loc[vol_df.index]
    z = aligned / vol_df
    z = z.replace([np.inf, -np.inf], np.nan).dropna()

    # Unconditional correlation of standardized residuals
    S = np.cov(z.values.T)
    # Normalize S to correlation
    d = np.sqrt(np.diag(S))
    S = (S / d).T / d

    # Step 2: Estimate DCC parameters a, b
    x0 = np.array([0.02, 0.95])
    bounds = [(1e-6, 0.5), (1e-6, 0.998)]
    cons = ({'type': 'ineq', 'fun': lambda x: 0.998 - (x[0] + x[1])},)
    res = minimize(_dcc_loglik, x0, args=(z.values, S), method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500})
    a, b = res.x

    # Reconstruct dynamic correlations
    T, N = z.shape
    Q = S.copy()
    R_list = []
    for t in range(T):
        if t > 0:
            outer = np.outer(z.values[t - 1], z.values[t - 1])
            Q = (1 - a - b) * S + a * outer + b * Q
        d = np.sqrt(np.diag(Q))
        Dinv = np.diag(1.0 / d)
        R = Dinv @ Q @ Dinv
        R_list.append(R)

    R_array = np.stack(R_list, axis=0)
    idx = z.index

    # Package pairwise correlations as series
    pairwise: Dict[Tuple[str, str], pd.Series] = {}
    cols = list(z.columns)
    for i in range(N):
        for j in range(i + 1, N):
            series = pd.Series(R_array[:, i, j], index=idx, name=f"corr_{cols[i]}_{cols[j]}")
            pairwise[(cols[i], cols[j])] = series

    summary = f"DCC(1,1) estimated. a={a:.4f}, b={b:.4f}, a+b={a+b:.4f}. Converged={res.success}"
    return {
        "summary": summary,
        "params": {"a": a, "b": b},
        "uncond_corr": pd.DataFrame(S, index=cols, columns=cols),
        "pairwise_corr_series": pairwise,
    }