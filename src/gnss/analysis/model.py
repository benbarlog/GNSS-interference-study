"""Multivariate models: does conflict (and what kind) predict interference once traffic, fleet mix,
coverage, latitude and geomagnetic activity are controlled for?

1. Cross-section per window: one row per hex, outcome = logit of the EB-smoothed pooled bad rate
   (or persistence). OLS → Moran's I on residuals / LM tests → spatial lag and spatial error models
   (spreg, sparse LU). Predictors are z-scored so coefficients are comparable.
2. GWR (optional, small samples): does the conflict effect vary by region?
3. Hex-day panel with hex fixed effects: day-to-day change in a hex's rate vs nearby conflict events
   in the previous few days and the daily Kp index; SEs clustered by hex.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .spatial import HexWeights, eb_rate, h3_weights


def cross_section(panel: pd.DataFrame, min_n: int = 5, bad_rate: float = 0.10,
                  min_days: int = 10) -> pd.DataFrame:
    """Per hex over the panel's days: days observed, persistence, pooled counts, EB rate, logit."""
    obs = panel[panel["n"] >= min_n].assign(degraded=lambda x: x["rate"] > bad_rate)
    g = obs.groupby("hex").agg(days_obs=("date", "size"), persistence=("degraded", "mean"),
                               bad=("bad", "sum"), n=("n", "sum"), n_mean=("n", "mean"))
    g = g[g["days_obs"] >= min_days].reset_index()
    g["pooled_rate"] = g["bad"] / g["n"]
    g["eb_rate"] = eb_rate(g["bad"], g["n"])
    eps = 0.5 / g["n"].clip(lower=1)
    p = g["eb_rate"].clip(lower=eps, upper=1 - eps)
    g["logit_eb_rate"] = np.log(p / (1 - p))
    if "log_n_aircraft" not in g:
        g["log_n_aircraft"] = np.log(g["n_mean"])
    return g


def vif(X: pd.DataFrame) -> pd.Series:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    A = np.c_[np.ones(len(X)), X.to_numpy(float)]
    return pd.Series([variance_inflation_factor(A, i + 1) for i in range(X.shape[1])], index=X.columns)


def design(df: pd.DataFrame, y: str, xs: list[str], max_vif: float = 10.0) -> tuple:
    """Drop missing columns/rows, constant columns, then iteratively the highest-VIF column above
    max_vif. Returns (y array, X z-scored frame, kept rows, dropped dict)."""
    dropped = {}
    xs = [x for x in xs if x in df.columns]
    for x in [x for x in xs if df[x].notna().mean() < 0.8]:
        dropped[x] = "missing"
    xs = [x for x in xs if x not in dropped]
    d = df[[y, *xs]].replace([np.inf, -np.inf], np.nan).dropna()
    for x in [x for x in xs if d[x].std() == 0]:
        dropped[x] = "constant"
    xs = [x for x in xs if x not in dropped]
    while len(xs) > 1:
        v = vif(d[xs])
        if v.max() <= max_vif:
            break
        dropped[v.idxmax()] = f"vif {v.max():.1f}"
        xs.remove(v.idxmax())
    X = (d[xs] - d[xs].mean()) / d[xs].std()
    return d[y].to_numpy(float).reshape(-1, 1), X, d.index, dropped


def _tidy(m, names, model: str, extra: str = "") -> pd.DataFrame:
    beta = np.asarray(m.betas).ravel()
    se = np.asarray(m.std_err).ravel()
    zs = getattr(m, "z_stat", None) or getattr(m, "t_stat", None)
    p = np.array([t[1] for t in zs])
    labels = list(names)
    if len(beta) > len(labels):      # ML models append rho (lag) or lambda (error)
        labels = labels + [extra] * (len(beta) - len(labels))
    return pd.DataFrame({"model": model, "term": labels[:len(beta)], "coef": beta, "se": se, "p": p})


def fit_spatial(df: pd.DataFrame, y: str, xs: list[str], island_policy: str = "k2",
                max_vif: float = 10.0) -> dict:
    """OLS + diagnostics, ML spatial lag, ML spatial error on one hex cross-section."""
    import spreg
    yv, X, rows, dropped = design(df, y, xs, max_vif)
    sub = df.loc[rows]
    hw = h3_weights(sub["hex"].to_numpy(), island_policy)
    keep = pd.Index(sub["hex"]).get_indexer(hw.cells)
    yv, X, sub = yv[keep], X.iloc[keep], sub.iloc[keep]
    names = ["CONSTANT", *X.columns]
    Xa = X.to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = spreg.OLS(yv, Xa, w=hw.w, spat_diag=True, moran=True, name_y=y, name_x=list(X.columns))
        lag = spreg.ML_Lag(yv, Xa, w=hw.w, method="LU", name_y=y, name_x=list(X.columns))
        err = spreg.ML_Error(yv, Xa, w=hw.w, method="LU", name_y=y, name_x=list(X.columns))
    coefs = pd.concat([_tidy(ols, names, "OLS"), _tidy(lag, names, "Spatial lag", "rho (W_y)"),
                       _tidy(err, names, "Spatial error", "lambda (W_e)")], ignore_index=True)
    lam = float(np.asarray(err.betas).ravel()[-1])
    diag = {
        "n": int(ols.n), "k": int(ols.k), "dropped": dropped, "island_dropped": hw.n_dropped,
        "ols_r2": float(ols.r2), "ols_aic": float(ols.aic),
        "moran_res_I": float(ols.moran_res[0]), "moran_res_p": float(ols.moran_res[2]),
        "lm_lag": [float(v) for v in ols.lm_lag], "lm_error": [float(v) for v in ols.lm_error],
        "rlm_lag": [float(v) for v in ols.rlm_lag], "rlm_error": [float(v) for v in ols.rlm_error],
        "lag_aic": float(lag.aic), "lag_rho": float(lag.rho), "lag_pr2": float(lag.pr2),
        "err_aic": float(err.aic), "err_lambda": lam,
    }
    resid = pd.DataFrame({"hex": sub["hex"].to_numpy(), "ols_resid": np.asarray(ols.u).ravel(),
                          "err_resid": np.asarray(err.e_filtered).ravel()})
    return {"coefs": coefs, "diag": diag, "resid": resid, "ols_summary": ols.summary,
            "lag_summary": lag.summary, "err_summary": err.summary, "weights": hw}


def fit_gwr(df: pd.DataFrame, y: str, xs: list[str], max_n: int = 2500) -> pd.DataFrame | None:
    """Single-bandwidth GWR (adaptive bisquare, spherical). Local coefficients per hex."""
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW

    from .data import centroids
    yv, X, rows, _ = design(df, y, xs)
    if len(rows) > max_n or len(rows) < 50:
        return None
    sub = df.loc[rows]
    lat, lon = centroids(sub["hex"].to_numpy())
    coords = np.c_[lon, lat]
    bw = Sel_BW(coords, yv, X.to_numpy(float), spherical=True).search()
    res = GWR(coords, yv, X.to_numpy(float), bw, spherical=True).fit()
    out = pd.DataFrame(res.params, columns=["CONSTANT", *X.columns])
    out.insert(0, "hex", sub["hex"].to_numpy())
    tv = res.filter_tvals()
    for i, c in enumerate(X.columns, start=1):
        out[f"{c}_sig"] = tv[:, i] != 0
    out.attrs["bw"] = float(bw)
    out.attrs["aicc"] = float(res.aicc)
    return out


def fit_panel(panel: pd.DataFrame, conflict: pd.DataFrame, kp: pd.DataFrame, min_n: int = 5) -> dict:
    """rate_it = a_i + sum_k b_k log1p(count_k,it) + c kp_max_t + d log n_it + day-of-week + e_it,
    hex fixed effects by within-transformation, SEs clustered by hex, weighted by n (rates from more
    aircraft are more precise). `conflict` holds hex, date and any count columns (UCDP: events and
    deaths; ACLED: standoff / ground / intercept events), each entered as log1p."""
    import statsmodels.api as sm
    counts = [c for c in conflict.columns if c not in ("hex", "date")]
    d = panel[panel["n"] >= min_n].merge(conflict, on=["hex", "date"], how="left")
    d = d.merge(kp[["date", "kp_max"]], on="date", how="left")
    d[counts] = d[counts].fillna(0)
    logs = []
    for c in counts:
        name = "log_" + c.replace("conflict_", "")
        d[name] = np.log1p(d[c])
        logs.append(name)
    d["log_n"] = np.log(d["n"])
    dow = pd.get_dummies(d["date"].dt.dayofweek, prefix="dow", drop_first=True, dtype=float)
    cols = ["rate", *logs, "kp_max", "log_n"]
    d = pd.concat([d, dow], axis=1).dropna(subset=cols)
    cols += list(dow.columns)
    counts = d.groupby("hex")["date"].transform("size")
    d = d[counts >= 2]
    wts = d["n"].to_numpy(float)
    # weighted within-transformation
    gm = d[cols].mul(wts, axis=0).groupby(d["hex"]).transform("sum").div(
        pd.Series(wts, index=d.index).groupby(d["hex"]).transform("sum"), axis=0)
    dm = d[cols] - gm
    keep = [c for c in cols[1:] if dm[c].std() > 0]
    m = sm.WLS(dm["rate"], dm[keep], weights=wts).fit(cov_type="cluster",
                                                     cov_kwds={"groups": d["hex"].to_numpy()})
    tidy = pd.DataFrame({"term": m.params.index, "coef": m.params.values, "se": m.bse.values,
                         "p": m.pvalues.values})
    return {"coefs": tidy, "n_obs": int(m.nobs), "n_hex": int(d["hex"].nunique()),
            "summary": m.summary().as_text()}


def event_study(region_daily: pd.DataFrame, event_day, pre: int = 14, post: int = 21) -> pd.DataFrame:
    """Daily region series (date, value columns) re-indexed to days relative to event_day, with the
    pre-event mean and SD of each value column for a ±2 SD baseline band."""
    e = pd.Timestamp(event_day)
    d = region_daily.copy()
    d["rel_day"] = (d["date"] - e).dt.days
    d = d[(d["rel_day"] >= -pre) & (d["rel_day"] <= post)]
    base = d[d["rel_day"] < 0]
    for c in [c for c in d.columns if c not in ("date", "rel_day", "region")]:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[f"{c}_base_mean"] = base[c].mean()
            d[f"{c}_base_sd"] = base[c].std()
    return d


__all__ = ["HexWeights", "cross_section", "event_study", "fit_gwr", "fit_panel", "fit_spatial"]
