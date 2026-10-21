"""adsb.lol (own NIC pipeline) vs GPSJAM at H3 res 4, hex-day matched (GNSS_interference_plan.md §6.7).

Both sides use GPSJAM-style counts: aircraft per hex-day, "bad" aircraft. Only hex-days with
≥ min_n aircraft in both sources are compared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def join_sources(ours: pd.DataFrame, gpsjam: pd.DataFrame, min_n: int = 5) -> pd.DataFrame:
    j = ours[["date", "hex", "n", "bad", "rate"]].merge(
        gpsjam[["date", "hex", "n", "bad", "rate"]], on=["date", "hex"], suffixes=("_a", "_g"))
    return j[(j["n_a"] >= min_n) & (j["n_g"] >= min_n)].reset_index(drop=True)


def agreement_by_day(j: pd.DataFrame, bad_rate: float = 0.10) -> pd.DataFrame:
    def f(g):
        a, b = g["rate_a"] > bad_rate, g["rate_g"] > bad_rate
        return pd.Series({
            "hexes": len(g),
            "spearman": g["rate_a"].corr(g["rate_g"], method="spearman") if len(g) >= 10 else np.nan,
            "pearson": g["rate_a"].corr(g["rate_g"]) if len(g) >= 10 else np.nan,
            "agree_pct": 100 * (a == b).mean(),
            "both_bad": int((a & b).sum()), "only_ours": int((a & ~b).sum()),
            "only_gpsjam": int((~a & b).sum()),
            "n_ratio_median": float((g["n_a"] / g["n_g"]).median()),
        })
    return j.groupby("date").apply(f, include_groups=False).reset_index()


def confusion(j: pd.DataFrame, bad_rate: float = 0.10) -> pd.DataFrame:
    a = np.where(j["rate_a"] > bad_rate, "bad", "good")
    g = np.where(j["rate_g"] > bad_rate, "bad", "good")
    return pd.crosstab(pd.Series(a, name="adsb.lol"), pd.Series(g, name="GPSJAM"))
