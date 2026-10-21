"""Spatial weights on H3 cells and the ESDA steps (GNSS_interference_plan.md §6.1–6.5).

- Weights: ring-1 H3 contiguity among the cells present that day, row-standardized. Islands (no
  present ring-1 neighbour) either get ring-2 neighbours (`k2`) or are dropped. KNN on centroids is
  the sensitivity alternative.
- Rates: Empirical Bayes smoothing (global, Marshall) because n ranges from 5 to hundreds.
- Global Moran's I on rates (`Moran_Rate`, EB-standardized) and LISA (`Moran_Local_Rate`), with
  permutation inference and FDR-adjusted significance.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import h3
import numpy as np
import pandas as pd
from esda.moran import Moran_Local_Rate, Moran_Rate
from libpysal.weights import KNN, W

from .data import centroids, observed

QUAD = {1: "HH", 2: "LH", 3: "LL", 4: "HL"}


@dataclass
class HexWeights:
    w: W
    cells: np.ndarray        # cell order = row order of w
    n_islands: int           # cells with no ring-1 neighbour
    n_dropped: int           # islands still without neighbours after the policy


def h3_weights(cells, island_policy: str = "k2") -> HexWeights:
    cells = np.asarray(cells, dtype="int64")
    pos = {int(c): i for i, c in enumerate(cells)}
    strs = [h3.int_to_str(int(c)) for c in cells]
    nbrs: dict[int, list[int]] = {}
    islands = 0
    for i, s in enumerate(strs):
        ring = [pos[k] for k in (h3.str_to_int(x) for x in h3.grid_disk(s, 1)) if k in pos and pos[k] != i]
        if not ring:
            islands += 1
            if island_policy == "k2":
                ring = [pos[k] for k in (h3.str_to_int(x) for x in h3.grid_disk(s, 2))
                        if k in pos and pos[k] != i]
        nbrs[i] = ring
    keep = [i for i, r in nbrs.items() if r]
    dropped = len(cells) - len(keep)
    if dropped:
        # re-index without the still-isolated cells; their neighbour lists are empty, so no one
        # else refers to them
        remap = {old: new for new, old in enumerate(keep)}
        nbrs = {remap[i]: [remap[j] for j in nbrs[i]] for i in keep}
        cells = cells[keep]
    # k2 links are not necessarily symmetric (A's ring-2 neighbour B may have its own ring-1); symmetrize
    for i, r in list(nbrs.items()):
        for j in r:
            if i not in nbrs[j]:
                nbrs[j].append(i)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = W(nbrs, silence_warnings=True)
    w.transform = "r"
    return HexWeights(w=w, cells=cells, n_islands=islands, n_dropped=dropped)


def knn_weights(cells, k: int = 6) -> HexWeights:
    cells = np.asarray(cells, dtype="int64")
    lat, lon = centroids(cells)
    # equirectangular around the sample's mean latitude: fine within a region
    xy = np.c_[np.radians(lon) * np.cos(np.radians(lat.mean())), np.radians(lat)] * 6371.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = KNN.from_array(xy, k=min(k, len(cells) - 1))
    w.transform = "r"
    return HexWeights(w=w, cells=cells, n_islands=0, n_dropped=0)


def eb_rate(bad, n) -> np.ndarray:
    """Global Empirical Bayes (Marshall 1991) smoothed rate: shrinks small-n hexes to the mean."""
    bad = np.asarray(bad, float)
    n = np.asarray(n, float)
    m = bad.sum() / n.sum()
    if m <= 0:
        return np.zeros_like(n)
    var = max((n * (bad / n - m) ** 2).sum() / n.sum() - m / n.mean(), 0.0)
    shrink = var / (var + m / n)
    return m + shrink * (bad / n - m)


def global_moran(bad, n, hw: HexWeights, permutations: int = 999, seed: int | None = None) -> dict:
    if seed is not None:
        np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mr = Moran_Rate(np.asarray(bad, float), np.asarray(n, float), hw.w, permutations=permutations)
    return {"I": mr.I, "EI": mr.EI, "z_sim": mr.z_sim, "p_sim": mr.p_sim}


def fdr_threshold(p: np.ndarray, alpha: float) -> float:
    """Benjamini–Hochberg cut-off: the largest p_(i) with p_(i) <= alpha * i / m (0 if none)."""
    p = np.sort(np.asarray(p))
    m = len(p)
    ok = p <= alpha * np.arange(1, m + 1) / m
    return float(p[ok].max()) if ok.any() else 0.0


def local_moran(bad, n, hw: HexWeights, permutations: int = 999, alpha: float = 0.05,
                seed: int | None = None, n_jobs: int = 1) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lm = Moran_Local_Rate(np.asarray(bad, float), np.asarray(n, float), hw.w,
                              permutations=permutations, seed=seed, keep_simulations=False,
                              n_jobs=n_jobs)
    p = lm.p_sim
    thr = fdr_threshold(p, alpha)
    quad = pd.Series(lm.q).map(QUAD).to_numpy()
    sig = p < alpha
    sig_fdr = p <= thr if thr > 0 else np.zeros_like(sig)
    return pd.DataFrame({
        "hex": hw.cells, "Ii": lm.Is, "p_sim": p, "quad": quad,
        "cluster": np.where(sig, quad, "ns"), "cluster_fdr": np.where(sig_fdr, quad, "ns"),
    })


def warmup() -> float:
    """Run a tiny LISA so esda's numba kernels compile now (~45 s, once per Python process) instead
    of during the first real map. Returns seconds taken."""
    import time
    t = time.time()
    cells = np.array([h3.str_to_int(c) for c in h3.grid_disk(h3.latlng_to_cell(0.0, 0.0, 4), 2)])
    n = np.full(len(cells), 20.0)
    bad = np.arange(len(cells), dtype=float) % 5
    local_moran(bad, n, h3_weights(cells), permutations=99, seed=0)
    return time.time() - t


def daily_esda(panel: pd.DataFrame, min_n: int = 5, permutations: int = 999, alpha: float = 0.05,
               island_policy: str = "k2", seed: int = 12345, lisa: bool = True,
               min_cells: int = 20, n_jobs: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per day: global Moran's I (rate) + LISA. `panel` should already be one region.

    Returns (daily, lisa): daily has date, n_cells, n_islands, n_dropped, I, z_sim, p_sim,
    mean_rate, share_degraded inputs; lisa has date, hex, rate, eb_rate, cluster labels."""
    rows, lisas = [], []
    obs = observed(panel, min_n)
    for d, g in obs.groupby("date", sort=True):
        if len(g) < min_cells:
            continue
        hw = h3_weights(g["hex"].to_numpy(), island_policy)
        if hw.w.n < min_cells:
            continue
        g = g.set_index("hex").loc[hw.cells]
        gm = global_moran(g["bad"], g["n"], hw, permutations, seed)
        rows.append({"date": d, "n_cells": hw.w.n, "n_islands": hw.n_islands, "n_dropped": hw.n_dropped,
                     "bad": int(g["bad"].sum()), "n": int(g["n"].sum()),
                     "pooled_rate": g["bad"].sum() / g["n"].sum(), **gm})
        if lisa:
            lm = local_moran(g["bad"], g["n"], hw, permutations, alpha, seed, n_jobs)
            lm.insert(0, "date", d)
            lm["n"] = g["n"].to_numpy()
            lm["bad"] = g["bad"].to_numpy()
            lm["rate"] = (g["bad"] / g["n"]).to_numpy()
            lm["eb_rate"] = eb_rate(g["bad"], g["n"])
            lisas.append(lm)
    daily = pd.DataFrame(rows)
    lis = pd.concat(lisas, ignore_index=True) if lisas else pd.DataFrame()
    return daily, lis


def persistence(lisa: pd.DataFrame, bad_rate: float = 0.10, min_days: int = 10) -> pd.DataFrame:
    """Per hex over the days in `lisa`: share of observed days that were significant HH (raw and
    FDR) and share degraded (rate > bad_rate). Hexes with < min_days observed days are dropped."""
    g = lisa.assign(hh=lisa["cluster"].eq("HH"), hh_fdr=lisa["cluster_fdr"].eq("HH"),
                    degraded=lisa["rate"] > bad_rate)
    out = g.groupby("hex").agg(days_obs=("date", "size"), hh=("hh", "mean"), hh_fdr=("hh_fdr", "mean"),
                               persistence=("degraded", "mean"), bad=("bad", "sum"), n=("n", "sum"))
    out["pooled_rate"] = out["bad"] / out["n"]
    return out[out["days_obs"] >= min_days].reset_index()


def hex_persistence(panel: pd.DataFrame, min_n: int = 5, bad_rate: float = 0.10,
                    min_days: int = 10) -> pd.DataFrame:
    """Persistence from the panel alone (no LISA): degraded days / observed days, pooled counts."""
    obs = observed(panel, min_n).assign(degraded=lambda x: x["rate"] > bad_rate)
    out = obs.groupby("hex").agg(days_obs=("date", "size"), persistence=("degraded", "mean"),
                                 bad=("bad", "sum"), n=("n", "sum"), n_mean=("n", "mean"))
    out["pooled_rate"] = out["bad"] / out["n"]
    return out[out["days_obs"] >= min_days].reset_index()
