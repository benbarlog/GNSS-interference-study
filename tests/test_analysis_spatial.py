"""Spatial weights, EB smoothing, FDR, Moran/LISA on synthetic hex data."""

from __future__ import annotations

import h3
import numpy as np
import pandas as pd

from gnss.analysis import spatial as sp

CENTER = h3.latlng_to_cell(57.0, 21.0, 4)


def disk(k=6):
    return np.array([h3.str_to_int(c) for c in h3.grid_disk(CENTER, k)], dtype="int64")


def test_weights_symmetric_and_row_standardized():
    hw = sp.h3_weights(disk(3))
    assert hw.n_islands == 0 and hw.n_dropped == 0
    for i, nb in hw.w.neighbors.items():
        for j in nb:
            assert i in hw.w.neighbors[j]
    assert np.allclose([sum(v) for v in hw.w.weights.values()], 1.0)


def test_islands_k2_and_drop():
    cells = list(disk(1))
    lonely = h3.str_to_int(h3.grid_ring(CENTER, 3)[0])   # 2 steps from ring 1: island at k=1, linked at k=2
    hw = sp.h3_weights(np.array(cells + [lonely]), "k2")
    assert hw.n_islands == 1 and hw.n_dropped == 0
    hw = sp.h3_weights(np.array(cells + [lonely]), "drop")
    assert hw.n_dropped == 1 and len(hw.cells) == len(cells)


def test_eb_shrinks_small_n_more():
    bad = np.array([1, 50, 0, 5])
    n = np.array([1, 500, 400, 50])
    eb = sp.eb_rate(bad, n)
    m = bad.sum() / n.sum()
    raw = bad / n
    assert abs(eb[0] - m) < abs(raw[0] - m)              # 1/1 pulled hard toward the mean
    assert abs(eb[1] - raw[1]) < abs(eb[0] - raw[0])     # big n barely moves
    assert np.all(sp.eb_rate(np.zeros(3), np.ones(3)) == 0)


def test_fdr_threshold_bh():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    assert sp.fdr_threshold(p, 0.05) == 0.008
    assert sp.fdr_threshold(np.array([0.5, 0.9]), 0.05) == 0.0


def synthetic_panel(days=3, seed=0):
    rng = np.random.default_rng(seed)
    cells = disk(6)
    lat = np.array([h3.cell_to_latlng(h3.int_to_str(int(c)))[0] for c in cells])
    hot = lat > np.median(lat) + 0.6
    rows = []
    for d in pd.date_range("2025-06-01", periods=days):
        n = rng.integers(10, 80, len(cells))
        p = np.where(hot, 0.6, 0.02)
        rows.append(pd.DataFrame({"date": d, "hex": cells, "n": n, "bad": rng.binomial(n, p)}))
    df = pd.concat(rows, ignore_index=True)
    df["rate"] = df["bad"] / df["n"]
    return df, set(cells[hot])


def test_daily_esda_finds_planted_cluster():
    df, hot = synthetic_panel()
    daily, lis = sp.daily_esda(df, permutations=199, seed=1)
    assert len(daily) == 3
    assert (daily["I"] > 0.5).all() and (daily["p_sim"] < 0.05).all()
    hh = set(lis.loc[lis["cluster_fdr"] == "HH", "hex"])
    assert hh and len(hh & hot) / len(hh) > 0.9
    pers = sp.persistence(lis, min_days=3)
    assert pers.set_index("hex").loc[next(iter(hh & hot)), "hh_fdr"] > 0


def test_knn_weights():
    hw = sp.knn_weights(disk(3), k=6)
    assert hw.w.n == len(disk(3)) and min(hw.w.cardinalities.values()) == 6
