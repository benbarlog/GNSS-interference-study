"""The adsb.lol side of the analysis on a synthetic Tier 1 (real schema, fake values): loading via
Tier 2 SQL, confounder filters, controls, ablation, materialized-vs-live agreement, validation join,
and the spatial regression. Real-data checks happen on the desktop (see adsb_lol_pipeline_plan.md §13).
"""

from __future__ import annotations

from datetime import date

import h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gnss.adsblol.schemas import NIC_BINS, TIER1
from gnss.analysis.config import load_analysis_config

DAYS = ["2025-06-13", "2025-06-14", "2025-06-15"]
CENTER5 = h3.latlng_to_cell(57.0, 21.0, 5)


def _row(icao, hx, hour, nic, secs=300.0, version=2, category="A3", roll_s=0.0, alt=35000, mlat=0):
    r = {f.name: None for f in TIER1}
    r.update(hour_utc=hour, icao=icao, hex5=hx, callsign="TST", n_pts=20 + mlat, n_pts_adsb=20,
             n_pts_mlat=mlat, n_pts_other=0, s_obs=secs, s_adsb=secs, sil_min=3, version=version,
             category=category, s_high_roll=roll_s, max_abs_roll=45.0 if roll_s else 2.0,
             alt_min=alt, alt_max=alt, alt_sum_s=float(alt) * secs, s_alt=secs, s_ground=0.0,
             gap_n=0, gap_abs_sum=0.0, gap_abs_max=0.0, n_jumps=0, max_implied_kt=450.0,
             spd_mismatch_max=5.0, first_ts=0, last_ts=0, sq_7500=False, sq_7600=False, sq_7700=False,
             pipeline_version="test", config_hash="test")
    for i in range(NIC_BINS):
        r[f"nic_s_{i}"] = secs if i == nic else 0.0
        r[f"nacp_s_{i}"] = secs if i == 10 else 0.0
    r["nic_s_unk"] = 0.0
    r["nacp_s_unk"] = 0.0
    return r


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("gnss-data")
    rng = np.random.default_rng(0)
    cells = [h3.str_to_int(c) for c in h3.grid_disk(CENTER5, 10)]
    lat = np.array([h3.cell_to_latlng(h3.int_to_str(c))[0] for c in cells])
    hot = lat > 57.3
    for d in DAYS:
        rows = []
        for k, (hx, is_hot) in enumerate(zip(cells, hot)):
            for a in range(8):
                nic = 0 if (is_hot and rng.random() < 0.7) else 8
                rows.append(_row(f"{k:04x}{a:02x}", hx, a % 24, nic,
                                 version=0 if a == 7 else 2,
                                 category="A1" if a == 6 else "A3",
                                 roll_s=300.0 if (a == 5 and k % 3 == 0) else 0.0,
                                 alt=5000 if a == 6 else 35000, mlat=4 if k % 2 else 0))
        # one faulty aircraft: NIC 0 in every cold cell it crosses, all day
        for hx in [c for c, h in zip(cells, hot) if not h][:40]:
            rows.append(_row("bad001", hx, 12, 0, secs=60.0))
        # one jamming victim: NIC 0 all day, but only ever inside the hot zone -> must NOT be flagged
        for hx in [c for c, h in zip(cells, hot) if h][:30]:
            rows.append(_row("jam001", hx, 13, 0, secs=60.0))
        out = root / "adsblol" / "tier1_aircraft_hex_hour" / f"date={d}"
        out.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=TIER1), out / "part-0.parquet")
    cfg = load_analysis_config(data_root=str(root))
    return cfg, set(cells), hot


def test_load_adsblol_counts_and_rates(world):
    from gnss.analysis.data import load_adsblol
    cfg, cells, _ = world
    p = load_adsblol(cfg, date(2025, 6, 13), date(2025, 6, 15), res=4)
    assert set(p["date"].dt.strftime("%Y-%m-%d")) == set(DAYS)
    assert (p["bad"] <= p["n"]).all() and p["rate"].between(0, 1).all()
    p5 = load_adsblol(cfg, date(2025, 6, 14), date(2025, 6, 14), res=5)
    assert set(p5["hex"]) <= cells
    # distinct aircraft at res 4 never exceed the sum over its res-5 children
    par = p5.assign(h4=p5["hex"].map(lambda c: h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(c)), 4))))
    s5 = par.groupby("h4")["n"].sum()
    p4 = p[p["date"] == "2025-06-14"].set_index("hex")["n"]
    assert (p4 <= s5.reindex(p4.index)).all()


def test_filters_remove_faulty_aircraft(world):
    from gnss.analysis.data import load_adsblol
    cfg, _, _ = world
    a = load_adsblol(cfg, date(2025, 6, 14), date(2025, 6, 14), res=5)
    b = load_adsblol(cfg, date(2025, 6, 14), date(2025, 6, 14), res=5, filters={"faulty": True})
    c = load_adsblol(cfg, date(2025, 6, 14), date(2025, 6, 14), res=5,
                     filters={"roll": True, "v0": True, "faulty": True})
    assert b["bad"].sum() == a["bad"].sum() - 40       # bad001 was bad in 40 hexes; jam001 kept
    assert c["n"].sum() < b["n"].sum()


def test_faulty_flags_only_low_nic_in_clean_airspace(world):
    from gnss.analysis.data import adsblol_connect
    cfg, _, _ = world
    con = adsblol_connect(cfg)
    flagged = set(con.sql("SELECT DISTINCT icao FROM faulty_aircraft").df()["icao"])
    assert "bad001" in flagged          # low NIC where everyone else was fine: faulty
    assert "jam001" not in flagged      # low NIC only inside the jammed zone: a victim, kept


def test_materialized_tier2_matches_live(world):
    from gnss.adsblol.tier2 import rebuild
    from gnss.analysis.data import load_adsblol
    cfg, _, _ = world
    live = load_adsblol(cfg, date(2025, 6, 13), date(2025, 6, 15), res=4)
    rebuild(cfg.paths, resolutions=(4,), nic_lt=7)
    mat = load_adsblol(cfg, date(2025, 6, 13), date(2025, 6, 15), res=4)
    k = ["date", "hex"]
    m = live.sort_values(k).reset_index(drop=True).merge(mat, on=k, suffixes=("_l", "_m"))
    assert len(m) == len(live) and (m["n_l"] == m["n_m"]).all() and (m["bad_l"] == m["bad_m"]).all()


def test_controls_and_ablation(world):
    from gnss.analysis.controls import ablation, bbox_where, hex_controls
    from gnss.analysis.data import adsblol_connect
    cfg, _, _ = world
    con = adsblol_connect(cfg)
    ctl = hex_controls(con, date(2025, 6, 13), date(2025, 6, 15), res=4)
    for c in ["share_high_alt", "share_light_ac", "share_v0", "share_high_roll", "share_mlat",
              "share_faulty"]:
        assert ctl[c].dropna().between(0, 1).all(), c
    assert ctl["share_v0"].max() > 0 and ctl["share_light_ac"].max() > 0
    assert ctl["share_faulty"].max() > 0
    ab = ablation(con, date(2025, 6, 13), date(2025, 6, 15), 7, bbox_where([50, 65, 10, 35]))
    assert list(ab["filter"]) == ["none", "roll", "v0", "faulty", "all"]
    row = ab.set_index("filter")
    assert row.loc["all", "s_known"] < row.loc["none", "s_known"]
    assert row.loc["faulty", "bad_kept"] < 1


def test_validation_join(world):
    from gnss.analysis import validation as vl
    from gnss.analysis.data import load_adsblol
    cfg, _, _ = world
    ours = load_adsblol(cfg, date(2025, 6, 13), date(2025, 6, 15), res=4)
    gj = ours.assign(bad=(ours["bad"] * 0.9).round().astype(int))
    gj["rate"] = gj["bad"] / gj["n"]
    j = vl.join_sources(ours, gj, min_n=5)
    by = vl.agreement_by_day(j)
    assert len(by) == 3 and (by["spearman"] > 0.9).all()


def test_cross_section_spatial_models(world):
    from gnss.analysis import model as md
    from gnss.analysis.data import load_adsblol
    cfg, _, _ = world
    p = load_adsblol(cfg, date(2025, 6, 13), date(2025, 6, 15), res=5)
    xs = md.cross_section(p, min_n=5, min_days=3)
    lat = np.array([h3.cell_to_latlng(h3.int_to_str(int(c)))[0] for c in xs["hex"]])
    rng = np.random.default_rng(1)
    xs["near_conflict"] = -lat + rng.normal(0, 0.05, len(xs))    # proxy that tracks the hot north
    xs["noise"] = rng.normal(size=len(xs))
    xs["constant_col"] = 1.0
    fit = md.fit_spatial(xs, "logit_eb_rate", ["near_conflict", "noise", "log_n_aircraft", "constant_col"])
    c = fit["coefs"]
    ols = c[c["model"] == "OLS"].set_index("term")
    assert ols.loc["near_conflict", "coef"] < 0 and ols.loc["near_conflict", "p"] < 0.01
    assert {"OLS", "Spatial lag", "Spatial error"} == set(c["model"])
    assert fit["diag"]["dropped"] == {"constant_col": "constant"}   # dropped, not a singular crash
