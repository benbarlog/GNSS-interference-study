import math

import numpy as np
import pytest

from conftest import DAY0, det, gz, point, straight_flight, trace_doc
from gnss.adsblol import trace
from gnss.adsblol.process import worker_params
from gnss.adsblol.schemas import REASON_CONTEXT, REASON_JUMP, REASON_NIC_LOW


@pytest.fixture
def P(cfg):
    return worker_params(cfg)


def run(docs, P):
    batch = [(f"./traces/23/trace_full_{d['icao']}.json", gz(d)) for d in docs]
    return trace.process_batch(batch, DAY0, P, sample_n=len(batch))


def df(tbl):
    return tbl.to_pandas()


def test_parse_fixture_trace():
    pts = straight_flight(10)
    pts.append(point(4000, 50.5, 10.5, alt="ground", gs=10, details=None))
    pts.append(point(4010, None, None, alt=None, gs=None))
    p = trace.parse_trace(gz(trace_doc(points=pts, r="D-ABCD", t="A320")), {})
    assert p.icao == "abc123" and p.meta["r"] == "D-ABCD"
    assert len(p.t) == 12 and p.t[0] == DAY0 + 3000
    assert p.ground.tolist() == [False] * 10 + [True, False]
    assert math.isnan(p.alt[10]) and math.isnan(p.alt[11]) and math.isnan(p.lat[11])
    assert p.ff[trace.FF_NIC, 0] == 8 and math.isnan(p.ff[trace.FF_NIC, 1])
    assert trace.SRC_CLASS[p.src[0]] == 0


def test_forward_fill_staleness_cap():
    t = np.array([0.0, 50, 100, 119, 121, 300, 301])
    vals = np.full((1, 7), np.nan, np.float32)
    vals[0, 0] = 8
    vals[0, 5] = 3
    tid = np.zeros(7, np.int32)
    out = trace.ffill_capped(vals, t, tid, np.array([120.0]))[0]
    assert out[:4].tolist() == [8, 8, 8, 8]
    assert math.isnan(out[4])            # 121 s after the detail: past the cap → unknown
    assert out[5:].tolist() == [3, 3]
    tid2 = np.array([0, 0, 0, 1, 1, 1, 1], np.int32)
    out2 = trace.ffill_capped(vals, t, tid2, np.array([1e18]))[0]
    assert math.isnan(out2[3])           # never fills across traces


def test_dt_weighting_and_legs():
    tid = np.zeros(6, np.int32)
    t = np.array([0.0, 4, 8, 200, 204, 1000])
    flags = np.array([0, 0, 0, 0, 2, 0], np.int16)    # point 4 starts a new leg
    ll = np.full(6, 50.0)
    b = np.zeros(6, bool)
    dt, leg, *_ = trace.point_metrics(tid, t, ll, ll, b, ~b, flags, np.full(6, np.nan, np.float32),
                                      60.0, 1.0, 30.0, 1000.0)
    assert leg.tolist() == [0, 0, 0, 0, 1, 1]
    # next-point interval capped at 60 s; last point in a leg reuses its previous interval
    assert dt.tolist() == [4, 4, 60, 60, 60, 60]


def test_implied_speed_and_jump():
    tid = np.zeros(3, np.int32)
    t = np.array([0.0, 10, 20])
    lat = np.array([50.0, 50.0, 52.0])                 # 2° jump in 10 s ≈ 43,000 kt
    lon = np.array([10.0, 10.0 + 1.25 / 38.57, 10.0])
    b = np.zeros(3, bool)
    gs = np.full(3, 450, np.float32)
    _, _, implied, jump, mm = trace.point_metrics(tid, t, lat, lon, b, ~b, np.zeros(3, np.int16), gs,
                                                  60.0, 1.0, 30.0, 1000.0)
    assert math.isnan(implied[0])
    assert implied[1] == pytest.approx(450, rel=0.01)
    assert mm[1] == pytest.approx(0, abs=5)
    assert jump.tolist() == [False, False, True]


def test_histogram_additivity(P):
    """nic bins sum to s_adsb per row; hourly rows sum to the day totals in aircraft_dim."""
    a = straight_flight(1500, dt=4.0, t0=1000.0)                 # spans ~1.7 h and many hexes
    b = straight_flight(400, dt=4.0, t0=40000.0, lat0=56.0, nic=5)
    b += [point(41700 + i * 30, 56.0, 12.0, alt="ground", gs=0) for i in range(5)]
    res = run([trace_doc("aaa111", a), trace_doc("bbb222", b, r="X-B")], P)
    t1, dim = df(res["tier1"]), df(res["dim"]).set_index("icao")
    nic_cols = [f"nic_s_{i}" for i in range(12)] + ["nic_s_unk"]
    assert np.allclose(t1[nic_cols].sum(axis=1), t1["s_adsb"], atol=1e-3)
    assert t1["hour_utc"].nunique() > 1 and t1["hex5"].nunique() > 5
    per_ac = t1.groupby("icao")["s_adsb"].sum()
    assert per_ac["aaa111"] == pytest.approx(dim.loc["aaa111", "s_adsb_day"], rel=1e-5)
    # day rollup of hourly rows equals a direct sum
    assert t1.groupby("icao")["n_pts"].sum()["bbb222"] == dim.loc["bbb222", "n_pts_day"]
    assert dim.loc["bbb222", "nic_lt7_s_day"] == pytest.approx(dim.loc["bbb222", "s_adsb_day"], rel=1e-5)
    assert t1.loc[t1.icao == "bbb222", "s_ground"].sum() > 0
    assert set(t1["callsign"].dropna()) == {"TST123"}
    assert set(t1["category"].dropna()) == {"A3"}


def test_stale_and_out_of_day_dropped(P):
    pts = straight_flight(20)
    pts[3][6] = 1                                  # stale position
    pts.append(point(90000.0, 50, 11))              # tomorrow
    res = run([trace_doc(points=pts)], P)
    assert res["qa"]["points_stale"] == 1 and res["qa"]["points_out_of_day"] == 1
    assert res["qa"]["points_kept"] == 19


def test_exceptions_segments(P):
    pts = straight_flight(300, dt=4.0, t0=1000.0)
    for i in range(100, 130):
        pts[i][8] = det(nic=0)                      # 2 minutes of NIC 0
    pts[200][1] += 3.0                              # one position jump
    res = run([trace_doc(points=pts)], P)
    e = df(res["exc"])
    assert len(e) > 0
    assert (e["reason"] & REASON_NIC_LOW).any() and (e["reason"] & REASON_JUMP).any()
    assert (e["reason"] == REASON_CONTEXT).any()
    assert e["segment_id"].nunique() >= 2
    # ≤ 1 point per 10 s within a segment
    for _, g in e.groupby("segment_id"):
        assert (np.diff(np.sort(g["ts"].to_numpy())) >= 10 - 1e-6).all()
