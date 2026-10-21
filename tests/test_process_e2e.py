"""End to end on a tiny synthetic split tar: tar streaming → spawn pool → parquet → QA."""

import json

import pyarrow.parquet as pq
import pytest

from conftest import DAY, det, gz, make_tar, straight_flight, trace_doc
from gnss.adsblol.config import Paths
from gnss.adsblol.process import DayProcessor, iter_traces, make_pool, write_day_qa
from gnss.adsblol.qa import verify_day


@pytest.fixture
def setup(tmp_path, cfg):
    cfg = dict(cfg)
    cfg["process"] = dict(cfg["process"], batch_traces=3, max_inflight_batches=2)
    paths = Paths(tmp_path / "gnss-data")
    paths.ensure()
    members = {}
    for k in range(7):
        pts = straight_flight(300, t0=1000.0 + k * 3000, lat0=40.0 + k)
        if k == 3:
            for i in range(50, 80):
                pts[i][8] = det(nic=0)
        icao = f"a{k:05d}"
        members[f"./traces/{icao[-2:]}/trace_full_{icao}.json"] = gz(trace_doc(icao, pts, r=f"N{k}", t="B738"))
    parts = make_tar(members, n_parts=3)
    day = paths.scratch / DAY
    day.mkdir(parents=True)
    files = []
    for i, data in enumerate(parts):
        f = day / f"v2025.06.14-planes-readsb-prod-0.tar.a{chr(97 + i)}"
        f.write_bytes(data)
        files.append(f)
    return cfg, paths, files


def test_iter_traces_over_split_parts(setup):
    _, _, files = setup
    counters = {"members": 0, "other_members": 0}
    names = [n for n, _ in iter_traces(files, counters)]
    assert len(names) == 7 and counters["other_members"] == 2


def test_day_end_to_end(setup):
    cfg, paths, files = setup
    pool, n = make_pool(cfg, workers=2, numba_cache_dir=str(paths.state / "numba"), low_priority=False)
    try:
        q = DayProcessor(cfg, paths, pool, n).run(DAY, files, sample_per_batch=3)
    finally:
        pool.terminate()
        pool.join()
    assert q["trace_count"] == 7 and q["aircraft_count"] == 7 and q["parse_errors"] == 0
    assert q["nic_known_share"] > 0.9 and q["adsb_share"] == 1.0
    t1 = pq.read_table(paths.tier1 / f"date={DAY}").to_pandas()
    dim = pq.read_table(paths.aircraft_dim / f"date={DAY}").to_pandas()
    exc = pq.read_table(paths.exceptions / f"date={DAY}").to_pandas()
    assert len(t1) == q["rows"]["tier1"] and t1["icao"].nunique() == 7
    assert len(dim) == 7 and set(dim["t"]) == {"B738"}
    assert set(exc["icao"]) == {"a00003"}
    assert not (paths.tmp / DAY).exists()
    write_day_qa(paths, q)
    verify_day(paths, cfg, q)
    assert q["qa"]["ok"], q["qa"]
    assert any("GPSJAM" in w for w in q["qa"]["warnings"])
    saved = json.loads((paths.day_qa / f"date={DAY}.json").read_text())
    assert saved["verify"]["paths"] == {"./traces": 7}
