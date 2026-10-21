"""Orchestrator end to end, offline: local HTTP server, synthetic tars for two days,
P0 → Gate 1 (simulated 'continue') → P1a → stop (--until P1a)."""

import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from conftest import gz, make_tar, straight_flight, trace_doc

from gnss.adsblol import orchestrator as orch
from gnss.adsblol.config import Paths
from gnss.adsblol.manifest import Manifest
from gnss.adsblol.runtime import Events, Status

DAYS = {"2025-06-14": "P0", "2024-10-01": "P1a"}
BLOBS: dict[str, bytes] = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        data = BLOBS.get(self.path)
        self.send_response(200 if data is not None else 404)
        self.send_header("Content-Length", str(len(data or b"")))
        self.end_headers()
        self.wfile.write(data or b"")


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_p0_gate_p1a(tmp_path, cfg, server, monkeypatch):
    import calendar
    from datetime import date
    rows = []
    for d in DAYS:
        day0 = float(calendar.timegm(date.fromisoformat(d).timetuple()))
        members = {f"./traces/0{k}/trace_full_b0000{k}.json": gz(dict(trace_doc(f"b0000{k}", straight_flight(150, t0=2000.0 + 900 * k)), timestamp=day0))
                   for k in range(5)}
        parts = make_tar(members, 2)
        urls = []
        for i, data in enumerate(parts):
            path = f"/{d}/v{d}-planes-readsb-prod-0.tar.a{chr(97 + i)}"
            BLOBS[path] = data
            urls.append(server + path)
        rows.append({"date": d, "repo": "globe_history_test", "tag": f"v{d}-planes-readsb-prod-0",
                     "variant": "prod", "urls": urls, "part_sizes": [len(p) for p in parts],
                     "total_bytes": sum(len(p) for p in parts), "preferred_tag": None,
                     "selection_note": "preferred"})

    def fake_catalog(cfg_, paths_, manifest_, client_, events_=None, max_age_s=0.0):
        manifest_.upsert_catalog(rows)
        return {"days": 2, "new": 2, "changed": 0, "bytes": 0, "unresolved_sizes": 0}

    monkeypatch.setattr(orch, "run_catalog", fake_catalog)
    monkeypatch.setattr(orch, "toast", lambda *a: None)
    replies = iter(["continue"])
    monkeypatch.setattr(orch.Orchestrator, "_read_line", lambda self, timeout: next(replies, None))

    cfg = dict(cfg, data_root=str(tmp_path / "gnss-data"))
    cfg["process"] = dict(cfg["process"], workers=2)
    cfg["qa"] = dict(cfg["qa"], min_nic_known_share=0.0)
    paths = Paths.from_config(cfg)
    paths.ensure()
    m = Manifest(paths.manifest)
    args = argparse.Namespace(dates=None, tier=None, reprocess=None, max_days=None, until="P1a",
                              auto=False, no_follow=True, dry_run=False, skip_gpsjam=True)
    stop = threading.Event()
    o = orch.Orchestrator(cfg, paths, args, m, Status(), Events(console=False), stop)
    o.run()

    for d, tier in DAYS.items():
        r = m.get(d)
        assert r["tier"] == tier and r["status"] == "cleaned", (d, r["status"], r["last_error"])
        assert (paths.tier1 / f"date={d}" / "part-0.parquet").exists()
        assert (paths.raw_retained / d).exists() and not (paths.scratch / d).exists()   # P0/P1a retained
    assert m.kv_get("gate1_passed") is True and not m.kv_get("gate2_passed")
    assert (paths.state / "gate1_report.txt").read_text(encoding="utf-8").startswith("══")
    m.close()
