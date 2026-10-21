"""Failed days are retried at the end of their own tier, not after the whole archive."""

import argparse
import threading

from gnss.adsblol.config import Paths
from gnss.adsblol.manifest import Manifest
from gnss.adsblol.orchestrator import Orchestrator
from gnss.adsblol.runtime import Events, Status


def test_failed_day_retried_when_its_tier_is_done(tmp_path, cfg):
    cfg = dict(cfg, data_root=str(tmp_path))
    paths = Paths.from_config(cfg)
    paths.ensure()
    m = Manifest(paths.manifest)
    days = {"2024-09-15": "P1b", "2024-09-16": "P1b", "2026-10-01": "P2"}
    m.upsert_catalog([{"date": d, "repo": "r", "tag": f"v{d}", "variant": "prod",
                       "urls": [f"https://x/{d}.tar.aa"], "part_sizes": [3], "total_bytes": 3,
                       "preferred_tag": None, "selection_note": ""} for d in days])
    m.set_schedule([(d, t, i, False, None) for i, (d, t) in enumerate(days.items())])
    m.kv_set("gate1_passed", True)
    m.kv_set("gate2_passed", True)
    m.transition("2024-09-16", "failed", retry_count=1, last_error="QA")
    (paths.scratch / "2024-09-16").mkdir(parents=True)
    (paths.scratch / "2024-09-16" / "2024-09-16.tar.aa").write_bytes(b"abc")    # parts on disk

    args = argparse.Namespace(dates=None, tier=None, reprocess=None, max_days=None, until=None,
                              auto=False, no_follow=True, dry_run=False, skip_gpsjam=True)
    o = Orchestrator(cfg, paths, args, m, Status(), Events(console=False), threading.Event())

    assert not o._requeue_failed()                     # another P1b day is still pending
    for to in ("downloading", "downloaded", "processing", "processed", "verified", "cleaned"):
        m.transition("2024-09-15", to)
    assert o._requeue_failed()                         # P1b done; P2 still pending — retry now
    assert m.get("2024-09-16")["status"] == "downloaded"   # reprocessed from disk, not re-downloaded
    m.close()
