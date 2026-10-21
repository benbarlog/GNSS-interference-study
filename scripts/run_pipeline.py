"""The single entry point for the adsb.lol pipeline (adsb_lol_pipeline_plan.md §9.3, §10).

Run it in Windows Terminal (not the classic console, whose QuickEdit mode pauses programs):

    python scripts/run_pipeline.py

It runs everything in order: catalog → schedule → GPSJAM daily backfill → variation check →
P0 → Gate 1 → P1a → Gate 2 → P1b → P2 → P3a → P3b → follow mode. At each gate it prints a report
and waits for you to type `continue`. Re-run the same command after any stop or reboot; it resumes
from D:/gnss-data/state/manifest.sqlite. Ctrl+C stops gracefully; a second Ctrl+C forces exit.

Common flags:
    --auto                 skip gate pauses (only once both gates have been passed)
    --until P2             stop after a phase
    --workers N            worker processes (default from config, 8)
    --max-download-mbps X  bandwidth cap
    --heartbeat-sec S      heartbeat interval (default 60)
    --live                 full-screen dashboard instead of the scrolling ticker
    --no-follow            exit when the archive is complete

Advanced (manual runs; no gates, no follow mode):
    --tier P1b             only this tier
    --dates 2025-06-14,2025-06-10:2025-06-12
    --max-days N           only the next N unfinished days
    --reprocess DATES      reprocess retained raw days (after a schema change)
    --dry-run              build catalog + schedule, print the plan, download nothing
    --skip-gpsjam          skip the GPSJAM backfill and analysis
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnss.adsblol.config import Paths, load_config, write_config_copy
from gnss.adsblol.manifest import TIERS, Manifest


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to adsblol.yaml (default: config/adsblol.yaml)")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--until", choices=TIERS)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--max-download-mbps", type=float)
    ap.add_argument("--heartbeat-sec", type=float)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--no-follow", action="store_true")
    ap.add_argument("--tier", choices=TIERS)
    ap.add_argument("--dates")
    ap.add_argument("--max-days", type=int)
    ap.add_argument("--reprocess")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-gpsjam", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, {
        "process.workers": args.workers,
        "download.max_download_mbps": args.max_download_mbps,
        "pipeline.heartbeat_sec": args.heartbeat_sec,
    })
    paths = Paths.from_config(cfg)
    if not paths.root.parent.exists():
        print(f"data_root drive not found: {paths.root}", file=sys.stderr)
        return 2
    paths.ensure()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(paths.state / "numba_cache"))

    from gnss.adsblol.heartbeat import Heartbeat
    from gnss.adsblol.orchestrator import Orchestrator
    from gnss.adsblol.runtime import Events, Status, lower_priority, prevent_sleep, setup_logging

    setup_logging(paths.logs)
    lower_priority()
    if cfg["pipeline"].get("prevent_sleep", True) and not args.dry_run:
        prevent_sleep(True)
    write_config_copy(cfg, paths)

    lock = paths.state / "pipeline.lock"
    hb_age = time.time() - paths.heartbeat.stat().st_mtime if paths.heartbeat.exists() else 1e9
    if lock.exists() and hb_age < 300:
        print("Another pipeline instance seems to be running (heartbeat < 5 min old). "
              f"If not, delete {lock}.", file=sys.stderr)
        return 3
    lock.write_text(str(os.getpid()))

    manifest = Manifest(paths.manifest)
    status = Status()
    events = Events()
    stop = threading.Event()
    hb = Heartbeat(manifest, status, events, paths, float(cfg["pipeline"]["heartbeat_sec"]), args.live)

    for d, a, b in manifest.recover_on_start():
        events.info(f"recovered {d}: {a} → {b}")

    presses = [0]

    def on_signal(signum, frame):
        presses[0] += 1
        if presses[0] == 1:
            stop.set()
            events.warn("stopping gracefully (Ctrl+C again to force)…")
        else:
            os._exit(130)

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, on_signal)

    hb.start()
    code = 0
    try:
        Orchestrator(cfg, paths, args, manifest, status, events, stop, hb).run()
    except Exception as e:
        import traceback
        events.error(f"fatal: {type(e).__name__}: {e}")
        events.info(traceback.format_exc())
        code = 1
    finally:
        hb.stop()
        manifest.close()
        lock.unlink(missing_ok=True)
        if cfg["pipeline"].get("prevent_sleep", True):
            prevent_sleep(False)
    return code


if __name__ == "__main__":
    sys.exit(main())
