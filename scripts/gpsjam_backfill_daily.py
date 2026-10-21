"""Fill every GPSJAM day from 2023-02-16 (config `archive.earliest_date`) into D:/gnss-data/gpsjam/.

`run_pipeline.py` already does this as its first data step; this script runs it on its own.
It reuses `gnss.ingest_gpsjam` (URL, cache naming, date ranges), copies the repo's existing
weekly cache first, and downloads politely: 1 connection, ≥ 1 s between files, Retry-After-aware
backoff, descriptive User-Agent. Idempotent.

    python scripts/gpsjam_backfill_daily.py
    python scripts/gpsjam_backfill_daily.py --start 2022-02-14 --end 2022-12-31
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnss.adsblol.config import Paths, load_config
from gnss.adsblol.gpsjam import backfill
from gnss.adsblol.http import Backoff, PoliteClient
from gnss.adsblol.runtime import Events, setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--start", type=date.fromisoformat)
    ap.add_argument("--end", type=date.fromisoformat)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = Paths.from_config(cfg)
    paths.ensure()
    setup_logging(paths.logs)
    events = Events(console=False)
    client = PoliteClient(cfg, Backoff(cfg, events))
    start = args.start or cfg["archive"]["earliest_date"]
    print(f"GPSJAM daily backfill {start} → {args.end or 'yesterday'} into {paths.gpsjam}", flush=True)
    res = backfill(paths, client, start, args.end, events=None)
    print(f"fetched {res['fetched']}, missing (404) {res['missing']}, copied from repo cache "
          f"{res['copied']}, files on disk {res['files']} ({res['seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
