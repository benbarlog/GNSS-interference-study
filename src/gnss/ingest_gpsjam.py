"""Polite, cached downloader and loader for GPSJAM daily H3 (res 4) CSVs.

Data: https://gpsjam.org (John Wiseman), derived from ADS-B Exchange.
Each file: hex, count_good_aircraft, count_bad_aircraft for one (UTC) day.
Earliest available date (probed 2026-10-14): 2022-02-14.
"""

from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

URL = "https://gpsjam.org/data/{d}-h3_4.csv"
USER_AGENT = "gnss-interference-study student project (github.com/benbarlog/GNSS-interference-study)"
EARLIEST = date(2022, 2, 14)
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "gpsjam"


def cache_path(d: date, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{d.isoformat()}-h3_4.csv"


def download_day(d: date, raw_dir: Path = RAW_DIR, session: requests.Session | None = None,
                 retries: int = 3) -> Path | None:
    """Download one day unless cached. Returns the path, or None if the date is missing (404)."""
    path = cache_path(d, raw_dir)
    if path.exists() and path.stat().st_size > 0:
        return path
    raw_dir.mkdir(parents=True, exist_ok=True)
    s = session or requests.Session()
    for attempt in range(retries):
        try:
            r = s.get(URL.format(d=d.isoformat()), headers={"User-Agent": USER_AGENT}, timeout=60)
        except requests.RequestException:
            time.sleep(2 ** (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        if r.ok:
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(r.content)
            tmp.replace(path)
            return path
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"failed to download {d} after {retries} attempts")


def download_dates(dates, raw_dir: Path = RAW_DIR, delay: float = 1.0) -> list[Path]:
    """Download many dates, sleeping `delay` seconds between network requests (not cache hits)."""
    paths = []
    with requests.Session() as s:
        for d in dates:
            cached = cache_path(d, raw_dir).exists()
            p = download_day(d, raw_dir, s)
            if p is None:
                print(f"missing: {d}")
            else:
                paths.append(p)
            if not cached:
                time.sleep(delay)
    return paths


def date_range(start: date, end: date, step_days: int = 1) -> list[date]:
    return [start + timedelta(days=i) for i in range(0, (end - start).days + 1, step_days)]


def load_panel(raw_dir: Path = RAW_DIR, dates=None) -> pd.DataFrame:
    """Load cached CSVs into a long panel: date, hex, good, bad, n, rate."""
    files = sorted(raw_dir.glob("*-h3_4.csv"))
    if dates is not None:
        wanted = {d.isoformat() for d in dates}
        files = [f for f in files if f.name[:10] in wanted]
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype={"hex": str})
        df.insert(0, "date", pd.Timestamp(f.name[:10]))
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True).rename(
        columns={"count_good_aircraft": "good", "count_bad_aircraft": "bad"})
    panel["n"] = panel["good"] + panel["bad"]
    panel["rate"] = panel["bad"] / panel["n"]
    return panel
