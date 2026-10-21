"""Download release parts for queued days into scratch, ahead of processing (plan §4, §9.1).

- ≤ `max_connections` (hard ceiling 2) concurrent part downloads, priority order
- HTTP Range resume into `<part>.partial`; size-verified, then renamed; finished parts and days are
  never requested again
- Retry-After-aware exponential backoff and the long pause come from the shared `Backoff`
- optional `max_download_mbps` token bucket shared by all connections
- stops starting new days above the buffer high-water mark or when D: free space is low
"""

from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import requests

from .http import PoliteClient, Stopped, retry_after_seconds

EST_DAY_BYTES = 7 * 2**30     # used for space checks when a day's size is not resolved yet


class TokenBucket:
    def __init__(self, mbps: float | None):
        self.rate = mbps * 1e6 / 8 if mbps else None
        self.tokens = 0.0
        self.t = time.monotonic()
        self.lock = threading.Lock()

    def take(self, n: int) -> None:
        if not self.rate:
            return
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.rate, self.tokens + (now - self.t) * self.rate)
            self.t = now
            self.tokens -= n
            wait = -self.tokens / self.rate if self.tokens < 0 else 0
        if wait > 0:
            time.sleep(wait)


def day_dir(paths, row: dict) -> Path:
    return Path(row["raw_dir"]) if row.get("raw_dir") else paths.scratch / row["date"]


def part_name(url: str) -> str:
    return url.rsplit("/", 1)[-1]


def part_paths(paths, row: dict) -> list[Path]:
    d = day_dir(paths, row)
    return [d / part_name(u) for u in row["urls"]]


def part_complete(p: Path, expected: int | None) -> bool:
    return p.exists() and (expected is None or p.stat().st_size == expected)


def day_complete(paths, row: dict) -> bool:
    sizes = row.get("part_sizes") or [None] * len(row["urls"])
    return all(part_complete(p, s) for p, s in zip(part_paths(paths, row), sizes))


class Downloader:
    def __init__(self, cfg: dict, paths, manifest, client: PoliteClient, status, events,
                 stop: threading.Event, eligible):
        """eligible() -> (tiers or None, dates or None): what may be downloaded now."""
        self.cfg = cfg
        self.paths = paths
        self.manifest = manifest
        self.client = client
        self.status = status
        self.events = events
        self.stop = stop
        self.eligible = eligible
        dl = cfg["download"]
        self.n = int(dl["max_connections"])
        self.hw = float(dl["buffer_high_water_gib"]) * 2**30
        self.min_free = float(dl["min_free_gib"]) * 2**30
        self.chunk = int(float(dl["chunk_mib"]) * 2**20)
        self.bucket = TokenBucket(dl.get("max_download_mbps"))
        self.lock = threading.Lock()
        self.assigned: set[tuple[str, int]] = set()
        self.threads: list[threading.Thread] = []
        self.idle = [True] * self.n

    def set_mbps(self, mbps: float | None) -> None:
        self.bucket = TokenBucket(mbps)

    def start(self) -> None:
        for i in range(self.n):
            th = threading.Thread(target=self._loop, args=(i,), name=f"download-{i}", daemon=True)
            th.start()
            self.threads.append(th)

    def join(self, timeout: float = 30) -> None:
        for th in self.threads:
            th.join(timeout)

    def busy(self) -> bool:
        return not all(self.idle)

    # ---- scheduling --------------------------------------------------------------------------
    def _claim(self) -> tuple[dict, int] | None:
        tiers, dates = self.eligible()
        with self.lock:
            rows = self.manifest.days(statuses=["downloading", "queued"], tiers=tiers, dates=dates)
            rows.sort(key=lambda r: (r["status"] != "downloading", r["priority"] or 0))
            buffer = self.manifest.buffer_bytes()
            free = shutil.disk_usage(self.paths.root).free
            if free < self.min_free:
                self.status.flag("DISK LOW", f"{free / 2**30:.0f} GiB free")
            else:
                self.status.unflag("DISK LOW")
            for row in rows:
                d = row["date"]
                sizes = row.get("part_sizes") or [None] * len(row["urls"])
                if row["status"] == "queued":
                    if buffer >= self.hw:
                        self.status.flag("BUFFER FULL", f"{buffer / 2**30:.0f} GiB")
                        return None
                    self.status.unflag("BUFFER FULL")
                    need = row.get("total_bytes") or EST_DAY_BYTES
                    if free - need < self.min_free:
                        self.status.flag("DISK LOW", f"{free / 2**30:.0f} GiB free")
                        return None
                    if day_complete(self.paths, row):
                        self.manifest.transition(d, "downloaded", "parts already on disk",
                                                 t_downloaded=time.time())
                        continue
                ppaths = part_paths(self.paths, row)
                for i, (p, s) in enumerate(zip(ppaths, sizes)):
                    if (d, i) in self.assigned or part_complete(p, s):
                        continue
                    if row["status"] == "queued":
                        self.manifest.transition(d, "downloading", t_download_start=time.time(),
                                                 raw_dir=None)
                        row["status"] = "downloading"
                    self.assigned.add((d, i))
                    return row, i
        return None

    def _loop(self, k: int) -> None:
        while not self.stop.is_set():
            try:
                self.client.backoff.wait(self.stop)
                task = self._claim()
            except Stopped:
                return
            except Exception as e:
                self.events.error(f"downloader scheduling error: {type(e).__name__}: {e}")
                self.stop.wait(30)
                continue
            if task is None:
                self.idle[k] = True
                self.stop.wait(5)
                continue
            self.idle[k] = False
            row, i = task
            try:
                self._download(row, i)
            except Stopped:
                return
            except Exception as e:
                self.events.error(f"download {row['date']} part {i}: {type(e).__name__}: {e}")
                self.client.backoff.failure(f"{type(e).__name__}")
            finally:
                with self.lock:
                    self.assigned.discard((row["date"], i))

    # ---- one part ----------------------------------------------------------------------------
    def _download(self, row: dict, i: int) -> None:
        d = row["date"]
        url = row["urls"][i]
        sizes = list(row.get("part_sizes") or [None] * len(row["urls"]))
        expected = sizes[i]
        final = part_paths(self.paths, row)[i]
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_name(final.name + ".partial")
        key = f"{d} {final.suffix}"
        offset = partial.stat().st_size if partial.exists() else 0
        if expected is not None and offset > expected:
            partial.unlink()
            offset = 0
        if expected is not None and offset == expected:
            partial.replace(final)
            self._maybe_done(d)
            return
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        t0 = time.time()
        try:
            r = self.client.session().get(url, headers=headers, stream=True,
                                          timeout=(30, float(self.cfg["download"]["timeout_s"])))
        except requests.RequestException as e:
            self.client.backoff.failure(f"{type(e).__name__} on {final.name}")
            return
        with r:
            if r.status_code in (404, 410):
                self._fail(d, f"HTTP {r.status_code} for {final.name}")
                return
            if r.status_code == 416:
                partial.unlink(missing_ok=True)
                self.client.backoff.failure(f"HTTP 416 on {final.name}; restarting part")
                return
            if r.status_code not in (200, 206):
                self.client.backoff.failure(f"HTTP {r.status_code} on {final.name}", retry_after_seconds(r))
                return
            if offset and r.status_code == 200:
                offset = 0          # server ignored Range: start over
            length = r.headers.get("Content-Length")
            total = expected if expected is not None else (offset + int(length) if length else None)
            done = offset
            with open(partial, "ab" if offset else "wb") as f:
                for chunk in r.iter_content(self.chunk):
                    if self.stop.is_set():
                        self.status.clear_download(key)
                        raise Stopped()
                    if not chunk:
                        continue
                    self.bucket.take(len(chunk))
                    f.write(chunk)
                    done += len(chunk)
                    self.status.download_meter.add(len(chunk))
                    self.status.set_download(key, done, total or 0)
                    if done % (256 * 2**20) < len(chunk):      # every ~256 MiB
                        self.manifest.update(d, bytes_done=self._day_bytes(row, i, done))
        self.status.clear_download(key)
        size = partial.stat().st_size
        if total is not None and size != total:
            if size > total:
                partial.unlink()
            self.client.backoff.failure(f"{final.name}: got {size} of {total} bytes")
            return
        partial.replace(final)
        self.client.backoff.success()
        if expected is None:
            sizes[i] = size
            self.manifest.update(d, part_sizes=sizes)
        self.events.info(f"downloaded {final.name} ({size / 2**30:.2f} GiB, "
                         f"{size / max(time.time() - t0, 1) / 2**20:.1f} MiB/s)")
        self._maybe_done(d)

    def _day_bytes(self, row: dict, i: int, done_i: int) -> int:
        tot = 0
        sizes = row.get("part_sizes") or [None] * len(row["urls"])
        for j, (p, s) in enumerate(zip(part_paths(self.paths, row), sizes)):
            if j == i:
                tot += done_i
            elif p.exists():
                tot += p.stat().st_size
        return tot

    def _maybe_done(self, d: str) -> None:
        with self.lock:
            row = self.manifest.get(d)
            if row and row["status"] == "downloading" and day_complete(self.paths, row):
                sizes = [p.stat().st_size for p in part_paths(self.paths, row)]
                start = row.get("t_download_start") or time.time()
                self.manifest.transition(d, "downloaded", t_downloaded=time.time(),
                                         download_sec=time.time() - start, part_sizes=sizes,
                                         total_bytes=sum(sizes), bytes_done=sum(sizes))
                self.events.info(f"{d} downloaded ({sum(sizes) / 2**30:.2f} GiB)", True)

    def _fail(self, d: str, why: str) -> None:
        with self.lock:
            row = self.manifest.get(d)
            if row and row["status"] in ("downloading", "queued"):
                self.manifest.transition(d, "failed", why, last_error=why,
                                         retry_count=(row["retry_count"] or 0) + 1)
        self.events.error(f"{d}: {why}")
