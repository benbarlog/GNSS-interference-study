"""Process one archive day (plan §6.1): stream the split tar, fan traces out to a worker pool,
write Tier 1 / aircraft_dim / exceptions parquet to a temp dir, then move it into place.

    parts (scratch or raw_retained) → one virtual stream → tarfile "r|" (no extraction)
      → batches of `batch_traces` trace_full members → Pool (BELOW_NORMAL, SIGINT ignored)
      → writer thread → _tmp/<date>/<table>/part-0.parquet → rename into date=<date>/

Windows uses `spawn`, so the worker entry points are module-level and receive bytes only.
"""

from __future__ import annotations

import calendar
import collections
import io
import json
import multiprocessing as mp
import os
import queue
import shutil
import signal
import tarfile
import threading
import time
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import PIPELINE_VERSION
from .config import config_hash
from .schemas import SCHEMAS

TABLE_KEYS = {"tier1": "tier1", "aircraft_dim": "dim", "exceptions": "exc"}


class Interrupted(Exception):
    pass


# ---- worker side -----------------------------------------------------------------------------
_P: dict = {}


def _worker_init(params: dict) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if params.get("numba_cache_dir"):
        os.environ["NUMBA_CACHE_DIR"] = params["numba_cache_dir"]
    from .runtime import lower_priority
    if params.get("low_priority", True):
        lower_priority()
    _P.update(params)
    from .trace import warmup
    warmup()


def _worker_run(batch: list[tuple[str, bytes]], day_start: float, sample_n: int) -> dict:
    from .trace import process_batch
    return process_batch(batch, day_start, _P, sample_n)


def worker_params(cfg: dict, numba_cache_dir: str | None = None, low_priority: bool = True) -> dict:
    p = dict(cfg["process"])
    p.update(cfg["exceptions"])
    p["pipeline_version"] = PIPELINE_VERSION
    p["config_hash"] = config_hash(cfg)
    p["numba_cache_dir"] = numba_cache_dir
    p["low_priority"] = low_priority
    return p


def make_pool(cfg: dict, workers: int | None = None, numba_cache_dir: str | None = None,
              low_priority: bool = True):
    ctx = mp.get_context("spawn")
    n = int(workers or cfg["process"]["workers"])
    return ctx.Pool(n, initializer=_worker_init,
                    initargs=(worker_params(cfg, numba_cache_dir, low_priority),),
                    maxtasksperchild=int(cfg["process"]["maxtasksperchild"])), n


# ---- tar streaming ---------------------------------------------------------------------------
class ChainedReader(io.RawIOBase):
    """Read several part files as one stream (split tar: .tar.aa, .tar.ab, ...)."""

    def __init__(self, paths: list[Path]):
        self.paths = list(paths)
        self.total = sum(p.stat().st_size for p in self.paths)
        self.pos = 0
        self._i = 0
        self._f = open(self.paths[0], "rb") if self.paths else None

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while self._f is not None:
            n = self._f.readinto(b)
            if n:
                self.pos += n
                return n
            self._f.close()
            self._i += 1
            self._f = open(self.paths[self._i], "rb") if self._i < len(self.paths) else None
        return 0

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
        super().close()


def iter_traces(paths: list[Path], counters: dict):
    """Yield (member_name, bytes) for every trace_full member; counts all members."""
    raw = ChainedReader(paths)
    counters["tar_bytes"] = raw.total
    counters["reader"] = raw
    stream = io.BufferedReader(raw, buffer_size=8 * 2**20)
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        for m in tf:
            counters["members"] += 1
            if not m.isfile():
                continue
            if "trace_full" not in m.name:
                counters["other_members"] += 1
                continue
            f = tf.extractfile(m)
            yield m.name, f.read()
    raw.close()


def day_start_epoch(d: str) -> float:
    return float(calendar.timegm(date.fromisoformat(d).timetuple()))


# ---- writer ----------------------------------------------------------------------------------
class DayWriter:
    """Buffers worker tables and writes each table as one parquet file with large row groups."""

    def __init__(self, tmp_dir: Path, rowgroup_rows: int = 500_000):
        self.tmp_dir = tmp_dir
        self.rowgroup_rows = rowgroup_rows
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.buffers: dict[str, list[pa.Table]] = collections.defaultdict(list)
        self.buffered: dict[str, int] = collections.Counter()
        self.rows: dict[str, int] = collections.Counter()
        for table, schema in SCHEMAS.items():
            d = tmp_dir / table
            d.mkdir(parents=True, exist_ok=True)
            self.writers[table] = pq.ParquetWriter(d / "part-0.parquet", schema, compression="zstd",
                                                   compression_level=6)

    def add(self, table: str, t: pa.Table | None) -> None:
        if t is None or t.num_rows == 0:
            return
        self.buffers[table].append(t)
        self.buffered[table] += t.num_rows
        if self.buffered[table] >= self.rowgroup_rows:
            self._flush(table)

    def _flush(self, table: str) -> None:
        if not self.buffers[table]:
            return
        t = pa.concat_tables(self.buffers[table])
        self.writers[table].write_table(t, row_group_size=max(self.rowgroup_rows, t.num_rows))
        self.rows[table] += t.num_rows
        self.buffers[table] = []
        self.buffered[table] = 0

    def close(self) -> dict[str, int]:
        for table, w in self.writers.items():
            self._flush(table)
            w.close()
        return dict(self.rows)

    def abort(self) -> None:
        for w in self.writers.values():
            try:
                w.close()
            except Exception:
                pass


def merge_counters(dst: dict, src: dict) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0) + v


def merge_verify(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, collections.Counter) or isinstance(v, dict):
            dst.setdefault(k, collections.Counter()).update(v)
        elif isinstance(v, list):
            lst = dst.setdefault(k, [])
            lst.extend(v[: max(0, 10 - len(lst))])
        else:
            dst[k] = dst.get(k, 0) + v


def publish(tmp_dir: Path, paths, d: str) -> dict[str, int]:
    """Move _tmp/<date>/<table> into <table_dir>/date=<date> (replacing an older version)."""
    sizes = {}
    for table in SCHEMAS:
        src = tmp_dir / table
        dst = paths.table_dir(table) / f"date={d}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        sizes[table] = sum(f.stat().st_size for f in src.glob("*.parquet"))
        if dst.exists():
            old = dst.with_name(f"{dst.name}.old-{int(time.time())}")
            os.replace(dst, old)
            os.replace(src, dst)
            shutil.rmtree(old, ignore_errors=True)
        else:
            os.replace(src, dst)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return sizes


# ---- day processing --------------------------------------------------------------------------
class DayProcessor:
    def __init__(self, cfg: dict, paths, pool, workers: int, status=None, events=None,
                 stop: threading.Event | None = None):
        self.cfg = cfg
        self.paths = paths
        self.pool = pool
        self.workers = workers
        self.status = status
        self.events = events
        self.stop = stop or threading.Event()

    def run(self, d: str, part_paths: list[Path], sample_per_batch: int = 2) -> dict:
        """Process a day. Returns the pre-QA day_qa dict. Raises Interrupted on stop."""
        pc = self.cfg["process"]
        batch_n = int(pc["batch_traces"])
        max_inflight = int(pc["max_inflight_batches"])
        t0 = time.time()
        day_start = day_start_epoch(d)
        tmp_dir = self.paths.tmp / d
        shutil.rmtree(tmp_dir, ignore_errors=True)
        writer = DayWriter(tmp_dir, int(pc.get("rowgroup_rows", 500_000)))
        results: queue.Queue = queue.Queue()
        sem = threading.BoundedSemaphore(max_inflight)
        qa: dict = {}
        verify: dict = {}
        counters: dict = collections.Counter()
        failure: list[BaseException] = []
        outstanding = [0]
        lock = threading.Lock()

        def on_result(res):
            results.put(res)

        def on_error(e):
            results.put(e)

        def drain(block: bool) -> None:
            while True:
                try:
                    res = results.get(block=block, timeout=1 if block else None)
                except queue.Empty:
                    return
                with lock:
                    outstanding[0] -= 1
                sem.release()
                if isinstance(res, BaseException):
                    failure.append(res)
                else:
                    for table, key in TABLE_KEYS.items():
                        writer.add(table, res[key])
                    merge_counters(qa, res["qa"])
                    merge_verify(verify, res["verify"])
                    if self.status:
                        self.status.set(process_traces=qa.get("traces", 0))
                block = False

        if self.status:
            self.status.set(process_day=d, process_traces=0, process_bytes_frac=0.0)
        try:
            batch: list[tuple[str, bytes]] = []
            it = iter_traces(part_paths, counters)

            def submit(b):
                while not sem.acquire(timeout=1):
                    drain(False)
                    if self.stop.is_set():
                        raise Interrupted()
                with lock:
                    outstanding[0] += 1
                self.pool.apply_async(_worker_run, (b, day_start, sample_per_batch),
                                      callback=on_result, error_callback=on_error)

            for name, raw in it:
                if self.stop.is_set():
                    raise Interrupted()
                batch.append((name, raw))
                if len(batch) >= batch_n:
                    submit(batch)
                    batch = []
                    drain(False)
                    if failure:
                        raise failure[0]
                    if self.status and counters.get("reader"):
                        r = counters["reader"]
                        self.status.set(process_bytes_frac=r.pos / max(r.total, 1))
            if batch:
                submit(batch)
            while True:
                with lock:
                    left = outstanding[0]
                if left == 0:
                    break
                if self.stop.is_set():
                    raise Interrupted()
                drain(True)
            if failure:
                raise failure[0]
            rows = writer.close()
        except BaseException:
            writer.abort()
            # let in-flight tasks finish into the void; their results are discarded
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        sizes = publish(tmp_dir, self.paths, d)
        secs = time.time() - t0
        counters.pop("reader", None)
        day_qa = {
            "date": d,
            "pipeline_version": PIPELINE_VERSION,
            "config_hash": config_hash(self.cfg),
            "tar_bytes": int(counters.get("tar_bytes", 0)),
            "members": int(counters.get("members", 0)),
            "other_members": int(counters.get("other_members", 0)),
            "trace_count": int(qa.get("traces", 0)),
            "parse_errors": int(qa.get("parse_errors", 0)),
            "aircraft_count": int(qa.get("aircraft", 0)),
            "point_count": int(qa.get("points_total", 0)),
            "points_kept": int(qa.get("points_kept", 0)),
            "points_stale": int(qa.get("points_stale", 0)),
            "points_out_of_day": int(qa.get("points_out_of_day", 0)),
            "adsb_share": _share(qa.get("pts_air_adsb", 0), qa.get("pts_air", 0)),
            "adsb_time_share": _share(qa.get("s_adsb", 0), qa.get("s_obs", 0) - qa.get("s_ground", 0)),
            "nic_known_share": _share(qa.get("s_adsb_nic_known", 0), qa.get("s_adsb", 0)),
            "s_obs": round(float(qa.get("s_obs", 0)), 1),
            "s_adsb": round(float(qa.get("s_adsb", 0)), 1),
            "rows": rows,
            "bytes": sizes,
            "processing_seconds": round(secs, 1),
            "workers": self.workers,
            "verify": _verify_json(verify),
        }
        return day_qa


def _share(a, b) -> float | None:
    return round(float(a) / float(b), 4) if b else None


def _verify_json(v: dict) -> dict:
    out = {}
    for k, val in v.items():
        if isinstance(val, collections.Counter) or isinstance(val, dict):
            out[k] = {str(kk): int(vv) for kk, vv in sorted(val.items(), key=lambda kv: -kv[1])[:60]}
        else:
            out[k] = val
    mp_, near = v.get("mlat_pts", 0), v.get("mlat_near_adsb", 0)
    out["mlat_concurrent_share"] = round(near / mp_, 4) if mp_ else None
    return out


def write_day_qa(paths, day_qa: dict) -> Path:
    p = paths.day_qa / f"date={day_qa['date']}.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(day_qa, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, p)
    return p
