"""Shared runtime plumbing: event bus, live status, logging, process priority, keep-awake.

Only the heartbeat thread writes to the console (plan §9.3). Everything else calls `Events.emit`,
which logs full detail to the rotating log file and queues a one-line message for the heartbeat.
"""

from __future__ import annotations

import collections
import logging
import logging.handlers
import queue
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("gnss.adsblol")


def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "pipeline.log", maxBytes=20 * 2**20, backupCount=10, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(threadName)s %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]
    for noisy in ("urllib3", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class Events:
    """Thread-safe event sink. `important` events print immediately between heartbeat ticks."""

    def __init__(self, console: bool = True):
        self.q: queue.Queue[tuple[float, str, str]] = queue.Queue()
        self.console = console
        self.warnings = 0
        self.errors = 0
        self._lock = threading.Lock()

    def emit(self, msg: str, level: str = "info", important: bool = True) -> None:
        getattr(log, "warning" if level == "warn" else level)(msg)
        with self._lock:
            if level == "warn":
                self.warnings += 1
            elif level == "error":
                self.errors += 1
        if important and self.console:
            self.q.put((time.time(), level, msg))

    def info(self, msg: str, important: bool = False) -> None:
        self.emit(msg, "info", important)

    def warn(self, msg: str) -> None:
        self.emit(msg, "warn", True)

    def error(self, msg: str) -> None:
        self.emit(msg, "error", True)


class RateMeter:
    """Rolling byte/event rate over a time window."""

    def __init__(self, horizon_s: float = 3600):
        self.horizon = horizon_s
        self.samples: collections.deque[tuple[float, float]] = collections.deque()
        self._lock = threading.Lock()

    def add(self, amount: float, t: float | None = None) -> None:
        t = time.time() if t is None else t
        with self._lock:
            self.samples.append((t, amount))
            while self.samples and self.samples[0][0] < t - self.horizon:
                self.samples.popleft()

    def rate(self, window_s: float) -> float:
        now = time.time()
        with self._lock:
            total = sum(a for t, a in self.samples if t >= now - window_s)
            first = next((t for t, _ in self.samples if t >= now - window_s), None)
        if first is None:
            return 0.0
        span = max(min(window_s, now - first), 1.0)
        return total / span


class Status:
    """Live, lock-protected status shared by the orchestrator, downloader, processor and heartbeat."""

    def __init__(self):
        self._lock = threading.Lock()
        self.started = time.time()
        self.phase = "starting"
        self.phase_detail = ""
        self.flags: dict[str, str] = {}          # name -> detail, e.g. {"BACKOFF": "until 14:50"}
        self.downloads: dict[str, tuple[int, int]] = {}   # "date part" -> (done, total)
        self.download_meter = RateMeter(3600)
        self.process_day: str | None = None
        self.process_traces = 0
        self.process_bytes_frac = 0.0
        self.last_qa: dict | None = None
        self.workers = 0
        self.gate: str | None = None

    def set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def flag(self, name: str, detail: str = "") -> None:
        with self._lock:
            self.flags[name] = detail

    def unflag(self, name: str) -> None:
        with self._lock:
            self.flags.pop(name, None)

    def set_download(self, key: str, done: int, total: int) -> None:
        with self._lock:
            self.downloads[key] = (done, total)

    def clear_download(self, key: str) -> None:
        with self._lock:
            self.downloads.pop(key, None)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "started": self.started, "phase": self.phase, "phase_detail": self.phase_detail,
                "flags": dict(self.flags), "downloads": dict(self.downloads),
                "process_day": self.process_day, "process_traces": self.process_traces,
                "process_bytes_frac": self.process_bytes_frac, "last_qa": self.last_qa,
                "workers": self.workers, "gate": self.gate,
            }


def lower_priority() -> None:
    """BELOW_NORMAL CPU priority and low I/O priority for this process (plan §4.1)."""
    try:
        import psutil
        p = psutil.Process()
        if sys.platform == "win32":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            p.ionice(psutil.IOPRIO_LOW)
        else:
            p.nice(10)
    except Exception as e:  # never fatal
        log.warning("could not lower priority: %s", e)


def prevent_sleep(enable: bool = True) -> None:
    """Ask Windows to stay awake while this process runs (released automatically on exit)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if enable else 0)
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except Exception as e:
        log.warning("could not set execution state: %s", e)
