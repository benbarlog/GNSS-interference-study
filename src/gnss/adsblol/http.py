"""Polite HTTP client for adsb.lol/GitHub and GPSJAM (plan §9.1, §9.2).

- descriptive User-Agent on every request
- ≥ `metadata_min_interval_s` between small requests to the same host
- on 429/403/5xx: honor Retry-After, else exponential backoff 30 s → 30 min with jitter;
  after N consecutive failures pause everything for an hour
- GitHub API: watch X-RateLimit-Remaining; when exhausted without a token, raise the token flag
  and keep going without API calls until the reset
"""

from __future__ import annotations

import logging
import os
import random
import sys
import threading
import time
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import requests

from .config import user_agent

log = logging.getLogger("gnss.adsblol")

RETRY_STATUS = {403, 429, 500, 502, 503, 504}


class RateLimited(Exception):
    def __init__(self, reset_ts: float | None, msg: str = "GitHub API rate limit"):
        super().__init__(msg)
        self.reset_ts = reset_ts


class Stopped(Exception):
    pass


def get_github_token() -> str | None:
    """GITHUB_TOKEN from this process's environment or, on Windows, the user's registry
    environment (so a token set while the pipeline runs is picked up without a restart)."""
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok.strip()
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                val, _ = winreg.QueryValueEx(k, "GITHUB_TOKEN")
                return str(val).strip() or None
        except OSError:
            return None
    return None


def retry_after_seconds(resp: requests.Response) -> float | None:
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return max(0.0, float(ra))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(ra).timestamp() - time.time())
        except Exception:
            return None


class Backoff:
    """Shared failure state. Every downloader thread and metadata call reports here."""

    def __init__(self, cfg: dict, events=None, status=None):
        dl = cfg["download"]
        self.initial = float(dl["backoff_initial_s"])
        self.cap = float(dl["backoff_max_s"])
        self.pause_after = int(dl["pause_after_failures"])
        self.pause_s = float(dl["pause_s"])
        self.events = events
        self.status = status
        self.consecutive = 0
        self.until = 0.0
        self._lock = threading.Lock()

    def failure(self, why: str, retry_after: float | None = None) -> float:
        with self._lock:
            self.consecutive += 1
            if self.consecutive >= self.pause_after:
                delay = max(self.pause_s, retry_after or 0)
                msg = f"{self.consecutive} consecutive failures ({why}); pausing downloads {delay/60:.0f} min"
                self.consecutive = 0
            elif retry_after is not None:
                delay = retry_after + random.uniform(0, 5)
                msg = f"server asked to wait {retry_after:.0f}s ({why})"
            else:
                base = min(self.cap, self.initial * 2 ** (self.consecutive - 1))
                delay = base * random.uniform(0.8, 1.2)
                msg = f"backing off {delay:.0f}s ({why})"
            self.until = max(self.until, time.time() + delay)
            until = self.until
        if self.status:
            self.status.flag("BACKOFF", "until " + time.strftime("%H:%M", time.localtime(until)))
        if self.events:
            self.events.warn(msg)
        return delay

    def success(self) -> None:
        with self._lock:
            self.consecutive = 0
        if self.status and time.time() >= self.until:
            self.status.unflag("BACKOFF")

    def wait(self, stop: threading.Event | None = None) -> None:
        while True:
            remaining = self.until - time.time()
            if remaining <= 0:
                if self.status:
                    self.status.unflag("BACKOFF")
                return
            if stop is not None:
                if stop.wait(min(remaining, 5)):
                    raise Stopped()
            else:
                time.sleep(min(remaining, 5))


class PoliteClient:
    def __init__(self, cfg: dict, backoff: Backoff | None = None, token_notifier=None,
                 stop: threading.Event | None = None):
        self.cfg = cfg
        self.ua = user_agent(cfg)
        self.min_interval = float(cfg["download"]["metadata_min_interval_s"])
        self.timeout = float(cfg["download"]["timeout_s"])
        self.backoff = backoff or Backoff(cfg)
        self.token_notifier = token_notifier
        self.stop = stop
        self._last: dict[str, float] = {}
        self._space_lock = threading.Lock()
        self._local = threading.local()
        self.api_remaining: int | None = None
        self.api_reset: float | None = None

    def session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = self.ua
            self._local.session = s
        return s

    def _space(self, url: str) -> None:
        host = urlparse(url).netloc
        with self._space_lock:
            wait = self._last.get(host, 0) + self.min_interval - time.time()
            if wait > 0:
                time.sleep(wait)
            self._last[host] = time.time()

    def _check_stop(self) -> None:
        if self.stop is not None and self.stop.is_set():
            raise Stopped()

    def meta(self, method: str, url: str, *, api: bool = False, ok_404: bool = True,
             max_attempts: int = 6, **kw) -> requests.Response:
        """A small, spaced metadata request with backoff. Returns the response (404 included
        when ok_404). Raises RateLimited for an exhausted GitHub API limit without a token."""
        headers = dict(kw.pop("headers", {}) or {})
        if api:
            headers.setdefault("Accept", "application/vnd.github+json")
            headers.setdefault("X-GitHub-Api-Version", "2022-11-28")
            token = get_github_token()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            elif self.api_remaining == 0 and self.api_reset and time.time() < self.api_reset:
                self._rate_limited("API calls exhausted")
        for attempt in range(max_attempts):
            self._check_stop()
            self.backoff.wait(self.stop)
            self._space(url)
            try:
                r = self.session().request(method, url, headers=headers, timeout=self.timeout,
                                           allow_redirects=True, **kw)
            except requests.RequestException as e:
                self.backoff.failure(f"{type(e).__name__} on {_short(url)}")
                continue
            if api:
                self._track_api(r)
                if r.status_code in (403, 429) and self._is_rate_limit(r):
                    if not headers.get("Authorization"):
                        self._rate_limited(f"HTTP {r.status_code} rate limit on {_short(url)}")
                    self.backoff.failure("API rate limit (token set)", retry_after_seconds(r)
                                         or max(60.0, (self.api_reset or time.time() + 60) - time.time()))
                    continue
            if r.status_code == 404 and ok_404:
                self.backoff.success()
                return r
            if r.status_code in RETRY_STATUS:
                self.backoff.failure(f"HTTP {r.status_code} on {_short(url)}", retry_after_seconds(r))
                continue
            if r.ok:
                self.backoff.success()
                if api and self.token_notifier and r.headers.get("X-RateLimit-Remaining"):
                    if int(r.headers["X-RateLimit-Remaining"]) > 0:
                        self.token_notifier.clear()
                return r
            r.raise_for_status()
        raise RuntimeError(f"giving up on {url} after {max_attempts} attempts")

    def get_text(self, url: str, **kw) -> str | None:
        r = self.meta("GET", url, **kw)
        return None if r.status_code == 404 else r.text

    def _track_api(self, r: requests.Response) -> None:
        try:
            if "X-RateLimit-Remaining" in r.headers:
                self.api_remaining = int(r.headers["X-RateLimit-Remaining"])
            if "X-RateLimit-Reset" in r.headers:
                self.api_reset = float(r.headers["X-RateLimit-Reset"])
        except ValueError:
            pass

    @staticmethod
    def _is_rate_limit(r: requests.Response) -> bool:
        return r.headers.get("X-RateLimit-Remaining") == "0" or "rate limit" in r.text.lower()

    def _rate_limited(self, reason: str):
        if self.token_notifier:
            self.token_notifier.raise_needed(reason, self.api_reset)
        raise RateLimited(self.api_reset, reason)


def _short(url: str) -> str:
    p = urlparse(url)
    tail = p.path.rsplit("/", 1)[-1]
    return f"{p.netloc}/…/{tail}" if tail else p.netloc
