"""Live heartbeat (plan §9.3): the only console writer.

Every `heartbeat_sec` it prints one compact block (or refreshes a `rich` dashboard with --live)
and atomically writes `state/heartbeat.json`, which `scripts/adsblol_status.py` reads. Important
events print immediately as single lines between ticks. Console writes happen only on this thread,
so a blocked console can never stall downloading or processing.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from datetime import datetime

from .manifest import DONE, TIERS

ANALYSIS_TIERS = ("P0", "P1a", "P1b", "P2")
TICK_ICONS = {"info": "·", "warn": "!", "error": "✗"}


def _fmt_dur(s: float) -> str:
    s = int(s)
    return f"{s // 86400}d {s % 86400 // 3600:02d}:{s % 3600 // 60:02d}"


def _bar(frac: float, width: int = 24) -> str:
    n = max(0, min(width, round(frac * width)))
    return "█" * n + "░" * (width - n)


def _gib(b: float) -> str:
    return f"{b / 2**30:,.0f}"


def build_snapshot(manifest, status, paths, windows_by_id: dict | None = None) -> dict:
    st = status.snapshot()
    counts = manifest.counts()
    tiers = {}
    total_days = done_days = failed = 0
    for t in TIERS:
        c = counts.get(t, {})
        n = sum(c.values())
        dn = sum(c.get(s, 0) for s in DONE)
        fl = c.get("failed", 0)
        if n:
            tiers[t] = {"done": dn, "total": n, "failed": fl}
        total_days += n
        done_days += dn
        failed += fl
    total_bytes = manifest.scalar("SELECT COALESCE(SUM(total_bytes),0) FROM days") or 0
    dl_bytes = manifest.scalar(
        "SELECT COALESCE(SUM(CASE WHEN status='queued' THEN 0 WHEN status='downloading' THEN bytes_done "
        "WHEN status='failed' THEN 0 ELSE total_bytes END),0) FROM days") or 0
    buf_days = manifest.scalar("SELECT COUNT(*) FROM days WHERE status IN ('downloaded','processing')") or 0
    buf_bytes = manifest.buffer_bytes()
    now = time.time()
    since = max(st["started"], now - 6 * 3600)
    recent = manifest.scalar("SELECT COUNT(*) FROM days WHERE t_verified >= ?", (since,)) or 0
    hours = max((now - since) / 3600, 1 / 60)
    days_per_h = recent / hours if recent else 0.0
    avg_sec = manifest.scalar(
        "SELECT AVG(process_sec) FROM (SELECT process_sec FROM days WHERE process_sec IS NOT NULL "
        "ORDER BY t_processed DESC LIMIT 10)")
    rem_analysis = sum(v["total"] - v["done"] - v["failed"] for t, v in tiers.items() if t in ANALYSIS_TIERS)
    rem_all = total_days - done_days - failed
    eta = lambda n: (now + n / days_per_h * 3600) if days_per_h > 0 and n > 0 else None
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        ram = (vm.used / 2**30, vm.total / 2**30)
    except Exception:
        cpu, ram = None, None
    try:
        free = shutil.disk_usage(paths.root).free
    except OSError:
        free = None
    phase_name = st["phase"]
    detail = st["phase_detail"]
    return {
        "ts": now, "started": st["started"], "phase": phase_name, "phase_detail": detail,
        "overall": {"done": done_days, "total": total_days, "failed": failed,
                    "dl_bytes": dl_bytes, "total_bytes": total_bytes},
        "tiers": tiers,
        "download": {"active": st["downloads"], "rate_1m": status.download_meter.rate(60),
                     "rate_1h": status.download_meter.rate(3600), "buffer_days": buf_days,
                     "buffer_bytes": buf_bytes},
        "process": {"day": st["process_day"], "traces": st["process_traces"],
                    "bytes_frac": st["process_bytes_frac"],
                    "avg_min": (avg_sec / 60) if avg_sec else None, "days_per_h": days_per_h},
        "eta": {"analysis": eta(rem_analysis), "full": eta(rem_all),
                "analysis_done": rem_analysis == 0 and total_days > 0},
        "system": {"cpu": cpu, "ram": ram, "workers": st["workers"], "free": free},
        "qa": st["last_qa"], "flags": st["flags"], "gate": st["gate"],
    }


def render(s: dict, warnings: int = 0, errors: int = 0) -> list[str]:
    t = datetime.fromtimestamp(s["ts"])
    head = f"── {t:%Y-%m-%d %H:%M} ── uptime {_fmt_dur(s['ts'] - s['started'])} ── PHASE {s['phase']}"
    if s["phase_detail"]:
        head += f" · {s['phase_detail']}"
    lines = [head + " ──"]
    o = s["overall"]
    frac = o["done"] / o["total"] if o["total"] else 0
    lines.append(f" OVERALL   {o['done']:,} / {o['total']:,} days verified ({frac:.1%})  {_bar(frac)}"
                 + (f"  · {o['failed']} failed" if o["failed"] else ""))
    if o["total_bytes"]:
        lines.append(f"           {_gib(o['dl_bytes'])} / {_gib(o['total_bytes'])} GiB downloaded "
                     f"({o['dl_bytes'] / o['total_bytes']:.1%})")
    parts = []
    for tier in TIERS:
        v = s["tiers"].get(tier)
        if not v:
            continue
        parts.append(f"{tier} ✓" if v["done"] + v["failed"] >= v["total"] else f"{tier} {v['done']}/{v['total']}")
    lines.append(" TIERS     " + "  ".join(parts))
    d = s["download"]
    act = "  ".join(f"{k} {(a / b if b else 0):.0%}" for k, (a, b) in sorted(d["active"].items())) or "idle"
    lines.append(f" DOWNLOAD  {act}  ·  {d['rate_1m'] / 1e6:.1f} MB/s (1 h avg {d['rate_1h'] / 1e6:.1f})"
                 f"  ·  buffer {d['buffer_days']} days / {_gib(d['buffer_bytes'])} GiB")
    p = s["process"]
    pd_ = (f"{p['day']}  traces {p['traces'] / 1000:.0f}k ({p['bytes_frac']:.0%} of tar)" if p["day"]
           else "idle")
    avg = f"avg {p['avg_min']:.1f} min/day" if p["avg_min"] else "avg –"
    lines.append(f" PROCESS   {pd_}  ·  {avg}  ·  {p['days_per_h']:.1f} days/h")
    e = s["eta"]
    fmt = lambda x: datetime.fromtimestamp(x).strftime("%b %d %H:%M") if x else "–"
    lines.append(f" ETA       analysis-ready (P2 done): {'done' if e['analysis_done'] else fmt(e['analysis'])}"
                 f"   ·   full archive: {fmt(e['full'])}")
    sy = s["system"]
    cpu = f"CPU {sy['cpu']:.0f}%" if sy["cpu"] is not None else "CPU –"
    ram = f"RAM {sy['ram'][0]:.1f}/{sy['ram'][1]:.0f} GB" if sy["ram"] else "RAM –"
    free = f"D: {sy['free'] / 2**40:.2f} TiB free" if sy["free"] is not None else ""
    lines.append(f" SYSTEM    {cpu} (BELOW_NORMAL, {sy['workers']} workers)  ·  {ram}  ·  {free}")
    q = s["qa"]
    if q:
        rho = f"GPSJAM ρ={q['spearman']:.2f}" if q.get("spearman") is not None else "GPSJAM ρ=–"
        nk = f"NIC-known {q['nic_known']:.0%}" if q.get("nic_known") is not None else ""
        qa = f"last day {q['date']}: {rho}, {nk}"
    else:
        qa = "no verified day yet"
    lines.append(f" QA        {qa}  ·  warnings {warnings}  ·  errors {errors}")
    flags = s["flags"]
    lines.append(" STATUS    " + ("  /  ".join(f"{k} {v}".strip() for k, v in flags.items()) if flags else "ok"))
    return lines


class Heartbeat(threading.Thread):
    def __init__(self, manifest, status, events, paths, interval: float = 60, live: bool = False):
        super().__init__(name="heartbeat", daemon=True)
        self.manifest = manifest
        self.status = status
        self.events = events
        self.paths = paths
        self.interval = interval
        self.live = live
        self.halt = threading.Event()
        self.force = threading.Event()
        self.last_lines: list[str] = []
        self._rich = None
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass

    def tick_now(self) -> None:
        self.force.set()

    def stop(self) -> None:
        self.halt.set()
        self.join(timeout=5)

    def _write_json(self, snap: dict, lines: list[str]) -> None:
        snap = dict(snap)
        snap["lines"] = lines
        snap["warnings"] = self.events.warnings
        snap["errors"] = self.events.errors
        tmp = self.paths.heartbeat.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(snap, default=str), encoding="utf-8")
            os.replace(tmp, self.paths.heartbeat)
        except OSError:
            pass

    def _print(self, text: str) -> None:
        if self._rich is not None:
            self._rich.console.print(text, markup=False, highlight=False)
        else:
            print(text, flush=True)

    def _drain_events(self) -> None:
        while True:
            try:
                ts, level, msg = self.events.q.get_nowait()
            except Exception:
                return
            self._print(f"{datetime.fromtimestamp(ts):%H:%M:%S} {TICK_ICONS.get(level, '·')} {msg}")

    def _tick(self) -> None:
        try:
            snap = build_snapshot(self.manifest, self.status, self.paths)
            lines = render(snap, self.events.warnings, self.events.errors)
        except Exception as e:
            self._print(f"(heartbeat error: {type(e).__name__}: {e})")
            return
        self.last_lines = lines
        self._write_json(snap, lines)
        if self._rich is not None:
            from rich.panel import Panel
            from rich.text import Text
            self._rich.update(Panel(Text("\n".join(lines[1:])), title=lines[0].strip("─ "), expand=True))
        elif self.status.gate:
            self._print(f"{datetime.now():%H:%M} · paused at {self.status.gate}; type 'continue' "
                        f"(downloads continue in the background)")
        else:
            self._print("\n".join(lines))

    def run(self) -> None:
        if self.live:
            try:
                from rich.live import Live
                self._rich = Live(refresh_per_second=1, transient=False)
                self._rich.start()
            except Exception:
                self._rich = None
        next_tick = time.time()
        try:
            while not self.halt.is_set():
                self._drain_events()
                if time.time() >= next_tick or self.force.is_set():
                    self.force.clear()
                    self._tick()
                    next_tick = time.time() + self.interval
                self.halt.wait(0.5)
            self._drain_events()
            self._tick()
        finally:
            if self._rich is not None:
                self._rich.stop()
