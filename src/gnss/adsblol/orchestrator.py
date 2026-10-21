"""The pipeline driver behind `scripts/run_pipeline.py` (plan §9.3, §10).

Sequence: catalog → provisional schedule → (downloads start on P0) → GPSJAM daily backfill →
variation check / window suggestions → final schedule → P0 → Gate 1 → P1a → Gate 2 → P1b → P2 →
P3a → P3b → follow mode (P4). Download and processing run concurrently; the manifest drives
everything, so re-running the same command resumes where it stopped.
"""

from __future__ import annotations

import json
import queue
import shutil
import sys
import threading
import time
import traceback
from datetime import date, datetime, timedelta

from . import PIPELINE_VERSION
from .catalog import run_catalog
from .fetch import Downloader, day_complete, part_paths
from .gpsjam import backfill, region_daily
from .heartbeat import ANALYSIS_TIERS
from .http import Backoff, PoliteClient, Stopped
from .manifest import DONE, TIERS
from .notify import TokenNotifier, toast
from .process import DayProcessor, Interrupted, make_pool, write_day_qa
from .qa import summarize, verify_day
from .schedule import build_schedule, load_windows


def parse_dates(spec: str) -> list[str]:
    out: list[str] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            a, b = (date.fromisoformat(x) for x in chunk.split(":"))
            while a <= b:
                out.append(a.isoformat())
                a += timedelta(days=1)
        else:
            out.append(date.fromisoformat(chunk).isoformat())
    return out


class Orchestrator:
    def __init__(self, cfg: dict, paths, args, manifest, status, events, stop: threading.Event,
                 heartbeat=None):
        self.cfg = cfg
        self.paths = paths
        self.args = args
        self.manifest = manifest
        self.status = status
        self.events = events
        self.stop = stop
        self.hb = heartbeat
        self.max_retries = int(cfg["pipeline"]["max_retries"])
        self.windows = load_windows(cfg)
        self.win_by_id = {w["id"]: w for w in self.windows["windows"]}
        self.token = TokenNotifier(paths.token_flag, events, status)
        self.backoff = Backoff(cfg, events, status)
        self.client = PoliteClient(cfg, self.backoff, self.token, stop)
        self.gpsjam_client = PoliteClient(cfg, Backoff(cfg, events, status), None, stop)
        self.manual = bool(args.dates or args.tier or args.reprocess)
        self.date_filter: list[str] | None = parse_dates(args.dates) if args.dates else None
        self.schedule_final = False
        self.last_catalog = 0.0
        self.pool = None
        self.downloader: Downloader | None = None
        self._stdin_q: queue.Queue | None = None

    # ---- eligibility -------------------------------------------------------------------------
    def _gate_passed(self, n: int) -> bool:
        return bool(self.manifest.kv_get(f"gate{n}_passed", False))

    def allowed_tiers(self, prefetch: bool = False) -> list[str] | None:
        if self.args.tier:
            return [self.args.tier]
        if self.manual and self.date_filter is not None:
            return None
        tiers = list(TIERS)
        if not self._gate_passed(1):
            tiers = ["P0", "P1a"] if prefetch and self.schedule_final else ["P0"]
        elif not self._gate_passed(2):
            tiers = ["P0", "P1a", "P1b"] if prefetch else ["P0", "P1a"]
        if self.args.until:
            tiers = [t for t in tiers if TIERS.index(t) <= TIERS.index(self.args.until)]
        return tiers

    def eligible_downloads(self):
        if not self.schedule_final:
            return ["P0"], self.date_filter
        return self.allowed_tiers(prefetch=True), self.date_filter

    def _pending(self) -> list[dict]:
        rows = self.manifest.days(tiers=self.allowed_tiers(), dates=self.date_filter)
        return [r for r in rows if r["status"] not in DONE and r["status"] != "failed"]

    # ---- prelude -----------------------------------------------------------------------------
    def prelude(self) -> None:
        self.status.set(phase="CATALOG", phase_detail="resolving releases")
        info = run_catalog(self.cfg, self.paths, self.manifest, self.client, self.events)
        self.last_catalog = time.time()
        self.events.info(f"catalog: {info['days']} days, {info['bytes'] / 2**30:,.0f} GiB "
                         f"({info['new']} new; {info['unresolved_sizes']} without sizes)", True)
        build_schedule(self.cfg, self.paths, self.manifest, None, None)
        if self.args.max_days and not self.date_filter:
            rows = [r for r in self.manifest.days(tiers=self.allowed_tiers())
                    if r["status"] not in DONE]
            self.date_filter = [r["date"] for r in rows[: self.args.max_days]]
        self._apply_reprocess()

    def gpsjam_and_schedule(self) -> None:
        series = None
        if not self.args.skip_gpsjam:
            self.status.set(phase="GPSJAM", phase_detail="daily backfill")
            today = date.today().isoformat()
            if self.manifest.kv_get("gpsjam_backfill_day") != today:
                res = backfill(self.paths, self.gpsjam_client, self.cfg["archive"]["earliest_date"],
                               events=self.events, stop=self.stop)
                self.events.info(f"GPSJAM backfill: {res['fetched']} fetched, {res['missing']} missing, "
                                 f"{res['files']} files on disk", True)
                self.manifest.kv_set("gpsjam_backfill_day", today)
            self.status.set(phase="ANALYZE", phase_detail="GPSJAM region series")
            series = region_daily(self.paths, self.windows["regions"],
                                  self.cfg["archive"]["earliest_date"], self.events)
        self.status.set(phase="SCHEDULE", phase_detail="")
        summary = build_schedule(self.cfg, self.paths, self.manifest, series, self.events)
        if summary.get("variation"):
            v = summary["variation"]
            self.events.info("GPSJAM day-to-day (lag-1 autocorr): " + ", ".join(
                f"{r} {x['lag1_autocorr']:.2f}" for r, x in sorted(v.items())), True)
        if summary.get("peak_source"):
            moved = {k: v for k, v in summary["peaks"].items()
                     if v != str(self.win_by_id[k]["peak"])}
            if moved:
                self.events.info(f"peak days moved to GPSJAM maxima: {moved}", True)
        if summary["missing_peak_days"]:
            self.events.warn(f"peak days not in the archive catalog: {summary['missing_peak_days']}")
        self.schedule_final = True

    def _apply_reprocess(self) -> None:
        if not self.args.reprocess:
            return
        for d in parse_dates(self.args.reprocess):
            row = self.manifest.get(d)
            retained = self.paths.raw_retained / d
            if not row:
                self.events.warn(f"--reprocess {d}: not in catalog")
            elif not retained.exists():
                self.events.warn(f"--reprocess {d}: no retained raw data; skipped")
            elif row["status"] in ("cleaned", "verified", "failed"):
                self.manifest.transition(d, "downloaded", "reprocess", raw_dir=str(retained),
                                         retry_count=0)
        self.date_filter = sorted(set(self.date_filter or []) | set(parse_dates(self.args.reprocess)))

    # ---- main --------------------------------------------------------------------------------
    def run(self) -> None:
        self.prelude()
        if self.args.dry_run:
            self.dry_run_report()
            return
        self.pool, n = make_pool(self.cfg, numba_cache_dir=str(self.paths.state / "numba_cache"))
        self.status.set(workers=n)
        self.processor = DayProcessor(self.cfg, self.paths, self.pool, n, self.status, self.events,
                                      self.stop)
        self.downloader = Downloader(self.cfg, self.paths, self.manifest, self.client, self.status,
                                     self.events, self.stop, self.eligible_downloads)
        self.downloader.start()
        try:
            if not self.manual:
                self.gpsjam_and_schedule()
            else:
                self.schedule_final = True
            self.loop()
        except Stopped:
            pass
        finally:
            self.shutdown()

    def loop(self) -> None:
        poll_s = float(self.cfg["pipeline"]["follow_poll_hours"]) * 3600
        while not self.stop.is_set():
            if not self.manual and time.time() - self.last_catalog > poll_s:
                self._refresh_catalog()
            for row in self.manifest.days(statuses=["processed"], dates=self.date_filter):
                self._verify(row)
            for row in self.manifest.days(statuses=["verified"]):
                self._cleanup(row)
            self._set_phase()
            nxt = self.manifest.days(statuses=["downloaded"], tiers=self.allowed_tiers(),
                                     dates=self.date_filter)
            if nxt:
                self._process(nxt[0])
                continue
            if self._requeue_failed():
                continue
            if self._pending():
                self.status.set(process_day=None)
                self.stop.wait(2)
                continue
            if self.manual or self.args.max_days:
                self.events.info("requested work complete", True)
                break
            if self.args.until and self.allowed_tiers()[-1] == self.args.until:
                self.events.info(f"stopped after {self.args.until} (--until)", True)
                break
            gate = self._gate_due()
            if gate:
                self._run_gate(gate)
                continue
            if self.args.no_follow:
                self.events.info("archive complete; exiting (--no-follow)", True)
                break
            self._follow_wait(poll_s)

    def shutdown(self) -> None:
        self.stop.set()
        if self.downloader:
            self.downloader.join(30)
        for row in self.manifest.days(statuses=["downloading"]):
            self.manifest.transition(row["date"], "queued", "shutdown; partial parts kept")
        for row in self.manifest.days(statuses=["processing"]):
            self.manifest.transition(row["date"], "downloaded", "shutdown")
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
        self.status.set(process_day=None, downloads={})
        c = self.manifest.counts()
        done = sum(v.get(s, 0) for v in c.values() for s in DONE)
        total = sum(sum(v.values()) for v in c.values())
        self.events.info(f"stopped cleanly: {done}/{total} days verified. Re-run the same command to resume.",
                         True)

    # ---- phases and gates --------------------------------------------------------------------
    def _set_phase(self) -> None:
        pend = self._pending()
        busy_day = self.status.process_day
        row = self.manifest.get(busy_day) if busy_day else (pend[0] if pend else None)
        if row is None:
            return
        detail = ""
        w = self.win_by_id.get(row.get("window_id") or "")
        if w and row["tier"] in ("P1b", "P1a", "P0"):
            days = (w["end"] - w["start"]).days + 1
            k = (date.fromisoformat(row["date"]) - w["start"]).days + 1
            detail = f"window {w['id']} ({w['name']}) day {k}/{days}"
        elif self.manual:
            detail = "manual run"
        self.status.set(phase=row["tier"] or "?", phase_detail=detail)

    def _gate_due(self) -> str | None:
        if self.manual or self.args.tier:
            return None
        if not self._gate_passed(1):
            return "Gate 1"
        if not self._gate_passed(2):
            return "Gate 2"
        return None

    def _run_gate(self, gate: str) -> None:
        n = 1 if gate == "Gate 1" else 2
        tier = "P0" if n == 1 else "P1a"
        if self.args.auto and self._gate_passed(n):
            return
        report = self.gate_report(tier, gate)
        rp = self.paths.state / f"gate{n}_report.txt"
        rp.write_text("\n".join(report), encoding="utf-8")
        for line in report:
            self.events.q.put((time.time(), "info", line))
        if self.args.auto:
            self.events.info("--auto never skips a gate that has not been passed once", True)
        steps = ("close the VERIFY items, update plan §4/§7 with real numbers" if n == 1 else
                 "review peak-day outputs, freeze the Tier 1 schema, bump PIPELINE_VERSION")
        self.events.info(f"{gate}: {steps}. Report saved to {rp}. "
                         "Type 'continue' + Enter to proceed, or 'quit' to stop.", True)
        toast(f"GNSS pipeline: {gate} reached", "Review the report and type 'continue'.")
        self.status.set(gate=gate)
        self.status.flag("PAUSED AT GATE", gate)
        try:
            while not self.stop.is_set():
                cmd = self._read_line(timeout=1.0)
                if cmd is None:
                    continue
                cmd = cmd.strip().lower()
                if cmd == "continue":
                    self.manifest.kv_set(f"gate{n}_passed", True)
                    self.manifest.kv_set(f"gate{n}_passed_at", datetime.now().isoformat(timespec="seconds"))
                    self.events.info(f"{gate} passed", True)
                    break
                if cmd in ("quit", "exit", "stop"):
                    self.stop.set()
                    break
                if cmd:
                    self.events.info(f"(type 'continue' or 'quit'; got {cmd!r})", True)
        finally:
            self.status.set(gate=None)
            self.status.unflag("PAUSED AT GATE")

    def _read_line(self, timeout: float) -> str | None:
        if self._stdin_q is None:
            self._stdin_q = queue.Queue()

            def reader():
                while True:
                    try:
                        line = sys.stdin.readline()
                    except Exception:
                        return
                    if not line:
                        return
                    self._stdin_q.put(line)
            threading.Thread(target=reader, name="stdin", daemon=True).start()
        try:
            return self._stdin_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def gate_report(self, tier: str, gate: str) -> list[str]:
        rows = self.manifest.days(tiers=[tier])
        L = [f"══════════ {gate}: {tier} complete ══════════"]
        verify_agg: dict = {}
        per_day_bytes = []
        for r in rows:
            p = self.paths.day_qa / f"date={r['date']}.json"
            if not p.exists():
                L.append(f" {r['date']}  status={r['status']}  {r.get('last_error') or ''}")
                continue
            q = json.loads(p.read_text(encoding="utf-8"))
            g = q.get("gpsjam") or {}
            b = q.get("bytes", {})
            per_day_bytes.append(sum(b.values()))
            L.append(
                f" {r['date']} [{r.get('window_id') or 'control'}] {q['processing_seconds'] / 60:.1f} min · "
                f"tar {q['tar_bytes'] / 2**30:.2f} GiB · traces {q['trace_count']:,} · aircraft "
                f"{q['aircraft_count']:,} · points {q['point_count'] / 1e6:.0f}M")
            L.append(
                f"            rows tier1 {q['rows'].get('tier1', 0):,} ({b.get('tier1', 0) / 2**20:.0f} MiB) · "
                f"dim {q['rows'].get('aircraft_dim', 0):,} · exceptions {q['rows'].get('exceptions', 0):,} "
                f"({b.get('exceptions', 0) / 2**20:.0f} MiB)")
            L.append(
                f"            NIC-known {q.get('nic_known_share')} · ADS-B share {q.get('adsb_share')} · "
                f"GPSJAM ρ={g.get('spearman')} agree={g.get('agree_pct')}% over {g.get('hexes')} hexes · "
                f"QA {'ok' if q.get('qa', {}).get('ok') else 'FAILED'}"
                + (f" · warnings: {'; '.join(q['qa']['warnings'])}" if q.get("qa", {}).get("warnings") else ""))
            for k, v in (q.get("verify") or {}).items():
                if isinstance(v, dict):
                    agg = verify_agg.setdefault(k, {})
                    for kk, vv in v.items():
                        agg[kk] = agg.get(kk, 0) + vv
                elif isinstance(v, (int, float)) and k != "mlat_concurrent_share":
                    verify_agg[k] = verify_agg.get(k, 0) + v
        if per_day_bytes:
            avg = sum(per_day_bytes) / len(per_day_bytes)
            total_days = sum(sum(v.values()) for v in self.manifest.counts().values())
            L.append(f" Output size: {avg / 2**20:.0f} MiB/day → ~{avg * total_days / 2**30:,.0f} GiB "
                     f"for {total_days:,} days (plan §7 estimate: ~60–200 MB/day)")
        L.append(" VERIFY findings (sampled traces):")
        st = verify_agg.get("sampled_traces", 0) or 1
        for key, title in (("paths", "member paths"), ("doc_keys", "trace JSON keys"),
                           ("det_keys", "detail-object keys"), ("src", "source types"),
                           ("flags", "flags values"), ("ncols", "point lengths"),
                           ("dbflags_bits", "dbFlags bits")):
            v = verify_agg.get(key) or {}
            top = sorted(v.items(), key=lambda kv: -kv[1])[:18]
            L.append(f"   {title}: " + ", ".join(f"{k}={c}" for k, c in top))
        mp_, near = verify_agg.get("mlat_pts", 0), verify_agg.get("mlat_near_adsb", 0)
        L.append(f"   MLAT points within 5 s of an ADS-B point: {near}/{mp_}"
                 + (" → ADS-B/MLAT divergence metric not feasible (dropped)" if mp_ and near / mp_ < 0.05 else ""))
        L.append(f"   sampled traces: {st}")
        fails = [r for r in rows if r["status"] == "failed"]
        if fails:
            L.append(" FAILED days: " + ", ".join(f"{r['date']} ({r['last_error']})" for r in fails))
        L.append(f" pipeline_version {PIPELINE_VERSION}")
        return L

    # ---- per-day steps -----------------------------------------------------------------------
    def _process(self, row: dict) -> None:
        d = row["date"]
        parts = part_paths(self.paths, row)
        if not day_complete(self.paths, row):
            self.events.warn(f"{d}: raw parts missing or incomplete; re-queueing download")
            self.manifest.transition(d, "queued", "parts missing", raw_dir=None)
            return
        self.manifest.transition(d, "processing", t_process_start=time.time())
        self._set_phase()
        sample = 10 if row.get("retain_raw") else 2
        try:
            day_qa = self.processor.run(d, parts, sample_per_batch=sample)
        except Interrupted:
            self.manifest.transition(d, "downloaded", "interrupted")
            return
        except Exception as e:
            self.events.error(f"{d}: processing failed: {type(e).__name__}: {e}")
            self.events.info(traceback.format_exc())
            self._fail(d, f"process: {type(e).__name__}: {e}"[:500])
            return
        finally:
            self.status.set(process_day=None)
        day_qa.update({"repo": row["repo"], "release_tag": row["tag"], "variant": row["variant"],
                       "tier": row["tier"], "window_id": row.get("window_id"),
                       "download_seconds": row.get("download_sec")})
        write_day_qa(self.paths, day_qa)
        self.manifest.transition(d, "processed", t_processed=time.time(),
                                 process_sec=day_qa["processing_seconds"])
        self._verify(self.manifest.get(d))

    def _verify(self, row: dict) -> None:
        d = row["date"]
        p = self.paths.day_qa / f"date={d}.json"
        if not p.exists():
            self.manifest.transition(d, "downloaded", "day_qa missing; reprocess")
            return
        day_qa = json.loads(p.read_text(encoding="utf-8"))
        verify_day(self.paths, self.cfg, day_qa)
        write_day_qa(self.paths, day_qa)
        q = day_qa["qa"]
        if q["ok"]:
            self.manifest.transition(d, "verified", t_verified=time.time(), last_error=None)
            s = summarize(day_qa)
            self.status.set(last_qa=s)
            rho = f"ρ={s['spearman']:.2f}" if s["spearman"] is not None else "ρ=–"
            self.events.info(f"{d} verified [{row['tier']}] {s['min']} min · NIC-known "
                             f"{s['nic_known']} · GPSJAM {rho}"
                             + (f" · {len(q['warnings'])} warnings" if q["warnings"] else ""), True)
            for w in q["warnings"]:
                self.events.emit(f"{d}: {w}", "warn", important=False)
        else:
            self._fail(d, "QA: " + "; ".join(q["errors"]))

    def _cleanup(self, row: dict) -> None:
        d = row["date"]
        src = self.paths.scratch / d
        raw_dir = row.get("raw_dir")
        if row.get("retain_raw") and not raw_dir:
            dst = self.paths.raw_retained / d
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.move(str(src), str(dst))
            raw_dir = str(dst)
        elif not raw_dir and src.exists():
            shutil.rmtree(src, ignore_errors=True)
        self.manifest.transition(d, "cleaned", t_cleaned=time.time(), raw_dir=raw_dir)

    def _fail(self, d: str, why: str) -> None:
        row = self.manifest.get(d)
        rc = (row["retry_count"] or 0) + 1
        self.manifest.transition(d, "failed", why, last_error=why, retry_count=rc)
        final = rc >= self.max_retries
        self.events.error(f"{d} failed ({rc}/{self.max_retries}{', giving up' if final else ''}): {why}")

    def _requeue_failed(self) -> bool:
        """Retry failed days once their own tier and every earlier tier have nothing pending
        (end of phase), so a failure is retried within its phase rather than after the whole
        archive. First reprocess from disk, then re-download."""
        rows = self.manifest.days(statuses=["failed"], tiers=self.allowed_tiers(), dates=self.date_filter)
        if not rows:
            return False
        rank = lambda t: TIERS.index(t) if t in TIERS else len(TIERS)  # noqa: E731
        busiest = min((rank(r["tier"]) for r in self._pending()), default=len(TIERS) + 1)
        moved = False
        for r in rows:
            if (r["retry_count"] or 0) >= self.max_retries or rank(r["tier"]) >= busiest:
                continue
            # 0 = parts changed after a catalog fix; 1 = first failure: reuse the files on disk
            if (r["retry_count"] or 0) <= 1 and day_complete(self.paths, r):
                self.manifest.transition(r["date"], "downloaded", "retry: reprocess")
            else:
                if not r.get("raw_dir"):
                    shutil.rmtree(self.paths.scratch / r["date"], ignore_errors=True)
                self.manifest.transition(r["date"], "queued", "retry: re-download")
            moved = True
        return moved

    # ---- follow mode -------------------------------------------------------------------------
    def _refresh_catalog(self) -> None:
        try:
            info = run_catalog(self.cfg, self.paths, self.manifest, self.client, self.events)
            self.last_catalog = time.time()
            if info["new"]:
                build_schedule(self.cfg, self.paths, self.manifest, None if self.args.skip_gpsjam else
                               region_daily(self.paths, self.windows["regions"],
                                            self.cfg["archive"]["earliest_date"]), self.events)
                self.events.info(f"{info['new']} new day(s) published and scheduled", True)
        except Stopped:
            raise
        except Exception as e:
            self.last_catalog = time.time()
            self.events.warn(f"catalog refresh failed: {type(e).__name__}: {e}")

    def _follow_wait(self, poll_s: float) -> None:
        nxt = self.last_catalog + poll_s
        self.status.set(phase="P4", phase_detail="follow mode, next poll "
                        + datetime.fromtimestamp(nxt).strftime("%H:%M"), process_day=None)
        while not self.stop.is_set() and time.time() < nxt:
            self.stop.wait(10)

    # ---- dry run -----------------------------------------------------------------------------
    def dry_run_report(self) -> None:
        rows = self.manifest.days()
        by_tier: dict[str, list[dict]] = {}
        for r in rows:
            by_tier.setdefault(r["tier"] or "unscheduled", []).append(r)
        lines = ["DRY RUN: schedule (no downloads, no processing)"]
        for t in [*TIERS, "unscheduled"]:
            rs = by_tier.get(t, [])
            if not rs:
                continue
            gib = sum(r["total_bytes"] or 0 for r in rs) / 2**30
            done = sum(1 for r in rs if r["status"] in DONE)
            first = ", ".join(r["date"] for r in rs[:6])
            lines.append(f"  {t:4} {len(rs):5} days  {gib:8,.0f} GiB  done {done:4}   first: {first}"
                         + (" …" if len(rs) > 6 else ""))
        an = sum(r["total_bytes"] or 0 for t in ANALYSIS_TIERS for r in by_tier.get(t, [])) / 2**30
        lines.append(f"  analysis set (P0–P2): {an:,.0f} GiB")
        if self.date_filter:
            lines.append(f"  date filter: {len(self.date_filter)} days ({self.date_filter[:5]}…)")
        for line in lines:
            self.events.q.put((time.time(), "info", line))
