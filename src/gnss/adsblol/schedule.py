"""Build the prioritized download/processing queue (plan §3) and write it into the manifest.

Tiers set the ORDER only; every catalog day is scheduled:

    P0   test day (windows.yaml `test_day`)
    P1a  peak day of each window (GPSJAM max when available) + quiet control days
    P1b  full case windows, daily, in windows.yaml order (chronological within a window)
    P2   the latest `recent.days` days (range frozen at first scheduling; newer days → P4)
    P3a  GPSJAM spike days (±pad) first, then every other remaining day by calendar parity,
         newest → oldest
    P3b  the rest, newest → oldest
    P4   days published after the P2 range was frozen (follow mode)

Re-runnable: re-reads windows.yaml and GPSJAM, recomputes tiers; day status is never touched.
`retain_raw` = P0 + P1a (plan §7).
"""

from __future__ import annotations

from datetime import date, timedelta

import yaml

from .config import windows_path
from .gpsjam import (
    _d,
    edge_suggestions,
    quiet_controls,
    refine_peaks,
    spike_days,
    variation_report,
    window_days,
)

TIER_BASE = {"P0": 0, "P1a": 1, "P1b": 2, "P2": 3, "P3a": 4, "P3b": 5, "P4": 6}
STEP = 1_000_000


def load_windows(cfg: dict) -> dict:
    with open(windows_path(cfg), encoding="utf-8") as f:
        w = yaml.safe_load(f)
    for win in w["windows"]:
        for k in ("start", "end", "peak"):
            win[k] = _d(win[k])
        for ph in win.get("phases", []) or []:
            ph["start"], ph["end"] = _d(ph["start"]), _d(ph["end"])
        unknown = [r for r in win["regions"] if r not in w["regions"]]
        if unknown:
            raise ValueError(f"window {win['id']}: unknown regions {unknown}")
    w["test_day"] = _d(w["test_day"])
    return w


def assign_tiers(catalog_days: list[str], windows: dict, *, peaks: dict[str, date] | None = None,
                 controls: list[date] | None = None, spikes: list[date] | None = None,
                 p2_range: tuple[date, date] | None = None,
                 earliest: date | None = None) -> list[tuple[str, str, int, bool, str | None]]:
    """Pure tier assignment. Returns [(date, tier, priority, retain_raw, window_id)]."""
    avail = set(catalog_days)
    out: dict[str, tuple[str, int, bool, str | None]] = {}
    counters = {t: 0 for t in TIER_BASE}

    def put(d: date | str, tier: str, window: str | None = None) -> None:
        ds = d if isinstance(d, str) else d.isoformat()
        if ds not in avail or ds in out:
            return
        out[ds] = (tier, TIER_BASE[tier] * STEP + counters[tier], tier in ("P0", "P1a"), window)
        counters[tier] += 1

    win_of: dict[str, str] = {}
    for w in windows["windows"]:
        for d in window_days(w):
            win_of.setdefault(d.isoformat(), w["id"])

    put(windows["test_day"], "P0", win_of.get(windows["test_day"].isoformat()))
    for w in windows["windows"]:
        put((peaks or {}).get(w["id"], w["peak"]), "P1a", w["id"])
    for c in controls or []:
        put(c, "P1a", None)
    for w in windows["windows"]:
        for d in window_days(w):
            put(d, "P1b", w["id"])
    if p2_range:
        d = p2_range[1]
        while d >= p2_range[0]:
            put(d, "P2", win_of.get(d.isoformat()))
            d -= timedelta(days=1)
        for ds in sorted(x for x in avail if x > p2_range[1].isoformat()):
            put(ds, "P4", win_of.get(ds))
    for d in spikes or []:
        put(d, "P3a", win_of.get(d.isoformat()))
    anchor = earliest or date(2023, 2, 16)
    rest = sorted((x for x in avail if x not in out), reverse=True)
    for ds in rest:
        if (date.fromisoformat(ds) - anchor).days % 2 == 0:
            put(ds, "P3a", win_of.get(ds))
    for ds in rest:
        put(ds, "P3b", win_of.get(ds))
    return [(d, *v) for d, v in sorted(out.items(), key=lambda kv: kv[1][1])]


def build_schedule(cfg: dict, paths, manifest, series=None, events=None) -> dict:
    """Compute and store the schedule. `series` is the GPSJAM region-daily frame (or None)."""
    windows = load_windows(cfg)
    days = [r["date"] for r in manifest.days(order="date")]
    if not days:
        raise RuntimeError("catalog is empty; run the catalog first")
    have_gpsjam = series is not None and not series.empty

    peaks, peak_src = {}, {}
    if have_gpsjam:
        for wid, (d, src) in refine_peaks(series, windows["windows"]).items():
            peaks[wid], peak_src[wid] = d, src
    controls = []
    if have_gpsjam:
        qc = windows["quiet_controls"]
        controls = quiet_controls(series, windows["primary_regions"], windows["windows"],
                                  qc["count"], qc["lookback_days"], qc["min_separation_days"])
    spikes = []
    if have_gpsjam:
        sp = windows["spikes"]
        spikes = spike_days(series, [r for r in windows["regions"] if r != "global"],
                            windows["windows"], sp["z_threshold"], sp["baseline_days"],
                            sp["pad_days"], sp["min_aircraft"])

    p2 = manifest.kv_get("p2_range")
    if p2 is None:
        last = date.fromisoformat(days[-1])
        p2 = [(last - timedelta(days=windows["recent"]["days"] - 1)).isoformat(), last.isoformat()]
        manifest.kv_set("p2_range", p2)
    p2_range = (date.fromisoformat(p2[0]), date.fromisoformat(p2[1]))

    assignments = assign_tiers(days, windows, peaks=peaks, controls=controls,
                               spikes=[d for d, _, _ in spikes], p2_range=p2_range,
                               earliest=cfg["archive"]["earliest_date"])
    manifest.set_schedule(assignments)

    missing = [w["id"] for w in windows["windows"]
               if (peaks.get(w["id"]) or w["peak"]).isoformat() not in set(days)]
    tiers: dict[str, int] = {}
    for _, t, *_ in assignments:
        tiers[t] = tiers.get(t, 0) + 1
    summary = {
        "tiers": tiers, "peaks": {k: str(v) for k, v in peaks.items()}, "peak_source": peak_src,
        "controls": [str(c) for c in controls], "spike_days": len(spikes),
        "p2_range": p2, "missing_peak_days": missing,
    }
    manifest.kv_set("schedule_summary", summary)
    if have_gpsjam:
        report = {
            "variation": variation_report(series, [r for r in windows["regions"]]),
            "edge_suggestions": edge_suggestions(series, windows["windows"]),
            "spikes_top20": [(str(d), r, round(z, 2)) for d, r, z in spikes[:20]],
        }
        (paths.state / "window_suggestions.yaml").write_text(
            yaml.safe_dump({"edges": report["edge_suggestions"], "peaks": summary["peaks"],
                            "peak_source": peak_src, "controls": summary["controls"],
                            "spikes_top20": report["spikes_top20"]}, sort_keys=False),
            encoding="utf-8")
        (paths.state / "gpsjam_variation_report.yaml").write_text(
            yaml.safe_dump(report["variation"], sort_keys=True), encoding="utf-8")
        summary["variation"] = report["variation"]
    if events:
        events.info("schedule: " + "  ".join(f"{t} {n}" for t, n in sorted(
            tiers.items(), key=lambda kv: TIER_BASE[kv[0]])) +
            f"  · controls {summary['controls'] or 'none (no GPSJAM)'} · spike days {len(spikes)}", True)
    return summary
