"""Validate a processed day before its raw tars may be deleted (plan §8.4, §9).

Hard failures (day → failed, scratch kept): no Tier 1 / dim rows, too many parse errors, too many
out-of-day points, far fewer traces than recent days (truncated tar), NIC-known or ADS-B share out
of range. Warnings only: weak GPSJAM agreement, outliers vs recent days.

GPSJAM agreement at res 4: an aircraft is "bad" in a hex when ≥ `aircraft_bad_share` of its
known-NIC ADS-B seconds there have NIC < `nic_bad_lt`; hex bad-share = bad / all aircraft, over hexes
with ≥ `gpsjam_min_aircraft` aircraft in both datasets. Reported: Spearman ρ of bad-share and the %
of hexes on which both agree about bad-share > `gpsjam_bad_rate`.
"""

from __future__ import annotations

import json
import statistics

import h3
import pandas as pd
import pyarrow.parquet as pq
from h3.api import basic_int as h3i

from .schemas import NIC_BINS


def gpsjam_agreement(paths, cfg: dict, d: str) -> dict | None:
    gp = paths.gpsjam / f"{d}-h3_4.csv"
    t1_dir = paths.tier1 / f"date={d}"
    if not gp.exists() or not t1_dir.exists():
        return None
    q = cfg["qa"]
    bad_lt = int(cfg["process"]["nic_bad_lt"])
    cols = ["icao", "hex5"] + [f"nic_s_{i}" for i in range(NIC_BINS)]
    t = pq.read_table(t1_dir, columns=cols).to_pandas()
    if t.empty:
        return None
    uniq = t["hex5"].unique()
    parent = {int(h): h3i.cell_to_parent(int(h), 4) for h in uniq}
    t["hex4"] = t["hex5"].map(parent)
    t["s_known"] = t[[f"nic_s_{i}" for i in range(NIC_BINS)]].sum(axis=1)
    t["s_bad"] = t[[f"nic_s_{i}" for i in range(bad_lt)]].sum(axis=1)
    ah = t.groupby(["hex4", "icao"], sort=False)[["s_known", "s_bad"]].sum()
    ah = ah[ah["s_known"] > 0]
    ah["bad"] = ah["s_bad"] / ah["s_known"] >= float(q["aircraft_bad_share"])
    ours = ah.groupby(level=0)["bad"].agg(["sum", "count"]).rename(columns={"sum": "bad", "count": "n"})
    g = pd.read_csv(gp, dtype={"hex": str})
    g["hex4"] = g["hex"].map(h3.str_to_int)
    g["n"] = g["count_good_aircraft"] + g["count_bad_aircraft"]
    g = g.set_index("hex4")[["count_bad_aircraft", "n"]].rename(columns={"count_bad_aircraft": "bad"})
    j = ours.join(g, lsuffix="_a", rsuffix="_g", how="inner")
    m = int(q["gpsjam_min_aircraft"])
    j = j[(j["n_a"] >= m) & (j["n_g"] >= m)]
    if len(j) < 10:
        return {"hexes": len(j), "spearman": None, "agree_pct": None}
    ra, rg = j["bad_a"] / j["n_a"], j["bad_g"] / j["n_g"]
    thr = float(q["gpsjam_bad_rate"])
    return {
        "hexes": len(j),
        "spearman": round(float(ra.corr(rg, method="spearman")), 4),
        "agree_pct": round(float(((ra > thr) == (rg > thr)).mean() * 100), 2),
        "red_hexes_ours": int((ra > thr).sum()),
        "red_hexes_gpsjam": int((rg > thr).sum()),
    }


def recent_history(paths, d: str, n: int = 14) -> list[dict]:
    files = sorted(paths.day_qa.glob("date=*.json"))
    out = []
    for f in reversed(files):
        if f.stem == f"date={d}":
            continue
        try:
            q = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if q.get("qa", {}).get("ok"):
            out.append(q)
        if len(out) >= n:
            break
    return out


def evaluate(day_qa: dict, history: list[dict], cfg: dict) -> tuple[list[str], list[str]]:
    q = cfg["qa"]
    errors, warnings = [], []
    rows = day_qa.get("rows", {})
    if rows.get("tier1", 0) <= 0:
        errors.append("no tier1 rows")
    if rows.get("aircraft_dim", 0) <= 0:
        errors.append("no aircraft_dim rows")
    traces = day_qa.get("trace_count", 0)
    if traces and day_qa.get("parse_errors", 0) / traces > float(q["max_parse_error_share"]):
        errors.append(f"parse errors {day_qa['parse_errors']}/{traces}")
    pts = day_qa.get("point_count", 0)
    if pts and day_qa.get("points_out_of_day", 0) / pts > 0.05:
        errors.append(f"{day_qa['points_out_of_day']}/{pts} points outside the day (timestamp format?)")
    nk = day_qa.get("nic_known_share")
    if nk is None or not (float(q["min_nic_known_share"]) <= nk <= float(q["max_nic_known_share"])):
        errors.append(f"NIC-known share {nk} out of range")
    ad = day_qa.get("adsb_share")
    if ad is None or ad < float(q["min_adsb_share"]):
        errors.append(f"ADS-B share {ad} below {q['min_adsb_share']}")
    if len(history) >= 3:
        med = statistics.median(h["trace_count"] for h in history)
        if traces < float(q["min_traces_frac_of_median"]) * med:
            errors.append(f"trace count {traces} < {q['min_traces_frac_of_median']} × recent median {med:.0f}"
                          " (truncated tar?)")
        for key in ("trace_count", "aircraft_count", "nic_known_share"):
            vals = [h.get(key) for h in history if h.get(key) is not None]
            if len(vals) >= 5 and day_qa.get(key) is not None:
                mu, sd = statistics.mean(vals), statistics.pstdev(vals)
                if sd > 0 and abs(day_qa[key] - mu) / sd > float(q["outlier_z"]):
                    warnings.append(f"{key} {day_qa[key]} is an outlier (recent mean {mu:.3g})")
    ag = day_qa.get("gpsjam")
    if ag is None:
        warnings.append("no GPSJAM file for agreement check")
    elif ag.get("spearman") is not None and ag["spearman"] < float(q["warn_spearman_below"]):
        warnings.append(f"GPSJAM agreement weak: rho={ag['spearman']}")
    return errors, warnings


def verify_day(paths, cfg: dict, day_qa: dict) -> dict:
    """Adds `gpsjam` and `qa` sections to day_qa (in place) and returns it."""
    try:
        day_qa["gpsjam"] = gpsjam_agreement(paths, cfg, day_qa["date"])
    except Exception as e:
        day_qa["gpsjam"] = {"error": f"{type(e).__name__}: {e}"}
    errors, warnings = evaluate(day_qa, recent_history(paths, day_qa["date"]), cfg)
    day_qa["qa"] = {"ok": not errors, "errors": errors, "warnings": warnings}
    return day_qa


def summarize(day_qa: dict) -> dict:
    g = day_qa.get("gpsjam") or {}
    return {"date": day_qa["date"], "spearman": g.get("spearman"), "agree_pct": g.get("agree_pct"),
            "nic_known": day_qa.get("nic_known_share"), "ok": day_qa.get("qa", {}).get("ok"),
            "min": round(day_qa.get("processing_seconds", 0) / 60, 1)}
