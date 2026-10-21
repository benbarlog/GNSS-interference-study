"""GPSJAM daily backfill and the region-level daily series the scheduler uses (plan §3, §5).

Backfill reuses `gnss.ingest_gpsjam` (URL pattern, cache file naming, date ranges) but downloads
through the polite client (≤ 1 connection, ≥ 1 s spacing, Retry-After-aware backoff).

Region series: for each day and region, `bad_share` = Σ bad / Σ (good + bad) aircraft over the
GPSJAM res-4 hexes whose centroid lies inside the region's bbox. Cached incrementally in
`state/gpsjam_region_daily.parquet`.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import date, timedelta

import h3
import numpy as np
import pandas as pd

from gnss.ingest_gpsjam import RAW_DIR as REPO_GPSJAM_DIR
from gnss.ingest_gpsjam import URL, cache_path, date_range

from .http import PoliteClient, Stopped

REGION_CACHE = "gpsjam_region_daily.parquet"


# ---- backfill --------------------------------------------------------------------------------
def backfill(paths, client: PoliteClient, start: date, end: date | None = None, events=None,
             stop=None) -> dict:
    """Fill every GPSJAM day in [start, end] into paths.gpsjam. Idempotent."""
    end = end or date.today() - timedelta(days=1)
    out_dir = paths.gpsjam
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    if REPO_GPSJAM_DIR.exists():
        for f in REPO_GPSJAM_DIR.glob("*-h3_4.csv"):
            dst = out_dir / f.name
            if not dst.exists() and f.stat().st_size > 0:
                shutil.copy2(f, dst)
                copied += 1
    missing_p = paths.state / "gpsjam_missing.json"
    known_missing = set(json.loads(missing_p.read_text())) if missing_p.exists() else set()
    recheck_after = end - timedelta(days=7)   # recent 404s may still appear; older ones are final
    todo = [d for d in date_range(start, end)
            if not cache_path(d, out_dir).exists()
            and not (d.isoformat() in known_missing and d < recheck_after)]
    if events and todo:
        events.info(f"GPSJAM: {len(todo)} daily files to fetch ({copied} copied from the repo cache)", True)
    got = missing = 0
    t0 = time.time()
    try:
        for i, d in enumerate(todo):
            if stop is not None and stop.is_set():
                raise Stopped()
            r = client.meta("GET", URL.format(d=d.isoformat()))
            if r.status_code == 404:
                known_missing.add(d.isoformat())
                missing += 1
                continue
            p = cache_path(d, out_dir)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(r.content)
            tmp.replace(p)
            got += 1
            if events and (i + 1) % 100 == 0:
                events.info(f"GPSJAM: {i + 1}/{len(todo)} fetched", True)
    finally:
        missing_p.write_text(json.dumps(sorted(known_missing)))
    have = len(list(out_dir.glob("*-h3_4.csv")))
    return {"fetched": got, "missing": missing, "copied": copied, "files": have,
            "seconds": round(time.time() - t0, 1)}


# ---- region series ---------------------------------------------------------------------------
def _hex_regions(hexes: np.ndarray, regions: dict, memo: dict) -> dict[str, np.ndarray]:
    """Boolean membership mask per region for an array of hex strings (centroid-in-bbox)."""
    lat = np.empty(len(hexes))
    lon = np.empty(len(hexes))
    for i, hx in enumerate(hexes):
        ll = memo.get(hx)
        if ll is None:
            ll = memo[hx] = h3.cell_to_latlng(hx)
        lat[i], lon[i] = ll
    masks = {}
    for name, bbox in regions.items():
        if bbox is None:
            masks[name] = np.ones(len(hexes), bool)
        else:
            la0, la1, lo0, lo1 = bbox
            masks[name] = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    return masks


def region_daily(paths, regions: dict, start: date, events=None) -> pd.DataFrame:
    """Columns: date, region, good, bad, n, n_hex, bad_share. Incremental over new files."""
    cache_p = paths.state / REGION_CACHE
    cached = pd.read_parquet(cache_p) if cache_p.exists() else pd.DataFrame()
    sig = json.dumps(regions, sort_keys=True)
    sig_p = paths.state / "gpsjam_region_sig.json"
    if not cached.empty and (not sig_p.exists() or sig_p.read_text() != sig):
        cached = pd.DataFrame()      # regions changed: rebuild
    done = set(cached["date"].dt.strftime("%Y-%m-%d")) if not cached.empty else set()
    files = sorted(f for f in paths.gpsjam.glob("*-h3_4.csv") if f.name[:10] >= start.isoformat())
    new_files = [f for f in files if f.name[:10] not in done]
    memo: dict[str, tuple[float, float]] = {}
    rows = []
    for f in new_files:
        try:
            df = pd.read_csv(f, dtype={"hex": str})
        except Exception:
            continue
        if df.empty:
            continue
        good = df["count_good_aircraft"].to_numpy()
        bad = df["count_bad_aircraft"].to_numpy()
        masks = _hex_regions(df["hex"].to_numpy(), regions, memo)
        for name, m in masks.items():
            g, b = int(good[m].sum()), int(bad[m].sum())
            rows.append((f.name[:10], name, g, b, g + b, int(m.sum())))
    if rows:
        new = pd.DataFrame(rows, columns=["date", "region", "good", "bad", "n", "n_hex"])
        new["date"] = pd.to_datetime(new["date"])
        cached = pd.concat([cached, new], ignore_index=True) if not cached.empty else new
        cached = cached.drop_duplicates(["date", "region"], keep="last").sort_values(["region", "date"])
        cached.to_parquet(cache_p, index=False)
        sig_p.write_text(sig)
        if events:
            events.info(f"GPSJAM region series: +{len(new_files)} days", False)
    if cached.empty:
        return cached
    cached = cached.copy()
    cached["bad_share"] = np.where(cached["n"] > 0, cached["bad"] / cached["n"].clip(lower=1), np.nan)
    return cached


def pivot(series: pd.DataFrame, regions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bad, n) wide frames indexed by daily date, one column per region (summed if several)."""
    sub = series[series["region"].isin(regions)]
    bad = sub.pivot_table(index="date", columns="region", values="bad", aggfunc="sum")
    n = sub.pivot_table(index="date", columns="region", values="n", aggfunc="sum")
    idx = pd.date_range(series["date"].min(), series["date"].max(), freq="D")
    return bad.reindex(idx), n.reindex(idx)


def combined_share(series: pd.DataFrame, regions: list[str]) -> pd.Series:
    bad, n = pivot(series, regions)
    tot_n = n.sum(axis=1, min_count=1)
    return (bad.sum(axis=1, min_count=1) / tot_n.where(tot_n > 0)).rename("bad_share")


# ---- analyses ---------------------------------------------------------------------------------
def variation_report(series: pd.DataFrame, regions: list[str]) -> dict:
    """Lag-1 autocorrelation and day-over-day change distribution per region (plan §3)."""
    out = {}
    for r in regions:
        s = combined_share(series, [r])
        if s.notna().sum() < 30:
            continue
        diff = s.diff().abs().dropna()
        out[r] = {
            "days": int(s.notna().sum()),
            "lag1_autocorr": round(float(s.autocorr(lag=1)), 4),
            "mean_bad_share": round(float(s.mean()), 5),
            "abs_dod_change_p50": round(float(diff.quantile(0.5)), 5),
            "abs_dod_change_p90": round(float(diff.quantile(0.9)), 5),
            "abs_dod_change_p99": round(float(diff.quantile(0.99)), 5),
        }
    return out


def window_days(win: dict) -> list[date]:
    return date_range(_d(win["start"]), _d(win["end"]))


def refine_peaks(series: pd.DataFrame, windows: list[dict]) -> dict[str, tuple[date, str]]:
    """{window_id: (peak_date, source)} — GPSJAM maximum within the window when ≥ half its days
    have data, else the provisional peak from windows.yaml."""
    out = {}
    for w in windows:
        s = combined_share(series, w["regions"]) if not series.empty else pd.Series(dtype=float)
        days = pd.DatetimeIndex(pd.to_datetime(window_days(w)))
        s = s.reindex(days)
        if s.notna().sum() >= max(1, len(days) // 2):
            out[w["id"]] = (s.idxmax().date(), "gpsjam")
        else:
            out[w["id"]] = (_d(w["peak"]), "provisional")
    return out


def quiet_controls(series: pd.DataFrame, primary: list[str], windows: list[dict], count: int,
                   lookback_days: int, min_sep: int) -> list[date]:
    if series.empty:
        return []
    shares = pd.concat({r: combined_share(series, [r]) for r in primary}, axis=1)
    shares = shares.dropna()
    if shares.empty:
        return []
    end = shares.index.max()
    shares = shares[shares.index > end - pd.Timedelta(days=lookback_days)]
    excluded = {d for w in windows for d in window_days(w)}
    score = shares.mean(axis=1)
    score = score[[d.date() not in excluded for d in score.index]].sort_values()
    picked: list[date] = []
    for ts in score.index:
        d = ts.date()
        if all(abs((d - p).days) >= min_sep for p in picked):
            picked.append(d)
        if len(picked) == count:
            break
    return picked


def spike_days(series: pd.DataFrame, regions: list[str], windows: list[dict], z: float,
               baseline_days: int, pad: int, min_aircraft: int) -> list[tuple[date, str, float]]:
    """Region-days with bad-share z > threshold vs a trailing baseline, outside case windows,
    padded ±pad days. Returns [(date, region, z)] sorted by z descending (deduped by date)."""
    if series.empty:
        return []
    excluded = {d for w in windows for d in window_days(w)}
    hits: dict[date, tuple[str, float]] = {}
    for r in regions:
        bad, n = pivot(series, [r])
        nn = n.sum(axis=1, min_count=1)
        s = (bad.sum(axis=1, min_count=1) / nn.where(nn >= min_aircraft))
        base = s.shift(1).rolling(baseline_days, min_periods=baseline_days // 2)
        zs = (s - base.mean()) / base.std()
        for ts, zv in zs[zs > z].items():
            for k in range(-pad, pad + 1):
                d = (ts + pd.Timedelta(days=k)).date()
                if d in excluded:
                    continue
                if d not in hits or hits[d][1] < zv:
                    hits[d] = (r, float(zv))
    return sorted(((d, r, zv) for d, (r, zv) in hits.items()), key=lambda x: -x[2])


def edge_suggestions(series: pd.DataFrame, windows: list[dict], pad: int = 14,
                     z: float = 2.0) -> dict[str, dict]:
    """Suggested onset/decay per window: first/last day near the window where the combined
    bad-share exceeds mean + z·std of the 28 days before the window. Written for review only."""
    out = {}
    for w in windows:
        s = combined_share(series, w["regions"]) if not series.empty else pd.Series(dtype=float)
        start, end = pd.Timestamp(_d(w["start"])), pd.Timestamp(_d(w["end"]))
        base = s[(s.index < start) & (s.index >= start - pd.Timedelta(days=28))].dropna()
        if len(base) < 14:
            out[w["id"]] = {"note": "not enough GPSJAM baseline"}
            continue
        thr = base.mean() + z * base.std()
        near = s[(s.index >= start - pd.Timedelta(days=pad)) & (s.index <= end + pd.Timedelta(days=pad))]
        above = near[near > thr]
        out[w["id"]] = {
            "current": [str(_d(w["start"])), str(_d(w["end"]))],
            "threshold_bad_share": round(float(thr), 5),
            "suggested_onset": str(above.index.min().date()) if len(above) else None,
            "suggested_decay": str(above.index.max().date()) if len(above) else None,
            "days_above": len(above),
        }
    return out


def _d(x) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x))
