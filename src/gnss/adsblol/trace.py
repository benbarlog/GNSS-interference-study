"""Worker-side trace parsing and per-batch aggregation (plan §6.1–6.3, §8).

readsb `trace_full_<icao>.json` (gzip'd; verified on 2025-06-14): top-level keys `icao`, `timestamp`
(day start, epoch s), `trace`, and optional `r`, `t`, `desc`, `ownOp`, `year`, `dbFlags`,
`noRegData`, `version`. Each trace point is a 14-element list:

    0 seconds after `timestamp`   5 track (deg)           10 geometric altitude (ft)
    1 lat                         6 flags                 11 geometric vertical rate
    2 lon                         7 vertical rate         12 indicated airspeed
    3 baro altitude | "ground"    8 details object | null 13 roll (deg)
    4 ground speed (kt)           9 source type

flags: 1 stale position, 2 start of a new leg, 4 vertical rate is geometric, 8 altitude is geometric.
The details object (≈1 in 4 points) carries `nic`, `nac_p`, `sil`, `version`, `category`,
`flight`, `squawk`, … and is forward-filled with a staleness cap.

Nothing point-level is stored except the `exceptions` table. Everything else is reduced to the
additive Tier 1 rows keyed (hour_utc, icao, hex5) and one aircraft_dim row per trace.
"""

from __future__ import annotations

import collections
import math
from itertools import zip_longest

import numpy as np
import orjson
import pyarrow as pa
from h3.api import basic_int as h3i
from isal import igzip
from numba import njit

from .schemas import (
    AIRCRAFT_DIM,
    EXCEPTIONS,
    NIC_BINS,
    REASON_CONTEXT,
    REASON_JUMP,
    REASON_NIC_LOW,
    REASON_NIC_UNKNOWN,
    TIER1,
)

SRC_NAMES = ["adsb_icao", "adsb_icao_nt", "adsb_other", "mlat", "adsr_icao", "adsr_other",
             "tisb_icao", "tisb_trackfile", "tisb_other", "adsc", "mode_s", "mode_ac", "other",
             "unknown"]
SRC_CODE = {n: i for i, n in enumerate(SRC_NAMES)}
SRC_UNKNOWN = SRC_CODE["unknown"]
ADSB_NAMES = {"adsb_icao", "adsb_icao_nt", "adsb_other"}
# 0 = ADS-B (integrity metrics), 1 = MLAT, 2 = other (ADS-R, TIS-B, Mode S, ...)
SRC_CLASS = np.array([0 if n in ADSB_NAMES else 1 if n == "mlat" else 2 for n in SRC_NAMES], np.int8)

CAT_NAMES = [f"{c}{i}" for c in "ABCD" for i in range(8)]
CAT_CODE = {n: i for i, n in enumerate(CAT_NAMES)}

# forward-filled detail fields, in this row order
FF_NIC, FF_NACP, FF_SIL, FF_VER, FF_CAT, FF_SQK, FF_CS = range(7)
N_FF = 7
_NUM = (int, float)
KTS_PER_RAD_S = 3440.065 * 3600.0     # earth radius in NM × s/h


def _num(v) -> float:
    return float(v) if v.__class__ in _NUM else math.nan


# ---- numba kernels ---------------------------------------------------------------------------
@njit(cache=True)
def ffill_capped(vals, t, trace_id, caps):
    """Forward-fill each row of vals (nan = no update) within a trace, dropping values older
    than caps[row] seconds."""
    k, n = vals.shape
    out = np.full((k, n), np.nan, dtype=np.float32)
    for r in range(k):
        last_v = np.nan
        last_t = -1e18
        cur = -1
        cap = caps[r]
        for i in range(n):
            if trace_id[i] != cur:
                cur = trace_id[i]
                last_v = np.nan
                last_t = -1e18
            v = vals[r, i]
            if not np.isnan(v):
                last_v = v
                last_t = t[i]
            if not np.isnan(last_v) and t[i] - last_t <= cap:
                out[r, i] = last_v
    return out


@njit(cache=True)
def point_metrics(trace_id, t, lat, lon, ground, adsb, flags, gs, dt_cap, min_dt, max_dt_mm,
                  jump_kt):
    """dt weights, leg ids, implied speed / jumps / speed mismatch (ADS-B airborne pairs)."""
    n = t.shape[0]
    dt = np.zeros(n, np.float32)
    leg = np.zeros(n, np.int32)
    implied = np.full(n, np.nan, np.float32)
    jump = np.zeros(n, np.bool_)
    mismatch = np.full(n, np.nan, np.float32)
    leg_id = -1
    for i in range(n):
        if i == 0 or trace_id[i] != trace_id[i - 1] or (flags[i] & 2) != 0:
            leg_id += 1
        leg[i] = leg_id
    for i in range(n):
        if i + 1 < n and leg[i + 1] == leg[i]:
            dt[i] = min(t[i + 1] - t[i], dt_cap)
        elif i > 0 and leg[i - 1] == leg[i]:
            dt[i] = min(t[i] - t[i - 1], dt_cap)
    j = -1
    for i in range(n):
        if j >= 0 and leg[j] != leg[i]:
            j = -1
        if not adsb[i] or ground[i]:
            continue
        if j >= 0:
            d_t = t[i] - t[j]
            if d_t >= min_dt:
                p1 = math.radians(lat[j])
                p2 = math.radians(lat[i])
                dp = p2 - p1
                dl = math.radians(lon[i] - lon[j])
                a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
                c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
                v = c * KTS_PER_RAD_S / d_t
                implied[i] = v
                if v > jump_kt:
                    jump[i] = True
                if d_t <= max_dt_mm and not np.isnan(gs[i]):
                    mismatch[i] = abs(v - gs[i])
        j = i
    return dt, leg, implied, jump, mismatch


@njit(cache=True)
def select_exceptions(trace_id, t, degraded, context_s, downsample_s):
    """Points within context_s of a degraded point, split into segments, ≤ 1 per downsample_s."""
    n = t.shape[0]
    near = np.zeros(n, np.bool_)
    cur = -1
    last = -1e18
    for i in range(n):
        if trace_id[i] != cur:
            cur = trace_id[i]
            last = -1e18
        if degraded[i]:
            last = t[i]
        if t[i] - last <= context_s:
            near[i] = True
    cur = -1
    nxt = 1e18
    for i in range(n - 1, -1, -1):
        if trace_id[i] != cur:
            cur = trace_id[i]
            nxt = 1e18
        if degraded[i]:
            nxt = t[i]
        if nxt - t[i] <= context_s:
            near[i] = True
    keep = np.zeros(n, np.bool_)
    seg = np.zeros(n, np.int32)
    cur = -1
    counter = -1
    prev = False
    last_kept = -1e18
    for i in range(n):
        if trace_id[i] != cur:
            cur = trace_id[i]
            counter = -1
            prev = False
        if near[i]:
            if not prev:
                counter += 1
                last_kept = -1e18
            seg[i] = counter
            if t[i] - last_kept >= downsample_s:
                keep[i] = True
                last_kept = t[i]
            prev = True
        else:
            prev = False
    return keep, seg


def warmup() -> None:
    """Compile (or load cached) kernels once per worker."""
    tid = np.zeros(3, np.int32)
    t = np.array([0.0, 5.0, 10.0])
    ll = np.array([50.0, 50.01, 50.02])
    b = np.array([False, False, False])
    ffill_capped(np.full((2, 3), np.nan, np.float32), t, tid, np.array([120.0, 1e18]))
    point_metrics(tid, t, ll, ll, b, ~b, np.zeros(3, np.int16), np.zeros(3, np.float32),
                  60.0, 1.0, 30.0, 1000.0)
    select_exceptions(tid, t, b, 120.0, 10.0)


# ---- parsing ---------------------------------------------------------------------------------
class ParsedTrace:
    __slots__ = (
        "alt",
        "alt_geom",
        "det_keys",
        "doc_keys",
        "ff",
        "flags",
        "ground",
        "gs",
        "icao",
        "lat",
        "lon",
        "meta",
        "ncols",
        "roll",
        "src",
        "src_names",
        "t",
        "track",
    )


def parse_trace(raw: bytes, callsigns: dict[str, int], sample: bool = False) -> ParsedTrace | None:
    data = igzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    doc = orjson.loads(data)
    tr = doc.get("trace") or []
    p = ParsedTrace()
    p.icao = str(doc.get("icao", "")).lower()
    p.meta = {k: doc.get(k) for k in ("r", "t", "desc", "ownOp", "year", "dbFlags")}
    p.doc_keys = list(doc.keys()) if sample else None
    n = len(tr)
    if n == 0:
        return None
    cols = list(zip_longest(*tr))
    p.ncols = len(cols)
    if len(cols) < 14:
        cols += [(None,) * n] * (14 - len(cols))
    p.t = float(doc["timestamp"]) + np.asarray(cols[0], np.float64)
    p.lat = np.array(cols[1], np.float64)
    p.lon = np.array(cols[2], np.float64)
    alt_c = cols[3]
    p.ground = np.fromiter((a == "ground" for a in alt_c), np.bool_, n)
    p.alt = np.array([a if a.__class__ in _NUM else None for a in alt_c], np.float32)
    p.gs = np.array(cols[4], np.float32)
    p.track = np.array(cols[5], np.float32)
    p.flags = np.nan_to_num(np.array(cols[6], np.float32)).astype(np.int16)
    p.src = np.fromiter((SRC_CODE.get(s, SRC_UNKNOWN) for s in cols[9]), np.int8, n)
    p.alt_geom = np.array(cols[10], np.float32)
    p.roll = np.array(cols[13], np.float32)
    ff = np.full((N_FF, n), np.nan, np.float32)
    det_keys = collections.Counter() if sample else None
    for i, x in enumerate(cols[8]):
        if x is None:
            continue
        if sample:
            det_keys.update(x.keys())
        if x.get("type") in ADSB_NAMES:
            ff[FF_NIC, i] = _num(x.get("nic"))
            ff[FF_NACP, i] = _num(x.get("nac_p"))
            ff[FF_SIL, i] = _num(x.get("sil"))
            ff[FF_VER, i] = _num(x.get("version"))
        cat = x.get("category")
        if cat is not None:
            ff[FF_CAT, i] = CAT_CODE.get(cat, math.nan)
        sq = x.get("squawk")
        if sq:
            try:
                ff[FF_SQK, i] = int(sq)
            except (TypeError, ValueError):
                pass
        fl = x.get("flight")
        if fl:
            fl = fl.strip()
            if fl:
                code = callsigns.get(fl)
                if code is None:
                    code = callsigns[fl] = len(callsigns)
                ff[FF_CS, i] = code
    p.ff = ff
    p.det_keys = det_keys
    p.src_names = collections.Counter(cols[9]) if sample else None
    return p


# ---- aggregation helpers ---------------------------------------------------------------------
def _mode(gid: np.ndarray, codes: np.ndarray, w: np.ndarray, G: int) -> np.ndarray:
    """Weighted mode of non-negative integer codes per group; -1 when none."""
    out = np.full(G, -1, np.int64)
    m = codes >= 0
    if not m.any():
        return out
    K = int(codes[m].max()) + 1
    key = gid[m].astype(np.int64) * K + codes[m]
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.bincount(inv, weights=w[m] + 1e-6)
    gg, cc = uk // K, uk % K
    order = np.lexsort((-sums, gg))
    first = np.ones(len(order), bool)
    first[1:] = gg[order][1:] != gg[order][:-1]
    out[gg[order][first]] = cc[order][first]
    return out


def _nullable(values: np.ndarray, valid: np.ndarray, typ: pa.DataType) -> pa.Array:
    vals = np.where(valid, values, 0)
    return pa.array(vals.astype(typ.to_pandas_dtype()), type=typ, mask=~valid)


def _strings(codes: np.ndarray, names: list[str]) -> pa.Array:
    lut = np.array(names + [None], dtype=object)
    return pa.array(lut[np.where(codes >= 0, codes, len(names))], type=pa.string())


def _meta_cols(n: int, version: str, chash: str) -> list[pa.Array]:
    return [pa.array([version] * n, pa.string()), pa.array([chash] * n, pa.string())]


# ---- batch -----------------------------------------------------------------------------------
def process_batch(batch: list[tuple[str, bytes]], day_start: float, P: dict, sample_n: int = 0) -> dict:
    """Parse and aggregate one batch of traces. Returns pyarrow tables + QA counters."""
    day_end = day_start + 86400.0
    callsigns: dict[str, int] = {}
    parsed: list[ParsedTrace] = []
    qa = collections.Counter()
    verify = {"doc_keys": collections.Counter(), "det_keys": collections.Counter(),
              "src": collections.Counter(), "flags": collections.Counter(),
              "ncols": collections.Counter(), "paths": collections.Counter(),
              "dbflags_bits": collections.Counter(), "mlat_pts": 0, "mlat_near_adsb": 0,
              "sampled_traces": 0, "errors": []}
    for k, (name, raw) in enumerate(batch):
        qa["traces"] += 1
        sample = k < sample_n
        try:
            pt = parse_trace(raw, callsigns, sample)
        except Exception as e:  # malformed member: count it, keep going
            qa["parse_errors"] += 1
            if len(verify["errors"]) < 5:
                verify["errors"].append(f"{name}: {type(e).__name__}: {e}"[:300])
            continue
        if sample:
            verify["paths"]["/".join(name.split("/")[:2])] += 1
        if pt is None:
            qa["empty_traces"] += 1
            continue
        if sample:
            _verify_sample(pt, verify)
        parsed.append(pt)
    if not parsed:
        return {"tier1": None, "dim": None, "exc": None, "qa": dict(qa), "verify": verify}

    sizes = np.array([len(p.t) for p in parsed])
    trace_id = np.repeat(np.arange(len(parsed), dtype=np.int32), sizes)
    cat = lambda attr: np.concatenate([getattr(p, attr) for p in parsed])
    t = cat("t")
    ff_raw = np.concatenate([p.ff for p in parsed], axis=1)
    caps = np.array([P["staleness_s"]] * (N_FF - 1) + [1e18], np.float64)
    ff = ffill_capped(ff_raw, t, trace_id, caps)
    qa["points_total"] += len(t)
    qa["n_details"] += int((~np.isnan(ff_raw[FF_NIC])).sum())

    lat, lon, flags = cat("lat"), cat("lon"), cat("flags")
    in_day = (t >= day_start) & (t < day_end)
    stale = (flags & 1) != 0
    finite = np.isfinite(lat) & np.isfinite(lon)
    keep = in_day & ~stale & finite
    qa["points_stale"] += int(stale.sum())
    qa["points_out_of_day"] += int((~in_day).sum())
    qa["points_nonfinite"] += int((~finite).sum())

    t, lat, lon, flags, trace_id = t[keep], lat[keep], lon[keep], flags[keep], trace_id[keep]
    ff = ff[:, keep]
    alt, ground, gs = cat("alt")[keep], cat("ground")[keep], cat("gs")[keep]
    track, src = cat("track")[keep], cat("src")[keep]
    alt_geom, roll = cat("alt_geom")[keep], cat("roll")[keep]
    n = len(t)
    qa["points_kept"] += n
    if n == 0:
        return {"tier1": None, "dim": None, "exc": None, "qa": dict(qa), "verify": verify}

    cls = SRC_CLASS[src]
    adsb = cls == 0
    air = ~ground
    air_adsb = air & adsb
    dt, _leg, implied, jump, mismatch = point_metrics(
        trace_id, t, lat, lon, ground, adsb, flags, gs, float(P["dt_cap_s"]),
        float(P["min_dt_implied_s"]), float(P["max_dt_mismatch_s"]), float(P["jump_kt"]))
    dt64 = dt.astype(np.float64)

    nic, nacp = ff[FF_NIC], ff[FF_NACP]
    nic_known = ~np.isnan(nic)
    qa["pts_air"] += int(air.sum())
    qa["pts_air_adsb"] += int(air_adsb.sum())
    qa["s_obs"] += float(dt64.sum())
    qa["s_adsb"] += float(dt64[air_adsb].sum())
    qa["s_adsb_nic_known"] += float(dt64[air_adsb & nic_known].sum())
    qa["s_ground"] += float(dt64[ground].sum())
    qa["unknown_src_pts"] += int((src == SRC_UNKNOWN).sum())

    # ---- Tier 1 ------------------------------------------------------------------------------
    hour = np.clip(((t - day_start) // 3600).astype(np.int64), 0, 23)
    res = int(P["h3_res"])
    hexes = np.fromiter((h3i.latlng_to_cell(a, b, res) for a, b in zip(lat.tolist(), lon.tolist())),
                        np.int64, n)
    o = np.lexsort((hexes, hour, trace_id))
    tid_s, hour_s, hex_s = trace_id[o], hour[o], hexes[o]
    new = np.empty(n, bool)
    new[0] = True
    new[1:] = (tid_s[1:] != tid_s[:-1]) | (hour_s[1:] != hour_s[:-1]) | (hex_s[1:] != hex_s[:-1])
    starts = np.flatnonzero(new)
    G = len(starts)
    gid = np.cumsum(new) - 1                      # group id in sorted order

    def s(x):                                     # sum per group
        return np.add.reduceat(x[o], starts)

    def mn(x, m):
        v = np.minimum.reduceat(np.where(m, x, np.inf)[o], starts)
        return v, np.isfinite(v)

    def mx(x, m):
        v = np.maximum.reduceat(np.where(m, x, -np.inf)[o], starts)
        return v, np.isfinite(v)

    w_adsb = np.where(air_adsb, dt64, 0.0)
    nic_bin = np.where(nic_known, np.clip(np.nan_to_num(nic), 0, NIC_BINS - 1), NIC_BINS).astype(np.int64)
    nacp_bin = np.where(~np.isnan(nacp), np.clip(np.nan_to_num(nacp), 0, NIC_BINS - 1), NIC_BINS).astype(np.int64)
    nic_h = np.bincount(gid * (NIC_BINS + 1) + nic_bin[o], weights=w_adsb[o],
                        minlength=G * (NIC_BINS + 1)).reshape(G, NIC_BINS + 1)
    nacp_h = np.bincount(gid * (NIC_BINS + 1) + nacp_bin[o], weights=w_adsb[o],
                         minlength=G * (NIC_BINS + 1)).reshape(G, NIC_BINS + 1)

    sil, ver = ff[FF_SIL], ff[FF_VER]
    sil_min, sil_ok = mn(sil.astype(np.float64), air_adsb & ~np.isnan(sil))
    ver_max, ver_ok = mx(ver.astype(np.float64), air_adsb & ~np.isnan(ver))
    cat_code = np.where(np.isnan(ff[FF_CAT]), -1, ff[FF_CAT]).astype(np.int64)
    cs_code = np.where(np.isnan(ff[FF_CS]), -1, ff[FF_CS]).astype(np.int64)
    cat_mode = _mode(gid, cat_code[o], dt64[o], G)
    cs_mode = _mode(gid, cs_code[o], dt64[o], G)
    cs_names = [None] * len(callsigns)
    for name, code in callsigns.items():
        cs_names[code] = name

    absroll = np.abs(roll.astype(np.float64))
    roll_ok = air & ~np.isnan(absroll)
    max_roll, max_roll_ok = mx(absroll, roll_ok)
    baro_ok = air & ~np.isnan(alt) & ((flags & 8) == 0)
    alt64 = alt.astype(np.float64)
    alt_min, alt_min_ok = mn(alt64, baro_ok)
    alt_max, alt_max_ok = mx(alt64, baro_ok)
    gap_ok = baro_ok & ~np.isnan(alt_geom)
    gap = np.abs(alt_geom.astype(np.float64) - alt64)
    gap_max, gap_max_ok = mx(gap, gap_ok)
    imp_max, imp_ok = mx(implied.astype(np.float64), ~np.isnan(implied))
    mm_max, mm_ok = mx(mismatch.astype(np.float64), ~np.isnan(mismatch))
    sqk = ff[FF_SQK]

    hi_roll = roll_ok & (absroll > float(P["high_roll_deg"]))
    cols = [
        pa.array(hour_s[starts].astype(np.uint8)),
        pa.array(np.array([p.icao for p in parsed], dtype=object)[tid_s[starts]], pa.string()),
        pa.array(hex_s[starts].astype(np.uint64)),
        _strings(cs_mode, cs_names),
        pa.array(np.add.reduceat(np.ones(n, np.int64), starts).astype(np.uint16)),
        pa.array(s(adsb.astype(np.int64)).astype(np.uint16)),
        pa.array(s((cls == 1).astype(np.int64)).astype(np.uint16)),
        pa.array(s((cls == 2).astype(np.int64)).astype(np.uint16)),
        pa.array(s(dt64).astype(np.float32)),
        pa.array(s(w_adsb).astype(np.float32)),
        *[pa.array(nic_h[:, i].astype(np.float32)) for i in range(NIC_BINS + 1)],
        *[pa.array(nacp_h[:, i].astype(np.float32)) for i in range(NIC_BINS + 1)],
        _nullable(sil_min, sil_ok, pa.uint8()),
        _nullable(ver_max, ver_ok, pa.uint8()),
        _strings(cat_mode, CAT_NAMES),
        pa.array(s(np.where(hi_roll, dt64, 0.0)).astype(np.float32)),
        _nullable(max_roll, max_roll_ok, pa.float32()),
        _nullable(alt_min, alt_min_ok, pa.int32()),
        _nullable(alt_max, alt_max_ok, pa.int32()),
        pa.array(s(np.where(baro_ok, alt64 * dt64, 0.0))),
        pa.array(s(np.where(baro_ok, dt64, 0.0)).astype(np.float32)),
        pa.array(s(np.where(ground, dt64, 0.0)).astype(np.float32)),
        pa.array(s(gap_ok.astype(np.int64)).astype(np.uint16)),
        pa.array(s(np.where(gap_ok, gap, 0.0)).astype(np.float32)),
        _nullable(gap_max, gap_max_ok, pa.float32()),
        pa.array(s(jump.astype(np.int64)).astype(np.uint16)),
        _nullable(imp_max, imp_ok, pa.float32()),
        _nullable(mm_max, mm_ok, pa.float32()),
        pa.array(np.floor(np.minimum.reduceat(t[o], starts)).astype(np.uint32)),
        pa.array(np.ceil(np.maximum.reduceat(t[o], starts)).astype(np.uint32)),
        pa.array(s((sqk == 7500).astype(np.int64)) > 0),
        pa.array(s((sqk == 7600).astype(np.int64)) > 0),
        pa.array(s((sqk == 7700).astype(np.int64)) > 0),
        *_meta_cols(G, P["pipeline_version"], P["config_hash"]),
    ]
    tier1 = pa.Table.from_arrays(cols, schema=TIER1)

    # ---- aircraft_dim ------------------------------------------------------------------------
    T = len(parsed)
    bad_lt = float(P["nic_bad_lt"])
    s_adsb_day = np.bincount(trace_id, weights=w_adsb, minlength=T)
    nic_lt = np.bincount(trace_id, weights=np.where(air_adsb & nic_known & (nic < bad_lt), dt64, 0.0),
                         minlength=T)
    npts = np.bincount(trace_id, minlength=T)
    present = npts > 0
    idx = np.flatnonzero(present)
    qa["aircraft"] += int(present.sum())

    def meta_str(key):
        return pa.array([_s(parsed[i].meta.get(key)) for i in idx], pa.string())

    def meta_int(key, hi):
        vals = [_i(parsed[i].meta.get(key), hi) for i in idx]
        return pa.array(vals, pa.uint16())

    dim = pa.Table.from_arrays([
        pa.array([parsed[i].icao for i in idx], pa.string()),
        meta_str("r"), meta_str("t"), meta_str("desc"), meta_str("ownOp"),
        meta_int("year", 3000), meta_int("dbFlags", 65535),
        pa.array(s_adsb_day[idx].astype(np.float32)),
        pa.array(nic_lt[idx].astype(np.float32)),
        pa.array(npts[idx].astype(np.uint32)),
        *_meta_cols(len(idx), P["pipeline_version"], P["config_hash"]),
    ], schema=AIRCRAFT_DIM)

    # ---- exceptions --------------------------------------------------------------------------
    nic_low = air_adsb & nic_known & (nic < bad_lt)
    nic_unk = air_adsb & ~nic_known & bool(P["include_nic_unknown"])
    degraded = nic_low | nic_unk | jump
    exc = None
    if degraded.any():
        ek, seg = select_exceptions(trace_id, t, degraded, float(P["context_s"]), float(P["downsample_s"]))
        e = np.flatnonzero(ek)
        reason = (nic_low[e] * REASON_NIC_LOW + nic_unk[e] * REASON_NIC_UNKNOWN
                  + jump[e] * REASON_JUMP).astype(np.uint8)
        reason[reason == 0] = REASON_CONTEXT
        nic_e, nacp_e = nic[e], nacp[e]
        exc = pa.Table.from_arrays([
            pa.array(t[e]),
            pa.array(np.array([p.icao for p in parsed], dtype=object)[trace_id[e]], pa.string()),
            pa.array(lat[e].astype(np.float32)),
            pa.array(lon[e].astype(np.float32)),
            _nullable(alt[e], ~np.isnan(alt[e]) & ((flags[e] & 8) == 0), pa.int32()),
            _nullable(alt_geom[e], ~np.isnan(alt_geom[e]), pa.int32()),
            pa.array(gs[e]), pa.array(track[e]), pa.array(roll[e]),
            pa.array(np.array(SRC_NAMES, dtype=object)[src[e]], pa.string()),
            _nullable(nic_e, ~np.isnan(nic_e), pa.uint8()),
            _nullable(nacp_e, ~np.isnan(nacp_e), pa.uint8()),
            pa.array(implied[e]),
            pa.array(np.minimum(seg[e], 65535).astype(np.uint16)),
            pa.array(reason),
            pa.array(ground[e]),
            *_meta_cols(len(e), P["pipeline_version"], P["config_hash"]),
        ], schema=EXCEPTIONS)
        qa["exception_rows"] += len(e)
    return {"tier1": tier1, "dim": dim, "exc": exc, "qa": dict(qa), "verify": verify}


def _s(v) -> str | None:
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _i(v, hi: int) -> int | None:
    try:
        x = int(v)
    except (TypeError, ValueError):
        return None
    return x if 0 <= x <= hi else None


def _verify_sample(p: ParsedTrace, v: dict) -> None:
    v["sampled_traces"] += 1
    v["doc_keys"].update(p.doc_keys or [])
    v["det_keys"].update(p.det_keys or {})
    v["src"].update(p.src_names or {})
    v["flags"].update(collections.Counter(p.flags.tolist()))
    v["ncols"][p.ncols] += 1
    db = _i(p.meta.get("dbFlags"), 65535)
    if db:
        for bit in range(8):
            if db & (1 << bit):
                v["dbflags_bits"][1 << bit] += 1
    cls = SRC_CLASS[p.src]
    ta = np.sort(p.t[cls == 0])
    tm = p.t[cls == 1]
    if len(ta) and len(tm):
        i = np.clip(np.searchsorted(ta, tm), 1, len(ta) - 1) if len(ta) > 1 else np.zeros(len(tm), int)
        near = np.minimum(np.abs(ta[i] - tm), np.abs(ta[np.maximum(i - 1, 0)] - tm))
        v["mlat_pts"] += len(tm)
        v["mlat_near_adsb"] += int((near <= 5).sum())
