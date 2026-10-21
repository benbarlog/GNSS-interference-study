"""Hex x day panels from either source, in one shape:

    date (datetime64), hex (int64 H3 index), n (aircraft), bad (aircraft), rate = bad / n

plus, for adsb.lol only, `s_bad`, `s_known`, `bad_share_time` (time-weighted NIC < nic_lt share).

- GPSJAM: daily res-4 CSVs. Coarser resolutions sum counts over `cell_to_parent` (counts of distinct
  aircraft per hex are not strictly additive across hexes, so coarse GPSJAM n is an upper bound;
  adsb.lol coarse resolutions count distinct aircraft properly).
- adsb.lol: Tier 2 `hex_day_res{res}_nic{nic}.parquet` if materialized, else the same SQL run live
  over Tier 1 (`gnss.adsblol.tier2.hex_day`). Optional filters drop high-roll rows, ADS-B version 0,
  and faulty-avionics aircraft-days (the confounder filters in GNSS_interference_plan.md §3).

Missing hexes stay missing: no zero filling anywhere.
"""

from __future__ import annotations

from datetime import date
from functools import cache

import h3
import numpy as np
import pandas as pd

from .config import AnalysisConfig


# ---- geometry helpers ------------------------------------------------------------------------
@cache
def _latlng(cell: int) -> tuple[float, float]:
    return h3.cell_to_latlng(h3.int_to_str(int(cell)))


def centroids(cells) -> tuple[np.ndarray, np.ndarray]:
    ll = np.array([_latlng(int(c)) for c in cells], dtype=float).reshape(-1, 2)
    return ll[:, 0], ll[:, 1]


def in_bbox(lat: np.ndarray, lon: np.ndarray, bbox) -> np.ndarray:
    if bbox is None:
        return np.ones(len(lat), bool)
    a, b, c, d = bbox
    return (lat >= a) & (lat <= b) & (lon >= c) & (lon <= d)


def select_region(df: pd.DataFrame, bbox, hex_col: str = "hex") -> pd.DataFrame:
    """Rows whose hex centroid lies inside bbox [lat_min, lat_max, lon_min, lon_max]."""
    if bbox is None or df.empty:
        return df
    cells = df[hex_col].unique()
    lat, lon = centroids(cells)
    keep = set(cells[in_bbox(lat, lon, bbox)].tolist())
    return df[df[hex_col].isin(keep)]


def select_regions(df: pd.DataFrame, cfg: AnalysisConfig, names) -> pd.DataFrame:
    """Union of several regions, with a `region` column (first matching name wins on overlap)."""
    cells = df["hex"].unique()
    lat, lon = centroids(cells)
    label = pd.Series(pd.NA, index=cells, dtype="object")
    for name in names:
        m = in_bbox(lat, lon, cfg.regions[name]) & label.isna().to_numpy()
        label.iloc[np.flatnonzero(m)] = name
    label = label.dropna()
    out = df[df["hex"].isin(label.index)].copy()
    out["region"] = out["hex"].map(label)
    return out


def to_parent(df: pd.DataFrame, res: int) -> pd.DataFrame:
    """Aggregate a res-4 GPSJAM panel to a coarser resolution by summing counts."""
    parents = {c: h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(c)), res))
               for c in df["hex"].unique()}
    out = (df.assign(hex=df["hex"].map(parents))
             .groupby(["date", "hex"], as_index=False)[["n", "bad"]].sum())
    out["rate"] = out["bad"] / out["n"]
    return out


# ---- GPSJAM ----------------------------------------------------------------------------------
def load_gpsjam(cfg: AnalysisConfig, start: date | None = None, end: date | None = None,
                dates=None, res: int = 4) -> pd.DataFrame:
    d = cfg.gpsjam_dir()
    files = sorted(d.glob("*-h3_4.csv"))
    if dates is not None:
        want = {x.isoformat() for x in dates}
        files = [f for f in files if f.name[:10] in want]
    else:
        files = [f for f in files if (start is None or f.name[:10] >= start.isoformat())
                 and (end is None or f.name[:10] <= end.isoformat())]
    frames = []
    for f in files:
        try:
            g = pd.read_csv(f, dtype={"hex": str})
        except Exception:
            continue
        if g.empty:
            continue
        frames.append(pd.DataFrame({
            "date": pd.Timestamp(f.name[:10]),
            "hex": g["hex"].map(h3.str_to_int).astype("int64"),
            "n": (g["count_good_aircraft"] + g["count_bad_aircraft"]).astype("int32"),
            "bad": g["count_bad_aircraft"].astype("int32"),
        }))
    if not frames:
        return pd.DataFrame(columns=["date", "hex", "n", "bad", "rate"])
    out = pd.concat(frames, ignore_index=True)
    out = out[out["n"] > 0]
    out["rate"] = out["bad"] / out["n"]
    return to_parent(out, res) if res < 4 else out


# ---- adsb.lol --------------------------------------------------------------------------------
def faulty_sql(nic_lt: int, min_known_s: float = 600.0, bad_share: float = 0.95,
               others_bad_below: float = 0.10, min_others: int = 2, clean_time_share: float = 0.5) -> str:
    """Aircraft-days treated as faulty avionics rather than jamming victims.

    Low NIC all day is not enough: an aircraft that spends the whole flight inside a jammed zone
    looks the same, and jamming is far more common than broken equipment. So an aircraft-day is
    flagged only if it is low-NIC for ≥ `bad_share` of ≥ `min_known_s` known seconds AND ≥
    `clean_time_share` of that time is spent in res-4 hexes where *other* aircraft were fine
    (≥ `min_others` others, their bad share < `others_bad_below`). Time in hexes with no other
    aircraft counts as "not clean": the benefit of the doubt goes to jamming."""
    from gnss.adsblol.tier2 import _bad_expr, _known_expr
    return f"""
        WITH ah AS (
            SELECT t.date, t.icao, l.hex4 AS hex,
                   SUM({_bad_expr(nic_lt)}) AS s_bad, SUM({_known_expr()}) AS s_known
            FROM tier1 t JOIN hex_lookup l ON l.hex5 = t.hex5
            GROUP BY ALL
        ), hx AS (
            SELECT date, hex, SUM(s_bad) AS hb, SUM(s_known) AS hk,
                   COUNT(*) FILTER (WHERE s_known > 0) AS na
            FROM ah GROUP BY ALL
        ), j AS (
            SELECT ah.date, ah.icao, ah.s_bad, ah.s_known,
                   (hx.hb - ah.s_bad) / NULLIF(hx.hk - ah.s_known, 0) AS others_share,
                   hx.na - (ah.s_known > 0)::INT AS others_n
            FROM ah JOIN hx USING (date, hex)
        ), d AS (
            SELECT date, icao, SUM(s_bad) AS b, SUM(s_known) AS k,
                   SUM(CASE WHEN others_n >= {min_others} AND others_share < {others_bad_below}
                            THEN s_known ELSE 0 END) AS k_clean
            FROM j GROUP BY ALL
        )
        SELECT date, icao FROM d
        WHERE k >= {min_known_s} AND b / k >= {bad_share} AND k_clean / k >= {clean_time_share}"""


def filter_where(roll: bool = False, v0: bool = False, faulty: bool = False) -> str:
    """Tier 1 row filter (alias t). Roll: drop rows where > 50% of ADS-B time had |roll| > 30°.
    Faulty: drop aircraft-days in the `faulty_aircraft` view (see `faulty_sql`)."""
    w = ["TRUE"]
    if roll:
        w.append("(t.s_high_roll <= 0.5 * t.s_adsb)")
    if v0:
        w.append("(t.version IS NULL OR t.version > 0)")
    if faulty:
        w.append("NOT EXISTS (SELECT 1 FROM faulty_aircraft b WHERE b.date = t.date AND b.icao = t.icao)")
    return " AND ".join(w)


def adsblol_connect(cfg: AnalysisConfig, threads: int = 6):
    from gnss.adsblol.tier2 import build_hex_lookup, connect
    con = connect(cfg.paths, threads=threads)
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    if "tier1" not in tables:
        raise FileNotFoundError(f"no Tier 1 parquet under {cfg.paths.tier1} (run on the desktop)")
    if "hex_lookup" not in tables:
        build_hex_lookup(con, cfg.paths)
    f = cfg["rate"].get("faulty") or {}
    con.execute(f"CREATE OR REPLACE TEMP VIEW faulty_aircraft AS {faulty_sql(cfg['rate']['nic_lt'], **f)}")
    return con


def load_adsblol(cfg: AnalysisConfig, start: date | None = None, end: date | None = None,
                 res: int | None = None, filters: dict | None = None, con=None) -> pd.DataFrame:
    """Hex x day from Tier 2 (materialized file when unfiltered and present, else live SQL)."""
    from gnss.adsblol.tier2 import hex_day
    r = cfg["rate"]
    res = res or r["res"]
    filters = filters or {}
    lo = f"'{start.isoformat()}'" if start else "'1900-01-01'"
    hi = f"'{end.isoformat()}'" if end else "'2999-12-31'"
    mat = cfg.paths.tier2 / f"hex_day_res{res}_nic{r['nic_lt']}.parquet"
    con = con or adsblol_connect(cfg)
    if not any(filters.values()) and mat.exists() and r["aircraft_bad_share"] == 0.5:
        sql = (f"SELECT * FROM read_parquet('{mat.as_posix()}') "
               f"WHERE date::DATE BETWEEN {lo} AND {hi}")
    else:
        where = f"t.date::DATE BETWEEN {lo} AND {hi} AND " + filter_where(**filters)
        sql = hex_day(res, r["nic_lt"], r["aircraft_bad_share"], where=where)
    df = con.sql(sql).df()
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"].astype(str)),
        "hex": df["hex"].astype("int64"),
        "n": df["n_aircraft"].astype("int32"),
        "bad": df["n_bad_aircraft"].astype("int32"),
        "s_bad": df["s_bad"].astype("float64"),
        "s_known": df["s_known"].astype("float64"),
        "bad_share_time": df["bad_share_time"].astype("float64"),
    })
    out = out[out["n"] > 0].reset_index(drop=True)
    out["rate"] = out["bad"] / out["n"]
    return out


def load_panel(cfg: AnalysisConfig, source: str | None = None, start: date | None = None,
               end: date | None = None, res: int | None = None, filters: dict | None = None,
               con=None) -> pd.DataFrame:
    source = source or cfg["source"]
    res = res or cfg["rate"]["res"]
    if source == "gpsjam":
        if res > 4:
            raise ValueError("GPSJAM is fixed at H3 res 4; finer resolutions need adsb.lol")
        return load_gpsjam(cfg, start, end, res=res)
    if source == "adsblol":
        return load_adsblol(cfg, start, end, res=res, filters=filters, con=con)
    raise ValueError(f"unknown source {source!r}")


def observed(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    return df[df["n"] >= min_n]
