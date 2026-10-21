"""Covariates per hex (and per hex-day): conflict events, military installations, distance to states
at war, geomagnetic activity. Downloaders cache under `covariates_dir`; builders are pure functions of
hex cells + dates so they can be tested on synthetic inputs.

Sources (attribute in the README/report):
- UCDP GED v26.1 (1989–2025) + UCDP Candidate Events 2026 (monthly releases), Uppsala Conflict Data
  Program. https://ucdp.uu.se/downloads/
- GFZ Potsdam Kp index (CC BY 4.0). https://kp.gfz-potsdam.de/
- OpenStreetMap military=airfield|naval_base|base via Overpass (ODbL). Coverage is uneven: absence
  of a mapped base is not absence of a base.
- Natural Earth admin-0 countries (public domain).
- ACLED events by category (see gnss.analysis.acled), when credentials/exports are present.

Distances are great-circle (haversine via chord distance on the unit sphere).
"""

from __future__ import annotations

import json
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

from .config import AnalysisConfig
from .data import centroids

R_EARTH_KM = 6371.0088
UCDP_GED_URL = "https://ucdp.uu.se/downloads/ged/ged261-csv.zip"
UCDP_CANDIDATE_URLS = [   # later files win on duplicate event ids
    "https://ucdp.uu.se/downloads/candidateged/GEDEvent_v26_01_26_06.csv",
    "https://ucdp.uu.se/downloads/candidateged/GEDEvent_v26_0_7.csv",
    "https://ucdp.uu.se/downloads/candidateged/GEDEvent_v26_0_8.csv",
]
KP_URL = "https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt"
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter"]


# ---- geometry --------------------------------------------------------------------------------
def unit_xyz(lat, lon) -> np.ndarray:
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    return np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]


def km_to_chord(km: float) -> float:
    return 2 * np.sin(km / (2 * R_EARTH_KM))


def chord_to_km(c) -> np.ndarray:
    return 2 * R_EARTH_KM * np.arcsin(np.clip(np.asarray(c) / 2, 0, 1))


def nearest_km(lat, lon, plat, plon) -> np.ndarray:
    """Great-circle km from each (lat, lon) to the nearest of (plat, plon); inf when none."""
    if len(plat) == 0:
        return np.full(len(lat), np.inf)
    d, _ = cKDTree(unit_xyz(plat, plon)).query(unit_xyz(lat, lon))
    return chord_to_km(d)


# ---- downloads -------------------------------------------------------------------------------
def _get(url: str, ua: str, out: Path, **kw) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers={"User-Agent": ua}, timeout=120, **kw)
    r.raise_for_status()
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(r.content)
    tmp.replace(out)
    time.sleep(1.0)
    return out


def download_ucdp(cfg: AnalysisConfig, ua: str, force: bool = False) -> list[Path]:
    d = cfg.covariates_dir / "ucdp"
    out = []
    for url in [UCDP_GED_URL, *UCDP_CANDIDATE_URLS]:
        p = d / url.rsplit("/", 1)[1]
        if force or not p.exists():
            _get(url, ua, p)
        out.append(p)
    return out


def download_kp(cfg: AnalysisConfig, ua: str, max_age_days: int = 3) -> Path:
    p = cfg.covariates_dir / "kp" / KP_URL.rsplit("/", 1)[1]
    if not p.exists() or (time.time() - p.stat().st_mtime) > max_age_days * 86400:
        _get(KP_URL, ua, p)
    return p


def _overpass(query: str, ua: str, tries: int = 6) -> dict:
    wait = 30.0
    last = None
    for attempt in range(tries):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        try:
            r = requests.post(url, data={"data": query}, headers={"User-Agent": ua}, timeout=400)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
            if r.status_code not in (429, 502, 503, 504):
                r.raise_for_status()
        except (requests.RequestException, ValueError) as e:
            last = repr(e)
        time.sleep(wait)
        wait = min(wait * 2, 600)
    raise RuntimeError(f"Overpass failed after {tries} tries: {last}")


def download_osm_military(cfg: AnalysisConfig, ua: str, regions=None, log=print) -> Path:
    """One Overpass query per (tag, padded region bbox); cached per pair; merged to a CSV."""
    cv = cfg["covariates"]
    pad = float(cv["osm_pad_deg"])
    if regions is None:   # every region any window or the model uses
        regions = list(dict.fromkeys([*cfg["model_regions"],
                                      *(r for w in cfg.windows.values() for r in w.regions)]))
    regions = [r for r in regions if cfg.regions.get(r)]
    raw_dir = cfg.covariates_dir / "osm" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for reg in regions:
        a, b, c, d = cfg.regions[reg]
        bbox = (max(a - pad, -90), max(c - pad, -180), min(b + pad, 90), min(d + pad, 180))
        for tag in cv["osm_military_tags"]:
            p = raw_dir / f"{tag}__{reg}.json"
            if not p.exists():
                q = (f'[out:json][timeout:300];nwr["military"="{tag}"]'
                     f'({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});out center tags;')
                log(f"Overpass: {tag} in {reg} {bbox}")
                try:
                    p.write_text(json.dumps(_overpass(q, ua)), encoding="utf-8")
                except RuntimeError as e:     # overloaded server: keep going, re-run later to fill in
                    log(f"Overpass: SKIPPED {tag} in {reg} ({e}); re-run to retry")
                    missing.append(f"{tag}__{reg}")
                    continue
                time.sleep(5)
            for e in json.loads(p.read_text(encoding="utf-8")).get("elements", []):
                lat = e.get("lat", (e.get("center") or {}).get("lat"))
                lon = e.get("lon", (e.get("center") or {}).get("lon"))
                if lat is None:
                    continue
                t = e.get("tags", {})
                rows.append({"osm_type": e["type"], "osm_id": e["id"], "military": tag,
                             "name": t.get("name:en") or t.get("name"), "lat": lat, "lon": lon})
    out = cfg.covariates_dir / "osm" / "military.csv"
    pd.DataFrame(rows).drop_duplicates(["osm_type", "osm_id"]).to_csv(out, index=False)
    (cfg.covariates_dir / "osm" / "MISSING.txt").write_text("\n".join(missing), encoding="utf-8")
    if missing:
        log(f"Overpass: {len(missing)} region/tag queries still missing (osm/MISSING.txt)")
    return out


# ---- loaders ---------------------------------------------------------------------------------
UCDP_COLS = ["id", "type_of_violence", "conflict_name", "side_a", "side_b", "where_prec",
             "latitude", "longitude", "country", "date_prec", "date_start", "date_end", "best"]


def load_ucdp(cfg: AnalysisConfig, min_where_prec: int | None = None) -> pd.DataFrame:
    """GED (through 2025) + Candidate (2026). Keeps events located at least to `min_where_prec`
    (1 exact … 4 adm2; 5+ are adm1/country-level and too coarse for 250 km radii)."""
    d = cfg.covariates_dir / "ucdp"
    min_where_prec = min_where_prec or cfg["covariates"]["ucdp_min_where_prec"]
    frames = []
    ged = sorted(d.glob("ged*-csv.zip"))
    if ged:
        z = zipfile.ZipFile(ged[-1])
        g = pd.read_csv(z.open(next(n for n in z.namelist() if n.endswith(".csv"))),
                        usecols=UCDP_COLS, low_memory=False)
        frames.append(g.assign(source="ged"))
    for url in UCDP_CANDIDATE_URLS:
        p = d / url.rsplit("/", 1)[1]
        if p.exists():
            frames.append(pd.read_csv(p, usecols=UCDP_COLS, low_memory=False).assign(source="candidate"))
    if not frames:
        raise FileNotFoundError(f"no UCDP files in {d}; run scripts/build_covariates.py")
    ev = pd.concat(frames, ignore_index=True).drop_duplicates("id", keep="last")
    ev["date_start"] = pd.to_datetime(ev["date_start"]).dt.normalize()
    ev["date_end"] = pd.to_datetime(ev["date_end"]).dt.normalize()
    # GED is final through its last year; use Candidate only after it
    ged_end = ev.loc[ev["source"] == "ged", "date_start"].max()
    if pd.notna(ged_end):
        ev = ev[(ev["source"] == "ged") | (ev["date_start"] > ged_end)]
    ev = ev[ev["where_prec"] <= min_where_prec]
    return ev.reset_index(drop=True)


def load_kp(cfg: AnalysisConfig) -> pd.DataFrame:
    """Daily: kp_max, kp_mean (of the eight 3-hourly values), ap (daily Ap)."""
    p = cfg.covariates_dir / "kp" / KP_URL.rsplit("/", 1)[1]
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        kp = np.array(f[7:15], float)
        kp = kp[kp >= 0]
        if kp.size == 0:
            continue
        rows.append((f"{f[0]}-{f[1]}-{f[2]}", kp.max(), kp.mean(), float(f[23])))
    df = pd.DataFrame(rows, columns=["date", "kp_max", "kp_mean", "ap"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_military(cfg: AnalysisConfig) -> pd.DataFrame:
    p = cfg.covariates_dir / "osm" / "military.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run scripts/build_covariates.py --osm")
    return pd.read_csv(p)


def load_countries(cfg: AnalysisConfig):
    import geopandas as gpd
    return gpd.read_file(cfg.naturalearth)


# ---- features --------------------------------------------------------------------------------
def conflict_features(cells, events: pd.DataFrame, start, end, radii_km=(100, 250, 500),
                      lag_days: int = 3) -> pd.DataFrame:
    """Per hex over [start - lag, end]: event counts and deaths within each radius, distance to the
    nearest event, and the state-based (type 1) share of nearby events."""
    cells = np.asarray(cells, dtype="int64")
    lat, lon = centroids(cells)
    s = pd.Timestamp(start) - pd.Timedelta(days=lag_days)
    e = pd.Timestamp(end)
    ev = events[(events["date_start"] >= s) & (events["date_start"] <= e)]
    out = pd.DataFrame({"hex": cells})
    out["dist_conflict_km"] = nearest_km(lat, lon, ev["latitude"].to_numpy(), ev["longitude"].to_numpy())
    if ev.empty:
        for r in radii_km:
            out[f"conflict_events_{r}"] = 0
            out[f"conflict_deaths_{r}"] = 0.0
            out[f"conflict_state_share_{r}"] = np.nan
        return out
    etree = cKDTree(unit_xyz(ev["latitude"], ev["longitude"]))
    htree = cKDTree(unit_xyz(lat, lon))
    deaths = ev["best"].fillna(0).to_numpy(float)
    state = (ev["type_of_violence"] == 1).to_numpy(float)
    for r in radii_km:
        lists = htree.query_ball_tree(etree, km_to_chord(r))
        cnt = np.array([len(x) for x in lists])
        out[f"conflict_events_{r}"] = cnt
        out[f"conflict_deaths_{r}"] = [deaths[x].sum() if x else 0.0 for x in lists]
        out[f"conflict_state_share_{r}"] = [state[x].mean() if x else np.nan for x in lists]
    return out


def conflict_hex_day(cells, events: pd.DataFrame, start, end, radius_km: float = 250,
                     lag_days: int = 3, max_date_prec: int = 3) -> pd.DataFrame:
    """Panel: per hex-day, events and deaths within radius over [day - lag, day]. Events dated
    only to a month or worse (date_prec > max_date_prec) are excluded from the daily panel."""
    cells = np.asarray(cells, dtype="int64")
    lat, lon = centroids(cells)
    s = pd.Timestamp(start) - pd.Timedelta(days=lag_days)
    ev = events[(events["date_start"] >= s) & (events["date_start"] <= pd.Timestamp(end))
                & (events["date_prec"] <= max_date_prec)]
    days = pd.date_range(start, end, freq="D")
    idx = pd.MultiIndex.from_product([cells, days], names=["hex", "date"])
    if ev.empty:
        return pd.DataFrame({"conflict_events": 0, "conflict_deaths": 0.0}, index=idx).reset_index()
    htree = cKDTree(unit_xyz(lat, lon))
    hits = htree.query_ball_point(unit_xyz(ev["latitude"], ev["longitude"]), km_to_chord(radius_km))
    pairs = [(cells[h], d, b) for hs, d, b in zip(hits, ev["date_start"], ev["best"].fillna(0)) for h in hs]
    if not pairs:
        return pd.DataFrame({"conflict_events": 0, "conflict_deaths": 0.0}, index=idx).reset_index()
    p = pd.DataFrame(pairs, columns=["hex", "date", "deaths"])
    daily = p.groupby(["hex", "date"]).agg(events=("deaths", "size"), deaths=("deaths", "sum"))
    full_days = pd.date_range(s, end, freq="D")
    daily = daily.reindex(pd.MultiIndex.from_product([cells, full_days], names=["hex", "date"]),
                          fill_value=0)
    roll = (daily.groupby(level="hex", group_keys=False)
                 .apply(lambda g: g.rolling(lag_days + 1, min_periods=1).sum()))
    roll = roll.reindex(idx, fill_value=0).rename(columns={"events": "conflict_events",
                                                             "deaths": "conflict_deaths"})
    return roll.reset_index()


def military_features(cells, mil: pd.DataFrame) -> pd.DataFrame:
    cells = np.asarray(cells, dtype="int64")
    lat, lon = centroids(cells)
    out = pd.DataFrame({"hex": cells})
    core = mil[mil["military"].isin(["airfield", "naval_base"])]
    out["dist_military_km"] = nearest_km(lat, lon, core["lat"].to_numpy(), core["lon"].to_numpy())
    base = mil[mil["military"] == "base"]
    if len(base):
        out["dist_base_km"] = nearest_km(lat, lon, base["lat"].to_numpy(), base["lon"].to_numpy())
    return out


def active_belligerents(belligerents: list[dict], start: date, end: date) -> list[str]:
    """ISO A3 codes of states at war at any point in [start, end]."""
    out = []
    for b in belligerents:
        b_end = b["end"] or date.max
        if b["start"] <= end and b_end >= start:
            out += list(b["iso3"])
    return sorted(set(out))


def belligerent_distance(cells, countries, iso3: list[str], step_deg: float = 0.05) -> np.ndarray:
    """Km from each hex centroid to the nearest territory of the given states (0 inside)."""
    import geopandas as gpd
    cells = np.asarray(cells, dtype="int64")
    lat, lon = centroids(cells)
    code = "ADM0_A3" if "ADM0_A3" in countries.columns else "adm0_a3"
    sel = countries[countries[code].isin(iso3)]
    if sel.empty:
        return np.full(len(cells), np.inf)
    geom = sel.geometry.union_all() if hasattr(sel.geometry, "union_all") else sel.geometry.unary_union
    edge = geom.boundary.segmentize(step_deg)
    parts = getattr(edge, "geoms", [edge])
    xy = np.vstack([np.asarray(p.coords) for p in parts])
    dist = nearest_km(lat, lon, xy[:, 1], xy[:, 0])
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=countries.crs)
    inside = pts.within(geom).to_numpy()
    dist[inside] = 0.0
    return dist


def hex_covariates(cfg: AnalysisConfig, cells, start: date, end: date, events=None, mil=None,
                   countries=None, acled=None) -> pd.DataFrame:
    """Cross-sectional covariates for one window: conflict (UCDP, plus ACLED by category when given),
    military, belligerent distance, latitude."""
    cv = cfg["covariates"]
    cells = np.asarray(cells, dtype="int64")
    events = load_ucdp(cfg) if events is None else events
    out = conflict_features(cells, events, start, end, cv["conflict_radii_km"], cv["conflict_lag_days"])
    try:
        mil = load_military(cfg) if mil is None else mil
        out = out.merge(military_features(cells, mil), on="hex")
    except FileNotFoundError:
        pass
    countries = load_countries(cfg) if countries is None else countries
    iso = active_belligerents(cv["belligerents"], start, end)
    out["dist_belligerent_km"] = belligerent_distance(cells, countries, iso)
    if acled is not None and len(acled):
        from .acled import acled_features
        out = out.merge(acled_features(cells, acled, start, end, cv["acled"]["radius_km"],
                                       cv["conflict_lag_days"]), on="hex")
    lat, _ = centroids(cells)
    out["abs_lat"] = np.abs(lat)
    for c in [c for c in out.columns if c.startswith(("conflict_events_", "conflict_deaths_"))]:
        out[f"log_{c}"] = np.log1p(out[c])
    for c in [c for c in out.columns if c.startswith("dist_") and c.endswith("_km")]:
        out[f"log_{c}"] = np.log1p(np.minimum(out[c], 5000.0))   # cap "none in range" at 5,000 km
    return out


def window_kp(kp: pd.DataFrame, start: date, end: date) -> dict:
    k = kp[(kp["date"] >= pd.Timestamp(start)) & (kp["date"] <= pd.Timestamp(end))]
    return {"kp_max": k["kp_max"].max(), "kp_mean": k["kp_mean"].mean(),
            "storm_days": int((k["kp_max"] >= 5).sum())}


def date_list(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]
