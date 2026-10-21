"""ACLED (Armed Conflict Location & Event Data) events: download via the myACLED API, normalize to the
UCDP-like columns the covariate builders use, and split by *how* the violence happened.

Why ACLED as well as UCDP: ACLED's `sub_event_type` separates stand-off violence (air/drone strikes,
shelling/artillery/missile attacks), ground fighting (armed clashes, territory changes), explosives
(IEDs, landmines) and "Disrupted weapons use" (interceptions: air defence at work). That is the
air-vs-ground distinction the hypothesis needs: counter-drone/anti-missile jamming should track
stand-off weapons and interceptions, not infantry clashes.

Access (checked 2026-10-20): myACLED account; OAuth password grant at https://acleddata.com/oauth/token
(client_id=acled), Bearer token on https://acleddata.com/api/acled/read. The account *tier* depends on
the email domain: a university email gets "Research" (event-level data with a time lag); gmail gets
"Open" (aggregated only). Credentials go in the repo's .env (gitignored), never in code:

    ACLED_EMAIL=you@university.edu
    ACLED_PASSWORD=...

Manual alternative: put CSV exports from the ACLED Data Export Tool in <covariates_dir>/acled/manual/.
Terms: attribute ACLED in every figure/report that uses it; do not redistribute the raw events.
"""

from __future__ import annotations

import io
import os
import time
from datetime import date
from urllib.parse import quote, urlencode
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from gnss.adsblol.config import REPO_ROOT

TOKEN_URL = "https://acleddata.com/oauth/token"
READ_URL = "https://acleddata.com/api/acled/read"
PAGE_LIMIT = 5000
FIELDS = ["event_id_cnty", "event_date", "year", "time_precision", "disorder_type", "event_type",
          "sub_event_type", "actor1", "actor2", "interaction", "country", "admin1", "location",
          "latitude", "longitude", "geo_precision", "fatalities"]
# Protests and riots are irrelevant here and dominate the counts in some countries (e.g. the US)
EVENT_TYPES = ["Battles", "Explosions/Remote violence", "Violence against civilians",
               "Strategic developments"]

CATEGORIES = {
    "air": ["Air/drone strike"],
    "shelling": ["Shelling/artillery/missile attack"],
    "ground": ["Armed clash", "Government regains territory", "Non-state actor overtakes territory",
               "Attack"],
    "explosive": ["Remote explosive/landmine/IED", "Suicide bomb", "Grenade"],
    "intercept": ["Disrupted weapons use"],
}
STANDOFF = ("air", "shelling")
KINETIC = ("air", "shelling", "ground", "explosive")

ACLED_ATTRIBUTION = "Armed Conflict Location & Event Data (ACLED), acleddata.com"


# ---- credentials & client --------------------------------------------------------------------
def read_env(path: Path | None = None) -> dict:
    """os.environ overlaid on a simple KEY=VALUE .env file (no python-dotenv dependency)."""
    out = {}
    p = path or REPO_ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    out.update({k: v for k, v in os.environ.items() if k.startswith("ACLED_")})
    return out


class AcledClient:
    def __init__(self, email: str, password: str, ua: str, min_interval_s: float = 1.0):
        self.email, self.password, self.ua = email, password, ua
        self.min_interval_s = min_interval_s
        self.session = requests.Session()
        self.session.headers["User-Agent"] = ua
        self._token = None
        self._refresh = None
        self._expires = 0.0
        self._last = 0.0

    def _auth(self, data: dict) -> None:
        r = self.session.post(TOKEN_URL, data=data, timeout=60)
        if r.status_code in (400, 401):
            raise PermissionError("ACLED rejected the credentials in .env (ACLED_EMAIL / ACLED_PASSWORD)")
        r.raise_for_status()
        j = r.json()
        self._token, self._refresh = j["access_token"], j.get("refresh_token")
        self._expires = time.time() + float(j.get("expires_in", 86400)) - 300

    def token(self) -> str:
        if self._token and time.time() < self._expires:
            return self._token
        if self._refresh:
            try:
                self._auth({"refresh_token": self._refresh, "grant_type": "refresh_token",
                            "client_id": "acled"})
                return self._token
            except (requests.RequestException, PermissionError):
                pass
        self._auth({"username": self.email, "password": self.password, "grant_type": "password",
                    "client_id": "acled", "scope": "authenticated"})
        return self._token

    @staticmethod
    def query_string(params: dict, or_params: dict | None = None) -> str:
        """Plain params URL-encoded; OR params joined ACLED-style (`k=a:OR:k=b`) with the `:OR:k=`
        separators left literal, which requests' own encoding would break."""
        qs = urlencode(params)
        for k, values in (or_params or {}).items():
            qs += f"&{k}=" + f":OR:{k}=".join(quote(str(v), safe="") for v in values)
        return qs

    def read(self, params: dict, or_params: dict | None = None, tries: int = 5) -> pd.DataFrame:
        """All pages of one query as a DataFrame (CSV format)."""
        frames, page = [], 1
        while True:
            q = self.query_string({**params, "_format": "csv", "limit": PAGE_LIMIT, "page": page},
                                  or_params)
            wait = 10.0
            for attempt in range(tries):
                gap = time.time() - self._last
                if gap < self.min_interval_s:
                    time.sleep(self.min_interval_s - gap)
                self._last = time.time()
                r = self.session.get(f"{READ_URL}?{q}", timeout=180,
                                     headers={"Authorization": f"Bearer {self.token()}"})
                if r.status_code == 200:
                    break
                if r.status_code in (401, 403) and attempt == 0:
                    self._expires = 0       # token expired: re-auth once
                    continue
                if r.status_code not in (429, 500, 502, 503, 504):
                    r.raise_for_status()
                time.sleep(float(r.headers.get("Retry-After", wait)))
                wait = min(wait * 2, 600)
            else:
                raise RuntimeError(f"ACLED read failed: HTTP {r.status_code}")
            text = r.text.lstrip("\ufeff").strip()     # an empty result is just a BOM
            df = pd.read_csv(io.StringIO(text)) if text else pd.DataFrame()
            frames.append(df)
            if len(df) < PAGE_LIMIT:
                break
            page += 1
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---- download & load -------------------------------------------------------------------------
def download_acled(cfg, ua: str, start: date | None = None, end: date | None = None, log=print) -> int:
    """Cache one CSV per (country, year) under <covariates_dir>/acled/raw/. Years before last year are
    final once cached; the current and previous year are refreshed if older than 7 days."""
    env = read_env()
    if not env.get("ACLED_EMAIL") or not env.get("ACLED_PASSWORD"):
        raise PermissionError("set ACLED_EMAIL and ACLED_PASSWORD in the repo's .env (see .env.example)")
    ac = cfg["covariates"]["acled"]
    start = start or date.fromisoformat(str(ac["start"]))
    end = end or date.today()
    raw = cfg.covariates_dir / "acled" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    client = AcledClient(env["ACLED_EMAIL"], env["ACLED_PASSWORD"], ua)
    n = 0
    for country in ac["countries"]:
        for year in range(start.year, end.year + 1):
            p = raw / f"{country.replace(' ', '_')}_{year}.csv"
            fresh = p.exists() and (year < end.year - 1 or time.time() - p.stat().st_mtime < 7 * 86400)
            if fresh:
                continue
            lo, hi = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
            params = {"country": country, "event_date": f"{lo.isoformat()}|{hi.isoformat()}",
                      "event_date_where": "BETWEEN", "fields": "|".join(FIELDS)}
            # One query per event type: the API's `:OR:` filter returned only the last type (2026-10-20)
            df = pd.concat([client.read({**params, "event_type": et}) for et in EVENT_TYPES],
                           ignore_index=True)
            tmp = p.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(p)
            log(f"ACLED: {country} {year}: {len(df):,} events")
            n += len(df)
    return n


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame({
        "id": df["event_id_cnty"].astype(str),
        "date_start": pd.to_datetime(df["event_date"]).dt.normalize(),
        "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
        "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
        "best": pd.to_numeric(df["fatalities"], errors="coerce").fillna(0),
        "country": df["country"],
        "event_type": df["event_type"],
        "sub_event_type": df["sub_event_type"],
        # ACLED precision codes → the UCDP-style scales the builders filter on
        "date_prec": df["time_precision"].map({1: 1, 2: 3, 3: 4}).fillna(5).astype(int),
        "where_prec": df["geo_precision"].map({1: 1, 2: 3, 3: 5}).fillna(6).astype(int),
    })
    cat = pd.Series("other", index=out.index)
    for name, subs in CATEGORIES.items():
        cat[out["sub_event_type"].isin(subs)] = name
    out["category"] = cat
    out["type_of_violence"] = np.nan     # UCDP-only field; kept so shared builders work
    return out.dropna(subset=["latitude", "longitude"])


def load_acled(cfg, min_where_prec: int | None = None) -> pd.DataFrame:
    d = cfg.covariates_dir / "acled"
    files = sorted((d / "raw").glob("*.csv")) + sorted((d / "manual").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no ACLED files in {d}; run scripts/build_covariates.py --acled")
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if len(df):
            frames.append(_norm(df))
    ev = pd.concat(frames, ignore_index=True).drop_duplicates("id", keep="last")
    min_where_prec = min_where_prec or cfg["covariates"]["ucdp_min_where_prec"]
    return ev[ev["where_prec"] <= min_where_prec].reset_index(drop=True)


# ---- features --------------------------------------------------------------------------------
def acled_features(cells, events: pd.DataFrame, start, end, radius_km: float = 250,
                   lag_days: int = 3) -> pd.DataFrame:
    """Per hex over [start - lag, end]: events by category within radius, deaths, and the stand-off
    share = (air + shelling) / all kinetic events (0 where there are none)."""
    from .covariates import conflict_features
    r = int(radius_km)
    out = None
    for cat in CATEGORIES:
        f = conflict_features(cells, events[events["category"] == cat], start, end, (r,), lag_days)
        f = f[["hex", f"conflict_events_{r}"]].rename(columns={f"conflict_events_{r}": f"acled_{cat}_{r}"})
        out = f if out is None else out.merge(f, on="hex")
    tot = conflict_features(cells, events[events["category"].isin(KINETIC)], start, end, (r,), lag_days)
    out = out.merge(tot[["hex", f"conflict_events_{r}", f"conflict_deaths_{r}"]].rename(columns={
        f"conflict_events_{r}": f"acled_kinetic_{r}", f"conflict_deaths_{r}": f"acled_deaths_{r}"}), on="hex")
    standoff = sum(out[f"acled_{c}_{r}"] for c in STANDOFF)
    out[f"acled_standoff_share_{r}"] = np.where(out[f"acled_kinetic_{r}"] > 0,
                                                standoff / out[f"acled_kinetic_{r}"].clip(lower=1), 0.0)
    for c in [c for c in out.columns if c.startswith("acled_") and "share" not in c]:
        out[f"log_{c}"] = np.log1p(out[c])
    return out


def acled_hex_day(cells, events: pd.DataFrame, start, end, radius_km: float = 250,
                  lag_days: int = 3) -> pd.DataFrame:
    """Panel counts per hex-day: stand-off (air + shelling), ground, intercept events within radius."""
    from .covariates import conflict_hex_day
    groups = {"standoff": list(STANDOFF), "ground": ["ground"], "intercept": ["intercept"]}
    out = None
    for name, cats in groups.items():
        p = conflict_hex_day(cells, events[events["category"].isin(cats)], start, end, radius_km, lag_days)
        p = p.rename(columns={"conflict_events": f"{name}_events"}).drop(columns="conflict_deaths")
        out = p if out is None else out.merge(p, on=["hex", "date"])
    return out
