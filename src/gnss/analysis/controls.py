"""adsb.lol-only nuisance controls per hex over a date range, plus the filter ablation.

Each control is a share of time (or points) in the hex, so it describes *who* and *how* the hex is
observed rather than whether it is jammed:

- share_high_alt:  barometric-altitude seconds in rows whose mean altitude ≥ 25,000 ft
- share_light_ac:  ADS-B seconds from emitter categories A1 (light), A2 (small), A7 (rotorcraft), B*
                   (gliders, balloons, UAVs) — GA/training fleets with weaker GNSS receivers and more
                   manoeuvring
- share_v0:        ADS-B seconds from ADS-B version 0 transponders (NIC encoded differently)
- share_high_roll: ADS-B seconds with |roll| > 30° (antenna masking)
- share_mlat:      MLAT points / all points: a receiver-coverage proxy (MLAT needs ≥ 4 receivers)
- share_faulty:    aircraft-days flagged as faulty avionics (low NIC where other aircraft were fine)
- log_n_aircraft:  log distinct aircraft (traffic / sample size)
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from gnss.adsblol.tier2 import _bad_expr, _known_expr

from .data import filter_where

LIGHT = "('A1', 'A2', 'A7', 'B1', 'B2', 'B4', 'B6')"


def hex_controls_sql(start: date, end: date, res: int = 4) -> str:
    hx = "t.hex5" if res == 5 else f"l.hex{res}"
    join = "" if res == 5 else "JOIN hex_lookup l ON l.hex5 = t.hex5"
    return f"""
    WITH t AS (
        SELECT t.*, {hx} AS hex,
               t.alt_sum_s / NULLIF(t.s_alt, 0) AS alt_mean,
               EXISTS (SELECT 1 FROM faulty_aircraft b WHERE b.date = t.date AND b.icao = t.icao) AS ab
        FROM tier1 t {join}
        WHERE t.date::DATE BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'
    )
    SELECT hex,
           COUNT(DISTINCT icao) AS n_aircraft_distinct,
           SUM(CASE WHEN alt_mean >= 25000 THEN s_alt ELSE 0 END) / NULLIF(SUM(s_alt), 0) AS share_high_alt,
           SUM(CASE WHEN category IN {LIGHT} THEN s_adsb ELSE 0 END) / NULLIF(SUM(s_adsb), 0) AS share_light_ac,
           SUM(CASE WHEN version = 0 THEN s_adsb ELSE 0 END) / NULLIF(SUM(s_adsb), 0) AS share_v0,
           SUM(s_high_roll) / NULLIF(SUM(s_adsb), 0) AS share_high_roll,
           SUM(n_pts_mlat) / NULLIF(SUM(n_pts), 0) AS share_mlat,
           COUNT(DISTINCT CASE WHEN ab THEN icao || date::VARCHAR END)
             / NULLIF(COUNT(DISTINCT icao || date::VARCHAR), 0) AS share_faulty,
           SUM(n_jumps) AS n_jumps,
           SUM(gap_abs_sum) / NULLIF(SUM(gap_n), 0) AS mean_alt_gap_ft
    FROM t GROUP BY hex"""


def hex_controls(con, start: date, end: date, res: int = 4) -> pd.DataFrame:
    import numpy as np
    df = con.sql(hex_controls_sql(start, end, res)).df()
    df["hex"] = df["hex"].astype("int64")
    df["log_n_aircraft"] = np.log(df["n_aircraft_distinct"].clip(lower=1))
    return df


def ablation(con, start: date, end: date, nic_lt: int = 7, where_region: str = "TRUE") -> pd.DataFrame:
    """Known-NIC seconds and bad seconds remaining after each filter, cumulatively and alone."""
    steps = {
        "none": {},
        "roll": {"roll": True},
        "v0": {"v0": True},
        "faulty": {"faulty": True},
        "all": {"roll": True, "v0": True, "faulty": True},
    }
    rows = []
    for name, f in steps.items():
        q = f"""
        SELECT SUM({_known_expr()}) AS s_known, SUM({_bad_expr(nic_lt)}) AS s_bad,
               COUNT(DISTINCT t.icao) AS n_aircraft
        FROM tier1 t JOIN hex_lookup l ON l.hex5 = t.hex5
        WHERE t.date::DATE BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'
          AND {where_region} AND {filter_where(**f)}"""
        r = con.sql(q).df().iloc[0].to_dict()
        rows.append({"filter": name, **r})
    out = pd.DataFrame(rows)
    out["bad_share"] = out["s_bad"] / out["s_known"]
    base = out.loc[out["filter"] == "none"].iloc[0]
    out["known_kept"] = out["s_known"] / base["s_known"]
    out["bad_kept"] = out["s_bad"] / base["s_bad"]
    return out


def bbox_where(bbox) -> str:
    """SQL predicate on hex_lookup centroid (alias l) for a region bbox."""
    if bbox is None:
        return "TRUE"
    a, b, c, d = bbox
    return f"(l.lat BETWEEN {a} AND {b} AND l.lon BETWEEN {c} AND {d})"
