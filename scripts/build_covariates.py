"""Download and sanity-check the covariate sources (small; cached under config/analysis.yaml
`covariates_dir`, which OneDrive syncs between laptop and desktop).

    python scripts/build_covariates.py            # UCDP + Kp + OSM military (skips cached files)
    python scripts/build_covariates.py --check    # only report what is cached and its coverage
    python scripts/build_covariates.py --acled    # also ACLED (needs ACLED_EMAIL/ACLED_PASSWORD in .env)

UCDP: GED v26.1 (1989–2025) + Candidate 2026 (Jan–Aug at time of writing). New Candidate months:
add their URLs to gnss.analysis.covariates.UCDP_CANDIDATE_URLS.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from gnss.adsblol.config import user_agent
from gnss.analysis import covariates as cv
from gnss.analysis.config import load_analysis_config


def check(cfg) -> int:
    problems = 0
    try:
        ev = cv.load_ucdp(cfg)
        print(f"UCDP: {len(ev):,} events (where_prec <= {cfg['covariates']['ucdp_min_where_prec']}), "
              f"{ev['date_start'].min():%Y-%m-%d} → {ev['date_start'].max():%Y-%m-%d}; "
              f"by source: {ev['source'].value_counts().to_dict()}")
        for w in cfg.windows.values():
            n = ((ev["date_start"] >= pd.Timestamp(w.start)) & (ev["date_start"] <= pd.Timestamp(w.end))).sum()
            flag = "" if n else "   <-- no events in window"
            if pd.Timestamp(w.end) > ev["date_start"].max():
                flag += f"   <-- window ends after UCDP coverage ({ev['date_start'].max():%Y-%m-%d})"
            print(f"  window {w.id:>2} {w.start} → {w.end}: {n:,} events{flag}")
    except FileNotFoundError as e:
        print(f"UCDP: {e}")
        problems += 1
    try:
        kp = cv.load_kp(cfg)
        print(f"Kp: {kp['date'].min():%Y-%m-%d} → {kp['date'].max():%Y-%m-%d}; "
              f"storm days (Kp ≥ 5) since 2023: {(kp[kp['date'] >= '2023-01-01']['kp_max'] >= 5).sum()}")
    except FileNotFoundError as e:
        print(f"Kp: {e}")
        problems += 1
    try:
        mil = cv.load_military(cfg)
        print(f"OSM military: {len(mil):,} sites; {mil['military'].value_counts().to_dict()}")
        miss = cfg.covariates_dir / "osm" / "MISSING.txt"
        if miss.exists() and miss.read_text().strip():
            print(f"OSM military: still missing {miss.read_text().split()}; re-run without --check")
            problems += 1
    except FileNotFoundError as e:
        print(f"OSM military: {e}")
        problems += 1
    try:
        from gnss.analysis.acled import load_acled
        ac = load_acled(cfg)
        print(f"ACLED: {len(ac):,} events, {ac['date_start'].min():%Y-%m-%d} → {ac['date_start'].max():%Y-%m-%d}; "
              f"by category: {ac['category'].value_counts().to_dict()}")
        for w in cfg.windows.values():
            covered = ac["date_start"].max() >= pd.Timestamp(w.end)
            print(f"  window {w.id:>2}: {'ACLED drivers' if covered else 'UCDP drivers (ACLED ends before window end)'}")
    except FileNotFoundError:
        print("ACLED: not downloaded (optional; models fall back to UCDP). See .env.example")
    countries = cv.load_countries(cfg)
    code = "ADM0_A3" if "ADM0_A3" in countries.columns else "adm0_a3"
    have = set(countries[code])
    for b in cfg["covariates"]["belligerents"]:
        for iso in b["iso3"]:
            if iso not in have:
                print(f"Natural Earth: belligerent code {iso} not found in {code}")
                problems += 1
    print("OK" if not problems else f"{problems} problem(s)")
    return problems


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only, download nothing")
    ap.add_argument("--no-osm", action="store_true")
    ap.add_argument("--force-ucdp", action="store_true", help="re-download UCDP files")
    ap.add_argument("--acled", action="store_true", help="download ACLED via the myACLED API")
    a = ap.parse_args()
    cfg = load_analysis_config()
    if not a.check:
        ua = user_agent(cfg.pipeline)
        print("UCDP:", [p.name for p in cv.download_ucdp(cfg, ua, force=a.force_ucdp)])
        print("Kp:", cv.download_kp(cfg, ua).name)
        if a.acled:
            from gnss.analysis.acled import download_acled
            print(f"ACLED: {download_acled(cfg, ua):,} new events")
        if not a.no_osm:
            print("OSM:", cv.download_osm_military(cfg, ua))
    sys.exit(1 if check(cfg) else 0)


if __name__ == "__main__":
    main()
