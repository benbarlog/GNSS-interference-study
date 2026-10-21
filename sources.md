# Data Sources

Every outside dataset this project uses, what we take from it, and how it may be used. Local cache
locations are relative to the repo root; `D:/gnss-data` is the external drive that holds the
large adsb.lol outputs.

## GNSS interference observations

### GPSJAM daily hex maps
- **Link:** https://gpsjam.org (FAQ: https://gpsjam.org/faq/)
- **Files:** `https://gpsjam.org/data/YYYY-MM-DD-h3_4.csv`, one per UTC day, from 2022-02-14 to the present
- **What it is:** For each H3 resolution-4 hex (~1,770 km²) and day, the number of aircraft that reported good vs. low navigation accuracy (`hex, count_good_aircraft, count_bad_aircraft`). Built by John Wiseman from ADS-B Exchange data. The exact "bad" threshold is not published.
- **Use here:** The primary hex × day interference panel (bad / total aircraft per hex). Also the reference for validating our own adsb.lol pipeline.
- **Cached in:** `data/raw/gpsjam/` (laptop sample) and `D:/gnss-data/gpsjam/` (daily, desktop)
- **Terms:** No API or license is published; these are the CSVs the website itself loads. We download each date once, cache it, space requests ≥ 1 s apart, and send a descriptive User-Agent. Credit GPSJAM / John Wiseman and ADS-B Exchange.

### ADS-B Exchange (upstream of GPSJAM)
- **Link:** https://www.adsbexchange.com
- **What it is:** A crowd-sourced network of ADS-B receivers. GPSJAM is derived from its data.
- **Use here:** Indirect only, through GPSJAM. We do not access it directly.
- **Terms:** Credit it alongside GPSJAM.

### adsb.lol historical ADS-B archive (`globe_history`)
- **Links:** https://github.com/adsblol/globe_history_2023, https://github.com/adsblol/globe_history_2024, https://github.com/adsblol/globe_history_2025, https://github.com/adsblol/globe_history_2026
- **What it is:** Daily dumps of readsb `globe_history`: per-aircraft trace files with position, altitude, speed, roll, and the intermittent ADS-B integrity fields (NIC, NACp, SIL, ADS-B version, emitter category). Published as split tar archives, one GitHub release per day, starting 2023-02-16.
- **Use here:**
  - Our own NIC < 7 interference pipeline at H3 res 5, which gives finer resolution than GPSJAM
  - Confounder filters (roll, ADS-B version 0, faulty avionics) and fleet/coverage controls
  - Spoofing indicators
  - Validation against GPSJAM
- **Format reference:** readsb trace JSON, https://github.com/wiedehopf/readsb/blob/dev/README-json.md#trace-jsons
- **Cached in:** `D:/gnss-data/adsblol/` (derived Tier 1/2 parquet; raw tars deleted after processing except a few retained days)
- **License:** Open Database License (ODbL 1.0). Figures and statistics need attribution. Any published derived *database* (e.g. Tier 1) must also be ODbL.

## Covariates

### UCDP Georeferenced Event Dataset (GED) and Candidate Events
- **Link:** https://ucdp.uu.se/downloads/
- **Files:**
  - GED v26.1 (1989–2025): https://ucdp.uu.se/downloads/ged/ged261-csv.zip
  - Candidate 2026, Jan–Jun: https://ucdp.uu.se/downloads/candidateged/GEDEvent_v26_01_26_06.csv
  - Candidate 2026, July: https://ucdp.uu.se/downloads/candidateged/GEDEvent_v26_0_7.csv
  - Candidate 2026, August: https://ucdp.uu.se/downloads/candidateged/GEDEvent_v26_0_8.csv
- **What it is:** Uppsala Conflict Data Program event data. Each lethal event of organized violence has a date, coordinates, location precision, type of violence (state-based / non-state / one-sided) and a best fatality estimate. The Candidate releases are preliminary monthly data for the current year.
- **Use here:**
  - Conflict event counts and deaths within 100/250/500 km of each hex
  - Distance to the nearest event
  - Share of state-based violence
  - Lagged daily event counts for the hex-day panel
- **Coverage note:** Currently ends 2026-08-31. Add newer Candidate releases as UCDP publishes them.
- **Cached in:** `data/raw/covariates/ucdp/`
- **Terms:** Free to use with citation. Cite per UCDP's guidance on the downloads page; the dataset paper is Sundberg & Melander (2013), "Introducing the UCDP Georeferenced Event Dataset," *Journal of Peace Research* 50(4).

### ACLED (Armed Conflict Location & Event Data), planned
- **Links:** https://acleddata.com, API docs: https://acleddata.com/api-documentation/getting-started, access FAQ: https://acleddata.com/myacled-faqs
- **What it is:** Event-level political violence data with event and sub-event types, e.g. "Air/drone strike", "Shelling/artillery/missile attack", "Armed clash", "Remote explosive/landmine/IED", and "Disrupted weapons use" (interceptions).
- **Use here (planned):**
  - Separating air/drone and missile activity from ground fighting
  - Interception events as a counter-drone signal
  - Additional conflict covariates alongside UCDP
- **Access:** Requires a myACLED account. The account tier depends on the email domain: a university email gives Research access (event-level data with a time lag); a generic email such as Gmail gives only aggregated Open access. API authentication is OAuth (email/password exchanged for a token); credentials go in a local `.env`, never in the repo.
- **Cached in:** `data/raw/covariates/acled/` (planned)
- **Terms:** Published work must state that ACLED data were used (see ACLED's Attribution Policy). Redistributing raw data or building substitute products is restricted.

### GFZ Potsdam geomagnetic Kp index
- **Link:** https://kp.gfz-potsdam.de/
- **File:** https://kp.gfz-potsdam.de/app/files/Kp_ap_Ap_SN_F107_since_1932.txt
- **What it is:** Eight 3-hourly Kp values per day plus the daily Ap index, sunspot number and F10.7 solar flux, 1932 to the present.
- **Use here:** Daily max/mean Kp as a natural-degradation control (geomagnetic storms disturb GNSS). Storm-day shading on time-series figures. The 2024-05 "Gannon" storm control window.
- **Cached in:** `data/raw/covariates/kp/`
- **License:** CC BY 4.0. Cite Matzka, J., Stolle, C., Yamazaki, Y., Bronkalla, O. & Morschhauser, A. (2021), "The geomagnetic Kp index and derived indices of geomagnetic activity," *Space Weather* 19, e2020SW002641.

### OpenStreetMap military installations (via Overpass API)
- **Links:** https://www.openstreetmap.org, Overpass endpoints: https://overpass-api.de/api/interpreter (main) and https://overpass.kumi.systems/api/interpreter (mirror)
- **What it is:** Features tagged `military=airfield` and `military=naval_base` (`military=base` was tried and dropped as too heavy to query), downloaded per study-region bounding box padded by 5°.
- **Use here:** Distance from each hex to the nearest military airfield or naval base, to test the hypothesis that jamming defends high-value military assets.
- **Cached in:** `data/raw/covariates/osm/` (raw Overpass JSON per tag × region, merged `military.csv`)
- **License:** ODbL 1.0. Attribute "© OpenStreetMap contributors." Mapping completeness varies by country, so a missing base is not evidence that no base exists.

### Natural Earth admin-0 countries (1:50m)
- **Link:** https://www.naturalearthdata.com/downloads/50m-cultural-vectors/50m-admin-0-countries-2/
- **What it is:** Country polygons with ISO codes (`ADM0_A3`).
- **Use here:** Basemap borders on every map. Distance from each hex to the territory of states at war (belligerent list and dates in `config/analysis.yaml`).
- **Cached in:** `data/raw/naturalearth/ne_50m_admin_0_countries.zip`
- **License:** Public domain.

## Event dates and case-study windows

- **What it is:** The case-study windows and peak days in `schedule/windows.yaml` (e.g. the Israel–Iran war of June 2025, the Lebanon escalation in Sep–Oct 2024, the 2026 US strikes on Iran) and the belligerent periods in `config/analysis.yaml`. Both were hand-compiled from public news reporting and checked against GPSJAM.
- **To do:** Add a citation (news source and date) for each window and belligerent period before the final write-up.

## Regulatory reference

- **14 CFR § 91.227** (ADS-B Out performance requirements): https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-91/subpart-C/section-91.227
- **Use here:** The basis for the NIC ≥ 7 requirement, and so for treating NIC < 7 as degraded.
