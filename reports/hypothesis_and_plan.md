# GNSS Interference Hotspots: Hypothesis & Plan

*Draft for the in-class proposal, 2026-10-15. Preliminary numbers come from `scripts/proposal_figures.py`,
using a GPSJAM sample (every pre-invasion day, one day per week from Feb 2022, and every day from 2026-07-16 to 2026-10-13).*

## The question in one line

**Is GPS interference random noise, or a geographic signature of active military conflict? And does it
track *fighting*, or merely *tension*?**

## Hypotheses

**H1. Interference is strongly spatially clustered.**
Degraded-GPS rates show significant positive global Moran's I, and LISA high-high clusters cover only a
small share of observed airspace.
*Preliminary:* only 869 of 43,899 observed hexes (2%) were degraded on at least half of the last 90 days,
and they fall into about six contiguous blobs.

**H2. Interference tracks active armed conflict, not geopolitical tension.**
Hexes closer to active armed conflict have higher interference persistence. Regions with high tension
but no active fighting (Taiwan Strait, Korean Peninsula) stay near baseline.
*Preliminary:* Baltic 35%, Black Sea 43%, Myanmar 26%, Persian Gulf 13%, Levant 12% mean persistence.
Taiwan Strait, Korea, and the continental-US control are all about 0%. The Taiwan idea becomes the
**falsification test**: it is where the "tension" explanation and the "conflict" explanation make
different predictions.

**H3. The Russia–Ukraine war produced a lasting but delayed increase on NATO's eastern flank.**
The share of degraded hexes in the Baltic and Black Sea regions rose after 24 Feb 2022. The Baltic rise
came in steps (mid-2023, then late 2024), not on invasion day.
*Preliminary:* Baltic mean persistence was 0.7% before the invasion vs. 35% now. Black Sea mean
persistence was 5.5% before vs. 43% now. Before the invasion, a thin degraded band already traced the
Russia–Ukraine border during the troop buildup.

**Counter-example worth presenting:** the Levant was the *most* degraded region before the invasion
(46%, Syria) and has since fallen and spiked with events. Interference turns on and off with operations.
It is not a permanent property of a place.

## Methods (maps to course requirements)

| Step | Method | Hypothesis |
|---|---|---|
| Rate per hex-day | bad / (good+bad), n ≥ 5 aircraft; Empirical Bayes smoothing for small n | all |
| Spatial weights | H3 ring-1 contiguity, row-standardized; islands reported; KNN sensitivity | H1 |
| Global clustering | Global Moran's I per region and per day (999 permutations) | H1 |
| Local clusters | LISA (Moran_Local_Rate) with FDR correction → persistence of HH membership | H1, H3 |
| Conflict link | Distance from each hex to UCDP GED conflict events; bivariate Moran's I / bivariate LISA | H2 |
| Time | Daily Moran's I and HH share, interrupted time series around 2022-02-24 | H3 |
| MAUP | Re-run at H3 res 3 and 2 (aggregated); res 5–6 from adsb.lol raw traces on 2–3 case-study days | robustness |

## Study design

- **Global layer (visual):** 90-day persistence raster at 0.1° (`data/processed/persistence_recent.tif`).
- **Case studies (analysis):** Baltic/Kaliningrad and Black Sea (H3); Levant or Myanmar as a
  non-Russia conflict contrast (H2); Taiwan Strait as the tension-without-conflict null (H2).
  The continental US is the control.
- **Higher resolution in case studies:** GPSJAM is fixed at H3 res 4 (~1,800 km²). Finer detail
  (res 5–6) means building our own NIC < 7 pipeline from adsb.lol raw traces for a few days, which is
  also the validation against GPSJAM.

## Data

- **GPSJAM** daily H3 CSVs (J. Wiseman / ADS-B Exchange), Feb 14 2022 to present. No auth; cached locally.
- **UCDP GED v25.1** georeferenced conflict events (open; confirmed downloadable, 29 MB, through 2024),
  plus UCDP Candidate events for 2025–26.
- **adsb.lol** raw ADS-B archive (ODbL) for the res 5–6 and validation days.
- **Natural Earth** boundaries.

## Limitations to state up front

- **No aircraft means no measurement.** Ukraine's own airspace is closed (black on the maps), so we
  measure interference *around* war zones, not inside them. Missing hexes are never treated as zero.
- **Pre-invasion baseline is short:** GPSJAM starts 2022-02-14, giving only 10 pre-war days.
- **Traffic bias:** busier airspace yields more reliable rates. Handled with an n threshold and
  Empirical Bayes smoothing.
- **Descriptive, not causal:** clusters show where receivers are degraded, not where the jammers are.
- **Black-box upstream:** GPSJAM's "bad" definition is not fully documented, which motivates the
  adsb.lol validation.

## Figures (in `figures/`)

1. `global_persistence.png`: headline global raster
2. `europe_before_after.png`: pre-invasion vs. last 90 days around Russia/Ukraine
3. `region_timeseries.png`: 2022–2026 degraded-hex share by region
4. `case_study_regions.png`: six candidate case studies at hex resolution
