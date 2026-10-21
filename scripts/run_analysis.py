"""First-pass analysis: figures + tables for each case window (schedule/windows.yaml) and P2.

    python scripts/run_analysis.py                                  # all steps, all windows, adsb.lol
    python scripts/run_analysis.py --source gpsjam --windows P2     # laptop: GPSJAM sample only
    python scripts/run_analysis.py --steps validate ablation --windows H
    python scripts/run_analysis.py --steps esda --windows H --regions levant iran

Steps (in order; each writes to <outputs_dir>/<source>/<window>/ and <figures_dir>/<source>/<window>/):
  validate    adsb.lol vs GPSJAM per hex-day: agreement table, scatter, peak-day disagreement map
  ablation    how much degraded time each confounder filter removes (adsb.lol only)
  eda         peak-day raw vs EB rate maps per region; daily share-degraded series
  esda        daily global Moran's I + LISA per region; LISA maps, Moran scatter, persistence maps
  resolution  peak-day LISA at H3 res 2/3/4 (+5 for adsb.lol)
  model       hex cross-section: OLS → spatial lag / error (+ GWR with --gwr), coefficient plot
  panel       hex-day fixed-effects model: conflict events (lagged) + Kp + traffic
  events      event study around each window's peak and phase starts
"""

from __future__ import annotations

import argparse
import sys
import traceback

import pandas as pd

from gnss.analysis import viz
from gnss.analysis.config import load_analysis_config
from gnss.analysis.pipeline import STEPS, Run


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles default to cp1252
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["adsblol", "gpsjam"], default=None)
    ap.add_argument("--windows", nargs="+", default=None, help="window ids (default: all + P2)")
    ap.add_argument("--regions", nargs="+", default=None, help="override each window's regions")
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    ap.add_argument("--gwr", action="store_true", help="also fit GWR (slow)")
    ap.add_argument("--data-root", default=None, help="override adsblol.yaml data_root")
    a = ap.parse_args()
    cfg = load_analysis_config(data_root=a.data_root)
    source = a.source or cfg["source"]
    viz.style()
    run = Run(cfg, source, regions=a.regions, gwr=a.gwr)
    wins = [cfg.windows[x] for x in (a.windows or list(cfg.windows))]
    fn = {"validate": run.validate, "ablation": run.ablation, "eda": run.eda, "esda": run.esda,
          "resolution": run.resolution, "model": run.model, "panel": run.panel_fe, "events": run.events_study}
    failures = []
    for w in wins:
        run.log(f"=== window {w.id}: {w.name} ({w.start} → {w.end}) · source {source}")
        for step in [s for s in STEPS if s in a.steps]:
            try:
                fn[step](w)
            except Exception as e:
                failures.append((w.id, step, repr(e)))
                run.log(f"{step} {w.id}: FAILED {e!r}")
                traceback.print_exc()
        run._cov = {k: v for k, v in run._cov.items() if not isinstance(k, tuple)}   # free panels
    if run.log_rows:
        p = cfg.outputs_dir / source / "weights_sensitivity.csv"
        pd.DataFrame(run.log_rows).to_csv(p, index=False)
    if failures:
        print("\nFAILED STEPS:")
        for f in failures:
            print("  ", *f)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
