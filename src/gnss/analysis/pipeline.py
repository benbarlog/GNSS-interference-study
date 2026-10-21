"""Analysis steps per case window, shared by `scripts/run_analysis.py` (CLI) and the notebook.

Each step writes tables to <outputs_dir>/<source>/<window>/ and figures to
<figures_dir>/<source>/<window>/. `Run(..., show=True)` also displays each figure inline (notebook).
`saved_figures` / `saved_table` read finished outputs back without recomputing (e.g. for a talk).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import pandas as pd

from . import covariates as cv
from . import model as md
from . import spatial as sp
from . import validation as vl
from . import viz
from .config import AnalysisConfig, Window
from .data import load_gpsjam, load_panel, observed, select_region, select_regions

STEPS = ["validate", "ablation", "eda", "esda", "resolution", "model", "panel", "events"]
CONTROL_REGIONS = ["taiwan_strait", "korea", "conus_control"]


class Run:
    def __init__(self, cfg: AnalysisConfig, source: str, regions: list[str] | None = None,
                 gwr: bool = False, show: bool = False):
        self.cfg, self.source = cfg, source
        self.regions_override, self.gwr, self.show = regions, gwr, show
        self.countries = cv.load_countries(cfg)
        self.src_line = viz.source_line(source)
        self._con = None
        self._cov = {}
        self.log_rows = []

    # -- shared --------------------------------------------------------------------------------
    @property
    def con(self):
        if self._con is None and self.source == "adsblol":
            from gnss.analysis.data import adsblol_connect
            self._con = adsblol_connect(self.cfg)
        return self._con

    def out(self, w: Window, name: str) -> Path:
        p = self.cfg.outputs_dir / self.source / w.id / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def fig(self, w: Window, name: str) -> Path:
        return self.cfg.figures_dir / self.source / w.id / name

    def save(self, fig, path: Path) -> Path:
        """Write the figure; in a notebook (show=True) also display the saved PNG inline."""
        viz.save(fig, path)
        if self.show:
            from IPython.display import Image, display
            display(Image(filename=str(path)))
        return path

    def log(self, msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def panel(self, w: Window, res: int | None = None, filters=None) -> pd.DataFrame:
        key = (w.id, res, json.dumps(filters, sort_keys=True))
        if key not in self._cov:
            self._cov[key] = load_panel(self.cfg, self.source, w.start, w.end, res=res, filters=filters,
                                        con=self.con)
        return self._cov[key]

    def regions(self, w: Window) -> list[str]:
        regs = self.regions_override or w.regions
        return [r for r in regs if r in self.cfg.regions and self.cfg.regions[r] is not None]

    def peak(self, w: Window, panel: pd.DataFrame, bbox) -> pd.Timestamp | None:
        days = set(panel["date"].unique())
        if w.peak and pd.Timestamp(w.peak) in days:
            return pd.Timestamp(w.peak)
        reg = observed(select_region(panel, bbox), self.cfg["rate"]["min_n"])
        if reg.empty:
            return None
        s = reg.groupby("date").apply(lambda g: g["bad"].sum() / g["n"].sum(), include_groups=False)
        return s.idxmax()

    def events(self, w: Window) -> list[tuple]:
        ev = []
        if w.peak:
            ev.append((w.peak, f"peak {w.peak:%b %d}"))
        for ph in w.phases:
            ev.append((ph["start"], ph["name"].replace("_", " ")))
        return ev

    def kp(self):
        if "kp" not in self._cov:
            try:
                self._cov["kp"] = cv.load_kp(self.cfg)
            except FileNotFoundError:
                self._cov["kp"] = None
        return self._cov["kp"]

    def ucdp(self):
        if "ucdp" not in self._cov:
            self._cov["ucdp"] = cv.load_ucdp(self.cfg)
        return self._cov["ucdp"]

    def acled(self):
        """ACLED events, or None when no credentials/exports have been downloaded."""
        if "acled" not in self._cov:
            from gnss.analysis.acled import load_acled
            try:
                self._cov["acled"] = load_acled(self.cfg)
            except FileNotFoundError:
                self._cov["acled"] = None
        return self._cov["acled"]

    def acled_covers(self, w: Window) -> bool:
        a = self.acled()
        return a is not None and len(a) > 0 and a["date_start"].max() >= pd.Timestamp(w.end)

    # -- steps ---------------------------------------------------------------------------------
    def validate(self, w: Window) -> None:
        if self.source != "adsblol":
            return self.log("validate: adsb.lol only, skipped")
        ours = self.panel(w)
        gj = load_gpsjam(self.cfg, w.start, w.end)
        if gj.empty:
            return self.log(f"validate {w.id}: no GPSJAM daily files for the window")
        r = self.cfg["rate"]
        j = vl.join_sources(ours, gj, r["min_n"])
        if j.empty:
            return self.log(f"validate {w.id}: no matched hex-days")
        by_day = vl.agreement_by_day(j, r["bad_rate"])
        by_day.to_csv(self.out(w, "validation_by_day.csv"), index=False)
        vl.confusion(j, r["bad_rate"]).to_csv(self.out(w, "validation_confusion.csv"))
        self.save(viz.validation_scatter(j, f"adsb.lol vs GPSJAM · window {w.id}"),
                 self.fig(w, "validation_scatter.png"))
        bbox = self.cfg.regions[self.regions(w)[0]]
        pk = self.peak(w, ours, bbox)
        jd = select_region(j[j["date"] == pk], bbox)
        if len(jd):
            self.save(viz.disagreement_map(jd, self.countries, bbox,
                                          f"Where the sources disagree · {pk:%Y-%m-%d}"),
                     self.fig(w, "validation_disagreement.png"))
        self.log(f"validate {w.id}: {len(j):,} hex-days, median daily Spearman "
                 f"{by_day['spearman'].median():.2f}, agreement {by_day['agree_pct'].median():.1f}%")

    def ablation(self, w: Window) -> None:
        if self.source != "adsblol":
            return self.log("ablation: adsb.lol only, skipped")
        from gnss.analysis.controls import ablation, bbox_where
        rows = []
        for reg in self.regions(w):
            ab = ablation(self.con, w.start, w.end, self.cfg["rate"]["nic_lt"],
                          bbox_where(self.cfg.regions[reg]))
            ab.insert(0, "region", reg)
            rows.append(ab)
            self.save(viz.ablation_bars(ab, f"Confounder filters · {reg.replace('_', ' ').title()} · {w.id}"),
                     self.fig(w, f"ablation_{reg}.png"))
        pd.concat(rows).to_csv(self.out(w, "ablation.csv"), index=False)
        self.log(f"ablation {w.id}: done for {len(rows)} regions")

    def eda(self, w: Window) -> None:
        p = self.panel(w)
        r = self.cfg["rate"]
        series = []
        for reg in self.regions(w):
            bbox = self.cfg.regions[reg]
            rp = observed(select_region(p, bbox), r["min_n"])
            if rp.empty:
                continue
            s = rp.groupby("date").agg(n_hex=("hex", "size"), bad=("bad", "sum"), n=("n", "sum"),
                                       share_degraded=("rate", lambda x: (x > r["bad_rate"]).mean()))
            s["pooled_rate"] = s["bad"] / s["n"]
            series.append(s.reset_index().assign(region=reg))
            pk = self.peak(w, p, bbox)
            day = rp[rp["date"] == pk].copy()
            if len(day) >= 5:
                day["eb_rate"] = sp.eb_rate(day["bad"], day["n"])
                self.save(viz.raw_vs_eb(day, self.countries, bbox,
                                       f"{reg.replace('_', ' ').title()} · {pk:%Y-%m-%d}", self.src_line),
                         self.fig(w, f"raw_vs_eb_{reg}.png"))
        if series:
            ser = pd.concat(series, ignore_index=True)
            ser.to_csv(self.out(w, "region_daily.csv"), index=False)
            if ser["date"].nunique() > 1:
                self.save(viz.region_share_timeseries(ser, ser["region"].unique().tolist(),
                                                     f"Degraded share by region · {w.name}", self.src_line,
                                                     self.events(w)),
                         self.fig(w, "region_share_timeseries.png"))
        # traffic distribution: the small-denominator problem in one number set
        q = p["n"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        self.log(f"eda {w.id}: {len(p):,} hex-days; aircraft per hex-day quantiles {q}")

    def esda(self, w: Window) -> None:
        p = self.panel(w)
        s, r = self.cfg["spatial"], self.cfg["rate"]
        daily_all, pers_all = {}, []
        for reg in self.regions(w):
            bbox = self.cfg.regions[reg]
            rp = select_region(p, bbox)
            daily, lis = sp.daily_esda(rp, r["min_n"], s["permutations"], s["alpha"], s["island_policy"],
                                       s["seed"], n_jobs=s.get("n_jobs", 1))
            if daily.empty:
                self.log(f"esda {w.id}/{reg}: too few observed hexes")
                continue
            daily.to_csv(self.out(w, f"moran_daily_{reg}.csv"), index=False)
            lis.to_parquet(self.out(w, f"lisa_{reg}.parquet"), index=False)
            daily_all[reg] = daily
            pk = self.peak(w, p, bbox)
            ld = lis[lis["date"] == pk]
            if len(ld):
                self.save(viz.lisa_maps(ld, self.countries, bbox,
                                       f"LISA clusters · {reg.replace('_', ' ').title()} · {pk:%Y-%m-%d}",
                                       self.src_line), self.fig(w, f"lisa_{reg}.png"))
                hw = sp.h3_weights(ld["hex"].to_numpy(), s["island_policy"])
                ldx = ld.set_index("hex").loc[hw.cells]
                self.save(viz.moran_scatter(ldx["eb_rate"].to_numpy(), hw.w,
                                           f"Moran scatterplot · {reg.replace('_', ' ').title()} · {pk:%Y-%m-%d}",
                                           self.src_line), self.fig(w, f"moran_scatter_{reg}.png"))
                # sensitivity: KNN weights on the same day
                knn = sp.knn_weights(ldx.index.to_numpy(), s["knn_k"])
                gk = sp.global_moran(ldx["bad"], ldx["n"], knn, s["permutations"], s["seed"])
                gh = daily.loc[daily["date"] == pk].iloc[0]
                self.log_rows.append({"window": w.id, "region": reg, "date": f"{pk:%Y-%m-%d}",
                                      "I_h3": gh["I"], "p_h3": gh["p_sim"], "I_knn": gk["I"],
                                      "p_knn": gk["p_sim"]})
            min_days = min(r["min_days"], max(3, math.ceil(lis["date"].nunique() / 2)))
            pers = sp.persistence(lis, r["bad_rate"], min_days)
            pers.to_csv(self.out(w, f"persistence_{reg}.csv"), index=False)
            pers_all.append(pers.assign(region=reg))
            if len(pers):
                cover = rp[["hex"]].drop_duplicates()
                self.save(viz.value_map(pers, "hh_fdr", self.countries, bbox,
                                       f"Hotspot persistence · {reg.replace('_', ' ').title()} · {w.id}",
                                       f"Share of observed days each hex was a significant High-High cluster "
                                       f"(FDR 5%), {w.start} → {w.end}", "Share of days in a hotspot",
                                       self.src_line, coverage=cover),
                         self.fig(w, f"persistence_{reg}.png"))
            self.log(f"esda {w.id}/{reg}: {len(daily)} days, median I = {daily['I'].median():.2f}, "
                     f"{(daily['p_sim'] < 0.05).mean():.0%} of days significant")
        if daily_all and max(len(d) for d in daily_all.values()) > 1:
            kp = self.kp()
            storms = None
            if kp is not None:
                storms = kp[(kp["date"] >= pd.Timestamp(w.start)) & (kp["date"] <= pd.Timestamp(w.end))
                            & (kp["kp_max"] >= 5)]
            self.save(viz.moran_timeseries(daily_all, self.events(w), f"Spatial clustering over time · {w.name}",
                                          self.src_line, storms), self.fig(w, "moran_timeseries.png"))

    def resolution(self, w: Window) -> None:
        s, r = self.cfg["spatial"], self.cfg["rate"]
        reg = self.regions(w)[0]
        bbox = self.cfg.regions[reg]
        pk = self.peak(w, self.panel(w), bbox)
        if pk is None:
            return
        resolutions = [2, 3, 4] + ([5] if self.source == "adsblol" else [])
        maps, rows = {}, []
        for res in resolutions:
            day = load_panel(self.cfg, self.source, pk.date(), pk.date(), res=res, con=self.con)
            day = select_region(day, bbox)
            daily, lis = sp.daily_esda(day, r["min_n"], s["permutations"], s["alpha"], s["island_policy"],
                                       s["seed"], min_cells=8, n_jobs=s.get("n_jobs", 1))
            if lis.empty:
                continue
            maps[res] = lis
            rows.append({"res": res, **daily.iloc[0].to_dict(),
                         "hh": int((lis["cluster"] == "HH").sum()),
                         "hh_fdr": int((lis["cluster_fdr"] == "HH").sum())})
        if maps:
            pd.DataFrame(rows).to_csv(self.out(w, f"resolution_{reg}.csv"), index=False)
            self.save(viz.resolution_panel(maps, "cluster_fdr", self.countries, bbox,
                                          f"Resolution sensitivity · {reg.replace('_', ' ').title()} · {pk:%Y-%m-%d}",
                                          self.src_line), self.fig(w, f"resolution_{reg}.png"))
            self.log(f"resolution {w.id}/{reg}: " + ", ".join(f"res {x['res']}: I={x['I']:.2f}" for x in rows))

    def model(self, w: Window) -> None:
        r, m = self.cfg["rate"], self.cfg["model"]
        regs = list(dict.fromkeys([*self.regions(w), *CONTROL_REGIONS]))
        p = select_regions(self.panel(w), self.cfg, regs)
        min_days = min(r["min_days"], max(3, math.ceil(p["date"].nunique() / 2)))
        xs = md.cross_section(p, r["min_n"], r["bad_rate"], min_days)
        xs = xs.merge(p[["hex", "region"]].drop_duplicates("hex"), on="hex")
        use_acled = self.acled_covers(w)
        if not use_acled and self.acled() is not None:
            self.log(f"model {w.id}: ACLED ends {self.acled()['date_start'].max():%Y-%m-%d} (account time lag?); "
                     f"using UCDP drivers for this window")
        cov = cv.hex_covariates(self.cfg, xs["hex"].to_numpy(), w.start, w.end, events=self.ucdp(),
                                countries=self.countries, acled=self.acled() if use_acled else None)
        drivers = m["drivers_acled"] if use_acled else m["drivers"]
        xs = xs.merge(cov, on="hex", how="left")
        ucdp_end = self.ucdp()["date_start"].max()
        if pd.Timestamp(w.end) > ucdp_end:
            self.log(f"model {w.id}: WARNING conflict covariates only cover the window up to "
                     f"{ucdp_end:%Y-%m-%d} (UCDP); add newer Candidate releases when published")
        if self.source == "adsblol":
            from gnss.analysis.controls import hex_controls
            ctl = hex_controls(self.con, w.start, w.end, r["res"])
            xs = xs.drop(columns=["log_n_aircraft"]).merge(ctl, on="hex", how="left")
        xs.to_parquet(self.out(w, "model_cross_section.parquet"), index=False)
        results = {}
        for y in m["outcomes"]:
            try:
                fit = md.fit_spatial(xs, y, drivers + m["controls"], self.cfg["spatial"]["island_policy"])
            except Exception as e:     # singular designs on tiny windows
                self.log(f"model {w.id}/{y}: failed ({e})")
                continue
            fit["coefs"].to_csv(self.out(w, f"model_coefs_{y}.csv"), index=False)
            fit["resid"].to_csv(self.out(w, f"model_resid_{y}.csv"), index=False)
            self.out(w, f"model_summary_{y}.txt").write_text(
                "\n\n".join([fit["ols_summary"], fit["lag_summary"], fit["err_summary"]]), encoding="utf-8")
            results[y] = {**fit["diag"], "conflict_source": "acled" if use_acled else "ucdp"}
            self.save(viz.coef_plot(fit["coefs"], f"What predicts interference? · {w.id} · {y.replace('_', ' ')}",
                                   f"n = {fit['diag']['n']:,} hexes, {xs['region'].nunique()} regions; OLS residual Moran's I = "
                                   f"{fit['diag']['moran_res_I']:.2f}", self.src_line),
                     self.fig(w, f"model_coefs_{y}.png"))
            if self.gwr:
                g = md.fit_gwr(xs, y, drivers + m["controls"], m["mgwr_max_n"])
                if g is not None:
                    g.to_csv(self.out(w, f"gwr_{y}.csv"), index=False)
                    results[y]["gwr_bw"] = g.attrs["bw"]
            self.log(f"model {w.id}/{y}: n={fit['diag']['n']}, OLS R²={fit['diag']['ols_r2']:.2f}, "
                     f"resid I={fit['diag']['moran_res_I']:.2f}, AIC OLS/lag/err = "
                     f"{fit['diag']['ols_aic']:.0f}/{fit['diag']['lag_aic']:.0f}/{fit['diag']['err_aic']:.0f}; "
                     f"dropped {fit['diag']['dropped']}")
        self.out(w, "model_diagnostics.json").write_text(json.dumps(results, indent=2, default=str))

    def panel_fe(self, w: Window) -> None:
        r, m = self.cfg["rate"], self.cfg["model"]
        kp = self.kp()
        if kp is None:
            return self.log("panel: no Kp file; run scripts/build_covariates.py")
        regs = self.regions(w)
        p = select_regions(self.panel(w), self.cfg, regs)
        p = observed(p, r["min_n"])
        use_acled = self.acled_covers(w)
        events = self.acled() if use_acled else self.ucdp()
        ev_end = events["date_start"].max()
        if p["date"].max() > ev_end:
            self.log(f"panel {w.id}: {'ACLED' if use_acled else 'UCDP'} ends {ev_end:%Y-%m-%d}; later days "
                     f"dropped (no events is not zero events)")
            p = p[p["date"] <= ev_end]
        if p["date"].nunique() < 5:
            return self.log(f"panel {w.id}: fewer than 5 days, skipped")
        lag = self.cfg["covariates"]["conflict_lag_days"]
        if use_acled:
            from gnss.analysis.acled import acled_hex_day
            conf = acled_hex_day(p["hex"].unique(), events, w.start, w.end, m["panel_radius_km"], lag)
        else:
            conf = cv.conflict_hex_day(p["hex"].unique(), events, w.start, w.end, m["panel_radius_km"], lag)
        fit = md.fit_panel(p, conf, kp, r["min_n"])
        fit["coefs"].to_csv(self.out(w, "panel_coefs.csv"), index=False)
        self.out(w, "panel_summary.txt").write_text(fit["summary"], encoding="utf-8")
        c = fit["coefs"].set_index("term")
        self.log(f"panel {w.id} ({'ACLED' if use_acled else 'UCDP'}): {fit['n_obs']:,} hex-days / "
                 f"{fit['n_hex']:,} hexes; "
                 + ", ".join(f"{t}={c.loc[t, 'coef']:.4f} (p={c.loc[t, 'p']:.3f})"
                             for t in c.index if not t.startswith("dow_") and t != "log_n"))

    def events_study(self, w: Window) -> None:
        f = self.out(w, "region_daily.csv")
        if not f.exists():
            self.eda(w)
        if not f.exists():
            return
        ser = pd.read_csv(f, parse_dates=["date"])
        anchors = [(d, lab) for d, lab in self.events(w)]
        for d0, lab in anchors:
            es = {}
            for reg, g in ser.groupby("region"):
                e = md.event_study(g[["date", "share_degraded", "pooled_rate"]], d0)
                if (e["rel_day"] < 0).sum() >= 3 and (e["rel_day"] >= 0).sum() >= 3:
                    es[reg.replace("_", " ").title()] = e
            if es:
                safe = str(lab).replace(" ", "_").replace("/", "-")
                self.save(viz.event_study_plot(es, "share_degraded", f"Event study · {lab} · {w.id}",
                                              "Share of hexes degraded", self.src_line),
                         self.fig(w, f"event_study_{safe}.png"))
        self.log(f"events {w.id}: {len(anchors)} anchors")


def saved_figures(cfg: AnalysisConfig, source: str, window: str, pattern: str = "*.png",
                  show: bool = True) -> list[Path]:
    """Finished figures for a window (no recomputation); displayed inline when show=True."""
    files = sorted((cfg.figures_dir / source / window).glob(pattern))
    if show:
        from IPython.display import Image, Markdown, display
        if not files:
            display(Markdown(f"*No saved figures match `{source}/{window}/{pattern}`; run the step first.*"))
        for f in files:
            display(Image(filename=str(f)))
    return files


def saved_table(cfg: AnalysisConfig, source: str, window: str, name: str) -> pd.DataFrame:
    p = cfg.outputs_dir / source / window / name
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".json":
        return pd.DataFrame(json.loads(p.read_text())).T
    return pd.read_csv(p)
