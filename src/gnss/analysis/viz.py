"""Figures with one shared style (same dark theme as scripts/proposal_figures.py).

Colour jobs:
- magnitude (rates, persistence): inferno, the same sequential map as the proposal figures
- LISA clusters: diverging red (high) / blue (low); strong = HH / LL, light = HL / LH outliers;
  not significant = the "covered" fill. CVD- and contrast-checked on this surface.
- model comparison (3 series): blue / orange / aqua, validated all-pairs on this surface.
Every map: Natural Earth borders, legend or colour bar, and a source line.
"""

from __future__ import annotations

from pathlib import Path

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BG = "#0b0f19"
LAND_EDGE = "#3b4a63"
COVERED = "#1d2a40"
INK = "#e8ecf3"
INK2 = "#9aa6b8"
# inferno from 12% up: its near-black start is indistinguishable from BG, which would hide observed zeros
HEAT = ListedColormap(plt.get_cmap("inferno")(np.linspace(0.12, 1.0, 256)), name="inferno_bg")
SERIES = ["#3987e5", "#d95926", "#199e70"]
LISA_COLORS = {"HH": "#e66767", "HL": "#f2b0b0", "LH": "#86b6ef", "LL": "#3987e5", "ns": COVERED}
LISA_NAMES = {"HH": "High-High (hotspot)", "HL": "High-Low (outlier)", "LH": "Low-High (outlier)",
              "LL": "Low-Low (coldspot)", "ns": "Not significant"}

SRC_GPSJAM = "Data: GPSJAM.org (J. Wiseman) / ADS-B Exchange"
SRC_ADSBLOL = "Data: adsb.lol globe_history (ODbL), own NIC < 7 pipeline"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": LAND_EDGE, "font.size": 11, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    })


def source_line(source: str) -> str:
    return SRC_ADSBLOL if source == "adsblol" else SRC_GPSJAM


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


# ---- hex drawing -----------------------------------------------------------------------------
def _polys(cells) -> list[np.ndarray]:
    out = []
    for c in cells:
        b = np.array(h3.cell_to_boundary(h3.int_to_str(int(c))))[:, ::-1]   # (lon, lat)
        if b[:, 0].max() - b[:, 0].min() > 180:                               # antimeridian
            b[:, 0] = np.where(b[:, 0] < 0, b[:, 0] + 360, b[:, 0])
        out.append(b)
    return out


def draw_hexes(ax, cells, values=None, colors=None, cmap=HEAT, vmin=0, vmax=1, edge="none"):
    pc = PolyCollection(_polys(cells), edgecolors=edge, linewidths=0.2)
    if colors is not None:
        pc.set_facecolor(colors)
    else:
        pc.set_array(np.asarray(values, float))
        pc.set_cmap(cmap)
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    return pc


def basemap(ax, countries, bbox, pad: float = 0.5) -> None:
    a, b, c, d = bbox
    if countries is not None:
        countries.boundary.plot(ax=ax, color=LAND_EDGE, linewidth=0.5, zorder=3)
    ax.set_xlim(c - pad, d + pad)
    ax.set_ylim(a - pad, b + pad)
    ax.set_aspect(1 / np.cos(np.radians((a + b) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _title(fig, title, subtitle=None, source=None):
    fig.text(0.02, 0.98, title, fontsize=17, weight="bold", va="top")
    if subtitle:
        fig.text(0.02, 0.935, subtitle, fontsize=11, color=INK2, va="top")
    if source:
        fig.text(0.02, 0.01, source, fontsize=8.5, color=INK2)


# ---- maps ------------------------------------------------------------------------------------
def rate_label(source_line: str) -> str:
    if source_line == SRC_ADSBLOL:
        return "Share of aircraft with degraded GNSS (NIC < 7)"
    return "Share of aircraft with low navigation accuracy (GPSJAM definition)"


def raw_vs_eb(day: pd.DataFrame, countries, bbox, title, source) -> plt.Figure:
    """day: hex, rate, eb_rate, n (one region, one day)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    fig.subplots_adjust(top=0.86, bottom=0.12, wspace=0.03)
    for ax, col, lab in zip(axes, ["rate", "eb_rate"], ["Raw rate (bad / all aircraft)",
                                                         "Empirical Bayes smoothed rate"]):
        pc = draw_hexes(ax, day["hex"], day[col])
        basemap(ax, countries, bbox)
        ax.set_title(lab, color=INK, fontsize=12, loc="left")
    cax = fig.add_axes([0.33, 0.07, 0.34, 0.022])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label(rate_label(source), color=INK2)
    _title(fig, title, "Smoothing pulls low-traffic hexes toward the regional mean; "
           "hexes with < 5 aircraft are not shown", source)
    return fig


def lisa_maps(lis: pd.DataFrame, countries, bbox, title, source) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    fig.subplots_adjust(top=0.86, bottom=0.12, wspace=0.03)
    for ax, col, lab in zip(axes, ["cluster", "cluster_fdr"],
                            ["Pseudo p < 0.05 (unadjusted)", "False discovery rate controlled at 5%"]):
        draw_hexes(ax, lis["hex"], colors=[LISA_COLORS[c] for c in lis[col]])
        basemap(ax, countries, bbox)
        counts = lis[col].value_counts()
        ax.set_title(f"{lab}  ·  HH = {counts.get('HH', 0)}", color=INK, fontsize=12, loc="left")
    handles = [Patch(color=LISA_COLORS[k], label=LISA_NAMES[k]) for k in ["HH", "HL", "LH", "LL", "ns"]]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.04))
    _title(fig, title, "Local Moran's I on EB-standardized rates, H3 ring-1 contiguity, 999 permutations",
           source)
    return fig


def value_map(df: pd.DataFrame, col: str, countries, bbox, title, subtitle, label, source,
              vmin=0, vmax=1, cmap=HEAT, coverage: pd.DataFrame | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    fig.subplots_adjust(top=0.86, bottom=0.12)
    if coverage is not None:
        draw_hexes(ax, coverage["hex"], colors=[COVERED] * len(coverage))
    # zero = observed but never hot: leave it in the covered fill so it reads apart from "no data"
    hot = df[df[col] > 0]
    pc = draw_hexes(ax, hot["hex"], hot[col], cmap=cmap, vmin=vmin, vmax=vmax)
    basemap(ax, countries, bbox)
    cax = fig.add_axes([0.25, 0.07, 0.5, 0.022])
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
    cb.set_label(label, color=INK2)
    _title(fig, title, subtitle, source)
    return fig


def disagreement_map(j: pd.DataFrame, countries, bbox, title, bad_rate=0.10) -> plt.Figure:
    a, g = j["rate_a"] > bad_rate, j["rate_g"] > bad_rate
    cls = np.select([a & g, a & ~g, ~a & g], ["both", "ours", "gpsjam"], "neither")
    col = {"both": "#e66767", "ours": SERIES[1], "gpsjam": SERIES[0], "neither": COVERED}
    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    fig.subplots_adjust(top=0.86, bottom=0.12)
    draw_hexes(ax, j["hex"], colors=[col[c] for c in cls])
    basemap(ax, countries, bbox)
    lab = {"both": "Both > 10%", "ours": "adsb.lol only", "gpsjam": "GPSJAM only", "neither": "Neither"}
    fig.legend(handles=[Patch(color=col[k], label=f"{lab[k]} ({(cls == k).sum()})") for k in col],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.04))
    _title(fig, title, "Hex-days with ≥ 5 aircraft in both sources", SRC_ADSBLOL + "; " + SRC_GPSJAM)
    return fig


# ---- charts ----------------------------------------------------------------------------------
def moran_timeseries(daily: dict[str, pd.DataFrame], events: list[tuple] | None, title, source,
                     storms: pd.DataFrame | None = None) -> plt.Figure:
    """daily: {region: frame with date, I, p_sim}; one small multiple per region, shared x."""
    n = len(daily)
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.9 * n + 1.6), sharex=True, squeeze=False)
    fig.subplots_adjust(top=1 - 1.15 / (1.9 * n + 1.6), bottom=0.08, hspace=0.35)
    for ax, (reg, d) in zip(axes[:, 0], daily.items()):
        d = d.sort_values("date")
        ax.plot(d["date"], d["I"], color=SERIES[0], lw=2)
        sig = d["p_sim"] < 0.05
        ax.scatter(d.loc[~sig, "date"], d.loc[~sig, "I"], s=14, facecolor=BG, edgecolor=SERIES[0],
                   zorder=3, lw=1)
        ax.axhline(0, color=LAND_EDGE, lw=0.8)
        ax.set_ylabel("Moran's I")
        ax.text(0.005, 0.92, reg.replace("_", " ").title(), transform=ax.transAxes, va="top",
                fontsize=11, weight="bold")
        if storms is not None:
            for dd in storms["date"]:
                ax.axvspan(dd - pd.Timedelta(hours=12), dd + pd.Timedelta(hours=12), color="#c98500",
                           alpha=0.25, lw=0)
        for (d0, lab) in events or []:
            ax.axvline(pd.Timestamp(d0), color=INK2, lw=0.8, ls=":")
    for (d0, lab) in events or []:
        axes[0, 0].annotate(lab, (pd.Timestamp(d0), 1.02), xycoords=("data", "axes fraction"),
                            fontsize=8.5, color=INK2, rotation=0, ha="left", va="bottom")
    handles = [Line2D([], [], color=SERIES[0], lw=2, marker="o", ms=0, label="Global Moran's I (rate)"),
               Line2D([], [], color=SERIES[0], lw=0, marker="o", mfc=BG, label="not significant (p ≥ 0.05)")]
    if storms is not None and len(storms):
        handles.append(Patch(color="#c98500", alpha=0.4, label="Geomagnetic storm (Kp ≥ 5)"))
    axes[-1, 0].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    _title(fig, title, "Daily global spatial autocorrelation of the degraded-GNSS rate", source)
    return fig


def region_share_timeseries(series: pd.DataFrame, regions: list[str], title, source,
                            events=None) -> plt.Figure:
    fig, axes = plt.subplots(len(regions), 1, figsize=(13, 1.7 * len(regions) + 1.5), sharex=True,
                             squeeze=False)
    fig.subplots_adjust(top=1 - 1.1 / (1.7 * len(regions) + 1.5), bottom=0.07, hspace=0.3)
    for ax, reg in zip(axes[:, 0], regions):
        d = series[series["region"] == reg].sort_values("date")
        ax.plot(d["date"], d["share_degraded"], color=SERIES[1], lw=2)
        ax.set_ylim(0, max(0.05, d["share_degraded"].max() * 1.1) if len(d) else 1)
        ax.set_ylabel("Share degraded")
        ax.text(0.005, 0.92, reg.replace("_", " ").title(), transform=ax.transAxes, va="top",
                fontsize=11, weight="bold")
        for (d0, _) in events or []:
            ax.axvline(pd.Timestamp(d0), color=INK2, lw=0.8, ls=":")
    _title(fig, title, "Share of observed hexes (≥ 5 aircraft) with > 10% of aircraft degraded", source)
    return fig


def coef_plot(coefs: pd.DataFrame, title, subtitle, source) -> plt.Figure:
    spatial = coefs[coefs["term"].isin(["rho (W_y)", "lambda (W_e)"])]
    c = coefs[~coefs["term"].isin(["CONSTANT", "rho (W_y)", "lambda (W_e)"])].copy()
    if len(spatial):
        subtitle += "  ·  " + ", ".join(f"{t.split()[0]} = {v:.2f}" for t, v in zip(spatial["term"], spatial["coef"]))
    terms = [t for t in c["term"].unique()]
    models = list(c["model"].unique())
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(terms) + 2.4))
    fig.subplots_adjust(top=1 - 1.0 / (0.5 * len(terms) + 2.4), left=0.3, bottom=0.24)
    off = np.linspace(-0.22, 0.22, len(models))
    for k, m in enumerate(models):
        d = c[c["model"] == m].set_index("term").reindex(terms)
        y = np.arange(len(terms)) + off[k]
        ax.errorbar(d["coef"], y, xerr=1.96 * d["se"], fmt="o", color=SERIES[k % 3], ms=7,
                    lw=2, capsize=0, label=m)
    ax.axvline(0, color=LAND_EDGE, lw=1)
    ax.set_yticks(np.arange(len(terms)), [t.replace("_", " ") for t in terms])
    ax.invert_yaxis()
    ax.set_xlabel("Standardized coefficient (95% CI); spatial-lag values are direct effects only")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.17), ncol=len(models))
    _title(fig, title, subtitle, source)
    return fig


def validation_scatter(j: pd.DataFrame, title) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    fig.subplots_adjust(top=0.86, bottom=0.12)
    hb = ax.hexbin(j["rate_g"], j["rate_a"], gridsize=40, bins="log", cmap=HEAT, mincnt=1,
                   extent=(0, 1, 0, 1))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.set_xlabel("GPSJAM bad-aircraft share")
    ax.set_ylabel("adsb.lol bad-aircraft share (NIC < 7)")
    rho = j["rate_a"].corr(j["rate_g"], method="spearman")
    ax.text(0.03, 0.95, f"Spearman ρ = {rho:.2f}\n{len(j):,} hex-days", transform=ax.transAxes,
            va="top", color=INK)
    fig.colorbar(hb, ax=ax, label="hex-days (log)")
    _title(fig, title, "Same hex, same UTC day, ≥ 5 aircraft in both", SRC_ADSBLOL + "; " + SRC_GPSJAM)
    return fig


def ablation_bars(ab: pd.DataFrame, title) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    fig.subplots_adjust(top=0.78, bottom=0.18, left=0.2)
    d = ab[ab["filter"] != "none"]
    y = np.arange(len(d))
    ax.barh(y, d["bad_kept"], color=SERIES[1], height=0.5)
    ax.barh(y, d["known_kept"] - d["bad_kept"], left=d["bad_kept"], color="none")
    for yi, (bk, kk) in enumerate(zip(d["bad_kept"], d["known_kept"])):
        ax.text(bk + 0.01, yi, f"{bk:.0%} of degraded time kept · {kk:.0%} of all time kept",
                va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, d["filter"].str.replace("_", " "))
    ax.set_xlim(0, 1.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    _title(fig, title, "How much degraded (NIC < 7) time each confounder filter removes", SRC_ADSBLOL)
    return fig


def event_study_plot(es: dict[str, pd.DataFrame], col: str, title, ylabel, source) -> plt.Figure:
    n = len(es)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n + 1, 4.2), sharey=False, squeeze=False)
    fig.subplots_adjust(top=0.78, bottom=0.16, wspace=0.3)
    for ax, (lab, d) in zip(axes[0], es.items()):
        d = d.sort_values("rel_day")
        m, s = d[f"{col}_base_mean"].iloc[0], d[f"{col}_base_sd"].iloc[0]
        ax.axhspan(m - 2 * s, m + 2 * s, color=COVERED, lw=0)
        ax.axhline(m, color=INK2, lw=0.8)
        ax.plot(d["rel_day"], d[col], color=SERIES[1], lw=2)
        ax.axvline(0, color=INK2, lw=0.8, ls=":")
        ax.set_title(lab, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Days from event")
    axes[0, 0].set_ylabel(ylabel)
    _title(fig, title, "Band = pre-event mean ± 2 SD", source)
    return fig


def moran_scatter(rate, w, title, source) -> plt.Figure:
    """Moran scatterplot of standardized EB rate vs its spatial lag."""
    from libpysal.weights import lag_spatial
    z = (rate - rate.mean()) / rate.std()
    lz = lag_spatial(w, z)
    b = np.polyfit(z, lz, 1)[0]
    fig, ax = plt.subplots(figsize=(7, 6.6))
    fig.subplots_adjust(top=0.84, bottom=0.12)
    ax.scatter(z, lz, s=10, color=SERIES[0], alpha=0.6, lw=0)
    xs = np.array([z.min(), z.max()])
    ax.plot(xs, b * xs, color=SERIES[1], lw=2)
    ax.axhline(0, color=LAND_EDGE, lw=0.8)
    ax.axvline(0, color=LAND_EDGE, lw=0.8)
    for (x, y, q) in [(0.97, 0.97, "HH"), (0.03, 0.97, "LH"), (0.03, 0.03, "LL"), (0.97, 0.03, "HL")]:
        ax.text(x, y, q, transform=ax.transAxes, ha="right" if x > 0.5 else "left",
                va="top" if y > 0.5 else "bottom", color=INK2)
    ax.set_xlabel("EB rate (z-score)")
    ax.set_ylabel("Spatial lag of EB rate")
    _title(fig, title, f"Slope = Moran's I ≈ {b:.2f}", source)
    return fig


def resolution_panel(maps: dict[int, pd.DataFrame], col: str, countries, bbox, title, source) -> plt.Figure:
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.4), squeeze=False)
    fig.subplots_adjust(top=0.8, bottom=0.14, wspace=0.03)
    for ax, (res, d) in zip(axes[0], sorted(maps.items())):
        if col in ("cluster", "cluster_fdr"):
            draw_hexes(ax, d["hex"], colors=[LISA_COLORS[c] for c in d[col]])
        else:
            draw_hexes(ax, d["hex"], d[col])
        basemap(ax, countries, bbox)
        ax.set_title(f"H3 res {res}  ·  {len(d):,} hexes", color=INK, fontsize=11, loc="left")
    if col in ("cluster", "cluster_fdr"):
        fig.legend(handles=[Patch(color=LISA_COLORS[k], label=LISA_NAMES[k])
                            for k in ["HH", "HL", "LH", "LL", "ns"]],
                   loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.03))
    _title(fig, title, "Same day and method, different hex size (modifiable areal unit problem)", source)
    return fig

