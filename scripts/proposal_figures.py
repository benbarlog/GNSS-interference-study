"""Rough figures for the hypothesis/plan presentation.

Inputs:  data/raw/gpsjam/*.csv  (see scripts/download_gpsjam.py)
Outputs: data/processed/persistence_recent.tif, figures/*.png, data/processed/region_stats.csv

A hex-day counts as "degraded" when n >= MIN_N aircraft and bad/n > BAD_RATE
(GPSJAM's own red threshold). Persistence = degraded days / observed days.
"""

from pathlib import Path

import geopandas as gpd
import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import Polygon

from gnss.ingest_gpsjam import load_panel

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PROC = ROOT / "data" / "processed"
COUNTRIES = ROOT / "data" / "raw" / "naturalearth" / "ne_50m_admin_0_countries.zip"

MIN_N = 5
BAD_RATE = 0.10
MIN_DAYS = 10
RECENT_DAYS = 90
INVASION = pd.Timestamp("2022-02-24")
PRE_END = pd.Timestamp("2022-02-23")

# (lat_min, lat_max, lon_min, lon_max)
REGIONS = {
    "Baltic / Kaliningrad": (53, 61, 13, 30),
    "Black Sea / Ukraine periphery": (41, 50, 24, 42),
    "Eastern Med / Levant": (30, 37, 28, 37),
    "Persian Gulf": (23, 31, 47, 57),
    "Taiwan Strait": (20, 28, 115, 125),
    "Korean Peninsula": (34, 40, 124, 130),
    "Myanmar": (14, 24, 92, 100),
    "Continental US (control)": (30, 45, -110, -80),
}

BG = "#0b0f19"
LAND_EDGE = "#3b4a63"
COVERED = "#1d2a40"
INK = "#e8ecf3"
INK2 = "#9aa6b8"
HEAT = plt.get_cmap("inferno")

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": LAND_EDGE, "font.size": 11,
})


def hex_stats(panel: pd.DataFrame) -> pd.DataFrame:
    obs = panel[panel["n"] >= MIN_N].copy()
    obs["degraded"] = obs["rate"] > BAD_RATE
    s = obs.groupby("hex").agg(days_obs=("degraded", "size"), days_bad=("degraded", "sum"),
                               bad=("bad", "sum"), n=("n", "sum"))
    s["persistence"] = s["days_bad"] / s["days_obs"]
    s["pooled_rate"] = s["bad"] / s["n"]
    return s


def add_centroids(s: pd.DataFrame) -> pd.DataFrame:
    ll = np.array([h3.cell_to_latlng(h) for h in s.index])
    return s.assign(lat=ll[:, 0], lon=ll[:, 1])


def in_box(df, box):
    a, b, c, d = box
    return df[(df.lat >= a) & (df.lat <= b) & (df.lon >= c) & (df.lon <= d)]


def hex_gdf(s: pd.DataFrame) -> gpd.GeoDataFrame:
    geoms = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h)]) for h in s.index]
    return gpd.GeoDataFrame(s, geometry=geoms, crs="EPSG:4326")


# ---------- global raster ----------

def rasterize(stats: pd.DataFrame, res: float = 0.1):
    """Sample each pixel centre's res-4 H3 cell and look up its value. NaN = no coverage."""
    cache = PROC / f"pixel_cells_{res}.npy"
    lats = np.arange(90 - res / 2, -90, -res)
    lons = np.arange(-180 + res / 2, 180, res)
    if cache.exists():
        cells = np.load(cache)
    else:
        f = h3.api.basic_int.latlng_to_cell
        cells = np.array([[f(la, lo, 4) for lo in lons] for la in lats], dtype=np.uint64)
        np.save(cache, cells)
    val = pd.Series(stats["persistence"].values,
                    index=[h3.str_to_int(h) for h in stats.index])
    flat = val.reindex(cells.ravel()).to_numpy(dtype="float32")
    return flat.reshape(cells.shape), from_origin(-180, 90, res, res)


def write_tif(arr, transform, path):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                       count=1, dtype="float32", crs="EPSG:4326", transform=transform,
                       nodata=np.nan, compress="deflate") as dst:
        dst.write(arr, 1)


def to_robinson(arr, transform):
    crs = "ESRI:54030"
    h, w = arr.shape
    t, W, H = calculate_default_transform("EPSG:4326", crs, w, h, -180, -90, 180, 90)
    out = np.full((H, W), np.nan, dtype="float32")
    reproject(arr, out, src_transform=transform, src_crs="EPSG:4326", dst_transform=t,
              dst_crs=crs, resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    extent = (t.c, t.c + t.a * W, t.f + t.e * H, t.f)
    return out, extent, crs


def plot_global(arr, transform, world, title, subtitle, path):
    rob, extent, crs = to_robinson(arr, transform)
    fig = plt.figure(figsize=(16, 8.4))
    ax = fig.add_axes([0.01, 0.13, 0.98, 0.74])
    covered = np.where(np.isfinite(rob), 1.0, np.nan)
    ax.imshow(covered, extent=extent, cmap=ListedColormap([COVERED]), interpolation="nearest")
    heat = np.where(rob >= 0.02, rob, np.nan)
    im = ax.imshow(heat, extent=extent, cmap=HEAT, vmin=0, vmax=1, interpolation="nearest")
    world.to_crs(crs).boundary.plot(ax=ax, color=LAND_EDGE, linewidth=0.35)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(-6.0e6, 8.4e6)
    ax.set_axis_off()
    fig.text(0.03, 0.95, title, fontsize=22, weight="bold")
    fig.text(0.03, 0.915, subtitle, fontsize=12, color=INK2)
    cax = fig.add_axes([0.33, 0.075, 0.34, 0.018])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.set_title("Share of observed days with >10% of aircraft reporting degraded GPS",
                  color=INK2, fontsize=10)
    cb.outline.set_visible(False)
    fig.text(0.03, 0.02, "Dark blue: aircraft observed, interference on <2% of days.   "
             "Black: no ADS-B coverage (not the same as no interference).", fontsize=9, color=INK2)
    fig.text(0.97, 0.02, "Data: GPSJAM.org (J. Wiseman) / ADS-B Exchange · H3 res 4",
             fontsize=9, color=INK2, ha="right")
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------- regional maps ----------

def plot_region(ax, g, world, box, title, vmax=1):
    a, b, c, d = box
    world.plot(ax=ax, color="#121a29", edgecolor=LAND_EDGE, linewidth=0.5)
    g[g.persistence < 0.02].plot(ax=ax, color=COVERED, linewidth=0)
    hot = g[g.persistence >= 0.02]
    if len(hot):
        hot.plot(ax=ax, column="persistence", cmap=HEAT, vmin=0, vmax=vmax, linewidth=0)
    world.boundary.plot(ax=ax, color="#7f8ea6", linewidth=0.5)
    ax.set_xlim(c, d)
    ax.set_ylim(a, b)
    ax.set_aspect(1 / np.cos(np.radians((a + b) / 2)))
    ax.set_title(title, fontsize=13, loc="left", color=INK)
    ax.tick_params(labelsize=8)


def main():
    FIG.mkdir(exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)
    world = gpd.read_file(COUNTRIES)
    panel = load_panel()
    last = panel["date"].max()
    recent = panel[panel["date"] > last - pd.Timedelta(days=RECENT_DAYS)]
    pre = panel[panel["date"] <= PRE_END]
    print(f"{panel['date'].nunique()} days loaded; recent window {recent['date'].nunique()} days "
          f"ending {last.date()}; pre-invasion {pre['date'].nunique()} days")

    rs = add_centroids(hex_stats(recent))
    rs = rs[rs.days_obs >= MIN_DAYS]
    ps = add_centroids(hex_stats(pre))
    ps = ps[ps.days_obs >= 3]

    # 1. global raster
    arr, tr = rasterize(rs)
    write_tif(arr, tr, PROC / "persistence_recent.tif")
    d0 = (last - pd.Timedelta(days=RECENT_DAYS - 1)).date()
    plot_global(arr, tr, world, "Where GPS is persistently degraded",
                f"Fraction of days each ~1,800 km² hex had >10% of aircraft reporting degraded "
                f"navigation accuracy, {d0} to {last.date()}", FIG / "global_persistence.png")

    # 2. Europe before vs after the invasion
    europe = (40, 66, 5, 45)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.2))
    plot_region(axes[0], hex_gdf(in_box(ps, europe)), world, europe,
                "Before: 14–23 Feb 2022 (pre-invasion)")
    plot_region(axes[1], hex_gdf(in_box(rs, europe)), world, europe,
                f"After: last {RECENT_DAYS} days ({d0} – {last.date()})")
    sm = plt.cm.ScalarMappable(cmap=HEAT, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.07, shrink=0.5)
    cb.set_label("Share of observed days with >10% degraded aircraft", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle("GPS degradation around Russia & Ukraine, before vs. after 24 Feb 2022",
                 fontsize=18, weight="bold", x=0.03, ha="left")
    fig.savefig(FIG / "europe_before_after.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 3. case-study panel
    cases = ["Baltic / Kaliningrad", "Eastern Med / Levant", "Persian Gulf",
             "Black Sea / Ukraine periphery", "Myanmar", "Taiwan Strait"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, name in zip(axes.ravel(), cases):
        box = REGIONS[name]
        pad = (box[0] - 1, box[1] + 1, box[2] - 1.5, box[3] + 1.5)
        plot_region(ax, hex_gdf(in_box(rs, pad)), world, pad, name)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.03, pad=0.05, shrink=0.4)
    cb.set_label("Share of observed days with >10% degraded aircraft", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Candidate case-study regions, last {RECENT_DAYS} days", fontsize=18,
                 weight="bold", x=0.03, ha="left")
    fig.savefig(FIG / "case_study_regions.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # 4. regional time series (share of observed hexes that are degraded each day)
    cents = add_centroids(pd.DataFrame(index=panel["hex"].unique()))
    obs = panel[panel["n"] >= MIN_N].join(cents[["lat", "lon"]], on="hex")
    obs["degraded"] = obs["rate"] > BAD_RATE
    rows = []
    for name, box in REGIONS.items():
        sub = in_box(obs, box)
        daily = sub.groupby("date").agg(share=("degraded", "mean"), hexes=("degraded", "size"))
        rows.append(daily.assign(region=name).reset_index())
    ts = pd.concat(rows)
    ts.to_csv(PROC / "region_daily_share.csv", index=False)

    fig, axes = plt.subplots(len(REGIONS), 1, figsize=(14, 2.1 * len(REGIONS)), sharex=True)
    for ax, (name, g) in zip(axes, ts.groupby("region", sort=False)):
        g = g.sort_values("date")
        ax.plot(g["date"], g["share"] * 100, color="#f2a23a", lw=1.6)
        ax.fill_between(g["date"], g["share"] * 100, color="#f2a23a", alpha=0.15, lw=0)
        ax.axvline(INVASION, color=INK2, lw=0.8, ls="--")
        ax.set_ylim(0, max(5, g["share"].max() * 110))
        ax.text(0.005, 0.85, name, transform=ax.transAxes, fontsize=11, weight="bold")
        ax.grid(axis="y", color="#26314a", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("% hexes", fontsize=9)
    axes[-1].text(INVASION, axes[-1].get_ylim()[1] * 0.5, " invasion of Ukraine", color=INK2,
                  fontsize=9)
    fig.suptitle("Share of observed hexes with >10% degraded aircraft "
                 "(weekly samples + last 90 days daily)", fontsize=15, weight="bold",
                 x=0.03, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "region_timeseries.png", dpi=170)
    plt.close(fig)

    # 5. summary table
    summ = []
    for name, box in REGIONS.items():
        r, p = in_box(rs, box), in_box(ps, box)
        summ.append({
            "region": name,
            "hexes_recent": len(r),
            "mean_persistence_recent": r["persistence"].mean(),
            "pct_hexes_degraded_>=50%_days": (r["persistence"] >= 0.5).mean() * 100,
            "hexes_pre": len(p),
            "mean_persistence_pre": p["persistence"].mean(),
        })
    summ = pd.DataFrame(summ).round(3)
    summ.to_csv(PROC / "region_stats.csv", index=False)
    print(summ.to_string(index=False))

    top = rs.sort_values("persistence", ascending=False)
    top = top[top.persistence >= 0.5]
    print(f"\nhexes degraded on >=50% of recent days: {len(top)} of {len(rs)}")
    print(top[["lat", "lon"]].round(0).value_counts().head(25).to_string())


if __name__ == "__main__":
    main()
