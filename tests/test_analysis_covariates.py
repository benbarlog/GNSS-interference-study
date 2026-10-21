"""Covariate builders on synthetic inputs (no downloads)."""

from __future__ import annotations

from datetime import date

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
from shapely.geometry import box

from gnss.analysis import covariates as cv


def cell(lat, lon, res=4):
    return h3.str_to_int(h3.latlng_to_cell(lat, lon, res))


def test_nearest_km_one_degree_latitude():
    d = cv.nearest_km(np.array([50.0]), np.array([10.0]), np.array([51.0]), np.array([10.0]))
    assert abs(d[0] - 111.2) < 0.5
    assert np.isinf(cv.nearest_km(np.array([0.0]), np.array([0.0]), np.array([]), np.array([])))[0]


def events(rows):
    df = pd.DataFrame(rows, columns=["latitude", "longitude", "date_start", "best", "type_of_violence"])
    df["date_start"] = pd.to_datetime(df["date_start"])
    df["date_prec"] = 1
    return df


def test_conflict_features_counts_and_window():
    c = cell(50.0, 30.0)
    lat, lon = h3.cell_to_latlng(h3.int_to_str(c))
    ev = events([(lat + 0.5, lon, "2025-06-10", 3, 1),       # ~56 km
                 (lat + 2.0, lon, "2025-06-10", 10, 2),      # ~222 km
                 (lat + 0.1, lon, "2025-05-01", 99, 1)])     # outside window
    f = cv.conflict_features([c], ev, date(2025, 6, 9), date(2025, 6, 12), radii_km=(100, 250), lag_days=3)
    r = f.iloc[0]
    assert r["conflict_events_100"] == 1 and r["conflict_events_250"] == 2
    assert r["conflict_deaths_250"] == 13
    assert r["conflict_state_share_250"] == 0.5
    assert 50 < r["dist_conflict_km"] < 60


def test_conflict_hex_day_lagged_rolling_sum():
    c = cell(50.0, 30.0)
    lat, lon = h3.cell_to_latlng(h3.int_to_str(c))
    ev = events([(lat, lon, "2025-06-10", 2, 1)])
    p = cv.conflict_hex_day([c], ev, date(2025, 6, 9), date(2025, 6, 15), radius_km=50, lag_days=2)
    s = p.set_index("date")["conflict_events"]
    assert s["2025-06-09"] == 0
    assert s["2025-06-10"] == 1 and s["2025-06-12"] == 1      # counted on the day and 2 days after
    assert s["2025-06-13"] == 0
    assert len(p) == 7


def test_belligerent_distance_inside_zero_and_outside_positive():
    countries = gpd.GeoDataFrame({"ADM0_A3": ["AAA", "BBB"]},
                                 geometry=[box(20, 50, 25, 55), box(30, 50, 35, 55)], crs="EPSG:4326")
    inside, outside = cell(52.5, 22.5), cell(52.5, 27.5)
    d = cv.belligerent_distance([inside, outside], countries, ["AAA"])
    assert d[0] == 0
    lat, lon = h3.cell_to_latlng(h3.int_to_str(outside))
    expect = (lon - 25) * 111.2 * np.cos(np.radians(lat))
    assert abs(d[1] - expect) / expect < 0.05


def test_active_belligerents_overlap():
    b = [{"iso3": ["IRN"], "start": date(2025, 6, 13), "end": date(2025, 6, 24)},
         {"iso3": ["RUS", "UKR"], "start": date(2022, 2, 24), "end": None}]
    assert cv.active_belligerents(b, date(2025, 6, 1), date(2025, 6, 12)) == ["RUS", "UKR"]
    assert cv.active_belligerents(b, date(2025, 6, 20), date(2025, 7, 1)) == ["IRN", "RUS", "UKR"]


def test_load_kp_parses_gfz_format(tmp_path):
    from types import SimpleNamespace
    d = tmp_path / "kp"
    d.mkdir()
    (d / cv.KP_URL.rsplit("/", 1)[1]).write_text(
        "# header\n"
        "2024 05 10 33722 33722.5 2604  5  2.667  3.000  2.333  5.667  8.667  9.000  9.000  9.000"
        "   12   15    9   67  300  400  400  400   200  150    200.0    200.0 0\n")
    cfg = SimpleNamespace(covariates_dir=tmp_path)
    kp = cv.load_kp(cfg)
    assert kp.iloc[0]["kp_max"] == 9.0 and kp.iloc[0]["ap"] == 200
