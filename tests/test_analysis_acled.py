"""ACLED normalization and category features on synthetic rows (no network, no credentials)."""

from __future__ import annotations

from datetime import date

import h3
import numpy as np
import pandas as pd

from gnss.analysis import acled


def raw_rows(lat, lon):
    def r(i, sub, typ, d="2025-06-14", fat=0, tp=1, gp=1, dlat=0.2):
        return {"event_id_cnty": f"X{i}", "event_date": d, "year": 2025, "time_precision": tp,
                "disorder_type": "Political violence", "event_type": typ, "sub_event_type": sub,
                "actor1": "A", "actor2": "B", "interaction": 12, "country": "Testland",
                "admin1": "", "location": "", "latitude": lat + dlat, "longitude": lon,
                "geo_precision": gp, "fatalities": fat}
    return pd.DataFrame([
        r(1, "Air/drone strike", "Explosions/Remote violence", fat=2),
        r(2, "Shelling/artillery/missile attack", "Explosions/Remote violence", fat=1),
        r(3, "Armed clash", "Battles", fat=5),
        r(4, "Disrupted weapons use", "Strategic developments"),
        r(5, "Air/drone strike", "Explosions/Remote violence", gp=3),          # province-level: dropped
        r(6, "Armed clash", "Battles", d="2025-05-01"),                         # outside window
        r(7, "Peaceful protest", "Protests"),                                   # not a category
    ])


def test_query_string_keeps_or_separators_literal():
    q = acled.AcledClient.query_string({"country": "Israel", "fields": "a|b"},
                                       {"event_type": ["Battles", "Explosions/Remote violence"]})
    assert "event_type=Battles:OR:event_type=Explosions%2FRemote%20violence" in q
    assert "fields=a%7Cb" in q


def _events(tmp_path):
    c = h3.latlng_to_cell(48.0, 37.0, 4)
    lat, lon = h3.cell_to_latlng(c)
    d = tmp_path / "acled" / "raw"
    d.mkdir(parents=True)
    raw_rows(lat, lon).to_csv(d / "Testland_2025.csv", index=False)
    cfg = type("C", (), {"covariates_dir": tmp_path,
                         "__getitem__": lambda self, k: {"covariates": {"ucdp_min_where_prec": 4}}[k]})()
    return acled.load_acled(cfg), c


def test_normalize_categories_and_precision(tmp_path):
    ev, _ = _events(tmp_path)
    assert "X5" not in set(ev["id"])                         # geo_precision 3 → where_prec 5 → dropped
    cats = ev.set_index("id")["category"]
    assert cats["X1"] == "air" and cats["X2"] == "shelling" and cats["X3"] == "ground"
    assert cats["X4"] == "intercept" and cats["X7"] == "other"


def test_acled_features_standoff_share(tmp_path):
    ev, c = _events(tmp_path)
    f = acled.acled_features([h3.str_to_int(c)], ev, date(2025, 6, 13), date(2025, 6, 15), 100, 3).iloc[0]
    assert f["acled_air_100"] == 1 and f["acled_shelling_100"] == 1 and f["acled_ground_100"] == 1
    assert f["acled_intercept_100"] == 1
    assert f["acled_kinetic_100"] == 3                         # intercepts are not kinetic attacks
    assert np.isclose(f["acled_standoff_share_100"], 2 / 3)
    assert f["acled_deaths_100"] == 8


def test_acled_hex_day_and_panel_columns(tmp_path):
    ev, c = _events(tmp_path)
    p = acled.acled_hex_day([h3.str_to_int(c)], ev, date(2025, 6, 13), date(2025, 6, 16), 100, 1)
    s = p.set_index("date")
    assert list(p.columns) == ["hex", "date", "standoff_events", "ground_events", "intercept_events"]
    assert s.loc["2025-06-14", "standoff_events"] == 2 and s.loc["2025-06-15", "standoff_events"] == 2
    assert s.loc["2025-06-16", "standoff_events"] == 0 and s.loc["2025-06-13", "ground_events"] == 0
