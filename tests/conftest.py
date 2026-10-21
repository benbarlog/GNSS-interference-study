"""Synthetic readsb trace fixtures (same 14-element point layout as the real archive; see
gnss/adsblol/trace.py). Synthetic so no ODbL data is committed."""

from __future__ import annotations

import calendar
import gzip
import io
import tarfile
from datetime import date

import orjson
import pytest

from gnss.adsblol.config import load_config

DAY = "2025-06-14"
DAY0 = float(calendar.timegm(date.fromisoformat(DAY).timetuple()))


def det(nic=8, nac_p=10, sil=3, version=2, category="A3", flight="TST123  ", squawk="1000",
        typ="adsb_icao"):
    return {"type": typ, "nic": nic, "nac_p": nac_p, "sil": sil, "version": version,
            "category": category, "flight": flight, "squawk": squawk, "alt_geom": 0}


def point(t, lat, lon, alt=35000, gs=450.0, track=90.0, flags=0, details=None, src="adsb_icao",
          alt_geom=35500, roll=0.0):
    return [t, lat, lon, alt, gs, track, flags, 0, details, src, alt_geom, None, None, roll]


def trace_doc(icao="abc123", points=(), **meta):
    doc = {"icao": icao, "timestamp": DAY0, "version": "readsb test", "trace": list(points)}
    doc.update(meta)
    return doc


def gz(doc) -> bytes:
    return gzip.compress(orjson.dumps(doc))


def straight_flight(n=200, dt=4.0, t0=3000.0, lat0=50.0, lon0=10.0, nic=8, details_every=5):
    """Eastbound at ~450 kt: 450 kt ≈ 0.125 NM/s; at 50°N 1° lon ≈ 38.6 NM."""
    pts = []
    for i in range(n):
        t = t0 + i * dt
        lon = lon0 + (0.125 * i * dt) / 38.57
        d = det(nic=nic) if i % details_every == 0 else None
        pts.append(point(t, lat0, lon, details=d))
    return pts


@pytest.fixture
def cfg():
    return load_config()


def make_tar(members: dict[str, bytes], n_parts: int = 2) -> list[bytes]:
    """A tar like the archive (heatmap + traces), split into n parts."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in [("./README.txt", b"readme"), ("./heatmap/01.bin.ttf", b"\0" * 5000), *members.items()]:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    raw = buf.getvalue()
    step = len(raw) // n_parts + 1
    return [raw[i:i + step] for i in range(0, len(raw), step)]
