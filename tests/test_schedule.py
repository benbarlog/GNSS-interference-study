from datetime import date, timedelta

from gnss.adsblol.schedule import assign_tiers, load_windows


def days(a, b):
    a, b = date.fromisoformat(a), date.fromisoformat(b)
    return [(a + timedelta(i)).isoformat() for i in range((b - a).days + 1)]


def test_tiers_cover_every_day_once(cfg):
    w = load_windows(cfg)
    cat = days("2023-02-16", "2026-10-15")
    out = assign_tiers(cat, w, controls=[date(2026, 1, 5)], spikes=[date(2024, 2, 20)],
                       p2_range=(date(2026, 7, 17), date(2026, 10, 15)), earliest=date(2023, 2, 16))
    assert sorted(d for d, *_ in out) == cat                       # every day exactly once
    tier = {d: t for d, t, *_ in out}
    assert tier["2025-06-14"] == "P0"
    assert tier["2024-10-01"] == "P1a" and tier["2026-01-05"] == "P1a"
    assert tier["2025-06-01"] == "P1b" and tier["2026-07-16"] == "P1b"   # J ends where P2 begins
    assert tier["2026-07-17"] == "P2" and tier["2026-10-15"] == "P2"
    assert tier["2024-02-20"] == "P3a"
    # priorities are strictly increasing in tier order and unique
    pr = [p for _, _, p, _, _ in out]
    assert pr == sorted(pr) and len(set(pr)) == len(pr)
    retain = {d for d, _, _, r, _ in out if r}
    assert retain == {d for d, t, *_ in out if t in ("P0", "P1a")}


def test_p3_parity_is_uniform_and_newest_first(cfg):
    w = load_windows(cfg)
    cat = days("2023-02-16", "2026-10-15")
    out = assign_tiers(cat, w, p2_range=(date(2026, 7, 17), date(2026, 10, 15)),
                       earliest=date(2023, 2, 16))
    p3a = [d for d, t, *_ in out if t == "P3a"]
    p3b = [d for d, t, *_ in out if t == "P3b"]
    assert p3a == sorted(p3a, reverse=True) and p3b == sorted(p3b, reverse=True)
    anchor = date(2023, 2, 16)
    assert all((date.fromisoformat(d) - anchor).days % 2 == 0 for d in p3a)
    assert all((date.fromisoformat(d) - anchor).days % 2 == 1 for d in p3b)
    weekdays = {date.fromisoformat(d).weekday() for d in p3a}
    assert weekdays == set(range(7))


def test_days_after_frozen_p2_are_p4(cfg):
    w = load_windows(cfg)
    cat = days("2026-07-01", "2026-10-20")
    out = assign_tiers(cat, w, p2_range=(date(2026, 7, 17), date(2026, 10, 15)))
    tier = {d: t for d, t, *_ in out}
    assert tier["2026-10-16"] == "P4" and tier["2026-10-20"] == "P4"
