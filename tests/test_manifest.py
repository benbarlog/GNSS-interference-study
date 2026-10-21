import pytest

from gnss.adsblol.manifest import InvalidTransition, Manifest


def rows(*dates, size=100):
    return [{"date": d, "repo": "globe_history_2025", "tag": f"v{d}-planes-readsb-prod-0",
             "variant": "prod", "urls": [f"https://x/{d}.tar.aa"], "part_sizes": [size],
             "total_bytes": size, "preferred_tag": None, "selection_note": "preferred"} for d in dates]


@pytest.fixture
def m(tmp_path):
    mf = Manifest(tmp_path / "manifest.sqlite")
    mf.upsert_catalog(rows("2025-06-13", "2025-06-14"))
    yield mf
    mf.close()


def test_happy_path(m):
    d = "2025-06-14"
    for to in ("downloading", "downloaded", "processing", "processed", "verified", "cleaned"):
        m.transition(d, to)
    assert m.get(d)["status"] == "cleaned"
    hist = m.conn.execute("SELECT COUNT(*) FROM history WHERE date=?", (d,)).fetchone()[0]
    assert hist == 6


def test_invalid_transitions(m):
    with pytest.raises(InvalidTransition):
        m.transition("2025-06-14", "processing")      # queued → processing skips download
    m.transition("2025-06-14", "downloading")
    with pytest.raises(InvalidTransition):
        m.transition("2025-06-14", "verified")
    with pytest.raises(InvalidTransition):
        m.transition("2025-06-14", "downloaded", expect="queued")


def test_recover_on_start(m):
    m.transition("2025-06-13", "downloading")
    m.transition("2025-06-14", "downloading")
    m.transition("2025-06-14", "downloaded")
    m.transition("2025-06-14", "processing")
    moves = sorted(m.recover_on_start())
    assert moves == [("2025-06-13", "downloading", "queued"), ("2025-06-14", "processing", "downloaded")]


def test_failed_retry_paths(m):
    m.transition("2025-06-14", "failed", retry_count=1, last_error="x")
    m.transition("2025-06-14", "queued")
    assert m.get("2025-06-14")["retry_count"] == 1


def test_catalog_refresh_does_not_touch_started_days(m):
    m.transition("2025-06-14", "downloading")
    changed = rows("2025-06-13", "2025-06-14", size=999)
    new, _ = m.upsert_catalog(changed)
    assert new == 0
    assert m.get("2025-06-13")["total_bytes"] == 999
    assert m.get("2025-06-14")["total_bytes"] == 100


def test_failed_day_gets_fresh_retries_when_parts_change(m):
    m.transition("2025-06-14", "failed", retry_count=3, last_error="QA: trace count")
    fixed = rows("2025-06-14")
    fixed[0]["urls"] = ["https://x/2025-06-14.tar.aa", "https://x/2025-06-14.tar.ab"]
    m.upsert_catalog(fixed)
    assert m.get("2025-06-14")["retry_count"] == 0
    m.upsert_catalog(fixed)                     # unchanged parts: nothing reset again
    m.update("2025-06-14", retry_count=2)
    m.upsert_catalog(fixed)
    assert m.get("2025-06-14")["retry_count"] == 2


def test_buffer_and_kv(m):
    m.transition("2025-06-14", "downloading")
    m.update("2025-06-14", bytes_done=40)
    m.transition("2025-06-13", "downloading")
    m.transition("2025-06-13", "downloaded")
    assert m.buffer_bytes() == 140
    m.kv_set("gate1_passed", True)
    assert m.kv_get("gate1_passed") is True and m.kv_get("nope", 5) == 5
