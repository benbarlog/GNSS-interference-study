from gnss.adsblol.catalog import parse_preferred, parse_releases, parse_tag, select_release

BASE = "https://github.com/adsblol/globe_history_2026/releases/download"

PREFERRED = f"""\
{BASE}/v2026.10.15-planes-readsb-staging-0/v2026.10.15-planes-readsb-staging-0.tar.aa,{BASE}/v2026.10.15-planes-readsb-staging-0/v2026.10.15-planes-readsb-staging-0.tar.ab
{BASE}/v2026.10.14-planes-readsb-prod-0/v2026.10.14-planes-readsb-prod-0.tar
"""


def rel(tag, sizes):
    return {"tag_name": tag, "draft": False, "assets": [
        {"name": f"{tag}.tar.a{chr(97 + i)}", "size": s,
         "browser_download_url": f"{BASE}/{tag}/{tag}.tar.a{chr(97 + i)}"} for i, s in enumerate(sizes)]}


def test_parse_tag():
    assert parse_tag("v2023.02.16-planes-readsb-test-0") == ("2023-02-16", "test")
    assert parse_tag("nonsense") is None


def test_parse_preferred_split_and_single():
    mlat = f"{BASE}/v2026.05.06-planes-readsb-mlatonly-0/v2026.05.06-planes-readsb-mlatonly-0.tar\n"
    p = parse_preferred(PREFERRED + mlat, "globe_history_2026")
    assert "2026-05-06" not in p
    t = "v2024.09.16-planes-readsb-staging-0"
    stray = f"{BASE}/{t}/{t}.tar,{BASE}/{t}/{t}.tar.aa,{BASE}/{t}/{t}.tar.ab\n"
    assert [u[-3:] for u in parse_preferred(stray, "r")["2024-09-16"]["urls"]] == [".aa", ".ab"]
    assert p["2026-10-15"]["variant"] == "staging" and len(p["2026-10-15"]["urls"]) == 2
    assert p["2026-10-14"]["urls"][0].endswith(".tar")


def test_select_keeps_preferred_unless_significantly_smaller():
    pref = parse_preferred(PREFERRED, "globe_history_2026")["2026-10-15"]
    cands = parse_releases([
        rel("v2026.10.15-planes-readsb-staging-0", [100, 100]),
        rel("v2026.10.15-planes-readsb-prod-0", [100, 95]),
        {"tag_name": "v2026.10.15-planes-readsb-mlatonly-0", "assets": [
            {"name": "x.tar", "size": 999, "browser_download_url": "u"}]},
    ], "globe_history_2026")["2026-10-15"]
    assert {c["variant"] for c in cands} == {"staging", "prod"}           # mlatonly ignored
    s = select_release(pref, cands, 0.2)
    assert s["variant"] == "staging" and s["selection_note"] == "preferred" and s["total_bytes"] == 200


def test_select_switches_when_preferred_is_much_smaller():
    pref = parse_preferred(PREFERRED, "globe_history_2026")["2026-10-15"]
    cands = parse_releases([
        rel("v2026.10.15-planes-readsb-staging-0", [100, 10]),
        rel("v2026.10.15-planes-readsb-prod-0", [100, 100]),
    ], "globe_history_2026")["2026-10-15"]
    s = select_release(pref, cands, 0.2)
    assert s["variant"] == "prod" and "switched" in s["selection_note"]


def test_stray_whole_tar_beside_split_parts_is_ignored():
    """v2024.09.16 staging lists x.tar (0.8 GiB) plus x.tar.aa/.ab; only the split parts form the day."""
    tag = "v2026.10.15-planes-readsb-staging-0"
    r = rel(tag, [100, 50])
    r["assets"].insert(0, {"name": f"{tag}.tar", "size": 80,
                           "browser_download_url": f"{BASE}/{tag}/{tag}.tar"})
    cands = parse_releases([r], "globe_history_2026")["2026-10-15"]
    assert [u.rsplit(".", 1)[-1] for u in cands[0]["urls"]] == ["aa", "ab"]
    assert cands[0]["total"] == 150
    pref = parse_preferred(PREFERRED, "globe_history_2026")["2026-10-15"]
    s = select_release(pref, cands, 0.2)
    assert s["urls"] == pref["urls"] and s["part_sizes"] == [100, 50] and s["total_bytes"] == 150


def test_select_without_listing_uses_preferred_urls():
    pref = parse_preferred(PREFERRED, "globe_history_2026")["2026-10-14"]
    s = select_release(pref, [], 0.2)
    assert s["total_bytes"] is None and s["part_sizes"] == [None]
