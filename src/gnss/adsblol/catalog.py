"""Resolve every archive day to a release, its part URLs and sizes, and upsert into the manifest.

Sources (checked 2026-10-16):
- `PREFERRED_RELEASES.txt` (raw.githubusercontent.com, not rate limited): one line per day, newest
  first. A line is either one `.tar` URL or a comma-separated list of every part (`.tar.aa,.tar.ab,…`).
- GitHub releases API (paginated, ~10 calls per repo): asset sizes for every variant, used for the
  "significantly smaller" rule and for download size checks. Archived repos are cached permanently.
- Fallback when the API is unavailable (rate limited): HEAD probes for sizes and for further parts
  when a line lists only `.tar.aa`.

Selection rule (adsb.lol README): use prod, fall back to staging, skip whichever variant is
significantly smaller; ignore mlatonly. We start from the maintainers' preferred release and switch
only when another prod/staging/test release for the same day is > `significantly_smaller_frac`
larger.
"""

from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

from .http import PoliteClient, RateLimited

RAW_URL = "https://raw.githubusercontent.com/adsblol/globe_history_{year}/main/PREFERRED_RELEASES.txt"
API_URL = "https://api.github.com/repos/adsblol/globe_history_{year}/releases?per_page=100&page={page}"
TAG_RE = re.compile(r"v(\d{4})\.(\d{2})\.(\d{2})-planes-readsb-([a-z]+)-(\d+)")
VARIANT_RANK = {"prod": 0, "staging": 1, "test": 2}
IGNORED_VARIANTS = {"mlatonly"}
SPLIT_RE = re.compile(r"\.tar\.[a-z]{2}$")


def parse_tag(tag: str) -> tuple[str, str] | None:
    m = TAG_RE.search(tag)
    if not m:
        return None
    y, mo, d, variant, _ = m.groups()
    return f"{y}-{mo}-{d}", variant


def tag_from_url(url: str) -> str:
    # .../releases/download/<tag>/<asset>
    return url.rstrip("/").split("/")[-2]


def parse_preferred(text: str, repo: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls = [u.strip() for u in line.split(",") if u.strip()]
        if any(SPLIT_RE.search(u) for u in urls):
            # a few lines (e.g. 2024-09-16) also list a stray partial whole `.tar`; see release_parts
            urls = [u for u in urls if SPLIT_RE.search(u)]
        tag = tag_from_url(urls[0])
        info = parse_tag(tag)
        if not info or info[1] in IGNORED_VARIANTS:
            continue        # e.g. 2026-05-06 has only an mlatonly release: no ADS-B integrity data
        d, variant = info
        out.setdefault(d, {"date": d, "repo": repo, "tag": tag, "variant": variant, "urls": urls})
    return out


def release_parts(assets: list[dict]) -> list[dict]:
    """The assets that make up the day's tar, in order. Four releases (2024-03-22, 2024-04-22,
    2024-09-16, 2026-06-06) carry both split parts and a stray whole `.tar`; for 2024-09-16 that file
    is a self-contained partial archive (8,462 of ~70k traces). When split parts exist, only they
    count."""
    split = sorted((a for a in assets if SPLIT_RE.search(a["name"])), key=lambda a: a["name"])
    if split:
        return split
    return [a for a in assets if a["name"].endswith(".tar")][:1]


def parse_releases(releases: list[dict], repo: str) -> dict[str, list[dict]]:
    """{date: [candidate, ...]} where candidate = {tag, variant, urls, sizes, total}."""
    out: dict[str, list[dict]] = {}
    for rel in releases:
        if rel.get("draft"):
            continue
        info = parse_tag(rel.get("tag_name", ""))
        if not info or info[1] in IGNORED_VARIANTS:
            continue
        assets = release_parts(rel.get("assets", []))
        if not assets:
            continue
        out.setdefault(info[0], []).append({
            "repo": repo, "tag": rel["tag_name"], "variant": info[1],
            "urls": [a["browser_download_url"] for a in assets],
            "sizes": [int(a["size"]) for a in assets],
            "total": sum(int(a["size"]) for a in assets),
        })
    return out


def select_release(preferred: dict | None, candidates: list[dict], frac: float) -> dict | None:
    """Return {repo, tag, variant, urls, part_sizes, total_bytes, preferred_tag, selection_note}."""
    pref_tag = preferred["tag"] if preferred else None
    by_tag = {c["tag"]: c for c in candidates}
    if candidates:
        biggest = max(c["total"] for c in candidates)
        ok = [c for c in candidates if c["total"] >= (1 - frac) * biggest]
        if pref_tag in by_tag and by_tag[pref_tag] in ok:
            chosen, note = by_tag[pref_tag], "preferred"
        else:
            ok.sort(key=lambda c: (VARIANT_RANK.get(c["variant"], 9), -c["total"]))
            chosen = ok[0]
            note = ("preferred release significantly smaller; switched" if pref_tag in by_tag
                    else "rule-selected (no preferred entry)" if pref_tag is None
                    else "preferred tag not in release listing; rule-selected")
        urls, sizes = chosen["urls"], chosen["sizes"]
        if chosen["tag"] == pref_tag:
            # the maintainers' list is authoritative for which files form the tar; the listing
            # only supplies their sizes
            by_name = {u.rsplit("/", 1)[-1]: s for u, s in zip(urls, sizes)}
            urls = preferred["urls"]
            sizes = [by_name.get(u.rsplit("/", 1)[-1]) for u in urls]
        total = sum(sizes) if all(s is not None for s in sizes) else None
        return {"repo": chosen["repo"], "tag": chosen["tag"], "variant": chosen["variant"],
                "urls": urls, "part_sizes": sizes, "total_bytes": total,
                "preferred_tag": pref_tag, "selection_note": note}
    if preferred:
        return {"repo": preferred["repo"], "tag": preferred["tag"], "variant": preferred["variant"],
                "urls": preferred["urls"], "part_sizes": [None] * len(preferred["urls"]),
                "total_bytes": None, "preferred_tag": pref_tag,
                "selection_note": "preferred (sizes unresolved)"}
    return None


def probe_sizes(client: PoliteClient, urls: list[str]) -> tuple[list[str], list[int | None]]:
    """HEAD-probe sizes. A lone `.tar.aa` is extended with `.ab`, `.ac`, … until 404."""
    urls = list(urls)
    if len(urls) == 1 and urls[0].endswith(".tar.aa"):
        base = urls[0][:-2]
        for i in range(1, 26):
            nxt = base + "a" + chr(ord("a") + i)
            if client.meta("HEAD", nxt).status_code == 404:
                break
            urls.append(nxt)
    sizes: list[int | None] = []
    for u in urls:
        r = client.meta("HEAD", u)
        cl = r.headers.get("Content-Length")
        sizes.append(int(cl) if r.ok and cl else None)
    return urls, sizes


def repo_years(cfg: dict, today: date | None = None) -> list[int]:
    today = today or date.today()
    years = set(int(y) for y in cfg["archive"]["repos"])
    years.add(today.year)
    return sorted(years)


def is_archived(year: int, today: date | None = None) -> bool:
    today = today or date.today()
    return year < today.year and today >= date(year + 1, 2, 1)


class Catalog:
    def __init__(self, cfg: dict, cache_dir: Path, client: PoliteClient, events=None):
        self.cfg = cfg
        self.cache_dir = cache_dir
        self.client = client
        self.events = events

    def _say(self, msg: str, important: bool = False) -> None:
        if self.events:
            self.events.info(msg, important)

    def _cache(self, year: int) -> Path:
        p = self.cache_dir / f"globe_history_{year}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def preferred(self, year: int, max_age_s: float) -> dict[str, dict]:
        cache = self._cache(year) / "PREFERRED_RELEASES.txt"
        fresh = cache.exists() and (is_archived(year) or time.time() - cache.stat().st_mtime < max_age_s)
        if not fresh:
            text = self.client.get_text(RAW_URL.format(year=year))
            if text is None:
                self._say(f"globe_history_{year}: no PREFERRED_RELEASES.txt (repo missing?)")
                return {}
            cache.write_text(text, encoding="utf-8")
        return parse_preferred(cache.read_text(encoding="utf-8"), f"globe_history_{year}")

    def releases(self, year: int, max_age_s: float) -> dict[str, list[dict]] | None:
        cache = self._cache(year) / "releases.json"
        meta_p = self._cache(year) / "releases_meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        fresh = cache.exists() and meta.get("complete") and (
            is_archived(year) or time.time() - meta.get("fetched_at", 0) < max_age_s)
        if not fresh:
            rels: list[dict] = []
            page, complete = 1, False
            try:
                while True:
                    r = self.client.meta("GET", API_URL.format(year=year, page=page), api=True)
                    if r.status_code == 404:
                        complete = True
                        break
                    batch = r.json()
                    rels += [{"tag_name": x["tag_name"], "draft": x.get("draft", False),
                              "assets": [{"name": a["name"], "size": a["size"],
                                          "browser_download_url": a["browser_download_url"]}
                                         for a in x.get("assets", [])]} for x in batch]
                    if len(batch) < 100 or 'rel="next"' not in r.headers.get("Link", ""):
                        complete = True
                        break
                    page += 1
            except RateLimited:
                self._say(f"globe_history_{year}: release listing stopped by rate limit", True)
                if not cache.exists():
                    return None
                complete = False
            if rels and (complete or not cache.exists()):
                cache.write_text(json.dumps(rels), encoding="utf-8")
                meta_p.write_text(json.dumps({"fetched_at": time.time(), "complete": complete}))
        if not cache.exists():
            return None
        return parse_releases(json.loads(cache.read_text(encoding="utf-8")), f"globe_history_{year}")

    def build(self, max_age_s: float = 0.0, today: date | None = None) -> list[dict]:
        today = today or date.today()
        earliest = self.cfg["archive"]["earliest_date"]
        last = today - timedelta(days=1)
        frac = float(self.cfg["archive"]["significantly_smaller_frac"])
        preferred: dict[str, dict] = {}
        candidates: dict[str, list[dict]] = {}
        listing_ok = True
        for year in repo_years(self.cfg, today):
            for d, entry in self.preferred(year, max_age_s).items():
                # A date can appear in two repos around New Year; keep the repo matching its year.
                if d not in preferred or entry["repo"].endswith(d[:4]):
                    preferred[d] = entry
            rels = self.releases(year, max_age_s)
            if rels is None:
                listing_ok = False
                continue
            for d, cands in rels.items():
                candidates.setdefault(d, []).extend(cands)
        rows = []
        for d in sorted(set(preferred) | set(candidates)):
            if not (earliest.isoformat() <= d <= last.isoformat()):
                continue
            sel = select_release(preferred.get(d), candidates.get(d, []), frac)
            if sel:
                rows.append({"date": d, **sel})
        switched = sum(1 for r in rows if "switched" in (r["selection_note"] or ""))
        self._say(f"catalog: {len(rows)} days ({rows[0]['date'] if rows else '-'} → "
                  f"{rows[-1]['date'] if rows else '-'}), {switched} switched from preferred, "
                  f"release listing {'complete' if listing_ok else 'INCOMPLETE (sizes resolved at download)'}",
                  True)
        return rows


def run_catalog(cfg: dict, paths, manifest, client: PoliteClient, events=None,
                max_age_s: float = 0.0) -> dict:
    rows = Catalog(cfg, paths.catalog_cache, client, events).build(max_age_s=max_age_s)
    new, changed = manifest.upsert_catalog(rows)
    dropped = manifest.drop_unstarted(keep={r["date"] for r in rows})
    if dropped and events:
        events.info(f"catalog: dropped {dropped} unstarted day(s) no longer in the catalog", True)
    total = sum(r["total_bytes"] or 0 for r in rows)
    unresolved = sum(1 for r in rows if not r["total_bytes"])
    return {"days": len(rows), "new": new, "changed": changed, "bytes": total,
            "unresolved_sizes": unresolved}
