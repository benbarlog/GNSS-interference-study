"""Load `config/analysis.yaml` plus the regions/windows from `schedule/windows.yaml`."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import yaml

from gnss.adsblol.config import REPO_ROOT, Paths, load_config, windows_path

DEFAULT_ANALYSIS_CONFIG = REPO_ROOT / "config" / "analysis.yaml"


def _d(x) -> date | None:
    if x is None or isinstance(x, date):
        return x
    return date.fromisoformat(str(x))


def _repo_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass
class Window:
    id: str
    name: str
    start: date
    end: date
    regions: list[str]
    peak: date | None = None
    phases: list[dict] = field(default_factory=list)

    def days(self) -> list[date]:
        return [self.start + timedelta(days=i) for i in range((self.end - self.start).days + 1)]


@dataclass
class AnalysisConfig:
    raw: dict
    pipeline: dict           # adsblol.yaml
    regions: dict[str, list[float] | None]
    windows: dict[str, Window]
    paths: Paths

    @property
    def covariates_dir(self) -> Path: return _repo_path(self.raw["covariates_dir"])
    @property
    def naturalearth(self) -> Path: return _repo_path(self.raw["naturalearth"])
    @property
    def outputs_dir(self) -> Path: return _repo_path(self.raw["outputs_dir"])
    @property
    def figures_dir(self) -> Path: return _repo_path(self.raw["figures_dir"])

    def gpsjam_dir(self) -> Path:
        """Daily GPSJAM on the desktop (D:), else the repo sample on the laptop."""
        d = self.paths.gpsjam
        if d.exists() and any(d.glob("*-h3_4.csv")):
            return d
        return _repo_path(self.raw["fallback_gpsjam_dir"])

    @property
    def has_adsblol(self) -> bool:
        """True when Tier 1 parquet is reachable (desktop D:, or the USB drive on the laptop)."""
        return any(self.paths.tier1.glob("date=*/*.parquet")) if self.paths.tier1.exists() else False

    def __getitem__(self, k):
        return self.raw[k]


def find_data_root(configured: str) -> str | None:
    """The gnss-data folder: the configured path if it exists, else GNSS_DATA_ROOT (environment or
    .env), else the first drive letter with a `gnss-data/adsblol` folder (the USB drive can mount
    under a different letter on the laptop). None when nothing is found."""
    import os
    import string
    if Path(configured).exists():
        return configured
    env = os.environ.get("GNSS_DATA_ROOT")
    dot = REPO_ROOT / ".env"
    if not env and dot.exists():
        for line in dot.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("GNSS_DATA_ROOT="):
                env = line.split("=", 1)[1].strip().strip('"').strip("'")
    if env and Path(env).exists():
        return env
    if os.name == "nt":
        for letter in string.ascii_uppercase[2:]:          # skip A:, B:
            cand = Path(f"{letter}:/gnss-data")
            try:
                if (cand / "adsblol").is_dir():
                    return cand.as_posix()
            except OSError:                                 # empty card readers etc.
                continue
    return None


def load_analysis_config(path: str | Path | None = None, pipeline_cfg: dict | None = None,
                         data_root: str | None = None) -> AnalysisConfig:
    with open(path or DEFAULT_ANALYSIS_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    pcfg = pipeline_cfg or load_config(overrides={"data_root": data_root} if data_root else None)
    if not data_root and pipeline_cfg is None:
        found = find_data_root(pcfg["data_root"])
        if found:
            pcfg["data_root"] = found
        raw["_data_root_found"] = found is not None
    with open(windows_path(pcfg), encoding="utf-8") as f:
        wy = yaml.safe_load(f)
    regions = dict(wy["regions"])
    regions.update(raw.get("extra_regions") or {})
    windows = {}
    for w in wy["windows"]:
        windows[w["id"]] = Window(id=w["id"], name=w["name"], start=_d(w["start"]), end=_d(w["end"]),
                                  regions=list(w["regions"]), peak=_d(w.get("peak")),
                                  phases=w.get("phases") or [])
    # P2 = the recent daily block (plan §3): starts the day after window J ends.
    j_end = windows["J"].end if "J" in windows else date(2026, 7, 16)
    p2_start = j_end + timedelta(days=1)
    windows["P2"] = Window(id="P2", name="Recent 91 days (P2)", start=p2_start,
                           end=p2_start + timedelta(days=int(wy["recent"]["days"]) - 1),
                           regions=list(raw["model_regions"]))
    for b in raw["covariates"]["belligerents"]:
        b["start"], b["end"] = _d(b["start"]), _d(b.get("end"))
    return AnalysisConfig(raw=raw, pipeline=pcfg, regions=regions, windows=windows,
                          paths=Paths.from_config(pcfg))
