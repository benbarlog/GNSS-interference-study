"""Load `config/adsblol.yaml`, resolve data paths, and derive the config hash and User-Agent."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from . import PIPELINE_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "config" / "adsblol.yaml"

# Process keys that only tune speed/RAM; everything else in `process` and `exceptions` feeds the hash.
RUNTIME_PROCESS_KEYS = {"workers", "batch_traces", "max_inflight_batches", "maxtasksperchild"}
MAX_CONNECTIONS_CEILING = 2


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = cfg
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    dl = cfg["download"]
    dl["max_connections"] = max(1, min(int(dl.get("max_connections", 2)), MAX_CONNECTIONS_CEILING))
    if isinstance(cfg["archive"]["earliest_date"], str):
        cfg["archive"]["earliest_date"] = date.fromisoformat(cfg["archive"]["earliest_date"])
    return cfg


def output_config(cfg: dict) -> dict:
    proc = {k: v for k, v in cfg["process"].items() if k not in RUNTIME_PROCESS_KEYS}
    return {"process": proc, "exceptions": cfg["exceptions"], "pipeline_version": PIPELINE_VERSION}


def config_hash(cfg: dict) -> str:
    blob = json.dumps(output_config(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def user_agent(cfg: dict) -> str:
    return (f"gnss-interference-hotspots/{PIPELINE_VERSION} (student research project; "
            f"+{cfg['repo_url']}; contact: {cfg['contact_email']})")


def windows_path(cfg: dict) -> Path:
    p = Path(cfg["schedule"]["windows_file"])
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def from_config(cls, cfg: dict) -> Paths:
        return cls(Path(cfg["data_root"]))

    @property
    def gpsjam(self) -> Path: return self.root / "gpsjam"
    @property
    def adsblol(self) -> Path: return self.root / "adsblol"
    @property
    def scratch(self) -> Path: return self.adsblol / "scratch"
    @property
    def raw_retained(self) -> Path: return self.adsblol / "raw_retained"
    @property
    def tmp(self) -> Path: return self.adsblol / "_tmp"
    @property
    def tier1(self) -> Path: return self.adsblol / "tier1_aircraft_hex_hour"
    @property
    def aircraft_dim(self) -> Path: return self.adsblol / "aircraft_dim"
    @property
    def exceptions(self) -> Path: return self.adsblol / "exceptions"
    @property
    def day_qa(self) -> Path: return self.adsblol / "day_qa"
    @property
    def tier2(self) -> Path: return self.adsblol / "tier2"
    @property
    def state(self) -> Path: return self.root / "state"
    @property
    def manifest(self) -> Path: return self.state / "manifest.sqlite"
    @property
    def heartbeat(self) -> Path: return self.state / "heartbeat.json"
    @property
    def token_flag(self) -> Path: return self.state / "NEEDS_GITHUB_TOKEN.txt"
    @property
    def catalog_cache(self) -> Path: return self.state / "catalog_cache"
    @property
    def logs(self) -> Path: return self.root / "logs"
    @property
    def config_copy(self) -> Path: return self.root / "config" / "adsblol.yaml"

    def table_dir(self, table: str) -> Path:
        return {"tier1": self.tier1, "aircraft_dim": self.aircraft_dim,
                "exceptions": self.exceptions}[table]

    def ensure(self) -> None:
        for p in (self.gpsjam, self.scratch, self.raw_retained, self.tmp, self.tier1,
                  self.aircraft_dim, self.exceptions, self.day_qa, self.tier2, self.state,
                  self.catalog_cache, self.logs, self.config_copy.parent):
            p.mkdir(parents=True, exist_ok=True)


def write_config_copy(cfg: dict, paths: Paths) -> None:
    clean = copy.deepcopy({k: v for k, v in cfg.items() if not k.startswith("_")})
    clean["_pipeline_version"] = PIPELINE_VERSION
    clean["_config_hash"] = config_hash(cfg)
    tmp = paths.config_copy.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(clean, sort_keys=False, default_flow_style=None), encoding="utf-8")
    os.replace(tmp, paths.config_copy)
