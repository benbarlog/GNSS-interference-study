"""Read-only status of the adsb.lol pipeline, safe to run from a second terminal (plan §9.3).

Reads D:/gnss-data/state/heartbeat.json and the manifest (read-only); never touches the running
process.

    python scripts/adsblol_status.py            # one snapshot
    python scripts/adsblol_status.py --watch    # refresh every 30 s (Ctrl+C to quit)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gnss.adsblol.config import Paths, load_config
from gnss.adsblol.manifest import DONE, STATES, TIERS, Manifest

RED, YELLOW, GREEN, DIM, RESET = "\033[91m", "\033[93m", "\033[92m", "\033[2m", "\033[0m"


def dir_size(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def report(paths: Paths, sizes: bool) -> list[str]:
    out: list[str] = []
    if paths.token_flag.exists():
        out.append(f"{RED}TOKEN NEEDED — see {paths.token_flag}{RESET}")
    hb = None
    if paths.heartbeat.exists():
        try:
            hb = json.loads(paths.heartbeat.read_text(encoding="utf-8"))
        except Exception:
            hb = None
    if hb is None:
        out.append(f"{RED}no heartbeat.json — the pipeline has not run yet{RESET}")
    else:
        age = time.time() - hb["ts"]
        if age > 300:
            out.append(f"{RED}PIPELINE NOT RUNNING — last heartbeat {age / 60:.0f} min ago{RESET}")
        else:
            out.append(f"{GREEN}running{RESET} {DIM}(heartbeat {age:.0f} s ago){RESET}")
        out += hb.get("lines", [])
    if not paths.manifest.exists():
        return out
    m = Manifest(paths.manifest, readonly=True)
    try:
        counts = m.counts()
        out.append("")
        hdr = f" {'tier':5}" + "".join(f"{s[:10]:>12}" for s in STATES) + f"{'total':>9}"
        out.append(hdr)
        for t in [*TIERS, "unscheduled"]:
            c = counts.get(t)
            if not c:
                continue
            out.append(f" {t:5}" + "".join(f"{c.get(s, 0):>12}" for s in STATES) + f"{sum(c.values()):>9}")
        fails = m.days(statuses=["failed"], order="updated_at DESC")[:10]
        if fails:
            out.append(f"\n {RED}failed days{RESET}:")
            for r in fails:
                out.append(f"   {r['date']} [{r['tier']}] retries {r['retry_count']}: {r['last_error']}")
        done = m.scalar(f"SELECT COUNT(*) FROM days WHERE status IN ({','.join('?' * len(DONE))})", DONE)
        last = m.days(statuses=list(DONE), order="t_verified DESC")[:5]
        if last:
            out.append(f"\n last verified ({done} total): " + ", ".join(
                f"{r['date']} ({(r['process_sec'] or 0) / 60:.1f} min)" for r in last))
        gates = [f"gate{n} {'passed ' + str(m.kv_get(f'gate{n}_passed_at', '')) if m.kv_get(f'gate{n}_passed') else 'not passed'}"
                 for n in (1, 2)]
        out.append(" " + " · ".join(gates))
    finally:
        m.close()
    if sizes:
        out.append("\n disk usage:")
        for name in ("scratch", "raw_retained", "tier1_aircraft_hex_hour", "aircraft_dim", "exceptions",
                     "tier2"):
            out.append(f"   {name:24} {dir_size(paths.adsblol / name) / 2**30:9.2f} GiB")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--watch", nargs="?", const=30, type=float, metavar="SECONDS")
    ap.add_argument("--sizes", action="store_true", help="also report disk usage per output (slower)")
    args = ap.parse_args()
    if sys.platform == "win32":
        os.system("")          # enable ANSI colors in the Windows console
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    paths = Paths.from_config(load_config(args.config))
    while True:
        lines = report(paths, args.sizes)
        if args.watch:
            print("\033[2J\033[H", end="")
        print("\n".join(lines), flush=True)
        if not args.watch:
            return 0
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
