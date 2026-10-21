"""Tier 2: DuckDB views and materializations over Tier 1 (plan §9). Derived, rebuildable, never
authoritative. Thresholds are parameters, never stored (plan §6.3).

    from gnss.adsblol.tier2 import connect, hex_day
    con = connect()
    df = con.sql(hex_day(res=4, nic_lt=7)).df()

or rebuild the standard set:  python -m gnss.adsblol.tier2 [--res 2 3 4 5] [--nic-lt 7]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from h3.api import basic_int as h3i

from .config import Paths, load_config
from .schemas import NIC_BINS

ALT_BANDS = "CASE WHEN alt_mean IS NULL THEN 'unknown' WHEN alt_mean < 10000 THEN '<10k' " \
            "WHEN alt_mean < 25000 THEN '10-25k' ELSE '>=25k' END"


def _glob(p: Path) -> str:
    return (p / "*" / "*.parquet").as_posix()


def connect(paths: Paths | None = None, threads: int = 4) -> duckdb.DuckDBPyConnection:
    paths = paths or Paths.from_config(load_config())
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    for name, d in (("tier1", paths.tier1), ("aircraft_dim_raw", paths.aircraft_dim),
                    ("exceptions", paths.exceptions)):
        if any(d.glob("date=*/*.parquet")):
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM "
                        f"read_parquet('{_glob(d)}', hive_partitioning=true, union_by_name=true)")
    if any(paths.aircraft_dim.glob("date=*/*.parquet")):
        # one row per icao: the most recent non-null registry fields
        con.execute("""
            CREATE OR REPLACE VIEW aircraft_dim AS
            SELECT icao,
                   arg_max(r, date) FILTER (WHERE r IS NOT NULL) AS r,
                   arg_max(t, date) FILTER (WHERE t IS NOT NULL) AS t,
                   arg_max("desc", date) FILTER (WHERE "desc" IS NOT NULL) AS "desc",
                   arg_max(ownOp, date) FILTER (WHERE ownOp IS NOT NULL) AS ownOp,
                   arg_max(year, date) FILTER (WHERE year IS NOT NULL) AS year,
                   bit_or(dbFlags) AS dbFlags,
                   COUNT(*) AS days_seen
            FROM aircraft_dim_raw GROUP BY icao""")
    lut = paths.tier2 / "hex_lookup.parquet"
    if lut.exists():
        con.execute(f"CREATE OR REPLACE VIEW hex_lookup AS SELECT * FROM read_parquet('{lut.as_posix()}')")
    return con


def build_hex_lookup(con: duckdb.DuckDBPyConnection, paths: Paths) -> int:
    """hex5 → hex4/hex3/hex2 parents and centroid; incremental over new hex5 cells."""
    lut = paths.tier2 / "hex_lookup.parquet"
    paths.tier2.mkdir(parents=True, exist_ok=True)
    have = set(pq.read_table(lut, columns=["hex5"]).column(0).to_pylist()) if lut.exists() else set()
    cells = [c for c in con.sql("SELECT DISTINCT hex5 FROM tier1").fetchnumpy()["hex5"].tolist()
             if c not in have]
    if not cells:
        return 0
    rows = {"hex5": cells, "hex4": [], "hex3": [], "hex2": [], "lat": [], "lon": []}
    for c in cells:
        for r in (4, 3, 2):
            rows[f"hex{r}"].append(h3i.cell_to_parent(c, r))
        la, lo = h3i.cell_to_latlng(c)
        rows["lat"].append(la)
        rows["lon"].append(lo)
    new = pa.table({k: pa.array(v, pa.uint64() if k.startswith("hex") else pa.float64())
                    for k, v in rows.items()})
    if lut.exists():
        new = pa.concat_tables([pq.read_table(lut), new])
    pq.write_table(new, lut, compression="zstd")
    con.execute(f"CREATE OR REPLACE VIEW hex_lookup AS SELECT * FROM read_parquet('{lut.as_posix()}')")
    return len(cells)


def _bad_expr(nic_lt: int) -> str:
    return " + ".join(f"nic_s_{i}" for i in range(nic_lt)) or "0"


def _known_expr() -> str:
    return " + ".join(f"nic_s_{i}" for i in range(NIC_BINS))


def hex_day(res: int = 4, nic_lt: int = 7, aircraft_bad_share: float = 0.5,
            where: str = "TRUE") -> str:
    """Hex × day at H3 `res` (2–5): time-weighted bad share and GPSJAM-style aircraft counts."""
    hx = "t.hex5" if res == 5 else f"l.hex{res}"
    join = "" if res == 5 else "JOIN hex_lookup l ON l.hex5 = t.hex5"
    return f"""
    WITH a AS (
        SELECT t.date, {hx} AS hex, t.icao,
               SUM({_bad_expr(nic_lt)}) AS s_bad, SUM({_known_expr()}) AS s_known,
               SUM(t.s_adsb) AS s_adsb, SUM(t.s_obs) AS s_obs, SUM(t.n_jumps) AS n_jumps
        FROM tier1 t {join}
        WHERE {where}
        GROUP BY ALL
    )
    SELECT date, hex,
           COUNT(*) FILTER (WHERE s_known > 0) AS n_aircraft,
           COUNT(*) FILTER (WHERE s_known > 0 AND s_bad / s_known >= {aircraft_bad_share}) AS n_bad_aircraft,
           SUM(s_bad) AS s_bad, SUM(s_known) AS s_known,
           SUM(s_bad) / NULLIF(SUM(s_known), 0) AS bad_share_time,
           n_bad_aircraft / NULLIF(n_aircraft, 0) AS bad_share_aircraft,
           SUM(s_adsb) AS s_adsb, SUM(s_obs) AS s_obs, SUM(n_jumps) AS n_jumps
    FROM a GROUP BY date, hex"""


def aircraft_day(nic_lt: int = 7) -> str:
    return f"""
    SELECT t.date, t.icao, any_value(t.callsign) AS callsign, MAX(t.category) AS category,
           SUM(t.s_obs) AS s_obs, SUM(t.s_adsb) AS s_adsb,
           SUM({_bad_expr(nic_lt)}) AS s_bad, SUM({_known_expr()}) AS s_known,
           SUM(t.n_jumps) AS n_jumps, MAX(t.max_implied_kt) AS max_implied_kt,
           MIN(t.first_ts) AS first_ts, MAX(t.last_ts) AS last_ts,
           COUNT(DISTINCT t.hex5) AS n_hex5
    FROM tier1 t GROUP BY t.date, t.icao"""


def rollup(period: str = "week", res: int = 4, nic_lt: int = 7,
           by: tuple[str, ...] = ()) -> str:
    """Hex × period (day|week|month) bucketed by any of: category, type, alt_band, hour."""
    hx = "t.hex5" if res == 5 else f"l.hex{res}"
    join = "" if res == 5 else "JOIN hex_lookup l ON l.hex5 = t.hex5"
    dims = {"category": "t.category", "type": "d.t", "hour": "t.hour_utc",
            "alt_band": ALT_BANDS.replace("alt_mean", "(t.alt_sum_s / NULLIF(t.s_alt, 0))")}
    extra = [f"{dims[b]} AS {b}" for b in by]
    djoin = "LEFT JOIN aircraft_dim d USING (icao)" if "type" in by else ""
    sel = ", ".join([f"date_trunc('{period}', t.date::DATE) AS period", f"{hx} AS hex", *extra])
    return f"""
    SELECT {sel},
           COUNT(DISTINCT t.icao) AS n_aircraft,
           SUM({_bad_expr(nic_lt)}) AS s_bad, SUM({_known_expr()}) AS s_known,
           SUM({_bad_expr(nic_lt)}) / NULLIF(SUM({_known_expr()}), 0) AS bad_share_time,
           SUM(t.s_obs) AS s_obs, SUM(t.n_jumps) AS n_jumps
    FROM tier1 t {join} {djoin}
    GROUP BY ALL"""


def materialize(con: duckdb.DuckDBPyConnection, sql: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    con.execute(f"COPY ({sql}) TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    tmp.replace(out)
    return out


def rebuild(paths: Paths | None = None, resolutions=(2, 3, 4, 5), nic_lt: int = 7) -> list[Path]:
    paths = paths or Paths.from_config(load_config())
    con = connect(paths)
    build_hex_lookup(con, paths)
    outs = []
    for r in resolutions:
        outs.append(materialize(con, hex_day(r, nic_lt), paths.tier2 / f"hex_day_res{r}_nic{nic_lt}.parquet"))
    outs.append(materialize(con, aircraft_day(nic_lt), paths.tier2 / f"aircraft_day_nic{nic_lt}.parquet"))
    for period in ("week", "month"):
        outs.append(materialize(con, rollup(period, 4, nic_lt),
                                paths.tier2 / f"hex_{period}_res4_nic{nic_lt}.parquet"))
    return outs


def main() -> None:
    ap = argparse.ArgumentParser(description="Rebuild the standard Tier 2 tables.")
    ap.add_argument("--res", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--nic-lt", type=int, default=7)
    a = ap.parse_args()
    for p in rebuild(resolutions=a.res, nic_lt=a.nic_lt):
        print(p)


if __name__ == "__main__":
    main()
