"""Parquet schemas for the permanent outputs (plan §8). Zstd-compressed, hive-partitioned by date
(`date=YYYY-MM-DD/`), so `date` is a partition column rather than a stored one.

Deviations from the plan's §8 lists, all additive:
- tier1 `s_alt`: seconds with a valid barometric altitude, so `alt_sum_s / s_alt` is a correct mean.
- dim `n_pts_day`: points kept for the aircraft that day.
- exceptions `on_ground`: context points can be on the ground.
- exceptions `reason` is a bitmask (REASON_* below), not a string.
"""

import pyarrow as pa

NIC_BINS = 12   # values 0..11, plus an "unknown" bin

REASON_NIC_LOW = 1       # ADS-B airborne point with NIC < nic_bad_lt
REASON_NIC_UNKNOWN = 2   # ADS-B airborne point with no NIC within the staleness cap
REASON_JUMP = 4          # implied speed from the previous ADS-B point > jump_kt
REASON_CONTEXT = 8       # within context_s of a degraded point, not itself degraded

_meta = [pa.field("pipeline_version", pa.string()), pa.field("config_hash", pa.string())]

TIER1 = pa.schema([
    pa.field("hour_utc", pa.uint8()),
    pa.field("icao", pa.string()),
    pa.field("hex5", pa.uint64()),
    pa.field("callsign", pa.string()),
    pa.field("n_pts", pa.uint16()),
    pa.field("n_pts_adsb", pa.uint16()),
    pa.field("n_pts_mlat", pa.uint16()),
    pa.field("n_pts_other", pa.uint16()),
    pa.field("s_obs", pa.float32()),
    pa.field("s_adsb", pa.float32()),
    *[pa.field(f"nic_s_{i}", pa.float32()) for i in range(NIC_BINS)],
    pa.field("nic_s_unk", pa.float32()),
    *[pa.field(f"nacp_s_{i}", pa.float32()) for i in range(NIC_BINS)],
    pa.field("nacp_s_unk", pa.float32()),
    pa.field("sil_min", pa.uint8()),
    pa.field("version", pa.uint8()),
    pa.field("category", pa.string()),
    pa.field("s_high_roll", pa.float32()),
    pa.field("max_abs_roll", pa.float32()),
    pa.field("alt_min", pa.int32()),
    pa.field("alt_max", pa.int32()),
    pa.field("alt_sum_s", pa.float64()),
    pa.field("s_alt", pa.float32()),
    pa.field("s_ground", pa.float32()),
    pa.field("gap_n", pa.uint16()),
    pa.field("gap_abs_sum", pa.float32()),
    pa.field("gap_abs_max", pa.float32()),
    pa.field("n_jumps", pa.uint16()),
    pa.field("max_implied_kt", pa.float32()),
    pa.field("spd_mismatch_max", pa.float32()),
    pa.field("first_ts", pa.uint32()),
    pa.field("last_ts", pa.uint32()),
    pa.field("sq_7500", pa.bool_()),
    pa.field("sq_7600", pa.bool_()),
    pa.field("sq_7700", pa.bool_()),
    *_meta,
])

AIRCRAFT_DIM = pa.schema([
    pa.field("icao", pa.string()),
    pa.field("r", pa.string()),
    pa.field("t", pa.string()),
    pa.field("desc", pa.string()),
    pa.field("ownOp", pa.string()),
    pa.field("year", pa.uint16()),
    pa.field("dbFlags", pa.uint16()),
    pa.field("s_adsb_day", pa.float32()),
    pa.field("nic_lt7_s_day", pa.float32()),
    pa.field("n_pts_day", pa.uint32()),
    *_meta,
])

EXCEPTIONS = pa.schema([
    pa.field("ts", pa.float64()),
    pa.field("icao", pa.string()),
    pa.field("lat", pa.float32()),
    pa.field("lon", pa.float32()),
    pa.field("alt_baro", pa.int32()),
    pa.field("alt_geom", pa.int32()),
    pa.field("gs", pa.float32()),
    pa.field("track", pa.float32()),
    pa.field("roll", pa.float32()),
    pa.field("src", pa.string()),
    pa.field("nic", pa.uint8()),
    pa.field("nac_p", pa.uint8()),
    pa.field("implied_kt", pa.float32()),
    pa.field("segment_id", pa.uint16()),
    pa.field("reason", pa.uint8()),
    pa.field("on_ground", pa.bool_()),
    *_meta,
])

SCHEMAS = {"tier1": TIER1, "aircraft_dim": AIRCRAFT_DIM, "exceptions": EXCEPTIONS}
