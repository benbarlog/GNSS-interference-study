"""Download GPSJAM daily files.

Default sample for the proposal: every pre-invasion day (2022-02-14 .. 02-23),
one day per week across the full record, and every day of the last 90 days.

    python scripts/download_gpsjam.py                 # default sample
    python scripts/download_gpsjam.py 2025-01-01 2025-03-31   # explicit daily range
"""

import sys
from datetime import date, timedelta

from gnss.ingest_gpsjam import EARLIEST, date_range, download_dates


def main():
    if len(sys.argv) == 3:
        dates = date_range(date.fromisoformat(sys.argv[1]), date.fromisoformat(sys.argv[2]))
    else:
        yesterday = date.today() - timedelta(days=1)
        dates = set(date_range(EARLIEST, date(2022, 2, 23)))
        dates |= set(date_range(EARLIEST, yesterday, step_days=7))
        dates |= set(date_range(yesterday - timedelta(days=89), yesterday))
        dates = sorted(dates)
    print(f"{len(dates)} dates")
    paths = download_dates(dates)
    print(f"done: {len(paths)} files")


if __name__ == "__main__":
    main()
