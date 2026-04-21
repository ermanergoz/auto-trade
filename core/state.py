"""Shared mutable state — avoids circular imports between modules."""

from datetime import date
from typing import Optional

shutting_down = False

# Start-of-day equity snapshot, keyed by ET date. The daily-loss-limit
# baseline must be a stable reference point for the session — using live
# MTM equity would let the cap drift down as losses accumulate. Updated
# by scheduler.run_scan_cycle on the first scan of each new ET date.
start_of_day_equity: Optional[float] = None
start_of_day_date: Optional[date] = None
