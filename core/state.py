"""Shared mutable state — avoids circular imports between modules."""

import threading
from datetime import date
from typing import Optional

# threading.Event gives us documented atomic set/is_set semantics across
# threads. A bare bool relies on CPython's GIL to make assignment atomic
# and doesn't guarantee visibility — a signal handler setting the flag
# may not be observed immediately by an ib_insync asyncio callback.
shutting_down: threading.Event = threading.Event()

# Start-of-day equity snapshot, keyed by ET date. The daily-loss-limit
# baseline must be a stable reference point for the session — using live
# MTM equity would let the cap drift down as losses accumulate. Updated
# by scheduler.run_scan_cycle on the first scan of each new ET date.
start_of_day_equity: Optional[float] = None
start_of_day_date: Optional[date] = None
