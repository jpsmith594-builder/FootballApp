#!/usr/bin/env python3
"""
refresh_standings.py
Run this on a lighter, separate schedule from refresh_cache.py (GitHub
Actions cron — see .github/workflows/refresh_standings.yml) to refresh just
the league tables (engine.refresh_all_standings() -> db.replace_standings()).

Split out from the main predictions refresh deliberately: a league table
only changes after matches actually finish (once or twice a week per team),
not hourly, so re-fetching it on the same cadence as predictions wastes
GitHub Actions minutes and football-data.org request budget for no benefit.

Needs the same DATABASE_URL and FOOTBALL_DATA_API_KEY environment variables
as refresh_cache.py — nothing else (no odds key, no VAPID keys; this never
touches the model or push notifications).

Usage:
    python refresh_standings.py
Exit code is non-zero only if the whole run crashed outright — a single
league's standings failing (e.g. a transient API hiccup) is logged and
skipped, same as everywhere else in this app, so one bad league doesn't
stop the other nine from updating.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("refresh_standings")


def main():
    import engine  # imported inside main() so a missing env var fails loudly, not at import time
    log.info("Starting scheduled refresh of all league tables...")
    engine.refresh_all_standings()
    log.info("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("refresh_standings.py crashed")
        sys.exit(1)
