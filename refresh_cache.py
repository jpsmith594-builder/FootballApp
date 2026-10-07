#!/usr/bin/env python3
"""
refresh_cache.py
Run this on a schedule (GitHub Actions cron — see
.github/workflows/refresh.yml — or any machine's own cron) to do the heavy
lifting: fetch results/fixtures, fit the Dixon-Coles + Elo models for all 10
leagues, build hints and value bets, and write the result straight into
Supabase's `cached_fixtures` table (engine.refresh_league() does the actual
write, via db.replace_cached_fixtures()).

Deliberately NOT run inside the web server's request path or startup
anymore. The whole point of this script is that it runs somewhere else,
on its own schedule, so nobody opening the app is ever the one waiting for
a model fit — the data's already there.

Needs the same environment variables api_server.py used to:
DATABASE_URL, FOOTBALL_DATA_API_KEY, ODDS_API_KEY (optional),
VAPID_PRIVATE_KEY / VAPID_PUBLIC_KEY (optional, for push notifications).

Usage:
    python refresh_cache.py
Exit code is non-zero only if the whole run crashed outright — a single
league failing (e.g. a transient API hiccup) is logged and skipped, same
as it always was inside the web server, so one bad league doesn't stop the
other nine from updating.
"""

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("refresh_cache")


def main():
    import engine  # imported inside main() so a missing env var fails loudly, not at import time
    log.info("Starting scheduled refresh of all leagues...")
    engine.refresh_all()
    cached = sum(1 for v in engine._cache.values() if v.get("predictions"))
    log.info(f"Done. {cached}/{len(engine._cache)} leagues produced at least one cached fixture.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.exception("refresh_cache.py crashed")
        sys.exit(1)
