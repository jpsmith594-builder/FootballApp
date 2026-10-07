"""
api_server.py
A thin, OPTIONAL web API over engine.py — kept for local development
(uvicorn --reload) and as a fallback you can still deploy if you ever want a
live HTTP API again. It is NOT required in production any more: the actual
compute work (fitting models, building hints, writing results) now lives in
engine.py and is driven on a schedule by refresh_cache.py (see that file and
.github/workflows/refresh.yml), independent of whether this server is
running. The deployed frontend reads Supabase's `cached_fixtures` table
directly instead of calling this API — see frontend/index.html.

Why this split exists: Render's free tier cold-starts on every wake, and
fitting the model for 10 leagues at that moment made users wait on exactly
the request that should have been instant. Decoupling "compute" from
"serve" removes that wait entirely, since the data is already sitting in
Postgres before anyone opens the app.

Run locally:
    uvicorn api_server:app --reload --port 8000
Then visit http://localhost:8000/docs for interactive API docs (free, built
into FastAPI).
"""

import os
import threading
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from data_fetcher import LEAGUES
import engine
import db
import push

# Error tracking (optional): set SENTRY_DSN to get alerted the moment
# something breaks in production, instead of finding out when a friend
# complains the app looks empty. Free tier at sentry.io is plenty for this.
SENTRY_DSN = os.environ.get("SENTRY_DSN")
if SENTRY_DSN:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.1, integrations=[FastApiIntegration()])

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api_server")

REFRESH_HOURS = float(os.environ.get("REFRESH_HOURS", "6"))
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "")

app = FastAPI(title="Matchday Predictor API")

# Allow any frontend to call this — tighten to your actual frontend's domain
# once you know it, e.g. allow_origins=["https://your-app.netlify.app"]
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    # Only runs refresh_all() in-process if you've deliberately chosen to
    # deploy this as a live server again. If refresh_cache.py is already
    # populating Supabase on a GitHub Actions schedule, leave
    # RUN_SCHEDULER_HERE unset — running both would just mean two processes
    # hitting football-data.org's rate limit for no benefit.
    if os.environ.get("RUN_SCHEDULER_HERE", "").lower() in ("1", "true", "yes"):
        threading.Thread(target=engine.refresh_all, daemon=True).start()
        scheduler = BackgroundScheduler()
        scheduler.add_job(engine.refresh_all, "interval", hours=REFRESH_HOURS)
        scheduler.start()
    else:
        log.info("RUN_SCHEDULER_HERE not set — not fitting models in this process. "
                 "Reading whatever's already in engine._cache (empty until you call "
                 "refresh_cache.py, or set RUN_SCHEDULER_HERE=1 for local dev).")


@app.get("/")
def root():
    # This is a JSON API with no visual homepage — visiting the bare URL in a
    # browser is expected to look like "just data," not a 404. The actual
    # frontend (index.html) is what renders a real page, hosted separately.
    return {
        "service": "Matchday Predictor API",
        "note": "Optional in production — the frontend normally reads Supabase directly.",
        "try_these": ["/health", "/leagues", "/predictions/Premier League"],
    }


@app.get("/leagues")
def get_leagues():
    return {"leagues": list(LEAGUES.keys())}


@app.get("/predictions/{league_name}")
def get_predictions(league_name: str):
    if league_name not in LEAGUES:
        raise HTTPException(404, f"Unknown league. Choose from: {list(LEAGUES.keys())}")
    if league_name in engine._cache:
        return engine._cache[league_name]
    # Fall back to the database cache if this process hasn't fitted models
    # itself (the normal case now that refresh_cache.py does that elsewhere).
    rows = db.get_cached_fixtures(league_name)
    if not rows:
        raise HTTPException(503, "Data not ready yet — try again shortly")
    return {"predictions": rows, "fitted_at": None}


@app.get("/parlay-builder")
def get_parlay_builder(limit: int = 15):
    """
    Cross-league board of the highest-confidence hints currently cached,
    ranked descending. See engine.build_parlay_board()'s docstring for why
    no combined odds/payout figure is shown.
    """
    if not engine._cache:
        raise HTTPException(503, "Data not ready yet — try again shortly")
    return engine.build_parlay_board(engine._cache, limit=limit)


@app.get("/track-record/{league_name}")
def get_track_record(league_name: str):
    if league_name not in LEAGUES:
        raise HTTPException(404, f"Unknown league. Choose from: {list(LEAGUES.keys())}")
    rows = db.get_track_record(league_name)
    return {"league": league_name, "summary": db.summarize_track_record(rows), "history": rows}


@app.get("/calibration/{league_name}")
def get_calibration(league_name: str):
    """
    Transparency feature: shows whether the model's stated confidence can
    actually be trusted, using your own accumulated track record (see
    db.get_calibration_curve's docstring for the full explanation). This is
    the kind of honesty a paying customer should be able to check for
    themselves rather than take on faith — surfacing it in the app, not
    just internally, is deliberate.
    """
    if league_name not in LEAGUES:
        raise HTTPException(404, f"Unknown league. Choose from: {list(LEAGUES.keys())}")
    return db.get_calibration_curve(league_name)


def _check_admin(x_admin_secret: str = Header(default="")):
    if not ADMIN_SECRET or x_admin_secret != ADMIN_SECRET:
        raise HTTPException(403, "Invalid or missing admin secret")


@app.post("/admin/team-adjustment")
def set_team_adjustment(league: str, team: str, goal_multiplier: float, note: str = "",
                         x_admin_secret: str = Header(default="")):
    """
    Manually scale a team's expected goals — e.g. goal_multiplier=0.8 for
    'missing their top striker'. Requires the X-Admin-Secret header to match
    the ADMIN_SECRET environment variable, since this directly changes what
    predictions everyone sees. Takes effect on the next refresh (scheduled
    job or manual run of refresh_cache.py), not instantly.
    """
    _check_admin(x_admin_secret)
    if league not in LEAGUES:
        raise HTTPException(404, f"Unknown league. Choose from: {list(LEAGUES.keys())}")
    db.set_team_adjustment(league, team, goal_multiplier, note)
    return {"status": "set", "league": league, "team": team, "goal_multiplier": goal_multiplier}


@app.delete("/admin/team-adjustment")
def clear_team_adjustment(league: str, team: str, x_admin_secret: str = Header(default="")):
    _check_admin(x_admin_secret)
    db.clear_team_adjustment(league, team)
    return {"status": "cleared", "league": league, "team": team}


@app.get("/push/vapid-public-key")
def get_vapid_public_key():
    return {"publicKey": push.VAPID_PUBLIC_KEY}


@app.post("/push/subscribe")
def subscribe_to_push(subscription: dict):
    """Expects the browser's PushSubscription.toJSON() shape:
    {"endpoint": "...", "keys": {"p256dh": "...", "auth": "..."}}
    Kept here for local dev; the deployed frontend now writes subscriptions
    straight into Supabase's push_subscriptions table instead (see
    frontend/index.html and DEPLOY.md's RLS policy for that table)."""
    endpoint = subscription.get("endpoint")
    keys = subscription.get("keys", {})
    if not endpoint or "p256dh" not in keys or "auth" not in keys:
        raise HTTPException(400, "Malformed push subscription")
    db.save_push_subscription(endpoint, keys["p256dh"], keys["auth"])
    return {"status": "subscribed"}


@app.get("/health")
def health():
    return {"status": "ok", "cached_leagues": list(engine._cache.keys())}
