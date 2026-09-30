"""
api_server.py
Wraps data_fetcher + poisson_model + value_bet in a small web API, so any
frontend (a hosted web page, FlutterFlow, a phone app) can call it over HTTP
instead of needing Python itself.

Fitting the Dixon-Coles model on every request would be slow and would burn
through your football-data.org rate limit fast, so results are cached and
refit on a schedule (default: every 6 hours) rather than per-request.

Run locally:
    uvicorn api_server:app --reload --port 8000
Then visit http://localhost:8000/docs for interactive API docs (free, built
into FastAPI).

Deploy: see DEPLOY.md for the free Render.com walkthrough.
"""

import os
import threading
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from data_fetcher import FootballDataClient, LEAGUES
from poisson_model import DixonColesModel
from value_bet import fetch_odds, find_value_bets, ODDS_SPORT_KEYS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api_server")

REFRESH_HOURS = float(os.environ.get("REFRESH_HOURS", "6"))
ODDS_ENABLED = bool(os.environ.get("ODDS_API_KEY"))

app = FastAPI(title="Matchday Predictor API")

# Allow any frontend to call this — tighten to your actual frontend's domain
# once you know it, e.g. allow_origins=["https://your-app.netlify.app"]
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
)

# In-memory cache: {league_name: {"model": ..., "fixtures": [...], "fitted_at": ...}}
_cache: dict = {}


def refresh_league(league_name: str):
    league_code = LEAGUES[league_name]
    log.info(f"Refreshing {league_name}...")
    try:
        client = FootballDataClient()
        results = client.get_results(league_code, days_back=365)
        model = DixonColesModel()
        model.fit(results)
        fixtures = client.get_upcoming_fixtures(league_code, days_ahead=10)

        odds_by_fixture = {}
        if ODDS_ENABLED:
            try:
                quotes = fetch_odds(league_name)
                odds_by_fixture = {(q.home_team, q.away_team): q for q in quotes}
            except Exception as e:
                log.warning(f"Odds fetch failed for {league_name}: {e}")

        predictions = []
        for fx in fixtures:
            try:
                pred = model.predict(fx.home, fx.away)
            except ValueError:
                continue  # team with no fitted history (e.g. newly promoted)
            entry = {
                "home": fx.home, "away": fx.away, "date": fx.date,
                "home_crest": fx.home_crest, "away_crest": fx.away_crest,
                "home_win": round(pred["home_win"], 4),
                "draw": round(pred["draw"], 4),
                "away_win": round(pred["away_win"], 4),
                "predicted_score": pred["most_likely_score"],
                "expected_goals": pred["expected_goals"],
                "over_2_5": round(pred["over_2_5"], 4),
                "value_bets": [],
            }
            quote = odds_by_fixture.get((fx.home, fx.away))
            if quote:
                bets = find_value_bets(f"{fx.home} vs {fx.away}", pred, quote)
                entry["value_bets"] = [
                    {"market": b.market, "odds": b.odds, "edge": round(b.edge, 4),
                     "kelly_stake": round(b.kelly_stake, 4)}
                    for b in bets
                ]
            predictions.append(entry)

        _cache[league_name] = {
            "predictions": predictions,
            "fitted_at": datetime.now(timezone.utc).isoformat(),
        }
        log.info(f"{league_name}: {len(predictions)} fixtures cached")
    except Exception as e:
        log.error(f"Failed to refresh {league_name}: {e}")


def refresh_all():
    for name in LEAGUES:
        refresh_league(name)


@app.on_event("startup")
def startup():
    # IMPORTANT: don't call refresh_all() directly here. Fetching 5 leagues'
    # worth of data can take long enough that Render's port check times out
    # before the server finishes starting, producing a "no open ports
    # detected" deploy failure. Running it in a background thread lets
    # uvicorn bind the port and report healthy immediately, while the first
    # data load happens moments later. /health will show cached_leagues: []
    # for the first ~10-30 seconds — that's expected, not a bug.
    threading.Thread(target=refresh_all, daemon=True).start()
    scheduler = BackgroundScheduler()
    scheduler.add_job(refresh_all, "interval", hours=REFRESH_HOURS)
    scheduler.start()


@app.get("/")
def root():
    # This is a JSON API with no visual homepage — visiting the bare URL in a
    # browser is expected to look like "just data," not a 404. The actual
    # frontend (index.html) is what renders a real page, hosted separately.
    return {
        "service": "Matchday Predictor API",
        "try_these": ["/health", "/leagues", "/predictions/Premier League"],
    }


@app.get("/leagues")
def get_leagues():
    return {"leagues": list(LEAGUES.keys())}


@app.get("/predictions/{league_name}")
def get_predictions(league_name: str):
    if league_name not in LEAGUES:
        raise HTTPException(404, f"Unknown league. Choose from: {list(LEAGUES.keys())}")
    if league_name not in _cache:
        raise HTTPException(503, "Data not ready yet — try again shortly")
    return _cache[league_name]


@app.get("/health")
def health():
    return {"status": "ok", "cached_leagues": list(_cache.keys())}
