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
from poisson_model import DixonColesModel, blend_with_market
from value_bet import fetch_odds, find_value_bets, devig, ODDS_SPORT_KEYS
import db

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api_server")

REFRESH_HOURS = float(os.environ.get("REFRESH_HOURS", "6"))
ODDS_ENABLED = bool(os.environ.get("ODDS_API_KEY"))
# How much trust goes to the model vs. the market when blending the
# *displayed* prediction. 1.0 = ignore the market, 0.0 = ignore the model.
# Value-bet detection always uses the raw, unblended model output regardless
# of this setting — see poisson_model.blend_with_market()'s docstring.
BLEND_WEIGHT = float(os.environ.get("BLEND_WEIGHT", "0.5"))

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
                pred = model.predict(fx.home, fx.away)  # raw model — used for value bets & storage
            except ValueError:
                continue  # team with no fitted history (e.g. newly promoted)

            quote = odds_by_fixture.get((fx.home, fx.away))
            value_bets = []
            display_probs = {"home_win": pred["home_win"], "draw": pred["draw"], "away_win": pred["away_win"]}

            if quote:
                bets = find_value_bets(f"{fx.home} vs {fx.away}", pred, quote)
                value_bets = [
                    {"market": b.market, "odds": b.odds, "edge": round(b.edge, 4),
                     "kelly_stake": round(b.kelly_stake, 4)}
                    for b in bets
                ]
                # Blend for DISPLAY only. find_value_bets() above already ran
                # against the raw, unblended `pred` — that ordering matters.
                fair_h, fair_d, fair_a = devig(quote.home_odds, quote.draw_odds, quote.away_odds)
                display_probs = blend_with_market(pred, fair_h, fair_d, fair_a, weight=BLEND_WEIGHT)

            entry = {
                "home": fx.home, "away": fx.away, "date": fx.date,
                "home_crest": fx.home_crest, "away_crest": fx.away_crest,
                "home_win": round(display_probs["home_win"], 4),
                "draw": round(display_probs["draw"], 4),
                "away_win": round(display_probs["away_win"], 4),
                "predicted_score": pred["most_likely_score"],
                "expected_goals": pred["expected_goals"],
                "over_2_5": round(pred["over_2_5"], 4),
                "btts_yes": round(pred["btts_yes"], 4),
                "value_bets": value_bets,
            }
            predictions.append(entry)

            # Record for the track record tab. Uses the raw model prediction,
            # not the blended display value — the track record should judge
            # your model, not a moving target of model+market averages.
            try:
                db.record_prediction(league_name, fx.home, fx.away, fx.date, pred,
                                      value_bets[0] if value_bets else None)
            except Exception as e:
                log.warning(f"Could not record prediction for track record: {e}")

        # Reconcile any previously-recorded predictions whose matches have
        # since finished, using the same `results` fetched above.
        try:
            db.resolve_predictions(league_name, results)
        except Exception as e:
            log.warning(f"Could not resolve past predictions: {e}")

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


@app.get("/track-record/{league_name}")
def get_track_record(league_name: str):
    if league_name not in LEAGUES:
        raise HTTPException(404, f"Unknown league. Choose from: {list(LEAGUES.keys())}")
    rows = db.get_track_record(league_name)
    return {"league": league_name, "summary": db.summarize_track_record(rows), "history": rows}


@app.get("/health")
def health():
    return {"status": "ok", "cached_leagues": list(_cache.keys())}
