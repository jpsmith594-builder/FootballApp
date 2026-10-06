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

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from data_fetcher import FootballDataClient, LEAGUES
from poisson_model import DixonColesModel, blend_with_market
from elo_model import EloModel, ensemble as elo_ensemble
from value_bet import fetch_odds, find_value_bets, devig, ODDS_SPORT_KEYS
import db
import push
import billing

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
RESULTS_DAYS_BACK = int(os.environ.get("RESULTS_DAYS_BACK", "365"))
# football-data.org's free tier may not actually serve more than the current
# season regardless of this setting — raising it costs nothing to try, but
# don't be surprised if the API just returns what it already would have.
ODDS_ENABLED = bool(os.environ.get("ODDS_API_KEY"))
# How much trust goes to Dixon-Coles vs Elo when forming "the model's view"
# (used for value-bet detection, before any market blending happens).
ENSEMBLE_WEIGHT_DC = float(os.environ.get("ENSEMBLE_WEIGHT_DC", "0.7"))
# How much trust goes to the (ensembled) model vs. the market when blending
# the *displayed* prediction. 1.0 = ignore the market, 0.0 = ignore the model.
# Value-bet detection always uses the raw, unblended ensemble regardless of
# this setting — see poisson_model.blend_with_market()'s docstring.
BLEND_WEIGHT = float(os.environ.get("BLEND_WEIGHT", "0.5"))
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "")
# Three separate bars, not one shared threshold. "Win or Draw" mathematically
# covers a wider slice of outcomes than "Straight Win" (it's win probability
# PLUS draw probability), so it will almost always score higher on a raw
# comparison — a flat shared threshold would make Win-or-Draw dominate every
# single hint and Straight Win would almost never be shown. Instead, the
# model's confidence is checked against each market's own, harder-to-clear
# bar, in priority order from most specific/useful to safest.
HINT_STRAIGHT_WIN_MIN = float(os.environ.get("HINT_STRAIGHT_WIN_MIN", "0.58"))
HINT_WIN_OR_DRAW_MIN = float(os.environ.get("HINT_WIN_OR_DRAW_MIN", "0.78"))
HINT_BTTS_MIN = float(os.environ.get("HINT_BTTS_MIN", "0.58"))

# --- Tiers ---
# "free" sees fixtures and win/draw/away probabilities for every league, but
# never hints or value bets — those are the Pro differentiators, and giving
# them away for free removes the entire reason to upgrade. See
# MONETIZATION.md for the full reasoning behind where this line is drawn.
# A missing/invalid API key is treated as "free", not rejected outright —
# the app stays usable without signing up, which is deliberate: the free
# tier IS the funnel, not a locked door.
FREE_DAILY_REQUEST_LIMIT = int(os.environ.get("FREE_DAILY_REQUEST_LIMIT", "100"))
PRO_DAILY_REQUEST_LIMIT = int(os.environ.get("PRO_DAILY_REQUEST_LIMIT", "5000"))


def get_current_tier(x_api_key: str = Header(default=None)) -> dict:
    """
    FastAPI dependency: resolves the caller's tier from their API key.
    Deliberately permissive on a missing or unrecognized key — this
    endpoint still has to work for an anonymous visitor on the free tier,
    so "no key" means "free", not "403". Invalid/expired *paid* access is
    what actually gets rejected, and only where real money is on the line
    (billing-portal/webhook endpoints use their own stricter checks, not
    this one).
    """
    if not x_api_key:
        return {"tier": "free", "api_key": None, "email": None}
    user = db.get_user_by_api_key(x_api_key)
    if not user:
        return {"tier": "free", "api_key": x_api_key, "email": None}
    limit = PRO_DAILY_REQUEST_LIMIT if user["tier"] == "pro" else FREE_DAILY_REQUEST_LIMIT
    if not db.check_and_increment_rate_limit(x_api_key, daily_limit=limit):
        raise HTTPException(429, f"Daily request limit reached for the {user['tier']} tier")
    return {"tier": user["tier"], "api_key": x_api_key, "email": user["email"]}


def generate_hint(home: str, away: str, win_draw_away_probs: dict, btts_yes: float, btts_no: float):
    """
    Checks three bet types in priority order and returns the first one that
    clears its own confidence bar — not whichever has the single highest raw
    number (see the note above on why that comparison is unfair between
    markets of different width). Order: Straight Win (most specific and most
    useful if the model is confident enough) → Win or Draw (a safer hedge
    when no single outcome stands out) → BTTS (a different angle entirely,
    checked last since it answers a different question than who wins).
    Uses the same (possibly market-blended) win/draw/away probabilities shown
    in the fixture's own bar, so the hint always agrees with what's displayed
    elsewhere on the card. Returns None if nothing clears its bar — a close
    game should show no hint, not a misleadingly confident-looking one.
    """
    home_win = win_draw_away_probs["home_win"]
    draw = win_draw_away_probs["draw"]
    away_win = win_draw_away_probs["away_win"]

    if home_win >= HINT_STRAIGHT_WIN_MIN and home_win >= away_win:
        return {"market": "Straight Win", "selection": home, "confidence": round(home_win, 3)}
    if away_win >= HINT_STRAIGHT_WIN_MIN and away_win > home_win:
        return {"market": "Straight Win", "selection": away, "confidence": round(away_win, 3)}

    double_chance_1x = home_win + draw
    double_chance_x2 = away_win + draw
    if double_chance_1x >= HINT_WIN_OR_DRAW_MIN and double_chance_1x >= double_chance_x2:
        return {"market": "Win or Draw", "selection": f"{home} or Draw", "confidence": round(double_chance_1x, 3)}
    if double_chance_x2 >= HINT_WIN_OR_DRAW_MIN and double_chance_x2 > double_chance_1x:
        return {"market": "Win or Draw", "selection": f"{away} or Draw", "confidence": round(double_chance_x2, 3)}

    if btts_yes >= HINT_BTTS_MIN and btts_yes >= btts_no:
        return {"market": "BTTS", "selection": "Yes", "confidence": round(btts_yes, 3)}
    if btts_no >= HINT_BTTS_MIN and btts_no > btts_yes:
        return {"market": "BTTS", "selection": "No", "confidence": round(btts_no, 3)}

    return None


def generate_alternative_bet(primary_hint: dict, btts_yes: float, btts_no: float):
    """
    A second, supplementary line shown below the primary hint: BTTS Yes or
    No, whichever side the model actually favors, with its own confidence —
    always shown, with no minimum-confidence bar. Unlike generate_hint()
    above, this isn't a "here's what to play" recommendation gated by
    confidence; it's just the model's BTTS read laid out for the user to
    weigh themselves, since BTTS answers a different question (will BOTH
    sides score) than who wins, and is often worth seeing alongside the
    primary pick rather than only when it happens to win the priority order.

    Returns None only when the primary hint's market is already "BTTS" —
    showing the identical pick twice would be redundant, not alternative.
    """
    if primary_hint and primary_hint["market"] == "BTTS":
        return None
    if btts_yes >= btts_no:
        return {"market": "BTTS", "selection": "Yes", "confidence": round(btts_yes, 3)}
    return {"market": "BTTS", "selection": "No", "confidence": round(btts_no, 3)}

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
        results = client.get_results(league_code, days_back=RESULTS_DAYS_BACK)
        model = DixonColesModel()
        model.fit(results)
        # Teams with a thin sample (newly promoted sides, mainly) get pulled
        # toward a below-average "newcomer" prior rather than trusted at
        # face value — see poisson_model.apply_promoted_team_shrinkage's
        # docstring for why. No-op for any team with enough matches already.
        model.apply_promoted_team_shrinkage(results)
        elo = EloModel()
        elo.fit(results)
        adjustments = db.get_team_adjustments(league_name)
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
            home_mult = adjustments.get(fx.home, 1.0)
            away_mult = adjustments.get(fx.away, 1.0)
            try:
                pred = model.predict(fx.home, fx.away,
                                      home_goal_multiplier=home_mult, away_goal_multiplier=away_mult)
                elo_pred = elo.predict(fx.home, fx.away)
            except ValueError:
                continue  # team with no fitted history (e.g. newly promoted)

            # "The model's view" for value-bet purposes is the DC+Elo
            # ensemble, not Dixon-Coles alone — two models that fail
            # differently, averaged, before any market blending happens.
            ensembled = elo_ensemble(pred, elo_pred, weight_dc=ENSEMBLE_WEIGHT_DC)
            # most_likely_score/expected_goals/over_2_5/btts still come from
            # the raw Dixon-Coles scoreline grid — Elo has no concept of these.
            model_view = {**pred, "home_win": ensembled["home_win"],
                          "draw": ensembled["draw"], "away_win": ensembled["away_win"]}

            quote = odds_by_fixture.get((fx.home, fx.away))
            value_bets = []
            display_probs = {"home_win": model_view["home_win"], "draw": model_view["draw"],
                              "away_win": model_view["away_win"]}

            if quote:
                bets = find_value_bets(f"{fx.home} vs {fx.away}", model_view, quote)
                value_bets = [
                    {"market": b.market, "odds": b.odds, "edge": round(b.edge, 4),
                     "kelly_stake": round(b.kelly_stake, 4)}
                    for b in bets
                ]
                # Blend for DISPLAY only. find_value_bets() above already ran
                # against the raw, unblended `model_view` — order matters here.
                fair_h, fair_d, fair_a = devig(quote.home_odds, quote.draw_odds, quote.away_odds)
                display_probs = blend_with_market(model_view, fair_h, fair_d, fair_a, weight=BLEND_WEIGHT)

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
                "hint": generate_hint(fx.home, fx.away, display_probs, pred["btts_yes"], pred["btts_no"]),
            }
            entry["alternative_hint"] = generate_alternative_bet(
                entry["hint"], pred["btts_yes"], pred["btts_no"])
            predictions.append(entry)

            # Record for the track record tab. Uses the ensembled model view,
            # not the blended display value — the track record should judge
            # your model(s), not a moving target of model+market averages.
            try:
                was_new = db.record_prediction(league_name, fx.home, fx.away, fx.date, model_view,
                                                value_bets[0] if value_bets else None)
                # Only notify the FIRST time this fixture's value bet is
                # recorded — otherwise every 6-hour refresh re-fires it.
                if was_new and value_bets:
                    push.maybe_notify_value_bet(league_name, fx.home, fx.away, value_bets[0])
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
    predictions everyone sees. Takes effect on the next scheduled refresh,
    not instantly.
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
    {"endpoint": "...", "keys": {"p256dh": "...", "auth": "..."}}"""
    endpoint = subscription.get("endpoint")
    keys = subscription.get("keys", {})
    if not endpoint or "p256dh" not in keys or "auth" not in keys:
        raise HTTPException(400, "Malformed push subscription")
    db.save_push_subscription(endpoint, keys["p256dh"], keys["auth"])
    return {"status": "subscribed"}


@app.get("/health")
def health():
    return {"status": "ok", "cached_leagues": list(_cache.keys())}
