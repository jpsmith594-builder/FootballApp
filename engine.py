"""
engine.py
The actual "compute predictions" work — fetching data, fitting models,
building hints, detecting value bets, writing results to the database. This
used to live inside api_server.py's request-handling code, which meant it
only ever ran inside the (cold-starting) web server process.

Pulled out on its own so it can be driven from two different places without
duplicating logic:
  - refresh_cache.py: a standalone script meant to run on a schedule
    (GitHub Actions cron, in production) completely independent of whether
    any web server is awake.
  - api_server.py: kept around for local development (uvicorn --reload) and
    as an optional thin API, but no longer required in production — the
    frontend can read Supabase directly once this module has populated it.

Either caller ends up with the same two things: `_cache` (in-process, handy
for local testing) and the `cached_fixtures` table in Postgres (what the
deployed frontend actually reads).
"""

import os
import logging
from datetime import datetime, timezone

from data_fetcher import FootballDataClient, LEAGUES
from poisson_model import DixonColesModel, blend_with_market
from elo_model import EloModel, ensemble as elo_ensemble
from value_bet import fetch_odds, find_value_bets, devig
import calibration
import db
import push

log = logging.getLogger("engine")

RESULTS_DAYS_BACK = int(os.environ.get("RESULTS_DAYS_BACK", "365"))
ODDS_ENABLED = bool(os.environ.get("ODDS_API_KEY"))
# How much trust goes to Dixon-Coles vs Elo when forming "the model's view"
# (used for value-bet detection, before any market blending happens).
ENSEMBLE_WEIGHT_DC = float(os.environ.get("ENSEMBLE_WEIGHT_DC", "0.7"))
# How much trust goes to the (ensembled) model vs. the market when blending
# the *displayed* prediction. 1.0 = ignore the market, 0.0 = ignore the model.
# Value-bet detection always uses the raw, unblended ensemble regardless of
# this setting — see poisson_model.blend_with_market()'s docstring.
BLEND_WEIGHT = float(os.environ.get("BLEND_WEIGHT", "0.5"))
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

    # "side" is a normalized grading key the Hint Accuracy tab uses to check
    # this tip against the real result later, without having to re-parse a
    # free-text `selection` string like "Arsenal or Draw" months from now —
    # it isn't shown anywhere in the UI, just carried through to db.py.
    if home_win >= HINT_STRAIGHT_WIN_MIN and home_win >= away_win:
        return {"market": "Straight Win", "selection": home, "confidence": round(home_win, 3), "side": "Home"}
    if away_win >= HINT_STRAIGHT_WIN_MIN and away_win > home_win:
        return {"market": "Straight Win", "selection": away, "confidence": round(away_win, 3), "side": "Away"}

    double_chance_1x = home_win + draw
    double_chance_x2 = away_win + draw
    if double_chance_1x >= HINT_WIN_OR_DRAW_MIN and double_chance_1x >= double_chance_x2:
        return {"market": "Win or Draw", "selection": f"{home} or Draw",
                "confidence": round(double_chance_1x, 3), "side": "HomeOrDraw"}
    if double_chance_x2 >= HINT_WIN_OR_DRAW_MIN and double_chance_x2 > double_chance_1x:
        return {"market": "Win or Draw", "selection": f"{away} or Draw",
                "confidence": round(double_chance_x2, 3), "side": "AwayOrDraw"}

    if btts_yes >= HINT_BTTS_MIN and btts_yes >= btts_no:
        return {"market": "BTTS", "selection": "Yes", "confidence": round(btts_yes, 3), "side": "BTTS_YES"}
    if btts_no >= HINT_BTTS_MIN and btts_no > btts_yes:
        return {"market": "BTTS", "selection": "No", "confidence": round(btts_no, 3), "side": "BTTS_NO"}

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


# In-memory cache: {league_name: {"predictions": [...], "fitted_at": ...}}
# Handy for local dev (api_server.py) and for tests. Not what the deployed
# frontend reads — that goes through db.cached_fixtures instead, which this
# module writes to as part of refresh_league() below.
_cache: dict = {}

# Competitions whose standings aren't a single flat domestic-style table —
# Champions League is group/league-phase, a different shape entirely (see
# FootballDataClient.get_standings's docstring). Skipped rather than
# mishandled; add others here if a future league turns out the same way.
STANDINGS_SKIP_LEAGUES = {"Champions League"}


def refresh_league(league_name: str):
    league_code = LEAGUES[league_name]
    log.info(f"Refreshing {league_name}...")
    try:
        client = FootballDataClient()
        results = client.get_results(league_code, days_back=RESULTS_DAYS_BACK)

        # Pure display data — never feeds the model — so it's fetched and
        # written independently, with its own try/except: a standings hiccup
        # should never block predictions from refreshing.
        if league_name not in STANDINGS_SKIP_LEAGUES:
            try:
                table_rows = client.get_standings(league_code)
                db.replace_standings(league_name, table_rows)
            except Exception as e:
                log.warning(f"Could not refresh standings for {league_name}: {e}")
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
        # Built once per league per refresh (not per fixture) — it's the
        # same correction curve for every fixture in this league this run.
        # See calibration.py for why this only touches what's displayed,
        # not value-bet detection or the track record.
        correction_fn = calibration.build_correction(league_name)

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

            # Calibration-corrected view, for DISPLAY only — find_value_bets()
            # below still runs against the raw `model_view`, and so does the
            # track record recording further down. See calibration.py.
            calibrated = calibration.apply_correction(
                {"home_win": model_view["home_win"], "draw": model_view["draw"],
                 "away_win": model_view["away_win"]},
                correction_fn,
            )
            calibrated_view = {**model_view, **calibrated}

            quote = odds_by_fixture.get((fx.home, fx.away))
            value_bets = []
            display_probs = {"home_win": calibrated_view["home_win"], "draw": calibrated_view["draw"],
                              "away_win": calibrated_view["away_win"]}

            if quote:
                bets = find_value_bets(f"{fx.home} vs {fx.away}", model_view, quote)
                value_bets = [
                    {"market": b.market, "odds": b.odds, "edge": round(b.edge, 4),
                     "kelly_stake": round(b.kelly_stake, 4)}
                    for b in bets
                ]
                # Blend for DISPLAY only, and blend the CALIBRATED view (not
                # the raw one) with the market — find_value_bets() above
                # already ran against the raw, unblended `model_view`, so
                # which view feeds display here doesn't affect value-bet
                # detection either way.
                fair_h, fair_d, fair_a = devig(quote.home_odds, quote.draw_odds, quote.away_odds)
                display_probs = blend_with_market(calibrated_view, fair_h, fair_d, fair_a, weight=BLEND_WEIGHT)

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
                                                value_bets[0] if value_bets else None, entry["hint"])
                # Only notify the FIRST time this fixture's value bet is
                # recorded — otherwise every refresh re-fires it.
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

        fitted_at = datetime.now(timezone.utc).isoformat()
        _cache[league_name] = {"predictions": predictions, "fitted_at": fitted_at}

        # This is the line that makes the "serve" side fast: write the
        # computed slate straight into Postgres so a cold-started server (or
        # the frontend, reading Supabase directly) never has to wait for a
        # model fit — it's already sitting there.
        try:
            db.replace_cached_fixtures(league_name, predictions, fitted_at)
        except Exception as e:
            log.warning(f"Could not write cached_fixtures for {league_name}: {e}")

        log.info(f"{league_name}: {len(predictions)} fixtures cached")
    except Exception as e:
        log.error(f"Failed to refresh {league_name}: {e}")


def refresh_all():
    for name in LEAGUES:
        refresh_league(name)


def build_parlay_board(cache: dict, limit: int = 15):
    """
    Pulls together every cached fixture, across every league, that already
    has a hint attached (generate_hint() only returns one once a market
    clears its own confidence bar — see its docstring), and ranks them by
    that hint's confidence, highest first.

    Deliberately does NOT compute or display a combined parlay price. Real
    odds are only ever fetched for the 1X2 market (value_bet.py), while
    these hints frequently recommend Win-or-Draw or BTTS — markets with no
    fetched price. Inventing a combined payout for an untraded market would
    be a fabricated number dressed up as a stat, so this only ever reports
    probabilities, never odds or payouts.
    """
    picks = []
    for league_name, data in cache.items():
        for entry in data.get("predictions", []):
            hint = entry.get("hint")
            if not hint:
                continue
            picks.append({
                "league": league_name,
                "home": entry["home"],
                "away": entry["away"],
                "date": entry["date"],
                "home_crest": entry.get("home_crest"),
                "away_crest": entry.get("away_crest"),
                "predicted_score": entry.get("predicted_score"),
                "hint": hint,
                "alternative_hint": entry.get("alternative_hint"),
            })
    picks.sort(key=lambda p: p["hint"]["confidence"], reverse=True)
    top = picks[:limit]
    combined_probability = 1.0
    for p in top:
        combined_probability *= p["hint"]["confidence"]
    return {
        "picks": top,
        "combined_probability_if_all_legs_selected": round(combined_probability, 6) if top else None,
        "total_candidates": len(picks),
    }
