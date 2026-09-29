"""
value_bet.py
This is the actual money-making layer. The model gives you a probability;
a bookmaker's odds imply their own probability (with their margin baked in).
When your estimate is meaningfully higher than theirs, that's a value bet —
over many bets, that edge should show a positive return, even though any
single bet is still a coin flip.

Odds source: bookmakers don't publish an API for individual bettors to pull
prices from directly. The standard route is The Odds API
(https://the-odds-api.com), which aggregates real bookmaker odds and has a
free tier (500 requests/month). Swap in another provider by matching the
OddsQuote shape below.
"""

import os
import requests
from dataclasses import dataclass
from typing import List, Optional

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

# The Odds API sport keys for the top 5 leagues
ODDS_SPORT_KEYS = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga1",
    "Ligue 1": "soccer_france_ligue_one",
}


@dataclass
class OddsQuote:
    home_team: str
    away_team: str
    home_odds: float   # decimal odds, e.g. 2.10
    draw_odds: float
    away_odds: float
    bookmaker: str


def fetch_odds(league: str, api_key: str = ODDS_API_KEY) -> List[OddsQuote]:
    if not api_key:
        raise ValueError("Set ODDS_API_KEY (get a free key at the-odds-api.com)")
    sport_key = ODDS_SPORT_KEYS[league]
    resp = requests.get(
        f"{ODDS_BASE_URL}/sports/{sport_key}/odds",
        params={"apiKey": api_key, "regions": "uk,eu", "markets": "h2h", "oddsFormat": "decimal"},
    )
    resp.raise_for_status()
    quotes = []
    for event in resp.json():
        if not event.get("bookmakers"):
            continue
        book = event["bookmakers"][0]  # first available bookmaker; average across
        market = next((m for m in book["markets"] if m["key"] == "h2h"), None)
        if not market:
            continue
        prices = {o["name"]: o["price"] for o in market["outcomes"]}
        quotes.append(OddsQuote(
            home_team=event["home_team"], away_team=event["away_team"],
            home_odds=prices.get(event["home_team"], 0),
            draw_odds=prices.get("Draw", 0),
            away_odds=prices.get(event["away_team"], 0),
            bookmaker=book["title"],
        ))
    return quotes


def devig(home_odds: float, draw_odds: float, away_odds: float) -> tuple:
    """Bookmaker odds always imply >100% probability (their margin/'vig').
    Strip it out proportionally so we're comparing our model to their true
    fair-odds view, not their inflated one."""
    imp_h, imp_d, imp_a = 1 / home_odds, 1 / draw_odds, 1 / away_odds
    overround = imp_h + imp_d + imp_a
    return imp_h / overround, imp_d / overround, imp_a / overround


@dataclass
class ValueBet:
    fixture: str
    market: str          # "Home", "Draw", or "Away"
    model_prob: float
    market_prob: float
    edge: float           # model_prob - market_prob, as a fraction
    odds: float
    kelly_stake: float    # fraction of bankroll, per the Kelly criterion


def find_value_bets(fixture_label: str, model_prediction: dict, odds: OddsQuote,
                     min_edge: float = 0.04) -> List[ValueBet]:
    """
    min_edge=0.04 means: only flag a bet if the model thinks the true
    probability is at least 4 percentage points higher than what the odds
    imply. This threshold exists because both the model and the market have
    noise — a 1-2% edge is within the model's own error bars and isn't
    reliably exploitable after bookmaker margin and bet-slippage.
    """
    fair_h, fair_d, fair_a = devig(odds.home_odds, odds.draw_odds, odds.away_odds)
    candidates = [
        ("Home", model_prediction["home_win"], fair_h, odds.home_odds),
        ("Draw", model_prediction["draw"], fair_d, odds.draw_odds),
        ("Away", model_prediction["away_win"], fair_a, odds.away_odds),
    ]
    bets = []
    for market, model_p, market_p, price in candidates:
        edge = model_p - market_p
        if edge >= min_edge:
            # Fractional Kelly (half-Kelly) — full Kelly is theoretically optimal
            # but assumes your model is perfectly calibrated, which it never is.
            # Half-Kelly trades some growth for much lower risk of ruin.
            b = price - 1
            full_kelly = (model_p * b - (1 - model_p)) / b if b > 0 else 0
            kelly_stake = max(0, full_kelly * 0.5)
            bets.append(ValueBet(
                fixture=fixture_label, market=market, model_prob=model_p,
                market_prob=market_p, edge=edge, odds=price, kelly_stake=kelly_stake,
            ))
    return bets


if __name__ == "__main__":
    # Self-test with made-up odds, no API key needed.
    fake_odds = OddsQuote(home_team="Strong FC", away_team="Weak Town",
                           home_odds=1.60, draw_odds=4.20, away_odds=6.00,
                           bookmaker="TestBook")
    fake_prediction = {"home_win": 0.689, "draw": 0.201, "away_win": 0.110}
    bets = find_value_bets("Strong FC vs Weak Town", fake_prediction, fake_odds)
    for b in bets:
        print(f"{b.fixture} | {b.market} | model={b.model_prob:.1%} vs market={b.market_prob:.1%} "
              f"| edge={b.edge:+.1%} | odds={b.odds} | stake={b.kelly_stake:.1%} of bankroll")
