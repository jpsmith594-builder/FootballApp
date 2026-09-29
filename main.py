"""
main.py
End-to-end pipeline: for each league, pull recent results, fit a Dixon-Coles
model, predict upcoming fixtures, pull bookmaker odds, and print any value
bets found.

Run:
    export FOOTBALL_DATA_API_KEY=your_key_here
    export ODDS_API_KEY=your_key_here          # optional — skips value bets if unset
    python main.py
"""

import os
from data_fetcher import FootballDataClient, LEAGUES
from poisson_model import DixonColesModel
from value_bet import fetch_odds, find_value_bets

ODDS_ENABLED = bool(os.environ.get("ODDS_API_KEY"))


def run_league(league_name: str, league_code: str):
    print(f"\n{'='*60}\n{league_name}\n{'='*60}")
    client = FootballDataClient()

    print("Fetching last 365 days of results...")
    results = client.get_results(league_code, days_back=365)
    print(f"  {len(results)} finished matches")

    print("Fitting Dixon-Coles model...")
    model = DixonColesModel()
    model.fit(results)

    print("Fetching upcoming fixtures...")
    fixtures = client.get_upcoming_fixtures(league_code, days_ahead=10)

    odds_by_fixture = {}
    if ODDS_ENABLED:
        try:
            odds_quotes = fetch_odds(league_name)
            odds_by_fixture = {(q.home_team, q.away_team): q for q in odds_quotes}
        except Exception as e:
            print(f"  (odds fetch failed: {e})")

    for fx in fixtures:
        try:
            pred = model.predict(fx.home, fx.away)
        except ValueError:
            continue  # newly promoted team with no fitted rating yet

        print(f"\n{fx.home} vs {fx.away}  ({fx.date[:10]})")
        print(f"  1X2: Home {pred['home_win']:.0%} / Draw {pred['draw']:.0%} / Away {pred['away_win']:.0%}")
        print(f"  Most likely score: {pred['most_likely_score'][0]}-{pred['most_likely_score'][1]}")
        print(f"  Over 2.5 goals: {pred['over_2_5']:.0%}")

        quote = odds_by_fixture.get((fx.home, fx.away))
        if quote:
            bets = find_value_bets(f"{fx.home} vs {fx.away}", pred, quote)
            for b in bets:
                print(f"  >> VALUE BET: {b.market} @ {b.odds} "
                      f"(edge {b.edge:+.1%}, suggested stake {b.kelly_stake:.1%} of bankroll)")


if __name__ == "__main__":
    for name, code in LEAGUES.items():
        try:
            run_league(name, code)
        except Exception as e:
            print(f"Skipping {name}: {e}")
