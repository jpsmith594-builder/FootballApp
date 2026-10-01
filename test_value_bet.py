import pytest
from value_bet import devig, find_value_bets, OddsQuote


def test_devig_removes_bookmaker_margin():
    # These odds imply more than 100% because of the bookmaker's margin
    fair_h, fair_d, fair_a = devig(2.0, 3.5, 4.0)
    assert abs((fair_h + fair_d + fair_a) - 1.0) < 1e-9
    # Relative ordering should be preserved — home was the shortest odds,
    # so it should still have the highest fair probability
    assert fair_h > fair_d > fair_a


def test_find_value_bets_flags_genuine_edge():
    odds = OddsQuote(home_team="A", away_team="B", home_odds=2.2, draw_odds=3.3,
                      away_odds=3.5, bookmaker="Test")
    # Model thinks home win is much more likely than the market implies
    prediction = {"home_win": 0.65, "draw": 0.20, "away_win": 0.15}
    bets = find_value_bets("A vs B", prediction, odds, min_edge=0.04)
    assert any(b.market == "Home" for b in bets)


def test_find_value_bets_ignores_small_edge():
    odds = OddsQuote(home_team="A", away_team="B", home_odds=2.0, draw_odds=3.3,
                      away_odds=4.0, bookmaker="Test")
    fair_h, _, _ = devig(2.0, 3.3, 4.0)
    # Model agrees almost exactly with the market — no real edge
    prediction = {"home_win": fair_h + 0.01, "draw": 0.28, "away_win": 0.22}
    bets = find_value_bets("A vs B", prediction, odds, min_edge=0.04)
    assert len(bets) == 0


def test_kelly_stake_is_never_negative():
    odds = OddsQuote(home_team="A", away_team="B", home_odds=1.5, draw_odds=4.0,
                      away_odds=6.0, bookmaker="Test")
    prediction = {"home_win": 0.55, "draw": 0.25, "away_win": 0.20}
    bets = find_value_bets("A vs B", prediction, odds, min_edge=0.01)
    for b in bets:
        assert b.kelly_stake >= 0
