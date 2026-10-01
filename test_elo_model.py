import random
import numpy as np
import pytest
from data_fetcher import Match
from elo_model import EloModel, ensemble


def test_elo_ranks_teams_correctly():
    random.seed(5)
    teams = ["Strong", "Mid", "Weak"]
    strength = {"Strong": 2.0, "Mid": 1.2, "Weak": 0.6}
    matches = []
    for day in range(60):
        h, a = random.sample(teams, 2)
        matches.append(Match(
            home=h, away=a, date=f"2026-01-{(day % 27) + 1:02d}T15:00:00Z",
            home_goals=int(np.random.poisson(strength[h] * 1.2)),
            away_goals=int(np.random.poisson(strength[a] * 0.9)),
            status="FINISHED",
        ))
    model = EloModel()
    model.fit(matches)
    assert model.ratings["Strong"] > model.ratings["Mid"] > model.ratings["Weak"]


def test_elo_predict_sums_to_one():
    model = EloModel()
    model.ratings = {"A": 1600, "B": 1400}
    pred = model.predict("A", "B")
    total = pred["home_win"] + pred["draw"] + pred["away_win"]
    assert abs(total - 1.0) < 1e-9


def test_unrated_teams_default_to_even_match():
    model = EloModel()
    pred = model.predict("Unknown A", "Unknown B")
    # No history for either team means only home advantage should matter,
    # so home should be favored but not overwhelmingly so.
    assert pred["home_win"] > pred["away_win"]
    assert pred["home_win"] < 0.7


def test_ensemble_weight_bounds():
    dc = {"home_win": 0.8, "draw": 0.15, "away_win": 0.05}
    elo = {"home_win": 0.4, "draw": 0.3, "away_win": 0.3}
    pure_dc = ensemble(dc, elo, weight_dc=1.0)
    assert pure_dc["home_win"] == pytest.approx(0.8, abs=1e-9)
    pure_elo = ensemble(dc, elo, weight_dc=0.0)
    assert pure_elo["home_win"] == pytest.approx(0.4, abs=1e-9)
