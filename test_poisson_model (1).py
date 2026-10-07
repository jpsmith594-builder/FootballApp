import random
import numpy as np
import pytest
from data_fetcher import Match
from poisson_model import DixonColesModel, blend_with_market


def make_synthetic_matches(seed=42, n=100):
    random.seed(seed)
    teams = ["Strong", "Mid", "Weak", "Solid", "Rising"]
    strength = {"Strong": 2.0, "Mid": 1.3, "Weak": 0.7, "Solid": 1.5, "Rising": 1.1}
    matches = []
    for day in range(n):
        h, a = random.sample(teams, 2)
        matches.append(Match(
            home=h, away=a, date=f"2026-{(day % 9) + 1:02d}-{(day % 27) + 1:02d}T15:00:00Z",
            home_goals=int(np.random.poisson(strength[h] * 1.2)),
            away_goals=int(np.random.poisson(strength[a] * 0.9)),
            status="FINISHED",
        ))
    return matches, strength


def test_fit_requires_minimum_matches():
    with pytest.raises(ValueError):
        DixonColesModel().fit([
            Match(home="A", away="B", date="2026-01-01T00:00:00Z", home_goals=1, away_goals=0, status="FINISHED")
        ])


def test_fit_recovers_relative_strength():
    matches, strength = make_synthetic_matches()
    model = DixonColesModel()
    model.fit(matches)
    # The strongest team should end up with the highest home attack rating
    strongest = max(strength, key=strength.get)
    weakest = min(strength, key=strength.get)
    assert model.ratings[strongest].attack_home > model.ratings[weakest].attack_home


def test_predict_probabilities_sum_to_one():
    matches, _ = make_synthetic_matches()
    model = DixonColesModel()
    model.fit(matches)
    pred = model.predict("Strong", "Weak")
    total = pred["home_win"] + pred["draw"] + pred["away_win"]
    assert abs(total - 1.0) < 1e-6


def test_predict_unknown_team_raises():
    matches, _ = make_synthetic_matches()
    model = DixonColesModel()
    model.fit(matches)
    with pytest.raises(ValueError):
        model.predict("Strong", "Never Seen This Team FC")


def test_goal_multiplier_reduces_win_probability():
    matches, _ = make_synthetic_matches()
    model = DixonColesModel()
    model.fit(matches)
    normal = model.predict("Strong", "Weak")
    injured = model.predict("Strong", "Weak", home_goal_multiplier=0.6)
    assert injured["home_win"] < normal["home_win"]


def test_blend_with_market_sums_to_one_and_respects_weight():
    model_probs = {"home_win": 0.7, "draw": 0.2, "away_win": 0.1}
    blended_full_model = blend_with_market(model_probs, 0.4, 0.3, 0.3, weight=1.0)
    assert blended_full_model == pytest.approx(model_probs, abs=1e-9)

    blended_full_market = blend_with_market(model_probs, 0.4, 0.3, 0.3, weight=0.0)
    assert blended_full_market["home_win"] == pytest.approx(0.4, abs=1e-9)

    blended_half = blend_with_market(model_probs, 0.4, 0.3, 0.3, weight=0.5)
    total = sum(blended_half.values())
    assert abs(total - 1.0) < 1e-9
