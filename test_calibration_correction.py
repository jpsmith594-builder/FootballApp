"""
Tests the correction layer in calibration.py — the piece that takes the
bucketed accuracy curve db.py already tracks and uses it to pull new
predictions' top-pick confidence toward what's actually been true
historically, rather than just reporting the gap.
"""
import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import db  # noqa: E402
from data_fetcher import Match  # noqa: E402
import calibration  # noqa: E402


def _seed_resolved(league, n, confidence, correct_fraction):
    n_correct = round(n * correct_fraction)
    for i in range(n):
        pred = {"home_win": confidence, "draw": (1 - confidence) * 0.6,
                "away_win": (1 - confidence) * 0.4, "most_likely_score": (2, 0)}
        date = f"2026-{(i % 9) + 1:02d}-{(i % 27) + 1:02d}T15:00:00Z"
        db.record_prediction(league, f"H{league}{i}", f"A{league}{i}", date, pred)
    finished = []
    for i in range(n):
        correct = i < n_correct
        date = f"2026-{(i % 9) + 1:02d}-{(i % 27) + 1:02d}T15:00:00Z"
        finished.append(Match(home=f"H{league}{i}", away=f"A{league}{i}", date=date,
                               home_goals=2 if correct else 0, away_goals=0 if correct else 2,
                               status="FINISHED"))
    db.resolve_predictions(league, finished)


def test_build_correction_returns_none_below_min_samples():
    _seed_resolved("TooFew", n=20, confidence=0.90, correct_fraction=0.5)
    fn = calibration.build_correction("TooFew")
    assert fn is None


def test_build_correction_returns_none_for_unseen_league():
    assert calibration.build_correction("NeverSeenThisLeagueEither") is None


def test_build_correction_pulls_overconfident_top_pick_down():
    # Model says 90% a lot, but is only actually right 55% of the time.
    _seed_resolved("Overconf2", n=60, confidence=0.90, correct_fraction=0.55)
    fn = calibration.build_correction("Overconf2")
    assert fn is not None
    assert abs(fn(0.90) - 0.55) < 1e-6


def test_apply_correction_is_noop_when_fn_is_none():
    probs = {"home_win": 0.72, "draw": 0.18, "away_win": 0.10}
    result = calibration.apply_correction(probs, None)
    assert result == probs
    assert result is not probs  # returns a copy, not the same object


def test_apply_correction_rescales_other_outcomes_proportionally():
    # Fake correction function: always says the true accuracy is 0.5,
    # regardless of the stated confidence.
    probs = {"home_win": 0.72, "draw": 0.18, "away_win": 0.10}
    result = calibration.apply_correction(probs, lambda c: 0.5)
    assert abs(result["home_win"] - 0.5) < 1e-9
    # Remaining 0.5 split between draw/away in their original 18:10 ratio.
    assert abs(result["draw"] - 0.5 * (0.18 / 0.28)) < 1e-9
    assert abs(result["away_win"] - 0.5 * (0.10 / 0.28)) < 1e-9
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_apply_correction_clamps_into_0_1_range():
    probs = {"home_win": 0.95, "draw": 0.03, "away_win": 0.02}
    result = calibration.apply_correction(probs, lambda c: 1.5)  # pathological fn
    assert result["home_win"] == 1.0
    assert result["draw"] == 0.0
    assert result["away_win"] == 0.0


def test_build_correction_disabled_via_env(monkeypatch):
    _seed_resolved("Disabled", n=60, confidence=0.90, correct_fraction=0.55)
    monkeypatch.setattr(calibration, "CALIBRATION_ENABLED", False)
    assert calibration.build_correction("Disabled") is None
