"""
Uses a fresh in-memory SQLite database for each test run via the DATABASE_URL
environment variable, set before db is imported — this is what lets these
tests run in CI with no real Postgres instance available.
"""
import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import db  # noqa: E402  (must import after setting DATABASE_URL above)
from data_fetcher import Match  # noqa: E402


def test_record_prediction_is_idempotent():
    pred = {"home_win": 0.5, "draw": 0.3, "away_win": 0.2, "most_likely_score": (1, 1)}
    first = db.record_prediction("League", "A", "B", "2026-01-01T15:00:00Z", pred)
    second = db.record_prediction("League", "A", "B", "2026-01-01T15:00:00Z", pred)
    assert first is True
    assert second is False  # must not overwrite the original prediction


def test_resolve_predictions_fills_in_actual_result():
    pred = {"home_win": 0.6, "draw": 0.25, "away_win": 0.15, "most_likely_score": (2, 0)}
    db.record_prediction("League2", "X", "Y", "2026-02-01T15:00:00Z", pred)
    finished = [Match(home="X", away="Y", date="2026-02-01T15:00:00Z",
                       home_goals=2, away_goals=0, status="FINISHED")]
    db.resolve_predictions("League2", finished)
    rows = db.get_track_record("League2")
    assert len(rows) == 1
    assert rows[0]["actual_home_goals"] == 2
    assert rows[0]["resolved"] is True


def test_summarize_track_record_computes_accuracy():
    rows = [
        {"actual_home_goals": 2, "actual_away_goals": 0,
         "predicted_home_win": 0.7, "predicted_draw": 0.2, "predicted_away_win": 0.1,
         "value_bet_market": None, "value_bet_odds": None},
        {"actual_home_goals": 0, "actual_away_goals": 2,
         "predicted_home_win": 0.7, "predicted_draw": 0.2, "predicted_away_win": 0.1,
         "value_bet_market": None, "value_bet_odds": None},
    ]
    summary = db.summarize_track_record(rows)
    assert summary["count"] == 2
    assert summary["accuracy"] == 0.5  # got the first right, second wrong


def test_team_adjustment_replaces_not_stacks():
    db.set_team_adjustment("League3", "TeamA", 0.8, "first note")
    db.set_team_adjustment("League3", "TeamA", 0.6, "updated note")
    adjustments = db.get_team_adjustments("League3")
    assert adjustments["TeamA"] == 0.6


def test_push_subscription_deduplicates():
    db.save_push_subscription("https://example.com/endpoint1", "p256dh", "auth")
    db.save_push_subscription("https://example.com/endpoint1", "p256dh", "auth")
    subs = [s for s in db.get_push_subscriptions() if s["endpoint"] == "https://example.com/endpoint1"]
    assert len(subs) == 1
