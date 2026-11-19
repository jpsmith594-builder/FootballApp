"""
Tests model-generation tagging: track-record rows record which model wrote
them, accuracy and calibration only judge the current generation, and the
database upgrades itself when a column is missing (the failure that broke the
Hint Accuracy tab once already).
"""
import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from datetime import datetime  # noqa: E402

from sqlalchemy import create_engine, inspect, insert, text  # noqa: E402

import calibration  # noqa: E402
import db  # noqa: E402
from data_fetcher import Match  # noqa: E402


def _pred(confidence=0.9):
    return {"home_win": confidence, "draw": (1 - confidence) * 0.6,
            "away_win": (1 - confidence) * 0.4, "most_likely_score": (2, 0)}


def _insert_resolved(league, n, model_version, confidence=0.9, correct_fraction=0.5):
    """Write finished rows directly, so a specific (or NULL) model_version can be set."""
    n_correct = round(n * correct_fraction)
    with db.engine.begin() as conn:
        for i in range(n):
            correct = i < n_correct
            conn.execute(insert(db.predictions).values(
                league=league, home=f"H{i}", away=f"A{i}", match_date=f"2026-01-{(i % 27) + 1:02d}T15:00:00Z",
                predicted_home_win=confidence, predicted_draw=(1 - confidence) * 0.6,
                predicted_away_win=(1 - confidence) * 0.4,
                predicted_score_home=2, predicted_score_away=0,
                hint_market="Straight Win", hint_selection=f"H{i}", hint_side="Home", hint_confidence=confidence,
                actual_home_goals=2 if correct else 0, actual_away_goals=0 if correct else 2,
                resolved=True, recorded_at=datetime(2026, 1, 1), model_version=model_version,
            ))


def test_record_prediction_tags_current_model_version():
    db.record_prediction("VerTag", "A", "B", "2026-10-10T15:00:00Z", _pred())
    with db.engine.begin() as conn:
        row = conn.execute(text("select model_version from predictions where league='VerTag'")).first()
    assert row[0] == db.CURRENT_MODEL_VERSION


def test_calibration_curve_defaults_to_current_version_only():
    _insert_resolved("CalVer", n=30, model_version=None, confidence=0.9, correct_fraction=0.2)   # old model
    _insert_resolved("CalVer", n=30, model_version=db.CURRENT_MODEL_VERSION, confidence=0.9, correct_fraction=0.9)
    current = db.get_calibration_curve("CalVer")
    assert current["total_resolved_predictions"] == 30
    assert current["buckets"][0]["actual_accuracy"] == 0.9
    assert db.get_calibration_curve("CalVer", model_version=None)["total_resolved_predictions"] == 60
    # version 1 means "before versioning", which is stored as NULL
    assert db.get_calibration_curve("CalVer", model_version=1)["total_resolved_predictions"] == 30


def test_track_record_readers_default_to_current_version():
    _insert_resolved("TRVer", n=5, model_version=None)
    _insert_resolved("TRVer", n=7, model_version=db.CURRENT_MODEL_VERSION)
    assert len(db.get_track_record("TRVer")) == 7
    assert len(db.get_hint_track_record("TRVer")) == 7
    assert len(db.get_track_record("TRVer", model_version=None)) == 12


def test_correction_ignores_rows_from_an_older_model():
    # 80 resolved rows exist, but all from the old model: nothing to learn from yet.
    _insert_resolved("OnlyOld", n=80, model_version=None, confidence=0.9, correct_fraction=0.4)
    assert calibration.build_correction("OnlyOld") is None


def test_correction_learns_from_current_model_rows():
    _insert_resolved("Mixed", n=80, model_version=None, confidence=0.9, correct_fraction=0.9)
    _insert_resolved("Mixed", n=60, model_version=db.CURRENT_MODEL_VERSION, confidence=0.9, correct_fraction=0.5)
    fn = calibration.build_correction("Mixed")
    assert fn is not None
    assert abs(fn(0.9) - 0.5) < 1e-6      # the old model's 90% hit rate must not leak in


def test_correction_never_raises_into_the_refresh(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("database hiccup")
    monkeypatch.setattr(db, "get_calibration_curve", boom)
    assert calibration.build_correction("Anything") is None


def test_add_missing_columns_upgrades_an_old_table(tmp_path):
    old_db = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with old_db.begin() as conn:
        # predictions as it looked before hints and model versions existed
        conn.execute(text("create table predictions (id integer primary key, league varchar not null, "
                          "home varchar not null, away varchar not null, match_date varchar not null, "
                          "predicted_home_win float, resolved boolean)"))
    before = {c["name"] for c in inspect(old_db).get_columns("predictions")}
    assert "model_version" not in before and "hint_market" not in before
    db.add_missing_columns(old_db)
    after = {c["name"] for c in inspect(old_db).get_columns("predictions")}
    assert {"model_version", "hint_market", "hint_side", "hint_confidence"} <= after
    db.add_missing_columns(old_db)   # idempotent
    assert {c["name"] for c in inspect(old_db).get_columns("predictions")} == after


# ---- upgrading rows recorded by an older model, before kickoff ----

def _future():
    from datetime import timedelta, timezone as tz
    return (datetime.now(tz.utc) + timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _past():
    from datetime import timedelta, timezone as tz
    return (datetime.now(tz.utc) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(league, home):
    with db.engine.begin() as conn:
        return conn.execute(text(
            "select predicted_home_win, model_version, hint_side from predictions "
            f"where league='{league}' and home='{home}'")).first()


def _insert_old_unresolved(league, home, match_date):
    with db.engine.begin() as conn:
        conn.execute(insert(db.predictions).values(
            league=league, home=home, away="Away", match_date=match_date,
            predicted_home_win=0.30, predicted_draw=0.35, predicted_away_win=0.35,
            predicted_score_home=1, predicted_score_away=1, resolved=False,
            recorded_at=datetime(2026, 1, 1), model_version=None))


def test_old_model_row_for_an_upcoming_match_is_upgraded_to_the_new_model():
    when = _future()
    _insert_old_unresolved("Upg", "Up", when)
    hint = {"market": "Straight Win", "selection": "Up", "confidence": 0.7, "side": "Home"}
    inserted = db.record_prediction("Upg", "Up", "Away", when, _pred(0.7), None, hint)
    assert inserted is False   # not a brand-new row, so no repeat push notification
    p_home, version, side = _row("Upg", "Up")
    assert (p_home, version, side) == (0.7, db.CURRENT_MODEL_VERSION, "Home")


def test_row_for_a_match_that_already_started_is_never_touched():
    when = _past()
    _insert_old_unresolved("Late", "Late", when)
    db.record_prediction("Late", "Late", "Away", when, _pred(0.9))
    p_home, version, _ = _row("Late", "Late")
    assert p_home == 0.30 and version is None


def test_current_model_rows_are_never_overwritten():
    when = _future()
    db.record_prediction("Frozen", "Fz", "Away", when, _pred(0.6))
    db.record_prediction("Frozen", "Fz", "Away", when, _pred(0.9))   # a later refresh
    assert _row("Frozen", "Fz")[0] == 0.6
