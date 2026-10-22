"""
Tests the Odds API credit-saving layer: db.py's odds cache and
engine.get_odds_by_fixture(), which only spends a credit when a league's
stored odds are stale (and never when there are no fixtures).
"""
import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from datetime import datetime, timedelta, timezone  # noqa: E402

import db  # noqa: E402
import engine  # noqa: E402
from value_bet import OddsQuote  # noqa: E402


def _quote_dict(home="A", away="B"):
    return {"home_team": home, "away_team": away, "home_odds": 2.0,
            "draw_odds": 3.4, "away_odds": 3.8, "bookmaker": "TestBook"}


def _backdate_fetch(league, hours):
    old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours)
    with db.engine.begin() as conn:
        conn.execute(db.update(db.odds_fetch_meta)
                     .where(db.odds_fetch_meta.c.league == league).values(fetched_at=old))


def _counting_fetch(monkeypatch, quotes=None, fail=False):
    calls = []

    def fake(league):
        calls.append(league)
        if fail:
            raise RuntimeError("quota exhausted")
        return quotes or []
    monkeypatch.setattr(engine, "fetch_odds", fake)
    monkeypatch.setattr(engine, "ODDS_ENABLED", True)
    return calls


def test_odds_cache_round_trips_and_replaces():
    db.replace_odds_cache("RT", [_quote_dict("A", "B"), _quote_dict("C", "D")])
    quotes, fetched_at = db.get_odds_cache("RT")
    assert {q["home_team"] for q in quotes} == {"A", "C"}
    assert fetched_at is not None
    db.replace_odds_cache("RT", [_quote_dict("E", "F")])
    quotes, _ = db.get_odds_cache("RT")
    assert [q["home_team"] for q in quotes] == ["E"]


def test_empty_fetch_still_stamps_fetch_time():
    db.replace_odds_cache("EmptyLg", [])
    quotes, fetched_at = db.get_odds_cache("EmptyLg")
    assert quotes == [] and fetched_at is not None


def test_unknown_league_has_no_cache():
    assert db.get_odds_cache("NeverFetched") == ([], None)


def test_fresh_cache_is_reused_without_calling_the_api(monkeypatch):
    db.replace_odds_cache("Fresh", [_quote_dict()])
    calls = _counting_fetch(monkeypatch)
    result = engine.get_odds_by_fixture("Fresh")
    assert calls == []
    assert isinstance(result[("A", "B")], OddsQuote)


def test_stale_cache_triggers_a_fetch_and_stores_it(monkeypatch):
    db.replace_odds_cache("Stale", [_quote_dict("Old", "Game")])
    _backdate_fetch("Stale", hours=engine.ODDS_MAX_AGE_HOURS + 1)
    fresh = [OddsQuote("New", "Game", 1.9, 3.5, 4.2, "TestBook")]
    calls = _counting_fetch(monkeypatch, quotes=fresh)
    result = engine.get_odds_by_fixture("Stale")
    assert calls == ["Stale"]
    assert ("New", "Game") in result and ("Old", "Game") not in result
    assert db.get_odds_cache("Stale")[0][0]["home_team"] == "New"


def test_never_fetched_league_fetches_once(monkeypatch):
    calls = _counting_fetch(monkeypatch, quotes=[OddsQuote("X", "Y", 2.0, 3.0, 4.0, "B")])
    engine.get_odds_by_fixture("Brand New")
    engine.get_odds_by_fixture("Brand New")  # second run reuses the stored copy
    assert calls == ["Brand New"]


def test_failed_fetch_falls_back_to_stale_copy(monkeypatch):
    db.replace_odds_cache("Fallback", [_quote_dict("Kept", "Quote")])
    _backdate_fetch("Fallback", hours=engine.ODDS_MAX_AGE_HOURS + 5)
    _counting_fetch(monkeypatch, fail=True)
    result = engine.get_odds_by_fixture("Fallback")
    assert ("Kept", "Quote") in result


def test_no_fixtures_means_no_api_call(monkeypatch):
    calls = _counting_fetch(monkeypatch)
    assert engine.get_odds_by_fixture("NoFixtures", has_fixtures=False) == {}
    assert calls == []


def test_odds_disabled_without_api_key(monkeypatch):
    calls = _counting_fetch(monkeypatch)
    monkeypatch.setattr(engine, "ODDS_ENABLED", False)
    assert engine.get_odds_by_fixture("Anything") == {}
    assert calls == []
