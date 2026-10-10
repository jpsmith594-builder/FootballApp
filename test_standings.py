"""
Tests the league-table feature: FootballDataClient.get_standings() shaping
the raw API response, and db.py's replace_standings/get_standings cache
(same wholesale-replace pattern as cached_fixtures). Purely display data —
none of this touches the prediction model.
"""
import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

import db  # noqa: E402
from data_fetcher import FootballDataClient  # noqa: E402


def _fake_standings_response(entries):
    """entries: list of (position, team_name, crest, played, won, drawn, lost, gf, ga, gd, points)"""
    return {
        "standings": [
            {
                "stage": "REGULAR_SEASON",
                "type": "TOTAL",
                "group": None,
                "table": [
                    {
                        "position": pos, "team": {"name": name, "crest": crest},
                        "playedGames": played, "won": won, "draw": drawn, "lost": lost,
                        "goalsFor": gf, "goalsAgainst": ga, "goalDifference": gd, "points": points,
                    }
                    for pos, name, crest, played, won, drawn, lost, gf, ga, gd, points in entries
                ],
            },
            {
                # A HOME-only table also comes back in the real API response —
                # get_standings must pick out TOTAL specifically, not just
                # grab the first entry.
                "stage": "REGULAR_SEASON", "type": "HOME", "group": None, "table": [],
            },
        ]
    }


def test_get_standings_shapes_total_table():
    client = FootballDataClient.__new__(FootballDataClient)  # skip __init__'s API-key check
    client._get = lambda path, params=None: _fake_standings_response([
        (1, "Arsenal", "https://crest/arsenal.png", 10, 8, 1, 1, 20, 8, 12, 25),
        (2, "Liverpool", "https://crest/liverpool.png", 10, 7, 2, 1, 22, 10, 12, 23),
    ])
    rows = client.get_standings("PL")
    assert len(rows) == 2
    assert rows[0] == {
        "position": 1, "team": "Arsenal", "crest": "https://crest/arsenal.png",
        "played": 10, "won": 8, "drawn": 1, "lost": 1,
        "goals_for": 20, "goals_against": 8, "goal_difference": 12, "points": 25,
    }


def test_get_standings_returns_empty_when_no_total_table():
    client = FootballDataClient.__new__(FootballDataClient)
    client._get = lambda path, params=None: {"standings": [{"type": "HOME", "table": []}]}
    assert client.get_standings("PL") == []


def test_replace_standings_round_trips():
    rows = [
        {"position": 1, "team": "Arsenal", "crest": None, "played": 10, "won": 8,
         "drawn": 1, "lost": 1, "goals_for": 20, "goals_against": 8,
         "goal_difference": 12, "points": 25},
        {"position": 2, "team": "Liverpool", "crest": None, "played": 10, "won": 7,
         "drawn": 2, "lost": 1, "goals_for": 22, "goals_against": 10,
         "goal_difference": 12, "points": 23},
    ]
    db.replace_standings("TestLeague", rows)
    stored = db.get_standings("TestLeague")
    assert len(stored) == 2
    assert stored[0]["team"] == "Arsenal"
    assert stored[0]["position"] == 1
    assert stored[1]["team"] == "Liverpool"


def test_replace_standings_fully_replaces_stale_rows():
    db.replace_standings("Stale", [
        {"position": 1, "team": "Old Leader", "crest": None, "played": 5, "won": 5,
         "drawn": 0, "lost": 0, "goals_for": 10, "goals_against": 2,
         "goal_difference": 8, "points": 15},
    ])
    assert len(db.get_standings("Stale")) == 1
    db.replace_standings("Stale", [
        {"position": 1, "team": "New Leader", "crest": None, "played": 6, "won": 6,
         "drawn": 0, "lost": 0, "goals_for": 14, "goals_against": 2,
         "goal_difference": 12, "points": 18},
    ])
    rows = db.get_standings("Stale")
    assert len(rows) == 1
    assert rows[0]["team"] == "New Leader"


def test_get_standings_unknown_league_returns_empty():
    assert db.get_standings("NeverRefreshedLeague") == []
