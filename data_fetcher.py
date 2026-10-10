"""
data_fetcher.py
Pulls fixtures and historical results from football-data.org (free tier covers
the top 5 European leagues at 10 req/min). Get a free key at:
https://www.football-data.org/client/register

Swap the BASE_URL/headers here if you'd rather use API-Football (RapidAPI) or
The Odds API for bookmaker prices — the rest of the pipeline (model, value bet
detection) doesn't care where the data came from, as long as it's shaped into
the Match records defined below.
"""

import os
import time
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")
BASE_URL = "https://api.football-data.org/v4"

# football-data.org competition codes. All ten of these are on the free tier.
LEAGUES = {
    "Premier League": "PL",
    "La Liga": "PD",
    "Serie A": "SA",
    "Bundesliga": "BL1",
    "Ligue 1": "FL1",
    "Champions League": "CL",
    "Eredivisie": "DED",
    "Championship": "ELC",       # English second tier
    "Brazil Serie A": "BSA",
    "Liga Portugal": "PPL",
}

# Champions League is a special case for the model: early in the season there
# aren't 20 played group-stage matches yet, so DixonColesModel.fit() will
# raise until enough fixtures have been played. That's expected, not a bug —
# api_server.py already catches this per-league and just leaves it out of
# the cache until there's enough data.


@dataclass
class Match:
    home: str
    away: str
    date: str
    home_goals: Optional[int] = None   # None for fixtures that haven't been played
    away_goals: Optional[int] = None
    status: str = "SCHEDULED"          # SCHEDULED | FINISHED | LIVE
    home_crest: Optional[str] = None   # URL to team badge, from football-data.org
    away_crest: Optional[str] = None


class FootballDataClient:
    def __init__(self, api_key: str = API_KEY):
        if not api_key:
            raise ValueError(
                "No API key set. Get a free one at football-data.org and set "
                "FOOTBALL_DATA_API_KEY as an environment variable."
            )
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": api_key})

    def _get(self, path: str, params: dict = None) -> dict:
        # timeout=15 is essential here: without it, a slow or unreachable API
        # can hang a request forever, which on Render can stall the whole
        # startup sequence and trigger a "no open ports detected" failure.
        resp = self.session.get(f"{BASE_URL}{path}", params=params or {}, timeout=15)
        if resp.status_code == 429:
            # Free tier rate limit — back off and retry once.
            time.sleep(6)
            resp = self.session.get(f"{BASE_URL}{path}", params=params or {}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_results(self, league_code: str, days_back: int = 365) -> List[Match]:
        """Finished matches, used to fit the prediction model."""
        date_from = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to = datetime.utcnow().strftime("%Y-%m-%d")
        data = self._get(
            f"/competitions/{league_code}/matches",
            params={"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"},
        )
        matches = []
        for m in data.get("matches", []):
            score = m.get("score", {}).get("fullTime", {})
            if score.get("home") is None:
                continue
            matches.append(Match(
                home=m["homeTeam"]["name"],
                away=m["awayTeam"]["name"],
                date=m["utcDate"],
                home_goals=score["home"],
                away_goals=score["away"],
                status="FINISHED",
                home_crest=m["homeTeam"].get("crest"),
                away_crest=m["awayTeam"].get("crest"),
            ))
        return matches

    def get_standings(self, league_code: str) -> List[dict]:
        """Current league table — position, points, goal difference, etc.
        Purely for display; the prediction model never reads this. Only
        handles the flat "TOTAL" table that a domestic league returns —
        Champions League (and other group-stage competitions) return a
        different, per-group shape, so callers should skip this for those
        competitions rather than rely on it here."""
        data = self._get(f"/competitions/{league_code}/standings")
        total = next((s for s in data.get("standings", []) if s.get("type") == "TOTAL"), None)
        if not total:
            return []
        rows = []
        for row in total.get("table", []):
            team = row.get("team", {})
            rows.append({
                "position": row.get("position"),
                "team": team.get("name"),
                "crest": team.get("crest"),
                "played": row.get("playedGames"),
                "won": row.get("won"),
                "drawn": row.get("draw"),
                "lost": row.get("lost"),
                "goals_for": row.get("goalsFor"),
                "goals_against": row.get("goalsAgainst"),
                "goal_difference": row.get("goalDifference"),
                "points": row.get("points"),
            })
        return rows

    def get_upcoming_fixtures(self, league_code: str, days_ahead: int = 10) -> List[Match]:
        """Scheduled matches to generate predictions for."""
        date_from = datetime.utcnow().strftime("%Y-%m-%d")
        date_to = (datetime.utcnow() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        data = self._get(
            f"/competitions/{league_code}/matches",
            params={"dateFrom": date_from, "dateTo": date_to, "status": "SCHEDULED"},
        )
        return [
            Match(home=m["homeTeam"]["name"], away=m["awayTeam"]["name"],
                  date=m["utcDate"], status="SCHEDULED",
                  home_crest=m["homeTeam"].get("crest"), away_crest=m["awayTeam"].get("crest"))
            for m in data.get("matches", [])
        ]


if __name__ == "__main__":
    # Quick manual test: set FOOTBALL_DATA_API_KEY then run this file directly.
    client = FootballDataClient()
    fixtures = client.get_upcoming_fixtures(LEAGUES["Premier League"])
    for f in fixtures[:5]:
        print(f"{f.date[:10]}  {f.home} vs {f.away}")
