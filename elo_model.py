"""
elo_model.py
A simple Elo rating system, used as a second, independent estimate that gets
ensembled with the Dixon-Coles model. Elo and a goals-based Poisson model
tend to make different mistakes — Elo reacts purely to win/draw/loss and a
team's recent run of results, while Dixon-Coles is driven by actual goal
difference. Averaging two models that fail in different ways is a standard
way to get something more robust than either alone — the same principle
behind blending with the market in poisson_model.py, just using a second
model instead of bookmaker odds.
"""

from typing import List, Dict
from data_fetcher import Match

DEFAULT_RATING = 1500.0


class EloModel:
    def __init__(self, k_factor: float = 20.0, home_advantage: float = 60.0):
        self.k = k_factor
        self.home_adv = home_advantage
        self.ratings: Dict[str, float] = {}

    def _get(self, team: str) -> float:
        return self.ratings.get(team, DEFAULT_RATING)

    def fit(self, matches: List[Match]):
        """Elo is inherently sequential (each result updates the next
        prediction), unlike Dixon-Coles' one-shot batch fit — so matches are
        processed in chronological order here."""
        played = [m for m in matches if m.home_goals is not None]
        played.sort(key=lambda m: m.date)
        self.ratings = {}
        for m in played:
            self._update(m)

    def _update(self, m: Match):
        r_home = self._get(m.home) + self.home_adv
        r_away = self._get(m.away)
        expected_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        if m.home_goals > m.away_goals:
            actual_home = 1.0
        elif m.home_goals < m.away_goals:
            actual_home = 0.0
        else:
            actual_home = 0.5
        # Margin-of-victory multiplier: a 3-0 win is stronger evidence of a
        # rating gap than a 1-0 win, so it should move ratings more.
        margin = abs(m.home_goals - m.away_goals)
        mov_multiplier = 1.0 if margin <= 1 else (1.0 + 0.25 * (margin - 1)) ** 0.5
        delta = self.k * mov_multiplier * (actual_home - expected_home)
        self.ratings[m.home] = self._get(m.home) + delta
        self.ratings[m.away] = self._get(m.away) - delta

    def predict(self, home: str, away: str) -> dict:
        r_home = self._get(home) + self.home_adv
        r_away = self._get(away)
        expected_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        # Elo natively only gives a single win-probability-like number, not
        # a draw/win/loss split — draw share is modeled separately here,
        # highest for evenly matched teams and shrinking as the gap widens.
        rating_gap = abs(r_home - r_away)
        draw = max(0.18, 0.30 - rating_gap / 1600)
        home_win = expected_home * (1 - draw)
        away_win = (1 - expected_home) * (1 - draw)
        return {"home_win": home_win, "draw": draw, "away_win": away_win}


def ensemble(dixon_coles_probs: dict, elo_probs: dict, weight_dc: float = 0.7) -> dict:
    """weight_dc is how much trust goes to Dixon-Coles vs Elo (0.7 = 70%
    Dixon-Coles, 30% Elo). Dixon-Coles gets more weight by default since it
    uses actual goal data, which carries more information than win/draw/loss
    alone — but Elo still catches momentum/form patterns DC can miss."""
    h = weight_dc * dixon_coles_probs["home_win"] + (1 - weight_dc) * elo_probs["home_win"]
    d = weight_dc * dixon_coles_probs["draw"] + (1 - weight_dc) * elo_probs["draw"]
    a = weight_dc * dixon_coles_probs["away_win"] + (1 - weight_dc) * elo_probs["away_win"]
    total = h + d + a
    return {"home_win": h / total, "draw": d / total, "away_win": a / total}


if __name__ == "__main__":
    import random
    import numpy as np
    random.seed(3)
    teams = ["Strong FC", "Mid United", "Weak Town"]
    strength = {"Strong FC": 2.0, "Mid United": 1.2, "Weak Town": 0.6}
    matches = []
    for day in range(40):
        h, a = random.sample(teams, 2)
        matches.append(Match(
            home=h, away=a, date=f"2026-01-{(day % 27) + 1:02d}T15:00:00Z",
            home_goals=int(np.random.poisson(strength[h] * 1.2)),
            away_goals=int(np.random.poisson(strength[a] * 0.9)),
            status="FINISHED",
        ))
    model = EloModel()
    model.fit(matches)
    print("Elo ratings:", {t: round(r, 1) for t, r in model.ratings.items()})
    pred = model.predict("Strong FC", "Weak Town")
    print("Elo prediction Strong vs Weak:", pred)

    fake_dc = {"home_win": 0.70, "draw": 0.18, "away_win": 0.12}
    print("Ensembled with a hypothetical Dixon-Coles output:", ensemble(fake_dc, pred))
