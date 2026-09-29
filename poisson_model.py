"""
poisson_model.py
Dixon-Coles model: the standard statistical approach for football score
prediction (Dixon & Coles, 1997). Improves on a naive Poisson model in two
ways:
  1. Adjusts the (0-0, 1-0, 0-1, 1-1) low-scoring cells with a rho parameter,
     since plain independent Poisson slightly misprices these.
  2. Weights recent matches more heavily via exponential time decay, so a
     team's current form matters more than results from a year ago.

Each team gets an "attack" and "defense" rating fitted from historical goals.
Expected goals for a fixture = attack_home * defense_away * home_advantage
(and symmetrically for the away side). From expected goals we build a full
scoreline probability matrix, sum it into 1X2 probabilities, and read off
the most likely scoreline.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple

from data_fetcher import Match


@dataclass
class TeamRating:
    attack: float
    defense: float


class DixonColesModel:
    def __init__(self, time_decay_halflife_days: float = 180):
        self.halflife = time_decay_halflife_days
        self.teams: List[str] = []
        self.ratings: Dict[str, TeamRating] = {}
        self.home_advantage: float = 0.25
        self.rho: float = -0.05

    # ---- fitting ----

    def fit(self, matches: List[Match]):
        played = [m for m in matches if m.home_goals is not None]
        if len(played) < 20:
            raise ValueError("Need at least ~20 played matches to fit a stable model")

        self.teams = sorted({m.home for m in played} | {m.away for m in played})
        n = len(self.teams)
        idx = {t: i for i, t in enumerate(self.teams)}

        latest_date = max(self._parse_date(m.date) for m in played)
        weights = np.array([
            0.5 ** (((latest_date - self._parse_date(m.date)).days) / self.halflife)
            for m in played
        ])

        # Parameter vector: [attack_0..attack_n, defense_0..defense_n, home_adv, rho]
        # Attack/defense are constrained to average 0 (in log-space) for identifiability.
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])

        def unpack(x):
            attack = x[0:n]
            defense = x[n:2 * n]
            home_adv = x[2 * n]
            rho = x[2 * n + 1]
            return attack, defense, home_adv, rho

        def neg_log_likelihood(x):
            attack, defense, home_adv, rho = unpack(x)
            ll = 0.0
            for w, m in zip(weights, played):
                hi, ai = idx[m.home], idx[m.away]
                lam_home = np.exp(attack[hi] + defense[ai] + home_adv)
                lam_away = np.exp(attack[ai] + defense[hi])
                p_home = poisson.pmf(m.home_goals, lam_home)
                p_away = poisson.pmf(m.away_goals, lam_away)
                tau = self._tau(m.home_goals, m.away_goals, lam_home, lam_away, rho)
                p = max(p_home * p_away * tau, 1e-10)
                ll += w * np.log(p)
            # Light L2 regularization keeps ratings from drifting to extremes
            # on teams with few matches.
            reg = 0.01 * np.sum(attack ** 2) + 0.01 * np.sum(defense ** 2)
            return -ll + reg

        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x[0:n])},       # attack sums to 0
            {"type": "eq", "fun": lambda x: np.sum(x[n:2 * n])},   # defense sums to 0
        ]
        result = minimize(neg_log_likelihood, x0, constraints=constraints, method="SLSQP",
                           options={"maxiter": 200, "ftol": 1e-6})

        attack, defense, home_adv, rho = unpack(result.x)
        self.home_advantage = float(home_adv)
        self.rho = float(rho)
        for t in self.teams:
            i = idx[t]
            self.ratings[t] = TeamRating(attack=float(attack[i]), defense=float(defense[i]))

    @staticmethod
    def _tau(hg, ag, lam_h, lam_a, rho):
        """Dixon-Coles low-score correlation adjustment."""
        if hg == 0 and ag == 0:
            return 1 - lam_h * lam_a * rho
        elif hg == 0 and ag == 1:
            return 1 + lam_h * rho
        elif hg == 1 and ag == 0:
            return 1 + lam_a * rho
        elif hg == 1 and ag == 1:
            return 1 - rho
        return 1.0

    @staticmethod
    def _parse_date(d: str) -> datetime:
        return datetime.fromisoformat(d.replace("Z", "+00:00")).replace(tzinfo=None)

    # ---- prediction ----

    def expected_goals(self, home: str, away: str) -> Tuple[float, float]:
        if home not in self.ratings or away not in self.ratings:
            raise ValueError(f"No rating for {home} or {away} — team not in fitted data")
        h, a = self.ratings[home], self.ratings[away]
        lam_home = np.exp(h.attack + a.defense + self.home_advantage)
        lam_away = np.exp(a.attack + h.defense)
        return float(lam_home), float(lam_away)

    def predict(self, home: str, away: str, max_goals: int = 8) -> dict:
        """Full scoreline matrix -> 1X2 probabilities, most likely scoreline,
        and over/under 2.5 goals (a common betting market)."""
        lam_h, lam_a = self.expected_goals(home, away)
        grid = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a) * self._tau(i, j, lam_h, lam_a, self.rho)
                grid[i, j] = p
        grid /= grid.sum()  # renormalize after the tau adjustment

        home_win = float(np.tril(grid, -1).sum())
        draw = float(np.trace(grid))
        away_win = float(np.triu(grid, 1).sum())
        best_score = np.unravel_index(np.argmax(grid), grid.shape)

        over25 = float(sum(grid[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j > 2.5))

        return {
            "home_win": home_win, "draw": draw, "away_win": away_win,
            "most_likely_score": (int(best_score[0]), int(best_score[1])),
            "expected_goals": (round(lam_h, 2), round(lam_a, 2)),
            "over_2_5": over25, "under_2_5": 1 - over25,
        }


if __name__ == "__main__":
    # Self-test with synthetic data so this can be verified without live API access.
    import random
    random.seed(7)
    teams = ["Strong FC", "Mid United", "Weak Town", "Solid City", "Rising Rovers"]
    true_strength = {"Strong FC": 2.0, "Mid United": 1.3, "Weak Town": 0.7,
                      "Solid City": 1.5, "Rising Rovers": 1.1}
    synthetic = []
    for day in range(60):
        h, a = random.sample(teams, 2)
        lam_h = true_strength[h] * 1.2
        lam_a = true_strength[a] * 0.9
        synthetic.append(Match(
            home=h, away=a,
            date=f"2026-{(day % 9)+1:02d}-{(day % 27)+1:02d}T15:00:00Z",
            home_goals=np.random.poisson(lam_h), away_goals=np.random.poisson(lam_a),
            status="FINISHED",
        ))

    model = DixonColesModel()
    model.fit(synthetic)
    print("Fitted ratings (higher attack / lower defense = stronger):")
    for t in teams:
        r = model.ratings[t]
        print(f"  {t:15s} attack={r.attack:+.2f}  defense={r.defense:+.2f}")

    print("\nPrediction: Strong FC vs Weak Town")
    print(model.predict("Strong FC", "Weak Town"))
