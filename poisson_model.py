"""
poisson_model.py
Dixon-Coles model with HOME/AWAY SPLIT RATINGS: each team gets four numbers
instead of two — attack_home, defense_home, attack_away, defense_away —
because a team's scoring and conceding patterns often differ meaningfully
between home and away (a team can be devastating at home and mediocre on
the road). A single combined rating averages that away; splitting it lets
the model actually use the difference.

    lam_home = exp(attack_home[home_team] + defense_away[away_team])
    lam_away = exp(attack_away[away_team] + defense_home[home_team])

Still uses the Dixon-Coles low-score correlation adjustment and exponential
time-decay weighting from before.

Also includes blend_with_market(): combines the model's own probabilities
with the bookmaker's de-vigged implied probabilities. Bookmaker odds react
to team news (injuries, suspensions, lineup changes) that a goals-only
model has no way to see, so a blended forecast is typically better
calibrated than the model alone — but that blended number must only be
used for *display*, never for value-bet detection, since blending toward
the market is exactly what would erase the edge you're trying to find.

CROSS-COMPETITION PRIORS: fit() accepts an optional `priors` dict mapping
team name -> TeamRating, used to regularize toward instead of toward zero.
This exists for competitions like the Champions League, where early in a
campaign there often aren't 20 played matches yet (or the ones that exist
are too few to pin down a reliable rating). Seeding a team's European
rating from their domestic league rating — where the same team usually has
a full season of data — gives the model a sensible starting point instead
of an uninformed guess, and lets actual European results pull the rating
away from that prior as they accumulate. A team not present in `priors`
(e.g. truly no data anywhere) still regularizes toward zero, same as before.
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
    attack_home: float
    defense_home: float
    attack_away: float
    defense_away: float


class DixonColesModel:
    def __init__(self, time_decay_halflife_days: float = 180):
        self.halflife = time_decay_halflife_days
        self.teams: List[str] = []
        self.ratings: Dict[str, TeamRating] = {}
        self.rho: float = -0.05

    # ---- fitting ----

    def fit(self, matches: List[Match], priors: Dict[str, TeamRating] = None,
             prior_weight: float = 0.02):
        """
        priors: optional {team_name: TeamRating}, e.g. that team's ratings
        fitted from their domestic league. A team found in `priors` is
        regularized toward that rating instead of toward zero; a team not
        in `priors` still regularizes toward zero exactly as before — so
        passing no priors reproduces the old behavior exactly.
        prior_weight: same role as the old fixed 0.02 L2 regularization
        constant, just now pulling toward a (possibly non-zero) prior
        instead of always toward zero.
        """
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

        priors = priors or {}
        prior_ah = np.zeros(n)
        prior_dh = np.zeros(n)
        prior_aa = np.zeros(n)
        prior_da = np.zeros(n)
        for t in self.teams:
            if t in priors:
                p = priors[t]
                i = idx[t]
                prior_ah[i], prior_dh[i] = p.attack_home, p.defense_home
                prior_aa[i], prior_da[i] = p.attack_away, p.defense_away

        # Parameter vector: [attack_home(n), defense_home(n), attack_away(n), defense_away(n), rho]
        # Splitting home/away doubles the parameter count versus a single
        # combined rating, so the L2 regularization weight is a bit higher
        # here to keep teams with thin data (e.g. newly promoted sides, or
        # anyone in a short competition like the Champions League group
        # stage) from swinging to extreme, overfit values. Starting the
        # optimizer at the priors (rather than always at zero) also helps
        # it converge faster when most teams already have a sensible
        # starting point from their domestic rating.
        n_params = 4 * n
        x0 = np.concatenate([prior_ah, prior_dh, prior_aa, prior_da, [-0.05]])

        def unpack(x):
            ah = x[0:n]
            dh = x[n:2*n]
            aa = x[2*n:3*n]
            da = x[3*n:4*n]
            rho = x[4*n]
            return ah, dh, aa, da, rho

        def neg_log_likelihood(x):
            ah, dh, aa, da, rho = unpack(x)
            ll = 0.0
            for w, m in zip(weights, played):
                hi, ai = idx[m.home], idx[m.away]
                lam_home = np.exp(ah[hi] + da[ai])
                lam_away = np.exp(aa[ai] + dh[hi])
                p_home = poisson.pmf(m.home_goals, lam_home)
                p_away = poisson.pmf(m.away_goals, lam_away)
                tau = self._tau(m.home_goals, m.away_goals, lam_home, lam_away, rho)
                p = max(p_home * p_away * tau, 1e-10)
                ll += w * np.log(p)
            reg = (prior_weight * np.sum((ah - prior_ah) ** 2)
                   + prior_weight * np.sum((dh - prior_dh) ** 2)
                   + prior_weight * np.sum((aa - prior_aa) ** 2)
                   + prior_weight * np.sum((da - prior_da) ** 2))
            return -ll + reg

        constraints = [
            {"type": "eq", "fun": lambda x: np.sum(x[0:n])},
            {"type": "eq", "fun": lambda x: np.sum(x[n:2*n])},
            {"type": "eq", "fun": lambda x: np.sum(x[2*n:3*n])},
            {"type": "eq", "fun": lambda x: np.sum(x[3*n:4*n])},
        ]
        result = minimize(neg_log_likelihood, x0, constraints=constraints, method="SLSQP",
                           options={"maxiter": 300, "ftol": 1e-6})

        ah, dh, aa, da, rho = unpack(result.x)
        self.rho = float(rho)
        for t in self.teams:
            i = idx[t]
            self.ratings[t] = TeamRating(
                attack_home=float(ah[i]), defense_home=float(dh[i]),
                attack_away=float(aa[i]), defense_away=float(da[i]),
            )

    def apply_promoted_team_shrinkage(self, matches: List[Match], min_matches: int = 15,
                                       newcomer_prior=(-0.15, 0.15, -0.20, 0.20)):
        """
        Call after fit(). A team with only a handful of matches can get an
        extreme fitted rating driven almost entirely by a lucky or unlucky
        run — three wins in their first three games says much less about
        true quality than three wins out of thirty. This pulls any
        under-sampled team's rating toward a "below-average newcomer" prior,
        proportionally to how little data they have: full shrink at 0
        matches, no shrink at `min_matches` or more, linear in between.

        The prior itself is deliberately below average (slightly worse
        attack, slightly worse defense on both sides) rather than neutral —
        newly promoted teams historically underperform a neutral-rated
        guess in their first season at a new level, so neutral would still
        be too generous.

        newcomer_prior = (attack_home, defense_home, attack_away, defense_away).
        This is independent of fit()'s cross-competition `priors` argument:
        that seeds a team's rating from a DIFFERENT competition's data
        (e.g. domestic league rating for a Champions League debut), while
        this handles a team with genuinely little data anywhere.
        """
        match_counts: Dict[str, int] = {}
        for m in matches:
            if m.home_goals is None:
                continue
            match_counts[m.home] = match_counts.get(m.home, 0) + 1
            match_counts[m.away] = match_counts.get(m.away, 0) + 1

        p_ah, p_dh, p_aa, p_da = newcomer_prior
        for team, rating in list(self.ratings.items()):
            n = match_counts.get(team, 0)
            if n >= min_matches:
                continue
            shrink = max(0.0, (min_matches - n) / min_matches)
            self.ratings[team] = TeamRating(
                attack_home=(1 - shrink) * rating.attack_home + shrink * p_ah,
                defense_home=(1 - shrink) * rating.defense_home + shrink * p_dh,
                attack_away=(1 - shrink) * rating.attack_away + shrink * p_aa,
                defense_away=(1 - shrink) * rating.defense_away + shrink * p_da,
            )

    @staticmethod
    def _tau(hg, ag, lam_h, lam_a, rho):
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
        lam_home = np.exp(h.attack_home + a.defense_away)
        lam_away = np.exp(a.attack_away + h.defense_home)
        return float(lam_home), float(lam_away)

    def predict(self, home: str, away: str, max_goals: int = 8,
                home_goal_multiplier: float = 1.0, away_goal_multiplier: float = 1.0) -> dict:
        """
        home_goal_multiplier / away_goal_multiplier scale the expected goals
        for that side before building the scoreline grid — this is the hook
        for manual injury/lineup adjustments (e.g. 0.8 for "missing their
        top striker"). No free API gives reliable real-time injury data, so
        this is meant to be set by hand via the /admin/team-adjustment
        endpoint in api_server.py when you know something the model can't.
        """
        lam_h, lam_a = self.expected_goals(home, away)
        lam_h *= home_goal_multiplier
        lam_a *= away_goal_multiplier
        grid = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a) * self._tau(i, j, lam_h, lam_a, self.rho)
                grid[i, j] = p
        grid /= grid.sum()

        home_win = float(np.tril(grid, -1).sum())
        draw = float(np.trace(grid))
        away_win = float(np.triu(grid, 1).sum())
        best_score = np.unravel_index(np.argmax(grid), grid.shape)
        over25 = float(sum(grid[i, j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i + j > 2.5))
        btts = float(sum(grid[i, j] for i in range(1, max_goals + 1) for j in range(1, max_goals + 1)))

        return {
            "home_win": home_win, "draw": draw, "away_win": away_win,
            "most_likely_score": (int(best_score[0]), int(best_score[1])),
            "expected_goals": (round(lam_h, 2), round(lam_a, 2)),
            "over_2_5": over25, "under_2_5": 1 - over25,
            "btts_yes": btts, "btts_no": 1 - btts,
        }


def blend_with_market(model_probs: dict, fair_h: float, fair_d: float, fair_a: float,
                       weight: float = 0.5) -> dict:
    """
    weight is how much trust goes to the model (1.0 = pure model, 0.0 = pure
    market). Use the result only for what's shown to users — value-bet
    detection must keep using the model's raw, unblended probabilities,
    since blending toward the market by definition shrinks any edge.
    """
    bh = weight * model_probs["home_win"] + (1 - weight) * fair_h
    bd = weight * model_probs["draw"] + (1 - weight) * fair_d
    ba = weight * model_probs["away_win"] + (1 - weight) * fair_a
    total = bh + bd + ba
    return {"home_win": bh / total, "draw": bd / total, "away_win": ba / total}


if __name__ == "__main__":
    import random
    random.seed(7)
    teams = ["Strong FC", "Mid United", "Weak Town", "Solid City", "Rising Rovers"]
    true_strength = {"Strong FC": 2.0, "Mid United": 1.3, "Weak Town": 0.7,
                      "Solid City": 1.5, "Rising Rovers": 1.1}
    synthetic = []
    for day in range(80):
        h, a = random.sample(teams, 2)
        lam_h = true_strength[h] * 1.25
        lam_a = true_strength[a] * 0.85
        synthetic.append(Match(
            home=h, away=a,
            date=f"2026-{(day % 9)+1:02d}-{(day % 27)+1:02d}T15:00:00Z",
            home_goals=np.random.poisson(lam_h), away_goals=np.random.poisson(lam_a),
            status="FINISHED",
        ))

    model = DixonColesModel()
    model.fit(synthetic)
    print("Fitted ratings:")
    for t in teams:
        r = model.ratings[t]
        print(f"  {t:15s} home: atk={r.attack_home:+.2f} def={r.defense_home:+.2f}"
              f"  |  away: atk={r.attack_away:+.2f} def={r.defense_away:+.2f}")

    print("\nPrediction: Strong FC (home) vs Weak Town (away)")
    pred = model.predict("Strong FC", "Weak Town")
    print(pred)

    print("\nBlending with a hypothetical market view (fair probs 0.55/0.25/0.20):")
    print(blend_with_market(pred, 0.55, 0.25, 0.20, weight=0.5))
