"""
Regression tests for the Dixon-Coles baseline fix: the fitted model has to
reproduce a league's scoring level and home advantage. Before the fix, team
ratings were forced to average zero with no league-wide intercept, so every
average match came out as 1.0 goals each side — which understated home
advantage, Over 2.5 and BTTS (and made 0-0 the "most likely score" far too often).
"""
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import time  # noqa: E402

import numpy as np  # noqa: E402

from data_fetcher import Match  # noqa: E402
from poisson_model import DixonColesModel  # noqa: E402

N_TEAMS = 20
MU_HOME, MU_AWAY = np.log(1.6), np.log(1.25)   # a realistic league: 1.6 / 1.25 goals


def _league(seed):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(N_TEAMS)]
    att = rng.normal(0, 0.25, N_TEAMS); att -= att.mean()
    dfn = rng.normal(0, 0.25, N_TEAMS); dfn -= dfn.mean()
    return teams, att, dfn


def _season(teams, att, dfn, season_seed):
    r = np.random.default_rng(season_seed)
    pairs = [(h, a) for h in range(N_TEAMS) for a in range(N_TEAMS) if h != a]
    r.shuffle(pairs)
    rows = []
    for k, (h, a) in enumerate(pairs):
        lh = np.exp(MU_HOME + att[h] + dfn[a])
        la = np.exp(MU_AWAY + att[a] + dfn[h])
        rows.append((h, a, int(r.poisson(lh)), int(r.poisson(la)), int(k * 330 / len(pairs))))
    return rows


def _as_matches(teams, rows):
    return [Match(home=teams[h], away=teams[a],
                  date=f"2026-{1 + d // 30:02d}-{1 + d % 28:02d}T15:00:00Z",
                  home_goals=hg, away_goals=ag, status="FINISHED")
            for h, a, hg, ag, d in rows]


def _fit_and_evaluate(seed):
    teams, att, dfn = _league(seed)
    train = _as_matches(teams, _season(teams, att, dfn, seed * 10 + 1))
    started = time.time()
    model = DixonColesModel()
    model.fit(train)
    fit_seconds = time.time() - started
    held_out = _season(teams, att, dfn, seed * 10 + 2)
    preds = [model.predict(teams[h], teams[a]) for h, a, *_ in held_out]
    # "Actual" rates come from re-simulating every held-out fixture many times
    # from the TRUE goal rates, so the comparison isn't muddied by the luck of
    # one 380-match sample (which alone can move average goals by ~0.1).
    rng = np.random.default_rng(seed * 10 + 3)
    reps = 40
    hg = np.concatenate([rng.poisson(np.exp(MU_HOME + att[h] + dfn[a]), reps) for h, a, *_ in held_out])
    ag = np.concatenate([rng.poisson(np.exp(MU_AWAY + att[a] + dfn[h]), reps) for h, a, *_ in held_out])
    mean = lambda key: float(np.mean([p[key] for p in preds]))  # noqa: E731
    return model, fit_seconds, {
        "pred_home_goals": float(np.mean([p["expected_goals"][0] for p in preds])),
        "act_home_goals": float(hg.mean()),
        "pred_home_win": mean("home_win"), "act_home_win": float(np.mean(hg > ag)),
        "pred_away_win": mean("away_win"), "act_away_win": float(np.mean(ag > hg)),
        "pred_over25": mean("over_2_5"), "act_over25": float(np.mean(hg + ag > 2.5)),
        "pred_btts": mean("btts_yes"), "act_btts": float(np.mean((hg > 0) & (ag > 0))),
        "zero_zero_share": float(np.mean([p["most_likely_score"] == (0, 0) for p in preds])),
    }


def test_fitted_model_reproduces_league_scoring_level_and_home_advantage():
    _, _, m = _fit_and_evaluate(seed=3)
    # Tolerances allow for the model being estimated from a single 380-match
    # season; the original model missed by far more on every one of these
    # (home goals ~0.4 short, Over 2.5 and BTTS ~11-13 points low, home advantage ~nil).
    assert abs(m["pred_home_goals"] - m["act_home_goals"]) < 0.15
    assert m["pred_home_win"] > m["pred_away_win"] + 0.03
    assert abs(m["pred_home_win"] - m["act_home_win"]) < 0.07
    assert abs(m["pred_over25"] - m["act_over25"]) < 0.07
    assert abs(m["pred_btts"] - m["act_btts"]) < 0.08


def test_zero_zero_is_no_longer_the_most_likely_score_for_a_third_of_fixtures():
    _, _, m = _fit_and_evaluate(seed=4)
    # Originally ~30% of fixtures had 0-0 as the single most likely score. It is
    # still somewhat above the true rate (the fitted rho picks up noise), which is
    # why the app should eventually show expected goals instead of one scoreline.
    assert m["zero_zero_share"] < 0.20


def test_baselines_are_learned_and_ratings_stay_centered():
    model, _, _ = _fit_and_evaluate(seed=5)
    assert model.mu_home > model.mu_away > 0
    assert abs(model.mu_home - MU_HOME) < 0.2
    ratings = list(model.ratings.values())
    for field in ("attack_home", "defense_home", "attack_away", "defense_away"):
        assert abs(sum(getattr(r, field) for r in ratings)) < 1e-3


def test_fit_is_fast_enough_to_backtest_with():
    # The old per-match Python loop took about a minute per league.
    _, fit_seconds, _ = _fit_and_evaluate(seed=6)
    assert fit_seconds < 15


def test_fit_with_priors_still_works():
    teams, att, dfn = _league(7)
    matches = _as_matches(teams, _season(teams, att, dfn, 71))
    first = DixonColesModel(); first.fit(matches)
    second = DixonColesModel()
    second.fit(matches, priors=dict(first.ratings))
    h, a = teams[0], teams[1]
    assert abs(first.predict(h, a)["home_win"] - second.predict(h, a)["home_win"]) < 0.05
