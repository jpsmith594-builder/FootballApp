"""
db.py
Stores each prediction the moment it's first made, then reconciles it
against the real result once the match finishes — this is what powers the
Track Record tab.

IMPORTANT — why this can't just be a local file: Render's free tier wipes
the local filesystem on every restart and every redeploy. A SQLite file
would silently lose your entire prediction history the next time you push
a code change. For anything actually deployed, set DATABASE_URL to a real
Postgres connection string.

Free, persistent option: create a project at supabase.com (free tier),
copy the connection string from Project Settings > Database, and set it as
the DATABASE_URL environment variable on Render. Without DATABASE_URL set,
this falls back to a local SQLite file — fine for testing on your own
machine, not fine for the deployed app.
"""

import os
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Float, Boolean,
    DateTime, select, and_, insert, update,
)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///matchday_local.db")
# Some providers (including Render's own Postgres, and some Supabase
# connection-string variants) hand out URLs starting with "postgres" rather
# than "postgresql" — SQLAlchemy dropped the "postgres" dialect alias, so
# "postgres://..." and "postgres+psycopg://..." both need the "postgresql"
# form or SQLAlchemy raises a dialect-loading error at engine creation time.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
elif DATABASE_URL.startswith("postgres+"):
    DATABASE_URL = DATABASE_URL.replace("postgres+", "postgresql+", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
metadata = MetaData()

team_adjustments = Table(
    "team_adjustments", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("league", String, nullable=False),
    Column("team", String, nullable=False),
    Column("goal_multiplier", Float, nullable=False),  # e.g. 0.8 = "missing a key attacker"
    Column("note", String, nullable=True),
    Column("set_at", DateTime),
)

push_subscriptions = Table(
    "push_subscriptions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("endpoint", String, nullable=False, unique=True),
    Column("p256dh", String, nullable=False),
    Column("auth", String, nullable=False),
    Column("created_at", DateTime),
)

predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("league", String, nullable=False),
    Column("home", String, nullable=False),
    Column("away", String, nullable=False),
    Column("match_date", String, nullable=False),
    Column("predicted_home_win", Float),
    Column("predicted_draw", Float),
    Column("predicted_away_win", Float),
    Column("predicted_score_home", Integer),
    Column("predicted_score_away", Integer),
    Column("value_bet_market", String, nullable=True),
    Column("value_bet_odds", Float, nullable=True),
    Column("value_bet_edge", Float, nullable=True),
    Column("actual_home_goals", Integer, nullable=True),
    Column("actual_away_goals", Integer, nullable=True),
    Column("resolved", Boolean, default=False),
    Column("recorded_at", DateTime),
)

metadata.create_all(engine)


def record_prediction(league: str, home: str, away: str, match_date: str, pred: dict,
                       value_bet: dict = None) -> bool:
    """Insert a prediction only if this exact fixture hasn't been recorded
    yet. This is what keeps the track record honest — once stored, a later
    refresh must never overwrite it, or the app could quietly 're-predict'
    with hindsight as kickoff approaches. Returns True if this call actually
    inserted a new row (used to decide whether to fire a push notification —
    without this check, every 6-hour refresh would re-notify for the same
    value bet)."""
    with engine.begin() as conn:
        exists = conn.execute(
            select(predictions.c.id).where(and_(
                predictions.c.league == league,
                predictions.c.home == home,
                predictions.c.away == away,
                predictions.c.match_date == match_date,
            ))
        ).first()
        if exists:
            return False
        conn.execute(insert(predictions).values(
            league=league, home=home, away=away, match_date=match_date,
            predicted_home_win=pred["home_win"], predicted_draw=pred["draw"],
            predicted_away_win=pred["away_win"],
            predicted_score_home=pred["most_likely_score"][0],
            predicted_score_away=pred["most_likely_score"][1],
            value_bet_market=value_bet["market"] if value_bet else None,
            value_bet_odds=value_bet["odds"] if value_bet else None,
            value_bet_edge=value_bet["edge"] if value_bet else None,
            resolved=False,
            recorded_at=datetime.now(timezone.utc),
        ))
        return True


def resolve_predictions(league: str, finished_matches: list):
    """Match finished results back to stored predictions and fill in what
    actually happened. finished_matches: Match objects with goals set."""
    with engine.begin() as conn:
        for m in finished_matches:
            conn.execute(
                update(predictions)
                .where(and_(
                    predictions.c.league == league,
                    predictions.c.home == m.home,
                    predictions.c.away == m.away,
                    predictions.c.match_date == m.date,
                    predictions.c.resolved == False,  # noqa: E712
                ))
                .values(actual_home_goals=m.home_goals, actual_away_goals=m.away_goals, resolved=True)
            )


def get_track_record(league: str, limit: int = 100) -> list:
    with engine.begin() as conn:
        rows = conn.execute(
            select(predictions).where(and_(
                predictions.c.league == league,
                predictions.c.resolved == True,  # noqa: E712
            )).order_by(predictions.c.match_date.desc()).limit(limit)
        ).mappings().all()
    return [dict(r) for r in rows]


def set_team_adjustment(league: str, team: str, goal_multiplier: float, note: str = None):
    """Manual override for when you know something the model doesn't —
    e.g. a star striker is injured. Replaces any existing adjustment for
    this team in this league rather than stacking multiple."""
    with engine.begin() as conn:
        conn.execute(delete_adjustment_stmt(league, team))
        conn.execute(insert(team_adjustments).values(
            league=league, team=team, goal_multiplier=goal_multiplier,
            note=note, set_at=datetime.now(timezone.utc),
        ))


def delete_adjustment_stmt(league: str, team: str):
    from sqlalchemy import delete
    return delete(team_adjustments).where(and_(
        team_adjustments.c.league == league, team_adjustments.c.team == team,
    ))


def clear_team_adjustment(league: str, team: str):
    with engine.begin() as conn:
        conn.execute(delete_adjustment_stmt(league, team))


def get_team_adjustments(league: str) -> dict:
    """Returns {team_name: goal_multiplier} for quick lookup during prediction."""
    with engine.begin() as conn:
        rows = conn.execute(
            select(team_adjustments.c.team, team_adjustments.c.goal_multiplier)
            .where(team_adjustments.c.league == league)
        ).all()
    return {r.team: r.goal_multiplier for r in rows}


def save_push_subscription(endpoint: str, p256dh: str, auth: str):
    with engine.begin() as conn:
        existing = conn.execute(
            select(push_subscriptions.c.id).where(push_subscriptions.c.endpoint == endpoint)
        ).first()
        if existing:
            return  # already subscribed, nothing to do
        conn.execute(insert(push_subscriptions).values(
            endpoint=endpoint, p256dh=p256dh, auth=auth,
            created_at=datetime.now(timezone.utc),
        ))


def get_push_subscriptions() -> list:
    with engine.begin() as conn:
        rows = conn.execute(select(push_subscriptions)).mappings().all()
    return [dict(r) for r in rows]


def remove_push_subscription(endpoint: str):
    from sqlalchemy import delete
    with engine.begin() as conn:
        conn.execute(delete(push_subscriptions).where(push_subscriptions.c.endpoint == endpoint))


def get_resolved_predictions_all(league: str = None) -> list:
    """Unlike get_track_record (capped at `limit`, newest-first, meant for
    the UI feed), this returns every resolved prediction with no cap —
    calibration needs the full history to get stable bucket counts, not
    just the most recent matches."""
    with engine.begin() as conn:
        query = select(predictions).where(predictions.c.resolved == True)  # noqa: E712
        if league:
            query = query.where(predictions.c.league == league)
        rows = conn.execute(query).mappings().all()
    return [dict(r) for r in rows]


def get_calibration_curve(league: str = None, bucket_size: float = 0.1,
                           min_per_bucket: int = 10) -> dict:
    """
    Checks whether the model's stated confidence can actually be trusted:
    among every past prediction where the model gave its top pick, say,
    70-80% confidence, was it right about 70-80% of the time, or
    systematically over/under-confident? This is standard model
    calibration, done here using your own accumulated track record rather
    than a held-out dataset, since the track record IS the held-out data —
    every row was predicted before the match was played.

    Returns {"buckets": [...], "well_calibrated": bool or None}. A bucket
    with fewer than `min_per_bucket` resolved predictions is dropped — with
    too few matches, "this bucket was 40% accurate" is mostly noise, not a
    real signal about the model.

    This is a DIAGNOSTIC, not a correction: nothing here automatically
    adjusts displayed probabilities. Blending that in uncritically could
    just as easily bake in a fluke from a small early sample as fix a real
    bias. Review the curve yourself once enough data has accumulated (the
    track record tab is the natural place to surface it) before deciding
    whether a correction is warranted.
    """
    rows = get_resolved_predictions_all(league)
    buckets: Dict[float, Dict[str, int]] = {}
    for r in rows:
        hg, ag = r["actual_home_goals"], r["actual_away_goals"]
        actual = "Home" if hg > ag else ("Away" if ag > hg else "Draw")
        predicted, confidence = max(
            [("Home", r["predicted_home_win"]), ("Draw", r["predicted_draw"]),
             ("Away", r["predicted_away_win"])],
            key=lambda x: x[1],
        )
        # A small epsilon before the floor division guards against float
        # imprecision — 0.90 // 0.1 naively evaluates to 8.0 (not 9.0) in
        # IEEE 754 float arithmetic, which would land a 90% confidence in
        # the 80-90% bucket instead of 90-100%.
        bucket_key = min(0.9, ((confidence + 1e-9) // bucket_size) * bucket_size)
        b = buckets.setdefault(bucket_key, {"correct": 0, "total": 0})
        b["total"] += 1
        if predicted == actual:
            b["correct"] += 1

    curve = []
    for bucket_key in sorted(buckets):
        b = buckets[bucket_key]
        if b["total"] < min_per_bucket:
            continue
        curve.append({
            "predicted_confidence_range": f"{bucket_key*100:.0f}-{(bucket_key+bucket_size)*100:.0f}%",
            "actual_accuracy": round(b["correct"] / b["total"], 3),
            "sample_size": b["total"],
        })

    # "Well calibrated" (informally) if every bucket's actual accuracy is
    # within 10 percentage points of its stated confidence range's midpoint.
    well_calibrated = None
    if curve:
        well_calibrated = all(
            abs(c["actual_accuracy"] - (float(c["predicted_confidence_range"].split("-")[0]) / 100 + bucket_size / 2)) <= 0.10
            for c in curve
        )
    return {"buckets": curve, "well_calibrated": well_calibrated,
            "total_resolved_predictions": len(rows)}


def summarize_track_record(rows: list) -> dict:
    if not rows:
        return {"count": 0}
    correct = 0
    brier_sum = 0.0
    staked_profit = 0.0
    bets_placed = 0
    for r in rows:
        hg, ag = r["actual_home_goals"], r["actual_away_goals"]
        actual = "Home" if hg > ag else ("Away" if ag > hg else "Draw")
        predicted = max(
            [("Home", r["predicted_home_win"]), ("Draw", r["predicted_draw"]), ("Away", r["predicted_away_win"])],
            key=lambda x: x[1],
        )[0]
        if predicted == actual:
            correct += 1
        outcome_vec = {"Home": (1, 0, 0), "Draw": (0, 1, 0), "Away": (0, 0, 1)}[actual]
        pred_vec = (r["predicted_home_win"], r["predicted_draw"], r["predicted_away_win"])
        brier_sum += sum((p - o) ** 2 for p, o in zip(pred_vec, outcome_vec))
        if r["value_bet_market"]:
            bets_placed += 1
            won = r["value_bet_market"] == actual
            staked_profit += (r["value_bet_odds"] - 1) if won else -1
    return {
        "count": len(rows),
        "accuracy": round(correct / len(rows), 3),
        "avg_brier_score": round(brier_sum / len(rows), 3),  # lower is better, 0 = perfect
        "value_bets_placed": bets_placed,
        "value_bets_profit_units": round(staked_profit, 2),  # assumes 1 unit staked per flagged bet
    }
