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
# Some providers (including Render's own Postgres) hand out URLs starting
# with postgres:// — SQLAlchemy 1.4+/2.0 requires postgresql:// instead.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
metadata = MetaData()

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
                       value_bet: dict = None):
    """Insert a prediction only if this exact fixture hasn't been recorded
    yet. This is what keeps the track record honest — once stored, a later
    refresh must never overwrite it, or the app could quietly 're-predict'
    with hindsight as kickoff approaches."""
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
            return
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
