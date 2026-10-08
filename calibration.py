"""
calibration.py
Turns the track record you've already been accumulating into a correction
layer: if the model says "70% confident" and, historically, picks in that
confidence band have actually been right 55% of the time, this nudges new
70%-ish predictions down toward 55% before they're shown — closing the gap
between stated and actual accuracy instead of just reporting it.

Deliberately built off db.get_calibration_curve's own buckets rather than a
separate model, so there's exactly one source of truth for "how calibrated
is this model" — the Track Record tab's calibration curve and this
correction are reading the same numbers.

This only touches what gets DISPLAYED (and therefore which hints clear
their confidence bar). Value-bet detection and the track record itself
keep using the model's raw, uncorrected probabilities — see engine.py's
refresh_league(). That split matters for two reasons:
  1. Value bets are a comparison against the market; correcting the model's
     own confidence first would mean an "edge" is partly measuring your own
     correction rather than a real model-vs-market disagreement.
  2. If the track record recorded corrected probabilities, next week's
     calibration curve would be measuring an already-corrected model, and
     the correction would compound on itself over time instead of staying
     anchored to the model's actual raw behavior.

Guardrails against overfitting a correction to noise:
  - CALIBRATION_MIN_SAMPLES: refuses to apply ANY correction until enough
    resolved predictions exist league-wide, full stop (returns None, which
    callers treat as a no-op).
  - CALIBRATION_MIN_PER_BUCKET (passed through to get_calibration_curve):
    individual buckets with too few matches are dropped before the curve is
    even built, so a correction can never anchor on a handful of fluke
    results in one confidence band.
  - Only the TOP-PICK probability is corrected; the other two outcomes are
    rescaled proportionally to preserve their relative ordering, rather
    than independently "calibrating" three correlated numbers against a
    curve that was only ever built from top-pick accuracy.
"""

import os
import logging

import db

log = logging.getLogger("calibration")

CALIBRATION_ENABLED = os.environ.get("CALIBRATION_ENABLED", "true").lower() not in ("false", "0", "")
# Below this many league-wide resolved predictions, there isn't enough
# history yet for a correction to mean anything — a new league or a
# freshly-launched app should run uncorrected for a while first.
CALIBRATION_MIN_SAMPLES = int(os.environ.get("CALIBRATION_MIN_SAMPLES", "50"))
CALIBRATION_MIN_PER_BUCKET = int(os.environ.get("CALIBRATION_MIN_PER_BUCKET", "10"))


def build_correction(league: str = None):
    """
    Returns a function mapping a raw top-pick confidence (0-1) to a
    calibrated one, built by linearly interpolating between the midpoints
    of db.get_calibration_curve's bucketed actual-accuracy values.

    Returns None when there isn't enough resolved history to trust a
    correction yet (either league-wide, or because every individual bucket
    got dropped for being too thin) — callers treat None as "no-op, use the
    raw probabilities as-is".
    """
    if not CALIBRATION_ENABLED:
        return None
    curve = db.get_calibration_curve(league=league, bucket_size=0.1,
                                      min_per_bucket=CALIBRATION_MIN_PER_BUCKET)
    if curve["total_resolved_predictions"] < CALIBRATION_MIN_SAMPLES or not curve["buckets"]:
        return None

    points = []
    for b in curve["buckets"]:
        lo = float(b["predicted_confidence_range"].split("-")[0]) / 100
        points.append((lo + 0.05, b["actual_accuracy"]))
    points.sort()

    def correct(confidence: float) -> float:
        if confidence <= points[0][0]:
            return points[0][1]
        if confidence >= points[-1][0]:
            return points[-1][1]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= confidence <= x1:
                if x1 == x0:
                    return y0
                t = (confidence - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)
        return confidence  # unreachable: bounds are covered above

    return correct


def apply_correction(probs: dict, correction_fn) -> dict:
    """
    probs: {"home_win": .., "draw": .., "away_win": ..}, summing to ~1.
    Corrects only the top pick toward its historically-actual accuracy; the
    other two outcomes are rescaled proportionally so the three still sum
    to 1 and keep their relative ordering. A no-op when correction_fn is
    None (not enough history yet — see build_correction).
    """
    if correction_fn is None:
        return dict(probs)
    top_outcome = max(probs, key=probs.get)
    top_p = probs[top_outcome]
    calibrated_top = min(max(correction_fn(top_p), 0.0), 1.0)
    others = [k for k in probs if k != top_outcome]
    other_sum = probs[others[0]] + probs[others[1]]
    remainder = 1 - calibrated_top
    if other_sum > 0:
        scale = remainder / other_sum
        return {
            top_outcome: calibrated_top,
            others[0]: probs[others[0]] * scale,
            others[1]: probs[others[1]] * scale,
        }
    return {top_outcome: calibrated_top, others[0]: remainder / 2, others[1]: remainder / 2}
