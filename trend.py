"""
Deterministic layer: turns the last N readings into a trend, and the trend
plus fixed thresholds into a decision. No LLM involved here on purpose --
this is plain arithmetic and should be fast, free, and auditable.

The only thing this module hands off is an `is_ambiguous` flag: cases where
the rules themselves don't give a confident answer (a predicted threshold
crossing that hasn't happened yet, or values flapping around a boundary).
Those are the only cases worth spending an LLM call on.
"""

from __future__ import annotations

from dataclasses import dataclass

WARN_THRESHOLD = 85.0
ALERT_THRESHOLD = 90.0

# how far below a threshold we still consider "close enough to flag as ambiguous"
BOUNDARY_BAND = 2.0
# minimum slope (deg per cycle) to call it a real trend rather than noise
TREND_EPSILON = 1.0


@dataclass
class Stats:
    values: list[float]          # oldest -> newest, includes the current reading
    slope: float                 # avg change per cycle
    trend: str                   # "rising" | "falling" | "stable"
    projected_next: float        # naive 1-step projection


def compute_stats(history: list[dict], current_value: float) -> Stats:
    values = [h["reading"]["value_c"] for h in history] + [current_value]

    if len(values) >= 2:
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        slope = sum(diffs) / len(diffs)
    else:
        slope = 0.0

    if slope > TREND_EPSILON:
        trend = "rising"
    elif slope < -TREND_EPSILON:
        trend = "falling"
    else:
        trend = "stable"

    return Stats(values=values, slope=slope, trend=trend, projected_next=current_value + slope)


def rule_based_decision(current_value: float, stats: Stats) -> tuple[str, bool, str]:
    """Returns (decision, is_ambiguous, note).

    decision     - the rule engine's best answer right now: ok | warn | alert
    is_ambiguous - True if this case deserves a second opinion (LLM) before acting
    note         - short human-readable justification, used whether or not the LLM gets involved
    """
    # Hard thresholds are never ambiguous -- current value already crossed the line.
    if current_value >= ALERT_THRESHOLD:
        return "alert", False, f"{current_value}C is at/above the {ALERT_THRESHOLD}C alert threshold."

    if current_value >= WARN_THRESHOLD:
        return "warn", False, f"{current_value}C is at/above the {WARN_THRESHOLD}C warn threshold."

    # Below both thresholds. Still check whether the trend says we're about
    # to cross one -- this is the "predictive" case, and it's genuinely
    # ambiguous: the rule alone (current value) says "ok", but a naive
    # 1-step projection says "about to be warn/alert".
    if stats.trend == "rising" and stats.projected_next >= WARN_THRESHOLD:
        return (
            "warn",
            True,
            f"{current_value}C is below {WARN_THRESHOLD}C but rising at {stats.slope:+.1f}C/cycle, "
            f"projected to reach {stats.projected_next:.1f}C next cycle.",
        )

    # Close to the warn boundary but not clearly rising or falling -- flapping risk.
    if WARN_THRESHOLD - BOUNDARY_BAND <= current_value < WARN_THRESHOLD and stats.trend == "stable":
        return (
            "ok",
            True,
            f"{current_value}C is within {BOUNDARY_BAND}C of the warn threshold with a flat trend "
            f"({stats.slope:+.1f}C/cycle) -- borderline.",
        )

    return "ok", False, f"{current_value}C is within normal range, trend is {stats.trend}."