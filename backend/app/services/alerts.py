"""Transition alert detection — identifies significant comfort changes in the forecast."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from app.models.schemas import ComfortLabel, ClothingRec
from app.services.comfort import score_to_comfort_label, score_to_clothing_rec


@dataclass
class TransitionAlert:
    """A significant weather transition detected in the forecast."""
    time: str
    message: str
    from_label: str
    to_label: str
    clothing_change: Optional[str] = None


def detect_transitions(
    hourly_forecasts: list[dict],
    threshold: float = 0.25,
) -> list[TransitionAlert]:
    """Scan hourly forecast for significant comfort transitions.

    A transition is detected when the comfort score changes by more than
    `threshold` between consecutive hours, or when the clothing recommendation
    changes.

    Args:
        hourly_forecasts: List of dicts with 'time', 'comfort_score', etc.
        threshold: Minimum score change to trigger an alert.

    Returns:
        List of TransitionAlert objects for significant transitions.
    """
    if len(hourly_forecasts) < 2:
        return []

    alerts = []
    prev = hourly_forecasts[0]

    for curr in hourly_forecasts[1:]:
        prev_score = prev.get("comfort_score", 0.0)
        curr_score = curr.get("comfort_score", 0.0)
        delta = curr_score - prev_score

        if abs(delta) < threshold:
            prev = curr
            continue

        prev_label = score_to_comfort_label(prev_score)
        curr_label = score_to_comfort_label(curr_score)

        # Only alert if the label actually changes
        if prev_label == curr_label:
            prev = curr
            continue

        prev_clothing = score_to_clothing_rec(prev_score)
        curr_clothing = score_to_clothing_rec(curr_score)

        time_str = curr.get("time", "")
        try:
            t = datetime.fromisoformat(time_str)
            friendly_time = t.strftime("%-I%p").lower()
        except (ValueError, TypeError):
            friendly_time = time_str

        # Build message
        if delta < 0:
            # Getting colder
            message = (
                f"Temperature dropping — based on your patterns, "
                f"you'll want a layer around {friendly_time}."
            )
        else:
            # Getting warmer
            message = (
                f"Warming up around {friendly_time} — "
                f"you can probably shed a layer."
            )

        clothing_change = None
        if prev_clothing != curr_clothing:
            clothing_change = f"{prev_clothing.value} → {curr_clothing.value}"

        alerts.append(TransitionAlert(
            time=time_str,
            message=message,
            from_label=prev_label.value,
            to_label=curr_label.value,
            clothing_change=clothing_change,
        ))

        prev = curr

    return alerts
