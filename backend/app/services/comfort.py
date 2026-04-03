"""Comfort prediction service — translates model outputs to user-facing labels."""

from __future__ import annotations

from app.models.schemas import ComfortLabel, ClothingRec


def score_to_comfort_label(score: float) -> ComfortLabel:
    """Map a continuous comfort score (-1 to 1) to a discrete label."""
    if score <= -0.75:
        return ComfortLabel.freezing
    if score <= -0.5:
        return ComfortLabel.cold
    if score <= -0.25:
        return ComfortLabel.chilly
    if score <= -0.05:
        return ComfortLabel.cool
    if score <= 0.2:
        return ComfortLabel.comfortable
    if score <= 0.5:
        return ComfortLabel.warm
    if score <= 0.75:
        return ComfortLabel.hot
    return ComfortLabel.sweltering


def score_to_clothing_rec(score: float) -> ClothingRec:
    """Map comfort score to clothing recommendation."""
    if score <= -0.6:
        return ClothingRec.heavy_coat
    if score <= -0.3:
        return ClothingRec.jacket
    if score <= -0.1:
        return ClothingRec.hoodie
    if score <= 0.2:
        return ClothingRec.light_layer
    if score <= 0.5:
        return ClothingRec.tshirt
    return ClothingRec.tank_top


def generate_description(label: ComfortLabel, clothing: ClothingRec, temp_c: float) -> str:
    """Generate a plain-language description for the prediction card."""
    templates = {
        ComfortLabel.freezing: "Bitterly cold — bundle up with a heavy coat, hat, and gloves.",
        ComfortLabel.cold: "Cold out there — a warm jacket is essential.",
        ComfortLabel.chilly: "Chilly — you'll want a jacket or thick hoodie.",
        ComfortLabel.cool: "Cool but manageable — a light hoodie should do.",
        ComfortLabel.comfortable: "Comfortable — enjoy it, light layers are fine.",
        ComfortLabel.warm: "Warm — a t-shirt should be enough.",
        ComfortLabel.hot: "Hot — stay hydrated and dress light.",
        ComfortLabel.sweltering: "Sweltering — minimize time outdoors if possible.",
    }
    base = templates.get(label, "Check conditions before heading out.")
    return f"{base} (Actual: {temp_c:.0f}\u00b0C)"
