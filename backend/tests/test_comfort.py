"""Tests for comfort prediction service."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.comfort import (
    score_to_comfort_label,
    score_to_clothing_rec,
    generate_description,
)
from app.models.schemas import ComfortLabel, ClothingRec


class TestComfortLabel:
    def test_freezing(self):
        assert score_to_comfort_label(-0.9) == ComfortLabel.freezing

    def test_comfortable(self):
        assert score_to_comfort_label(0.0) == ComfortLabel.comfortable

    def test_hot(self):
        assert score_to_comfort_label(0.6) == ComfortLabel.hot

    def test_sweltering(self):
        assert score_to_comfort_label(0.9) == ComfortLabel.sweltering

    def test_boundary_cold(self):
        assert score_to_comfort_label(-0.5) == ComfortLabel.cold


class TestClothingRec:
    def test_heavy_coat_for_freezing(self):
        assert score_to_clothing_rec(-0.8) == ClothingRec.heavy_coat

    def test_tshirt_for_warm(self):
        assert score_to_clothing_rec(0.4) == ClothingRec.tshirt

    def test_light_layer_for_comfortable(self):
        assert score_to_clothing_rec(0.0) == ClothingRec.light_layer


class TestDescription:
    def test_generates_string(self):
        desc = generate_description(ComfortLabel.comfortable, ClothingRec.light_layer, 20.0)
        assert "20" in desc
        assert len(desc) > 10

    def test_cold_description(self):
        desc = generate_description(ComfortLabel.cold, ClothingRec.jacket, 5.0)
        assert "jacket" in desc.lower() or "cold" in desc.lower()
