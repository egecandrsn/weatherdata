"""Tests for feature engineering module."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.features import (
    compute_heat_index,
    compute_wind_chill,
    compute_standard_apparent_temp,
    build_feature_vector,
    feature_vector_to_model_input,
    FEATURE_NAMES,
)


class TestHeatIndex:
    def test_returns_none_below_threshold(self):
        assert compute_heat_index(20.0, 50.0) is None

    def test_returns_value_above_threshold(self):
        hi = compute_heat_index(35.0, 80.0)
        assert hi is not None
        assert hi > 35.0  # heat index should exceed air temp in hot+humid

    def test_increases_with_humidity(self):
        hi_low = compute_heat_index(35.0, 50.0)
        hi_high = compute_heat_index(35.0, 90.0)
        assert hi_low is not None and hi_high is not None
        assert hi_high > hi_low


class TestWindChill:
    def test_returns_none_above_threshold(self):
        assert compute_wind_chill(15.0, 20.0) is None

    def test_returns_value_below_threshold(self):
        wc = compute_wind_chill(-5.0, 20.0)
        assert wc is not None
        assert wc < -5.0  # wind chill should be below air temp

    def test_colder_with_more_wind(self):
        wc_low = compute_wind_chill(0.0, 10.0)
        wc_high = compute_wind_chill(0.0, 40.0)
        assert wc_low is not None and wc_high is not None
        assert wc_high < wc_low


class TestApparentTemp:
    def test_returns_temp_in_moderate_range(self):
        # 15 C, low humidity, low wind — should return close to actual temp
        apparent = compute_standard_apparent_temp(15.0, 30.0, 1.0)
        assert abs(apparent - 15.0) < 2.0

    def test_cold_with_wind(self):
        apparent = compute_standard_apparent_temp(-10.0, 50.0, 10.0)
        assert apparent < -10.0

    def test_hot_with_humidity(self):
        apparent = compute_standard_apparent_temp(35.0, 85.0, 1.0)
        assert apparent > 35.0


class TestBuildFeatureVector:
    def test_basic_construction(self):
        fv = build_feature_vector(temp_c=22.0, humidity_pct=50.0, wind_speed_ms=3.0)
        assert fv.temp_c == 22.0
        assert fv.humidity_pct == 50.0
        assert fv.apparent_temp_delta is not None

    def test_derived_features_computed(self):
        fv = build_feature_vector(temp_c=-5.0, humidity_pct=60.0, wind_speed_ms=8.0)
        assert fv.wind_chill is not None
        assert fv.wind_chill < -5.0

    def test_model_input_length(self):
        fv = build_feature_vector(temp_c=20.0)
        raw = feature_vector_to_model_input(fv)
        assert len(raw) == len(FEATURE_NAMES)
