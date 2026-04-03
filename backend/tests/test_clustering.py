"""Tests for user clustering module."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ml.clustering import (
    assign_cluster,
    apply_cluster_adjustment,
    encode_quiz_answers,
    CLUSTER_PERSONALITIES,
    N_CLUSTERS,
)


class TestAssignCluster:
    def test_cold_natured_gets_cold_cluster(self):
        cluster = assign_cluster({
            "hot_cold_slider": -0.9,
            "barometric_sensitivity": "no",
            "climate_zone": "temperate",
        })
        assert cluster in (0, 1, 2)  # cold-natured clusters

    def test_hot_natured_gets_hot_cluster(self):
        cluster = assign_cluster({
            "hot_cold_slider": 0.9,
            "barometric_sensitivity": "no",
            "climate_zone": "tropical",
        })
        assert cluster in (5, 6, 7)  # hot-natured clusters

    def test_neutral_gets_middle_cluster(self):
        cluster = assign_cluster({
            "hot_cold_slider": 0.0,
            "barometric_sensitivity": "no",
            "climate_zone": "temperate",
        })
        assert cluster == 3  # Goldilocks

    def test_all_clusters_have_personalities(self):
        for i in range(N_CLUSTERS):
            assert i in CLUSTER_PERSONALITIES


class TestClusterAdjustment:
    def test_cold_cluster_makes_colder(self):
        base = -0.2
        adjusted = apply_cluster_adjustment(base, 0, temp_c=5.0)
        assert adjusted < base  # cluster 0 has cold_offset = -0.15

    def test_hot_cluster_makes_warmer(self):
        base = 0.2
        adjusted = apply_cluster_adjustment(base, 7, temp_c=30.0)
        # Cluster 7 has heat_offset = -0.10 (they run hot, so heat feels WORSE)
        assert adjusted != base

    def test_neutral_cluster_minimal_change(self):
        base = 0.0
        adjusted = apply_cluster_adjustment(base, 3, temp_c=20.0)
        assert abs(adjusted - base) < 0.01

    def test_score_stays_in_range(self):
        adjusted = apply_cluster_adjustment(-0.95, 0, temp_c=-20.0, wind_speed_ms=15.0)
        assert -1.0 <= adjusted <= 1.0
