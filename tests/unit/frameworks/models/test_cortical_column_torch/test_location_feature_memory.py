# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Unit tests for LocationFeatureMemory."""

import torch
import pytest

from tbp.monty.frameworks.models.cortical_column_torch.location_feature_memory import (
    LocationFeatureMemory,
)


@pytest.fixture
def lfm():
    return LocationFeatureMemory(
        d_loc=32, d_feat=32, beta=8.0, novelty_threshold=0.7, max_patterns=100
    )


def _rand(d, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randn(d)


class TestStoreAndCount:
    def test_store_increments_count(self, lfm):
        loc = _rand(32, seed=0)
        feat = _rand(32, seed=1)
        assert lfm.store(loc, feat, "mug")
        assert lfm.n_stored == 1

    def test_novelty_gating_rejects_duplicate(self, lfm):
        loc = _rand(32, seed=0)
        feat = _rand(32, seed=1)
        lfm.store(loc, feat, "mug")
        # Same pattern should be rejected
        assert not lfm.store(loc, feat, "mug")
        assert lfm.n_stored == 1

    def test_novelty_gating_accepts_different(self, lfm):
        lfm.store(_rand(32, seed=0), _rand(32, seed=1), "mug")
        lfm.store(_rand(32, seed=2), _rand(32, seed=3), "mug")
        assert lfm.n_stored == 2

    def test_novelty_disabled(self):
        lfm = LocationFeatureMemory(
            d_loc=32, d_feat=32, novelty_threshold=None
        )
        loc = _rand(32, seed=0)
        feat = _rand(32, seed=1)
        lfm.store(loc, feat, "mug")
        assert lfm.store(loc, feat, "mug")  # duplicate accepted
        assert lfm.n_stored == 2

    def test_ring_buffer(self):
        lfm = LocationFeatureMemory(
            d_loc=8, d_feat=8, novelty_threshold=None, max_patterns=5
        )
        for i in range(10):
            lfm.store(_rand(8, seed=i * 2), _rand(8, seed=i * 2 + 1), f"obj{i}")
        assert lfm.n_stored == 5  # capped

    def test_known_objects(self, lfm):
        lfm.store(_rand(32, seed=0), _rand(32, seed=1), "mug")
        lfm.store(_rand(32, seed=2), _rand(32, seed=3), "drill")
        assert set(lfm.known_objects) == {"mug", "drill"}


class TestQueryFeatures:
    def test_returns_evidence(self, lfm):
        lfm.store(_rand(32, seed=0), _rand(32, seed=1), "mug")
        lfm.store(_rand(32, seed=2), _rand(32, seed=3), "drill")
        _, evidence, _ = lfm.query_features(_rand(32, seed=1))
        assert "mug" in evidence
        assert "drill" in evidence
        assert abs(sum(evidence.values()) - 1.0) < 1e-5  # sums to 1

    def test_correct_object_dominates(self):
        """Features matching mug's stored pattern should give mug highest ev."""
        lfm = LocationFeatureMemory(d_loc=32, d_feat=32, beta=12.0,
                                     novelty_threshold=None)
        torch.manual_seed(42)
        # Store distinct patterns for each object
        mug_feat = torch.randn(32) + 2.0  # shifted positive
        drill_feat = torch.randn(32) - 2.0  # shifted negative
        lfm.store(_rand(32), mug_feat, "mug")
        lfm.store(_rand(32), drill_feat, "drill")

        # Query with mug-like features
        query = mug_feat + torch.randn(32) * 0.1  # small noise
        _, evidence, _ = lfm.query_features(query)
        assert evidence["mug"] > evidence["drill"]

    def test_empty_memory_returns_zeros(self, lfm):
        retrieved, evidence, _ = lfm.query_features(_rand(32))
        assert retrieved.shape == (64,)
        assert evidence == {}


class TestQueryLocation:
    def test_returns_evidence(self, lfm):
        lfm.store(_rand(32, seed=0), _rand(32, seed=1), "mug")
        _, evidence = lfm.query_location(_rand(32, seed=0))
        assert "mug" in evidence

    def test_matching_location_gives_correct_features(self):
        lfm = LocationFeatureMemory(d_loc=32, d_feat=32, beta=12.0,
                                     novelty_threshold=None)
        loc = torch.randn(32) + 3.0
        feat = torch.randn(32) + 5.0
        lfm.store(loc, feat, "mug")
        lfm.store(torch.randn(32) - 3.0, torch.randn(32) - 5.0, "drill")

        retrieved, _ = lfm.query_location(loc + torch.randn(32) * 0.01)
        # Retrieved feature part should be close to mug's features
        retrieved_feat = retrieved[32:]
        cos_sim = torch.nn.functional.cosine_similarity(
            retrieved_feat.unsqueeze(0), feat.unsqueeze(0)
        ).item()
        assert cos_sim > 0.8


class TestQueryFull:
    def test_full_query_sharper_than_partial(self):
        """Full query should give more concentrated evidence."""
        lfm = LocationFeatureMemory(d_loc=32, d_feat=32, beta=12.0,
                                     novelty_threshold=None)
        torch.manual_seed(42)
        mug_loc = torch.randn(32) + 2.0
        mug_feat = torch.randn(32) + 2.0
        drill_loc = torch.randn(32) - 2.0
        drill_feat = torch.randn(32) - 2.0

        lfm.store(mug_loc, mug_feat, "mug")
        lfm.store(drill_loc, drill_feat, "drill")

        # Feature-only query
        _, feat_ev, _ = lfm.query_features(mug_feat)
        # Full query
        _, full_ev = lfm.query_full(mug_loc, mug_feat)

        # Full should give even higher evidence for mug
        assert full_ev["mug"] >= feat_ev["mug"] - 0.01


class TestMultiObjectDiscrimination:
    def test_three_objects_correct_top1(self):
        """With 3 well-separated objects, feature query identifies correct one."""
        lfm = LocationFeatureMemory(d_loc=32, d_feat=64, beta=12.0,
                                     novelty_threshold=None)
        torch.manual_seed(0)
        objects = {}
        for i, name in enumerate(["mug", "drill", "banana"]):
            # Create well-separated feature clusters
            base = torch.zeros(64)
            base[i * 20: (i + 1) * 20] = 1.0  # non-overlapping active regions
            objects[name] = base
            # Store multiple viewpoints (slightly noisy)
            for _ in range(5):
                lfm.store(
                    torch.randn(32),
                    base + torch.randn(64) * 0.05,
                    name,
                )

        # Query each object
        for name, base_feat in objects.items():
            query = base_feat + torch.randn(64) * 0.05
            _, evidence, _ = lfm.query_features(query)
            top = max(evidence, key=evidence.get)
            assert top == name, f"Expected {name}, got {top}"

    def test_evidence_accumulates_over_queries(self):
        """Multiple queries for the same object should accumulate."""
        lfm = LocationFeatureMemory(d_loc=16, d_feat=32, beta=10.0,
                                     novelty_threshold=None)
        torch.manual_seed(7)
        mug_feat = torch.randn(32) + 3.0
        lfm.store(torch.randn(16), mug_feat, "mug")
        lfm.store(torch.randn(16), torch.randn(32) - 3.0, "drill")

        total_ev = {"mug": 0.0, "drill": 0.0}
        for _ in range(10):
            query = mug_feat + torch.randn(32) * 0.2
            _, ev, _ = lfm.query_features(query)
            for k, v in ev.items():
                total_ev[k] += v

        assert total_ev["mug"] > total_ev["drill"]
        assert total_ev["mug"] > 5.0  # accumulated significantly


class TestBidirectionalRetrieval:
    def test_features_retrieve_location(self):
        lfm = LocationFeatureMemory(d_loc=32, d_feat=32, beta=12.0,
                                     novelty_threshold=None)
        loc = torch.randn(32) + 5.0
        feat = torch.randn(32) + 5.0
        lfm.store(loc, feat, "mug")
        # Store a different pattern far away
        lfm.store(torch.randn(32) - 5.0, torch.randn(32) - 5.0, "drill")

        retrieved, _, _ = lfm.query_features(feat + torch.randn(32) * 0.01)
        retrieved_loc = retrieved[:32]
        cos_sim = torch.nn.functional.cosine_similarity(
            retrieved_loc.unsqueeze(0), loc.unsqueeze(0)
        ).item()
        assert cos_sim > 0.8

    def test_location_retrieves_features(self):
        lfm = LocationFeatureMemory(d_loc=32, d_feat=32, beta=12.0,
                                     novelty_threshold=None)
        loc = torch.randn(32) + 5.0
        feat = torch.randn(32) + 5.0
        lfm.store(loc, feat, "mug")
        lfm.store(torch.randn(32) - 5.0, torch.randn(32) - 5.0, "drill")

        retrieved, _ = lfm.query_location(loc + torch.randn(32) * 0.01)
        retrieved_feat = retrieved[32:]
        cos_sim = torch.nn.functional.cosine_similarity(
            retrieved_feat.unsqueeze(0), feat.unsqueeze(0)
        ).item()
        assert cos_sim > 0.8


class TestStateDictRoundtrip:
    def test_save_load_preserves_patterns(self, lfm):
        lfm.store(_rand(32, seed=0), _rand(32, seed=1), "mug")
        lfm.store(_rand(32, seed=2), _rand(32, seed=3), "drill")

        sd = lfm.state_dict()
        lfm2 = LocationFeatureMemory(d_loc=32, d_feat=32, beta=8.0)
        lfm2.load_state_dict(sd)

        assert lfm2.n_stored == 2
        assert set(lfm2.known_objects) == {"mug", "drill"}

        # Same query should produce same evidence
        q = _rand(32, seed=10)
        _, ev1, _ = lfm.query_features(q)
        _, ev2, _ = lfm2.query_features(q)
        for k in ev1:
            assert abs(ev1[k] - ev2[k]) < 1e-5


class TestClear:
    def test_clear_resets(self, lfm):
        lfm.store(_rand(32, seed=0), _rand(32, seed=1), "mug")
        assert lfm.n_stored == 1
        lfm.clear()
        assert lfm.n_stored == 0
        assert lfm.known_objects == []
