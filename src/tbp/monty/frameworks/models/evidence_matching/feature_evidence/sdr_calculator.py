# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import numpy as np

from tbp.monty.frameworks.models.evidence_matching.feature_evidence.calculator import (
    DefaultFeatureEvidenceCalculator,
)


class SDRFeatureEvidenceCalculator:
    @staticmethod
    def calculate(
        channel_feature_array: np.ndarray,
        channel_feature_order: list[str],
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
        input_channel: str,
    ) -> np.ndarray:
        """Calculates feature evidence for all nodes stored in a graph.

        This calculation tests if the input_channel is a learning_module. If so,
        a different function is used for feature comparison.

        Note: This assumes that learning modules always outputs 1 feature, object_id.
        If the learning modules output more than object_id features, we need to
        compare these according to their weights.

        Returns:
            The feature evidence for all nodes.
        """
        if input_channel.startswith("learning_module"):
            return SDRFeatureEvidenceCalculator.calculate_feature_evidence_sdr_for_all_nodes(  # noqa: E501
                channel_feature_array=channel_feature_array,
                channel_feature_order=channel_feature_order,
                channel_feature_weights=channel_feature_weights,
                channel_query_features=channel_query_features,
                channel_tolerances=channel_tolerances,
            )

        return DefaultFeatureEvidenceCalculator.calculate(
            channel_feature_array=channel_feature_array,
            channel_feature_order=channel_feature_order,
            channel_feature_weights=channel_feature_weights,
            channel_query_features=channel_query_features,
            channel_tolerances=channel_tolerances,
            input_channel=input_channel,
        )

    @staticmethod
    def calculate_feature_evidence_sdr_for_all_nodes(
        channel_feature_array: np.ndarray,
        channel_feature_order: list[str] | None,
        channel_feature_weights: dict,
        channel_query_features: dict,
        channel_tolerances: dict,
    ) -> np.ndarray:
        """Calculate overlap between stored and query SDR features.

        Calculates the overlap between the SDR features stored at every location in
        the graph and the query SDR feature. This overlap is then compared to the
        tolerance value and the result is used for adjusting the evidence score.

        We use the tolerance (in overlap bits) for generalization. If two objects are
        close enough, their overlap in bits should be higher than the set tolerance
        value.

        The tolerance sets the lowest overlap for adding evidence, the range
        [tolerance, sdr_on_bits] is mapped to [0,1] evidence points. Any overlap less
        than tolerance will not add any evidence. These evidence scores are then
        multiplied by the feature weight of object_ids which scales all of the
        evidence points to the range [0, channel_feature_weights["object_id"]].

        The below variables have the following shapes:
            - channel_feature_array: (n, sdr_length)
            - channel_query_features["object_id"]: (sdr_length)
            - query_feat: (sdr_length, 1)
            - np.matmul(channel_feature_array, query_feat): (n, 1)
            - overlaps: (n)

        Returns:
            The normalized overlaps.
        """
        if channel_feature_order is None:
            channel_feature_order = list(channel_query_features.keys())

        start_idx = 0
        feature_evidence = []
        feature_weights = []
        for feature in channel_feature_order:
            if feature not in channel_query_features:
                continue
            query_feat = np.asarray(channel_query_features[feature], dtype=np.float64)
            if query_feat.ndim == 0:
                query_feat = query_feat.reshape(1)
            else:
                query_feat = query_feat.reshape(-1)

            end_idx = start_idx + len(query_feat)
            stored_feat = channel_feature_array[:, start_idx:end_idx]
            tolerance = channel_tolerances[feature]

            if feature == "object_id" or feature.startswith("object_id_"):
                evidence = SDRFeatureEvidenceCalculator._calculate_object_id_evidence(
                    stored_feat,
                    query_feat,
                    tolerance,
                )
            elif feature == "object_support":
                evidence = (
                    SDRFeatureEvidenceCalculator._calculate_object_support_evidence(
                        stored_feat,
                        query_feat,
                        tolerance,
                    )
                )
            else:
                evidence = SDRFeatureEvidenceCalculator._calculate_dense_similarity(
                    stored_feat,
                    query_feat,
                    tolerance,
                )

            feature_evidence.append(evidence)
            feature_weights.append(channel_feature_weights[feature])
            start_idx = end_idx

        if len(feature_evidence) == 0:
            return np.zeros(channel_feature_array.shape[0])

        if len(feature_evidence) == 1:
            return feature_evidence[0] * feature_weights[0]

        stacked_evidence = np.vstack(feature_evidence)
        return np.average(stacked_evidence, weights=feature_weights, axis=0)

    @staticmethod
    def _calculate_object_id_evidence(
        stored_feat: np.ndarray,
        query_feat: np.ndarray,
        tolerance: float,
    ) -> np.ndarray:
        max_overlap = float(query_feat @ query_feat)
        max_overlap = max(max_overlap, tolerance + np.finfo(np.float64).eps)

        overlaps = stored_feat @ query_feat
        normalized_overlaps = (overlaps - tolerance) / (max_overlap - tolerance)
        normalized_overlaps[normalized_overlaps < 0] = 0.0
        normalized_overlaps[normalized_overlaps > 1.0] = 1.0
        return normalized_overlaps

    @staticmethod
    def _calculate_object_support_evidence(
        stored_feat: np.ndarray,
        query_feat: np.ndarray,
        tolerance: float,
    ) -> np.ndarray:
        query_norm = float(np.linalg.norm(query_feat))
        if np.isclose(query_norm, 0.0):
            return np.zeros(stored_feat.shape[0])

        stored_norms = np.linalg.norm(stored_feat, axis=1)
        similarities = np.zeros(stored_feat.shape[0], dtype=np.float64)
        valid = stored_norms > 0
        similarities[valid] = (stored_feat[valid] @ query_feat) / (
            stored_norms[valid] * query_norm
        )
        return SDRFeatureEvidenceCalculator._normalize_similarity(
            similarities,
            tolerance,
        )

    @staticmethod
    def _calculate_dense_similarity(
        stored_feat: np.ndarray,
        query_feat: np.ndarray,
        tolerance: float,
    ) -> np.ndarray:
        differences = np.mean(np.abs(stored_feat - query_feat), axis=1)
        similarities = 1.0 - differences
        return SDRFeatureEvidenceCalculator._normalize_similarity(
            similarities,
            tolerance,
        )

    @staticmethod
    def _normalize_similarity(
        similarities: np.ndarray,
        tolerance: float,
    ) -> np.ndarray:
        denominator = max(1.0 - tolerance, np.finfo(np.float64).eps)
        normalized = (similarities - tolerance) / denominator
        normalized[normalized < 0] = 0.0
        normalized[normalized > 1.0] = 1.0
        return normalized
