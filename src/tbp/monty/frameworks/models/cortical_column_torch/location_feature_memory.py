# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Location-feature Hopfield memory with bidirectional retrieval.

Stores composite patterns xi_i = [location_i ; features_i] tagged with object
IDs in a single shared Hopfield memory.  Retrieval via partial cues enables:

  - **Feature -> location** (localization): query with [0 ; features],
    retrieve associated locations and per-object evidence.
  - **Location -> features** (prediction): query with [location ; 0],
    retrieve predicted features for spatial prediction.
  - **Full query** (strongest retrieval): query with [location ; features],
    both parts contribute to attention.

Evidence per object = sum of softmax attention weights on that object's
stored patterns.  This replaces the EMA prototype + softmax-over-prototypes
approach with a many-pattern-per-object memory that preserves viewpoint-
specific features.

All operations are forward-only -- no autograd, no backprop.
"""

from __future__ import annotations

import numpy as np
import torch


class LocationFeatureMemory:
    """Hopfield memory storing (location, feature) composites with object tags.

    Parameters
    ----------
    d_loc : int
        Dimensionality of location encoding (grid cells).
    d_feat : int
        Dimensionality of feature encoding.
    beta : float
        Inverse temperature for softmax retrieval.
    novelty_threshold : float or None
        Cosine similarity threshold for novelty gating on the full
        composite.  None disables gating (store everything).
    max_patterns : int
        Maximum stored patterns (ring buffer when exceeded).
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        d_loc: int = 320,
        d_feat: int = 320,
        beta: float = 12.0,
        novelty_threshold: float | None = 0.7,
        max_patterns: int = 5000,
        device: str = "cpu",
    ):
        self.d_loc = d_loc
        self.d_feat = d_feat
        self.d_total = d_loc + d_feat
        self.beta = beta
        self.novelty_threshold = novelty_threshold
        self.max_patterns = max_patterns
        self.device = torch.device(device)

        self._patterns = torch.zeros(
            0, self.d_total, dtype=torch.float32, device=self.device
        )
        self._object_ids: list[str] = []
        self._raw_locations: list[np.ndarray] = []
        self._n_stored = 0

    @property
    def n_stored(self) -> int:
        """Number of patterns currently in memory (capped at max_patterns)."""
        return min(self._n_stored, self.max_patterns)

    @property
    def known_objects(self) -> list[str]:
        """Unique object IDs in memory."""
        return list(dict.fromkeys(self._object_ids[: self.n_stored]))

    def store(
        self,
        loc_enc: torch.Tensor,
        feat_enc: torch.Tensor,
        object_id: str,
        raw_location: np.ndarray | None = None,
        novelty_threshold: float | None = None,
    ) -> bool:
        """Store a (location, feature) composite tagged with object_id.

        Parameters
        ----------
        loc_enc : Tensor
            Grid-cell encoded location.
        feat_enc : Tensor
            Feature encoding.
        object_id : str
            Object label.
        raw_location : ndarray or None
            Raw 3D sensor location (world frame) for reference frame
            estimation.  If None, a zero vector is stored.
        novelty_threshold : float or None
            Override instance threshold for this store call.

        Returns True if stored, False if rejected by novelty gate.
        """
        with torch.no_grad():
            loc = loc_enc.detach().to(self.device).float()
            feat = feat_enc.detach().to(self.device).float()
            composite = torch.cat([loc, feat])

            thresh = (
                novelty_threshold
                if novelty_threshold is not None
                else self.novelty_threshold
            )

            if thresh is not None and self.n_stored > 0:
                n = self.n_stored
                c_norm = composite / (composite.norm() + 1e-8)
                p_norms = self._patterns[:n] / (
                    self._patterns[:n].norm(dim=1, keepdim=True) + 1e-8
                )
                sims = torch.mv(p_norms, c_norm)
                if sims.max().item() >= thresh:
                    return False

            raw_loc = (
                np.asarray(raw_location, dtype=np.float64)
                if raw_location is not None
                else np.zeros(3, dtype=np.float64)
            )

            if self._n_stored < self.max_patterns:
                self._patterns = torch.cat(
                    [self._patterns, composite.unsqueeze(0)]
                )
                self._object_ids.append(object_id)
                self._raw_locations.append(raw_loc)
            else:
                idx = self._n_stored % self.max_patterns
                self._patterns[idx] = composite
                self._object_ids[idx] = object_id
                self._raw_locations[idx] = raw_loc

            self._n_stored += 1
            return True

    def query_features(
        self,
        feat_enc: torch.Tensor,
        beta: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, np.ndarray]]:
        """Feature-only query: cosine similarity on feature dimensions only.

        Use when location is unknown (eval without reference frame).

        Returns
        -------
        retrieved : Tensor
            Attention-weighted composite pattern.
        evidence : dict[str, float]
            Per-object evidence (softmax attention sums).
        predicted_locations : dict[str, ndarray]
            Per-object weighted average of raw 3D training locations.
            Used by ReferenceFrameEstimator for rotation estimation.
        """
        if self.n_stored == 0:
            return torch.zeros(self.d_total, device=self.device), {}, {}

        with torch.no_grad():
            b = beta if beta is not None else self.beta
            feat = feat_enc.detach().to(self.device).float()
            q_norm = feat / (feat.norm() + 1e-8)

            n = self.n_stored
            stored_feats = self._patterns[:n, self.d_loc :]
            sf_norms = stored_feats / (
                stored_feats.norm(dim=1, keepdim=True) + 1e-8
            )

            sims = b * torch.mv(sf_norms, q_norm)
            sims = sims - sims.max()
            attention = torch.softmax(sims, dim=0)

            retrieved = torch.mv(self._patterns[:n].t(), attention)
            evidence = self._aggregate_evidence(attention, n)
            pred_locs = self._aggregate_raw_locations(attention, n)
            return retrieved, evidence, pred_locs

    def query_location(
        self,
        loc_enc: torch.Tensor,
        beta: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Location-only query: cosine similarity on location dimensions only.

        Use for spatial prediction (what should I see here?).
        Returns (retrieved_composite, per_object_evidence).
        """
        if self.n_stored == 0:
            return torch.zeros(self.d_total, device=self.device), {}

        with torch.no_grad():
            b = beta if beta is not None else self.beta
            loc = loc_enc.detach().to(self.device).float()
            q_norm = loc / (loc.norm() + 1e-8)

            n = self.n_stored
            stored_locs = self._patterns[:n, : self.d_loc]
            sl_norms = stored_locs / (
                stored_locs.norm(dim=1, keepdim=True) + 1e-8
            )

            sims = b * torch.mv(sl_norms, q_norm)
            sims = sims - sims.max()
            attention = torch.softmax(sims, dim=0)

            retrieved = torch.mv(self._patterns[:n].t(), attention)
            evidence = self._aggregate_evidence(attention, n)
            return retrieved, evidence

    def query_full(
        self,
        loc_enc: torch.Tensor,
        feat_enc: torch.Tensor,
        beta: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Full (location, feature) query for strongest retrieval.

        Both location and feature parts contribute to cosine similarity.
        Returns (retrieved_composite, per_object_evidence).
        """
        if self.n_stored == 0:
            return torch.zeros(self.d_total, device=self.device), {}

        with torch.no_grad():
            b = beta if beta is not None else self.beta
            loc = loc_enc.detach().to(self.device).float()
            feat = feat_enc.detach().to(self.device).float()
            query = torch.cat([loc, feat])
            q_norm = query / (query.norm() + 1e-8)

            n = self.n_stored
            p_norms = self._patterns[:n] / (
                self._patterns[:n].norm(dim=1, keepdim=True) + 1e-8
            )

            sims = b * torch.mv(p_norms, q_norm)
            sims = sims - sims.max()
            attention = torch.softmax(sims, dim=0)

            retrieved = torch.mv(self._patterns[:n].t(), attention)
            evidence = self._aggregate_evidence(attention, n)
            return retrieved, evidence

    def _aggregate_evidence(
        self, attention: torch.Tensor, n: int
    ) -> dict[str, float]:
        """Sum attention weights per object ID."""
        evidence: dict[str, float] = {}
        att_cpu = attention.cpu()
        for i in range(n):
            oid = self._object_ids[i]
            evidence[oid] = evidence.get(oid, 0.0) + float(att_cpu[i])
        return evidence

    def _aggregate_raw_locations(
        self, attention: torch.Tensor, n: int
    ) -> dict[str, np.ndarray]:
        """Per-object attention-weighted average of raw 3D training locations.

        Used by ReferenceFrameEstimator: the predicted object-space location
        for each object given the current feature query.
        """
        if not self._raw_locations:
            return {}
        weighted_locs: dict[str, np.ndarray] = {}
        weight_sums: dict[str, float] = {}
        att_cpu = attention.cpu().numpy()
        for i in range(n):
            oid = self._object_ids[i]
            w = float(att_cpu[i])
            if oid not in weighted_locs:
                weighted_locs[oid] = np.zeros(3, dtype=np.float64)
                weight_sums[oid] = 0.0
            weighted_locs[oid] += w * self._raw_locations[i]
            weight_sums[oid] += w
        for oid in weighted_locs:
            if weight_sums[oid] > 1e-12:
                weighted_locs[oid] /= weight_sums[oid]
        return weighted_locs

    def clear(self) -> None:
        """Remove all stored patterns."""
        self._patterns = torch.zeros(
            0, self.d_total, dtype=torch.float32, device=self.device
        )
        self._object_ids = []
        self._raw_locations = []
        self._n_stored = 0

    def state_dict(self) -> dict:
        raw_locs = np.array(self._raw_locations) if self._raw_locations else \
            np.zeros((0, 3), dtype=np.float64)
        return {
            "patterns": self._patterns.cpu(),
            "object_ids": list(self._object_ids),
            "raw_locations": raw_locs,
            "n_stored": self._n_stored,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._patterns = sd["patterns"].to(self.device)
        self._object_ids = list(sd["object_ids"])
        raw_locs = sd.get("raw_locations", np.zeros((0, 3), dtype=np.float64))
        if len(raw_locs) > 0:
            self._raw_locations = [raw_locs[i] for i in range(len(raw_locs))]
        else:
            self._raw_locations = [
                np.zeros(3, dtype=np.float64) for _ in self._object_ids
            ]
        self._n_stored = sd["n_stored"]
