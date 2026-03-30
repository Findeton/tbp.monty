# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Hopfield-based associative memory with attention readout.

Storage: Hebbian outer product  W += lr * x ⊗ label
Retrieval: softmax attention over stored object patterns.

Provides exponential storage capacity from the softmax retrieval, replacing
the linear readout of the numpy HeteroAssociativeMemory.
"""

from __future__ import annotations

import hashlib

import torch


class HopfieldAssociativeMemory:
    """Associative memory using modern Hopfield / attention readout.

    Parameters
    ----------
    n_cells : int
        Dimensionality of cell activation patterns.
    n_label_bits : int
        Dimensionality of label SDR.
    beta : float
        Inverse temperature for softmax retrieval.
    learning_rate : float
        Hebbian learning rate for storage.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        n_cells: int = 16384,
        n_label_bits: int = 256,
        beta: float = 8.0,
        learning_rate: float = 0.01,
        device: str = "cpu",
    ):
        self.n_cells = n_cells
        self.n_label_bits = n_label_bits
        self.beta = beta
        self.learning_rate = learning_rate
        self.device = torch.device(device)

        # Weight matrix: W @ x → label space
        self._W = torch.zeros(
            n_label_bits, n_cells, dtype=torch.float32, device=self.device
        )
        # Per-object prototypes for attention retrieval
        self._prototypes: dict[str, torch.Tensor] = {}  # obj_name → (n_cells,)
        self._proto_counts: dict[str, int] = {}          # obj_name → n_observations
        self._labels: dict[str, torch.Tensor] = {}      # obj_name → (n_label_bits,)

    @property
    def known_objects(self) -> list[str]:
        return list(self._prototypes.keys())

    def learn(
        self,
        x: torch.Tensor,
        object_name: str,
        lr: float | None = None,
    ) -> None:
        """Store an observation pattern associated with an object.

        Hebbian update: W += lr * label ⊗ x
        Prototype: EMA of observed patterns for this object.
        """
        with torch.no_grad():
            rate = lr if lr is not None else self.learning_rate
            x = x.detach().to(self.device).float()

            if object_name not in self._labels:
                self._labels[object_name] = self._make_label(object_name)
            label = self._labels[object_name]

            # Hebbian: W += lr * label ⊗ x
            self._W += rate * torch.outer(label, x)

            # Prototype: running mean (count-based) for stable averaging
            # across the full training trajectory, not just recent observations.
            if object_name not in self._prototypes:
                self._prototypes[object_name] = x.clone()
                self._proto_counts[object_name] = 1
            else:
                n = self._proto_counts[object_name] + 1
                self._proto_counts[object_name] = n
                # Online mean: proto = proto + (x - proto) / n
                self._prototypes[object_name] += (
                    x - self._prototypes[object_name]
                ) / n

    def recall(self, x: torch.Tensor) -> dict[str, float]:
        """Retrieve per-object evidence using attention over prototypes.

        Modern Hopfield retrieval: softmax(beta * cos_sim(protos, x)).
        Provides exponential storage capacity from the softmax retrieval.

        Returns dict of object_name → score.
        """
        if not self._prototypes:
            return {}

        with torch.no_grad():
            x = x.detach().to(self.device).float()
            x_norm = x / (x.norm() + 1e-8)

            names = list(self._prototypes.keys())
            protos = torch.stack([self._prototypes[n] for n in names])
            proto_norms = protos / (protos.norm(dim=1, keepdim=True) + 1e-8)

            # Attention: softmax(beta * proto @ x)
            similarities = self.beta * torch.mv(proto_norms, x_norm)
            similarities = similarities - similarities.max()
            attention = torch.softmax(similarities, dim=0)

            scores = {}
            for i, name in enumerate(names):
                scores[name] = float(attention[i])

            return scores

    def auto_label(self, activation_history: list[torch.Tensor]) -> str:
        """Generate a deterministic label from activation history."""
        # Hash the first few patterns
        h = hashlib.sha256()
        for pat in activation_history[:5]:
            h.update(pat.cpu().numpy().tobytes())
        return f"auto_{h.hexdigest()[:12]}"

    def _make_label(self, name: str) -> torch.Tensor:
        """Create a deterministic label SDR from object name."""
        h = hashlib.sha256(name.encode()).digest()
        bits = torch.zeros(self.n_label_bits, dtype=torch.float32,
                           device=self.device)
        for i in range(min(self.n_label_bits, len(h) * 8)):
            byte_idx = i // 8
            bit_idx = i % 8
            if h[byte_idx] & (1 << bit_idx):
                bits[i] = 1.0
        return bits

    def state_dict(self) -> dict:
        return {
            "W": self._W.cpu(),
            "prototypes": {k: v.cpu() for k, v in self._prototypes.items()},
            "proto_counts": dict(self._proto_counts),
            "labels": {k: v.cpu() for k, v in self._labels.items()},
        }

    def load_state_dict(self, sd: dict) -> None:
        self._W = sd["W"].to(self.device)
        self._prototypes = {k: v.to(self.device) for k, v in sd["prototypes"].items()}
        self._proto_counts = sd.get("proto_counts", {
            k: 1 for k in self._prototypes
        })
        self._labels = {k: v.to(self.device) for k, v in sd["labels"].items()}
