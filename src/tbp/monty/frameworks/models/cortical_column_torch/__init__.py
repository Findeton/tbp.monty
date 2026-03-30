# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Modern Hopfield cortical column implemented in PyTorch.

Replaces binary SDR computation with sparse continuous tensors and modern
Hopfield energy-based settling.  All learning is purely Hebbian — no backprop.
GPU-scalable via standard PyTorch ops.

Key classes:

- :class:`~.column.CorticalColumnTorch` — unified column with Hopfield dynamics
- :class:`~.hopfield.ModernHopfieldMemory` — energy-based attractor memory
- :class:`~.dendrites.SparseDendrites` — sparse matmul dendritic prediction
- :class:`~.associative_memory.HopfieldAssociativeMemory` — attention readout
- :class:`~.motor_prediction.ContrastiveMotorPrediction` — contrastive Hebbian
- :class:`~.neuromodulators.NeuromodulatoryGating` — ACh/NE/DA/5-HT gating
- :class:`~.learning_module.CorticalColumnTorchLM` — Monty LearningModule adapter

Track 10 (laminar biological depth):

- :class:`~.layers.LaminarConfig` — laminar cell allocation
- :class:`~.layers.L4Layer` — feedforward input layer
- :class:`~.layers.L23Layer` — representation / Hopfield layer
- :class:`~.layers.L5Layer` — output / apical integration layer
- :class:`~.layers.L6Layer` — feedback / thalamic modulation layer
- :class:`~.multihead_dendrites.MultiHeadDendrites` — multi-head attention
- :class:`~.plateau.PlateauPotentialMemory` — working memory
- :class:`~.interneurons.InterneuronCircuit` — PV+/SST+/VIP+ gating
- :class:`~.stdp.STDPRule` — spike-timing-dependent plasticity
- :class:`~.stdp.EligibilityTrace` — three-factor learning bridge
- :class:`~.oscillator.CorticalOscillator` — phase coding
- :class:`~.thalamic_relay.ThalamicRelay` — thalamocortical loop
"""

from tbp.monty.frameworks.models.cortical_column_torch.column import (
    CorticalColumnTorch,
)
from tbp.monty.frameworks.models.cortical_column_torch.learning_module import (
    CorticalColumnTorchLM,
)

__all__ = [
    "CorticalColumnTorch",
    "CorticalColumnTorchLM",
]
