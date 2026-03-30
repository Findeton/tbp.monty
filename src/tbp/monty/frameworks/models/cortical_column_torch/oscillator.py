# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Oscillatory phase coding for precise temporal interval encoding.

STDP captures temporal *ordering* but not *interval*. This module adds
an internal oscillator whose phase provides a reference signal. Cell
activations are augmented with phase tags, enabling the Hopfield memory
to distinguish patterns that occurred at different temporal positions.

Key concepts:
- **Oscillator**: Sinusoidal signal with configurable period.
- **Phase encoding**: Each cell's firing is tagged with (cos φ, sin φ).
- **Phase precession**: Successive sequence elements fire at progressively
  earlier phases, compressing temporal sequences.
- **Phase-augmented patterns**: pattern_aug = concat(x_spatial, phase_vec)

References:
- O'Keefe & Recce (1993) — theta-phase precession
- Lisman & Jensen (2013) — theta-gamma multi-item working memory
- Fries (2015) — "Rhythms for Cognition: Communication through Coherence"
"""

from __future__ import annotations

import math

import torch


class CorticalOscillator:
    """Internal oscillator providing phase reference signal.

    Parameters
    ----------
    period : float
        Oscillation period in steps (default 20 ≈ 50ms theta at 20ms/step).
    precession_rate : float
        Phase shift per sequence position (radians). Controls how much
        successive items precess relative to oscillation.
    device : str
        PyTorch device.
    """

    def __init__(
        self,
        period: float = 20.0,
        precession_rate: float = 0.3,
        device: str = "cpu",
    ):
        self.period = period
        self.precession_rate = precession_rate
        self.device = torch.device(device)

        self._step = 0
        self._sequence_position = 0

    @property
    def phase(self) -> float:
        """Current oscillator phase in [0, 2π)."""
        return (2.0 * math.pi * self._step / self.period) % (2.0 * math.pi)

    @property
    def oscillation(self) -> float:
        """Current oscillation value in [-1, 1]."""
        return math.cos(self.phase)

    def effective_phase(self) -> float:
        """Phase with precession shift applied.

        Items later in a sequence fire at earlier phases.
        """
        shift = -self.precession_rate * self._sequence_position
        return (self.phase + shift) % (2.0 * math.pi)

    def phase_vector(self) -> torch.Tensor:
        """2D phase encoding: [cos(φ), sin(φ)].

        Returns Tensor of shape (2,) for smooth wraparound encoding.
        """
        phi = self.effective_phase()
        return torch.tensor(
            [math.cos(phi), math.sin(phi)],
            dtype=torch.float32, device=self.device,
        )

    def augment_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate phase encoding to spatial pattern.

        Parameters
        ----------
        x : Tensor (n_cells,)
            Spatial activation pattern.

        Returns
        -------
        Tensor (n_cells + 2,) — spatial features + phase encoding.
        """
        pv = self.phase_vector()
        return torch.cat([x, pv])

    def strip_phase(self, x_aug: torch.Tensor, n_cells: int) -> torch.Tensor:
        """Remove phase encoding, return spatial-only pattern.

        Parameters
        ----------
        x_aug : Tensor (n_cells + 2,)
            Phase-augmented pattern.
        n_cells : int
            Original spatial dimensionality.

        Returns
        -------
        Tensor (n_cells,) — spatial features only.
        """
        return x_aug[:n_cells]

    def extract_phase(self, x_aug: torch.Tensor, n_cells: int) -> torch.Tensor:
        """Extract phase encoding from augmented pattern.

        Returns Tensor (2,).
        """
        return x_aug[n_cells:n_cells + 2]

    def phase_difference(
        self, phase_a: torch.Tensor, phase_b: torch.Tensor
    ) -> float:
        """Compute angular difference between two phase vectors.

        Returns value in [0, π].
        """
        dot = (phase_a * phase_b).sum().clamp(-1, 1)
        return float(torch.acos(dot))

    def step(self, is_new_element: bool = True) -> None:
        """Advance oscillator by one step.

        Parameters
        ----------
        is_new_element : bool
            If True, increment sequence position (for precession).
        """
        self._step += 1
        if is_new_element:
            self._sequence_position += 1

    def reset(self) -> None:
        """Reset oscillator state for new episode."""
        self._step = 0
        self._sequence_position = 0


class PhaseAugmentedHopfield:
    """Wrapper that stores/retrieves phase-augmented patterns.

    Works with an existing ModernHopfieldMemory by augmenting patterns
    before storage and stripping phase after retrieval.

    Parameters
    ----------
    hopfield : object
        A ModernHopfieldMemory instance.
    oscillator : CorticalOscillator
        The oscillator providing phase tags.
    n_spatial_cells : int
        Number of spatial cells (before augmentation).
    """

    def __init__(self, hopfield, oscillator: CorticalOscillator,
                 n_spatial_cells: int):
        self.hopfield = hopfield
        self.oscillator = oscillator
        self.n_spatial = n_spatial_cells

    def store(self, x: torch.Tensor) -> None:
        """Store a phase-augmented pattern."""
        x_aug = self.oscillator.augment_pattern(x)
        self.hopfield.store(x_aug)

    def settle(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, int]:
        """Settle with phase-augmented query, return spatial-only result."""
        x_aug = self.oscillator.augment_pattern(x)
        settled_aug, n_iters = self.hopfield.settle(x_aug, **kwargs)
        spatial = self.oscillator.strip_phase(settled_aug, self.n_spatial)
        return spatial, n_iters

    def retrieve(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Single-step retrieval with phase augmentation."""
        x_aug = self.oscillator.augment_pattern(x)
        result_aug = self.hopfield.retrieve(x_aug, **kwargs)
        return self.oscillator.strip_phase(result_aug, self.n_spatial)
