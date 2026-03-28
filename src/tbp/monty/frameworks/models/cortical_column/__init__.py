# Copyright 2025-2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""Biologically plausible cortical column implementation.

Implements the core computation of a neocortical column using sparse
distributed representations (SDRs), dendritic segments for context-dependent
prediction, recurrent attractor dynamics, and three-factor Hebbian learning.

Key classes:

- :class:`~.encoders.GridCellEncoder` — multi-scale periodic location encoding
- :class:`~.encoders.ScalarEncoder` — continuous features to SDR
- :class:`~.dendrites.DendriteSegments` — multi-segment context prediction
- :class:`~.recurrent.RecurrentConnections` — attractor dynamics via Hebbian recurrence
- :class:`~.associative_memory.HeteroAssociativeMemory` — weight-based recognition
- :class:`~.attractor_memory.AttractorMemory` — prototype-based object recognition
- :class:`~.column.CorticalColumn` — unified spatial + temporal column
- :class:`~.motor_prediction.MotorPrediction` — sensorimotor location prediction
- :class:`~.sdr_memory.SDRObjectMemory` — SDR-based object storage (legacy)
"""

from tbp.monty.frameworks.models.cortical_column.associative_memory import (
    HeteroAssociativeMemory,
)
from tbp.monty.frameworks.models.cortical_column.attractor_memory import (
    AttractorMemory,
)
from tbp.monty.frameworks.models.cortical_column.column import CorticalColumn
from tbp.monty.frameworks.models.cortical_column.dendrites import DendriteSegments
from tbp.monty.frameworks.models.cortical_column.encoders import (
    GridCellEncoder,
    ScalarEncoder,
)
from tbp.monty.frameworks.models.cortical_column.motor_prediction import (
    MotorEncoder,
    MotorPrediction,
)
from tbp.monty.frameworks.models.cortical_column.neuromodulators import (
    NeuromodulatoryState,
)
from tbp.monty.frameworks.models.cortical_column.recurrent import (
    RecurrentConnections,
)
from tbp.monty.frameworks.models.cortical_column.sdr_memory import SDRObjectMemory

__all__ = [
    "AttractorMemory",
    "CorticalColumn",
    "DendriteSegments",
    "GridCellEncoder",
    "HeteroAssociativeMemory",
    "MotorEncoder",
    "MotorPrediction",
    "NeuromodulatoryState",
    "RecurrentConnections",
    "ScalarEncoder",
    "SDRObjectMemory",
]
