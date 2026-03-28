# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging

import numpy as np

from tbp.monty.frameworks.models.graph_matching import GraphMemory
from tbp.monty.frameworks.models.object_model import (
    GridObjectModel,
    GridTooSmallError,
)

logger = logging.getLogger(__name__)


class EvidenceGraphMemory(GraphMemory):
    """Custom GraphMemory that stores GridObjectModel instead of GraphObjectModel."""

    def __init__(
        self,
        max_nodes_per_graph,
        max_graph_size,
        num_model_voxels_per_dim,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.max_nodes_per_graph = max_nodes_per_graph
        self.max_graph_size = max_graph_size
        self.num_model_voxels_per_dim = num_model_voxels_per_dim

    # =============== Public Interface Functions ===============

    # ------------------- Main Algorithm -----------------------

    # ------------------ Getters & Setters ---------------------
    def get_initial_hypotheses(self):
        return self.get_memory_ids()

    def get_states_in_graph(self, graph_id, input_channel):
        """Get list of state IDs if model is state-conditioned, else None.

        Args:
            graph_id: ID of graph.
            input_channel: Input channel identifier.

        Returns:
            List of state IDs or None if model is not state-conditioned.
        """
        model = self.models_in_memory[graph_id][input_channel]
        if hasattr(model, "get_states") and hasattr(model, "is_state_conditioned"):
            return model.get_states()
        return None

    def get_channel_model(self, graph_id, input_channel, state_id=None):
        """Get the GridObjectModel for a specific (graph, channel, state).

        For plain models, state_id is ignored and the model is returned directly.
        For StateConditionedModels, returns the sub-model for the given state.

        Args:
            graph_id: ID of graph.
            input_channel: Input channel identifier.
            state_id: Optional state ID for state-conditioned models.

        Returns:
            GridObjectModel instance.
        """
        model = self.models_in_memory[graph_id][input_channel]
        if state_id is not None and hasattr(model, "get_model_for_state"):
            return model.get_model_for_state(state_id)
        # For plain models or when state_id is None
        if hasattr(model, "get_model_for_state") and hasattr(model, "get_states"):
            # StateConditionedModel with no explicit state: use first state
            states = model.get_states()
            if states:
                return model.get_model_for_state(states[0])
        return model

    def get_locations_in_graph(self, graph_id, input_channel, state_id=None):
        """Get node locations, optionally for a specific state.

        Args:
            graph_id: ID of graph.
            input_channel: Input channel identifier.
            state_id: Optional state ID for state-conditioned models.

        Returns:
            Array of node locations.
        """
        model = self.get_channel_model(graph_id, input_channel, state_id)
        return model.pos

    def get_graph(self, graph_id, input_channel=None, state_id=None):
        """Return graph/model from memory, optionally for a specific state.

        Overrides base class to add state_id support.
        """
        if input_channel is None:
            return self.models_in_memory[graph_id]

        if input_channel == "first":
            first_channel = self.get_input_channels_in_graph(graph_id)[0]
            return self.get_channel_model(graph_id, first_channel, state_id)

        if input_channel in self.get_input_channels_in_graph(graph_id):
            return self.get_channel_model(graph_id, input_channel, state_id)

        raise ValueError(f"{graph_id} has no data stored for {input_channel}.")

    def get_graph_node_ids(self, graph_id, input_channel, state_id=None):
        """Get node IDs, optionally for a specific state."""
        model = self.get_channel_model(graph_id, input_channel, state_id)
        num_nodes = model.x.shape[0]
        return np.linspace(0, num_nodes - 1, num_nodes, dtype=int)

    def get_features_at_node(
        self, graph_id, input_channel, node_id, feature_keys=None, state_id=None
    ):
        """Get features at a node, optionally for a specific state."""
        if feature_keys is None:
            feature_keys = self.features_to_use[input_channel]
        node_features = {}
        graph = self.get_channel_model(graph_id, input_channel, state_id)
        if graph is None:
            logger.debug(
                f"{input_channel} not stored in graph {graph_id} yet. "
                "-> Input not used for matching."
            )
        else:
            for key in feature_keys:
                if key not in graph.feature_mapping:
                    continue
                key_ids = graph.feature_mapping[key]
                feature = graph.x[node_id, key_ids[0] : key_ids[1]]
                node_features[key] = feature
        return node_features

    def get_rotation_features_at_all_nodes(
        self, graph_id, input_channel, state_id=None
    ):
        """Get rotation features from all N nodes. Shape=(N, 3, 3).

        Args:
            graph_id: ID of graph.
            input_channel: Input channel identifier.
            state_id: Optional state ID for state-conditioned models.

        Returns:
            The rotation features from all N nodes. Shape=(N, 3, 3).
        """
        all_node_r_features = self.get_features_at_node(
            graph_id,
            input_channel,
            self.get_graph_node_ids(graph_id, input_channel, state_id=state_id),
            feature_keys=["pose_vectors"],
            state_id=state_id,
        )
        node_directions = all_node_r_features["pose_vectors"]
        num_nodes = len(node_directions)
        return node_directions.reshape((num_nodes, 3, 3))

    def initialize_feature_arrays(self):
        """Build feature arrays, including per-state arrays for state models."""
        super().initialize_feature_arrays()
        # Build per-state feature arrays for state-conditioned models
        self.state_feature_arrays = {}
        for graph_id in self.get_memory_ids():
            for input_channel in self.get_input_channels_in_graph(graph_id):
                states = self.get_states_in_graph(graph_id, input_channel)
                if states is not None:
                    if graph_id not in self.state_feature_arrays:
                        self.state_feature_arrays[graph_id] = {}
                    self.state_feature_arrays[graph_id][input_channel] = {}
                    for state_id in states:
                        node_ids = self.get_graph_node_ids(
                            graph_id, input_channel, state_id=state_id
                        ).astype(int)
                        feature_arrays, feature_order = (
                            self._build_feature_array_for_state(
                                graph_id, input_channel, state_id, node_ids
                            )
                        )
                        self.state_feature_arrays[graph_id][input_channel][
                            state_id
                        ] = (feature_arrays, feature_order)

    def get_state_feature_array(self, graph_id, input_channel, state_id):
        """Get cached feature array for a specific state.

        Returns:
            Tuple of (feature_array, feature_order) for the given state.
        """
        return self.state_feature_arrays[graph_id][input_channel][state_id]

    def _build_feature_array_for_state(
        self, graph_id, input_channel, state_id, node_ids
    ):
        """Build feature array for a specific state's sub-model."""
        model = self.get_channel_model(graph_id, input_channel, state_id)
        num_nodes = len(node_ids)
        feature_arrays = None
        feature_order = []

        for i, node_id in enumerate(node_ids):
            node_features = {}
            for key in self.features_to_use.get(input_channel, []):
                if key in ["pose_vectors", "pose_fully_defined"]:
                    continue
                if key not in model.feature_mapping:
                    continue
                key_ids = model.feature_mapping[key]
                node_features[key] = model.x[node_id, key_ids[0] : key_ids[1]]

            if feature_arrays is None and node_features:
                total_features = sum(len(v) for v in node_features.values())
                feature_arrays = np.zeros((num_nodes, total_features))

            start_idx = 0
            for feature in node_features:
                if i == 0:
                    feature_order.append(feature)
                end_idx = start_idx + len(node_features[feature])
                if feature_arrays is not None:
                    feature_arrays[node_id, start_idx:end_idx] = node_features[
                        feature
                    ]
                start_idx = end_idx

        if feature_arrays is None:
            feature_arrays = np.zeros((num_nodes, 0))
        return feature_arrays, feature_order

    # ======================= Private ==========================

    # ------------------- Main Algorithm -----------------------
    def _add_graph_to_memory(self, model, graph_id):
        """Add a pretrained graph to memory.

        Initializes GridObjectModel and calls set_graph. Handles both plain
        GridObjectModel and StateConditionedModel instances.

        Args:
            model: New model to be added to memory.
            graph_id: ID of the graph that should be added.

        """
        self.models_in_memory[graph_id] = {}
        for input_channel in model:
            channel_model = model[input_channel]
            try:
                # Handle StateConditionedModel (duck-type check)
                if hasattr(channel_model, "is_state_conditioned"):
                    self.models_in_memory[graph_id][input_channel] = channel_model
                    logger.info(
                        f"Loaded StateConditionedModel for {input_channel} "
                        f"with states {channel_model.get_states()}"
                    )
                    continue

                if not isinstance(channel_model, GridObjectModel):
                    # When loading a model trained with a different LM, need to convert
                    # it to the GridObjectModel (with use_original_graph == True)
                    loaded_graph = channel_model._graph
                    channel_model = self._initialize_model_with_graph(
                        graph_id, loaded_graph
                    )
                else:
                    # serialization seems to mess up the sparse tensors, so we need to
                    # coalesce them again.
                    if channel_model._observation_count is not None:
                        channel_model._observation_count = (
                            channel_model._observation_count.coalesce()
                        )
                    if channel_model._feature_grid is not None:
                        channel_model._feature_grid = (
                            channel_model._feature_grid.coalesce()
                        )
                    if channel_model._location_grid is not None:
                        channel_model._location_grid = (
                            channel_model._location_grid.coalesce()
                        )

                logger.info(f"Loaded {model} for {input_channel}")
                self.models_in_memory[graph_id][input_channel] = channel_model
            except GridTooSmallError:
                logger.info("Grid too small for given locations. Not adding to memory.")

    def _initialize_model_with_graph(self, graph_id, graph):
        model = GridObjectModel(
            object_id=graph_id,
            max_nodes=self.max_nodes_per_graph,
            max_size=self.max_graph_size,
            num_voxels_per_dim=self.num_model_voxels_per_dim,
        )
        # Keep benchmark results constant by still using original graph for
        # matching when loading pretrained models.
        model.use_original_graph = True
        model.set_graph(graph)
        return model

    def _build_graph(self, locations, features, graph_id, input_channel):
        """Build a graph from a list of features at locations and add it to memory.

        This initializes a new GridObjectModel and calls model.build_graph.

        Args:
            locations: List of x, y, z locations.
            features: List of features.
            graph_id: ID of the new graph.
            input_channel: Identifier of the input channel.
        """
        logger.info("Adding a new graph to memory.")

        model = GridObjectModel(
            object_id=graph_id,
            max_nodes=self.max_nodes_per_graph,
            max_size=self.max_graph_size,
            num_voxels_per_dim=self.num_model_voxels_per_dim,
        )
        try:
            model.build_model(locations=locations, features=features)

            if graph_id not in self.models_in_memory:
                self.models_in_memory[graph_id] = {}
            self.models_in_memory[graph_id][input_channel] = model

            logger.info(f"Added new graph with id {graph_id} to memory.")
            logger.info(model)
        except GridTooSmallError:
            logger.info(
                "Grid too small for given locations. Not building a model "
                f"for {graph_id}"
            )

    def _extend_graph(
        self,
        locations,
        features,
        graph_id,
        input_channel,
        object_location_rel_body,
        location_rel_model,
        object_rotation,
    ):
        """Add new observations into an existing graph.

        Args:
            locations: List of x, y, z locations.
            features: Features observed at the provided locations.
            graph_id: ID of the existing graph.
            input_channel: Identifier of the input channel.
            object_location_rel_body: Location of the sensor in the body reference
                frame.
            location_rel_model: Location of the sensor in the model reference frame.
            object_rotation: Rotation of the sensed object relative to the model.
        """
        logger.info(f"Updating existing graph for {graph_id}")

        try:
            self.models_in_memory[graph_id][input_channel].update_model(
                locations=locations,
                features=features,
                location_rel_model=location_rel_model,
                object_location_rel_body=object_location_rel_body,
                object_rotation=object_rotation,
            )
            logger.info(
                f"Extended graph {graph_id} with new points. New model:\n"
                f"{self.models_in_memory[graph_id]}"
            )
        except GridTooSmallError:
            logger.info("Grid too small for given locations. Not updating model.")
