# Copyright 2025-2026 Thousand Brains Project
# Copyright 2022-2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import copy
import logging

import numpy as np

from tbp.monty.frameworks.models.graph_matching import (
    MontyForGraphMatching,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    align_orthonormal_vectors,
)

logger = logging.getLogger(__name__)


class MontyForEvidenceGraphMatching(MontyForGraphMatching):
    """Monty model for evidence-based graphs.

    Customize voting and union of possible matches.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and reset LM."""
        self.authoritative_goal_sender_ids = set(
            kwargs.pop("authoritative_goal_sender_ids", []) or []
        )
        super().__init__(*args, **kwargs)

    def _select_motor_goal_states(self):
        goal_states = [
            goal_state
            for goal_state in getattr(self, "gsg_outputs", [])
            if goal_state is not None
            and bool(getattr(goal_state, "use_state", True))
        ]

        authoritative_goal_sender_ids = set(
            getattr(self, "authoritative_goal_sender_ids", set()) or []
        )
        if not authoritative_goal_sender_ids:
            return goal_states

        return [
            goal_state
            for goal_state in goal_states
            if getattr(goal_state, "sender_id", None)
            in authoritative_goal_sender_ids
        ]

    def _pass_infos_to_motor_system(self):
        """Pass processed observations and goal states to the motor system.

        Currently there is no complex connectivity or hierarchy, and all generated
        goal states are considered bound for the motor system. TODO M change this.
        """
        super()._pass_infos_to_motor_system()
        goal_states = self._select_motor_goal_states()

        # Check that the motor system can receive goal states
        if self.motor_system._policy.use_goal_state_driven_actions:
            if hasattr(self.motor_system._policy, "set_driving_goal_states"):
                self.motor_system._policy.set_driving_goal_states(goal_states)
                return

            best_goal_state = None
            best_goal_confidence = -np.inf
            for current_goal_state in goal_states:
                if (
                    current_goal_state is not None
                    and current_goal_state.confidence > best_goal_confidence
                ):
                    best_goal_state = current_goal_state
                    best_goal_confidence = current_goal_state.confidence

            self.motor_system._policy.set_driving_goal_state(best_goal_state)

    def _combine_votes(self, votes_per_lm):
        """Combine evidence from different LMs.

        Returns:
            The combined votes.
        """
        combined_votes = []
        for i in range(len(self.learning_modules)):
            lm_state_votes = {}
            aggregated_ranked_hypotheses = {}
            if votes_per_lm[i] is not None:
                receiving_lm_pose = votes_per_lm[i]["sensed_pose_rel_body"]
                for j in self.lm_to_lm_vote_matrix[i]:
                    if votes_per_lm[j] is not None:
                        for hypothesis in votes_per_lm[j].get(
                            "ranked_hypotheses", []
                        ):
                            object_id = str(hypothesis.get("object_id", ""))
                            if not object_id:
                                continue
                            entry = aggregated_ranked_hypotheses.setdefault(
                                object_id,
                                {
                                    "object_id": object_id,
                                    "probability": 0.0,
                                    "evidence": 0.0,
                                    "source_rank": int(hypothesis.get("rank", 10**6)),
                                    "sender_ids": set(),
                                },
                            )
                            entry["probability"] += float(
                                hypothesis.get("probability", 0.0)
                            )
                            entry["evidence"] = max(
                                entry["evidence"],
                                float(hypothesis.get("evidence", 0.0)),
                            )
                            entry["source_rank"] = min(
                                entry["source_rank"],
                                int(hypothesis.get("rank", 10**6)),
                            )
                            sender_id = hypothesis.get(
                                "sender_id",
                                votes_per_lm[j].get("sender_id"),
                            )
                            if sender_id is not None:
                                entry["sender_ids"].add(str(sender_id))

                        sending_lm_pose = votes_per_lm[j]["sensed_pose_rel_body"]
                        sensor_disp = np.array(receiving_lm_pose[0]) - np.array(
                            sending_lm_pose[0]
                        )
                        sensor_rotation_disp, _ = align_orthonormal_vectors(
                            sending_lm_pose[1:],
                            receiving_lm_pose[1:],
                            as_scipy=False,
                        )
                        logger.debug(
                            f"LM {j} to {i} - displacement: {sensor_disp}, "
                            f"rotation: "
                            f"{sensor_rotation_disp}"
                        )
                        for obj in votes_per_lm[j]["possible_states"]:
                            # Get the displacement between the sending and receiving
                            # sensor and take this into account when transmitting
                            # possible locations on the object.
                            # "If I am here, you should be there."
                            lm_states_for_object = votes_per_lm[j]["possible_states"][
                                obj
                            ]
                            # Take the location votes and transform them so they would
                            # apply to the receiving LM's sensor. Basically, if my
                            # sensor is here in this pose, then your sensor should be
                            # there in that pose.
                            # NOTE: rotation votes are not being used right now.
                            transformed_lm_states_for_object = []
                            for s in lm_states_for_object:
                                # need to make a copy because the same vote state may be
                                # transformed in different ways depending on the
                                # receiving LMs' poses
                                new_s = copy.deepcopy(s)
                                rotated_displacement = new_s.get_pose_vectors().dot(
                                    sensor_disp
                                )
                                new_s.transform_morphological_features(
                                    translation=rotated_displacement,
                                    rotation=sensor_rotation_disp,
                                )
                                transformed_lm_states_for_object.append(new_s)
                            if obj in lm_state_votes:
                                lm_state_votes[obj].extend(
                                    transformed_lm_states_for_object
                                )
                            else:
                                lm_state_votes[obj] = transformed_lm_states_for_object
            logger.debug(f"VOTE from LMs {self.lm_to_lm_vote_matrix[i]} to LM {i}")
            vote = {"possible_states": lm_state_votes}
            if aggregated_ranked_hypotheses:
                ranked_hypotheses = sorted(
                    aggregated_ranked_hypotheses.values(),
                    key=lambda item: (
                        -float(item["probability"]),
                        int(item["source_rank"]),
                        str(item["object_id"]),
                    ),
                )
                for rank, hypothesis in enumerate(ranked_hypotheses, start=1):
                    hypothesis["rank"] = rank
                    hypothesis["sender_ids"] = sorted(hypothesis["sender_ids"])
                    hypothesis.pop("source_rank", None)
                vote["ranked_hypotheses"] = ranked_hypotheses
            combined_votes.append(vote)
        return combined_votes

    def switch_to_exploratory_step(self):
        """Switch to an exploratory step.

        Also set MLH evidence high enough to generate output during exploration.
        """
        super().switch_to_exploratory_step()
        # Make sure the new object ID is communicated to higher-level LMs during
        # exploration.
        for lm in self.learning_modules:
            if hasattr(lm, "current_mlh") and hasattr(lm, "object_evidence_threshold"):
                lm.current_mlh["evidence"] = lm.object_evidence_threshold + 1
