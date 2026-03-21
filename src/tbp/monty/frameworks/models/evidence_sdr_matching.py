# Copyright 2025-2026 Thousand Brains Project
# Copyright 2024 Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import logging
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.evidence_matching.feature_evidence.sdr_calculator import (  # noqa: E501
    SDRFeatureEvidenceCalculator,
)
from tbp.monty.frameworks.models.evidence_matching.features_for_matching.all_selector import (  # noqa: E501
    AllFeaturesForMatchingSelector,
)
from tbp.monty.frameworks.models.evidence_matching.learning_module import (
    EvidenceGraphLM,
)
from tbp.monty.frameworks.models.states import State

logger = logging.getLogger(__name__)


class LoggerSDR:
    """A simple logger that saves the data passed to it.

    This logger maintains an episode counter and logs
    the data it receives under different files named
    by the episode counter.

    *See more information about what data is being logged
    under the `log_episode` function.*
    """

    # TODO: Needs to be removed. The logger should be part
    # of Monty loggers. Fix issue #328 first.
    def __init__(self, path):
        if path is None:
            logger.warning("EvidenceSDR log path is set to None.")
            return

        path = Path(path).expanduser()

        # overwrite existing logs
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

        self.path = path
        self.episode = 0

    def log_episode(self, data):
        """Receives data dictionary and saves it as a pth file.

        This function will save all the data passed to it. Here is
        a breakdown of the data to be logged by this function.

        The data dictionary contains these key-value pairs:
            - mask: 2d tensor of the available overlap targets after this
                    episode
            - target_overlap: 2d tensor of the target overlap at the end of
                    this episode
            - training: Dictionary of training statistics for every epoch.
                    Includes overlap_error, training_summed_distance,
                    dense representations, and sdrs
            - obj2id: Objects to ids dictionary mapping
            - id2obj: Ids to objects dictionary mapping
        """
        if hasattr(self, "path"):
            np.save(
                self.path / f"episode_{self.episode:03d}.npy",
                data,
            )
            self.episode += 1


class EncoderSDR:
    """The SDR Encoder class.

    This class keeps track of the dense representations, and trains them to output SDRs
    when binarized.  This class also contains its own optimizer and function to add more
    objects/representations.

    The representations are stored as dense vectors and binarized using
    top-k to convert them to SDRs. During training, the pairwise overlaps between the
    sdrs are compared to the target overlaps. This error signal trains the dense
    representations.

    Refer to the `self.train_sdrs` function for more information on the training details

    Attributes:
        sdr_length: The size of the SDRs (total number of bits).
        sdr_on_bits: The number of on bits in the SDRs. Controls sparsity.
        lr: The learning rate of the encoding algorithm.
        n_epochs: The number of training epochs per episode
        stability: The stability parameter controls by how much old SDRs
            change relative to new SDRs.  Value range is [0.0, 1.0], where 0.0 is no
            stability constraint applied and 1.0 is fixed SDRs. Values in between are
            for partial stability.
        log_flag: Flag to activate the logger.
    """

    def __init__(
        self,
        sdr_length=2048,
        sdr_on_bits=41,
        lr=1e-2,
        n_epochs=1000,
        stability=0.0,
        log_flag=False,
    ):
        if sdr_on_bits >= sdr_length or sdr_on_bits <= 0:
            logger.warning(
                f"Invalid sparsity: sdr_on_bits set to 2% ({round(sdr_length * 0.02)})"
            )
            sdr_on_bits = round(sdr_length * 0.02)

        self.sdr_length, self.sdr_on_bits = sdr_length, sdr_on_bits
        self.lr = lr
        self.n_epochs = n_epochs
        self.stability = stability
        if self.stability > 1.0 or self.stability < 0.0:
            self.stability = np.clip(self.stability, 0.0, 1.0)
            logger.warning(
                f"Invalid stability parameter: stability clamped to {self.stability}"
            )
        self.log_flag = log_flag

        # Initialize obj SDR array with arbitrary values
        self.obj_sdrs = np.zeros((0, self.sdr_length))

    @property
    def n_objects(self):
        """Return the available number of objects."""
        return self.obj_sdrs.shape[0]

    @property
    def sdrs(self):
        """Return the available SDRs."""
        return self.binarize(self.obj_sdrs)

    def state_dict(self):
        """Return the encoder state needed to reuse trained SDRs."""
        return dict(
            sdr_length=self.sdr_length,
            sdr_on_bits=self.sdr_on_bits,
            lr=self.lr,
            n_epochs=self.n_epochs,
            stability=self.stability,
            log_flag=self.log_flag,
            obj_sdrs=self.obj_sdrs.copy(),
            stable_ids=getattr(self, "stable_ids", np.array([], dtype=int)).copy(),
        )

    def load_state_dict(self, state_dict):
        """Restore encoder state from a serialized state dict."""
        self.sdr_length = state_dict["sdr_length"]
        self.sdr_on_bits = state_dict["sdr_on_bits"]
        self.lr = state_dict["lr"]
        self.n_epochs = state_dict["n_epochs"]
        self.stability = state_dict["stability"]
        self.log_flag = state_dict["log_flag"]
        self.obj_sdrs = state_dict["obj_sdrs"].copy()
        self.stable_ids = np.array(
            state_dict.get("stable_ids", np.array([], dtype=int)), dtype=int
        )

    def get_sdr(self, index):
        """Return the SDR at a specific index.

        This index refers to the object index in the SDRs dictionary
        (i.e., self.obj_sdrs)
        """
        return self.sdrs[index]

    def optimize(self, overlap_error, mask):
        """Compute and apply local gradient descent.

        Compute based on the overlap error and mask.

        Note there is no use of the chain rule, i.e. each SDR is optimized based on the
        derivative of its clustering error with respect to its values, with no
        intermediary functions.

        The overlap error helps correct the sign and also provides a magnitude for the
        representation updates.

        Args:
            overlap_error: The difference between target and predicted overlaps.
            mask: Mask indicating valid entries in the overlap matrix.

        Note:
            num_objects = self.n_objects

        Note:
            A vectorized version of the algorithm is provided below, although it
            would need to be modified to avoid repeated creation of arrays in order to
            be more efficient. Leaving for now as this algorithm is not a bottleneck
            (circa 10-20 seconds to learn 60 object SDRs).:

            # Initialize gradients
            grad = np.zeros_like(self.obj_sdrs)

            # Compute the pairwise differences between SDRs
            diff_matrix = (
                self.obj_sdrs[:, np.newaxis, :] - self.obj_sdrs[np.newaxis, :, :]
            )

            # Compute the absolute differences for each pair
            abs_diff = np.sum(np.abs(diff_matrix), axis=2)

            # Create a mask for non-zero differences
            non_zero_mask = abs_diff > 0

            # Apply the mask to the original mask
            valid_mask = mask & non_zero_mask

            # Calculate the summed distance and gradient contributions where the mask
            # is valid for logging
            summed_distance = np.sum(overlap_error * valid_mask * abs_diff)

            # Calculate the gradients
            grad_contrib = overlap_error[:, :, np.newaxis] * 2 * diff_matrix
            grad += np.sum(grad_contrib * valid_mask[:, :, np.newaxis], axis=1)
            grad -= np.sum(grad_contrib * valid_mask[:, :, np.newaxis], axis=0)

            # Update the SDRs using the gradient
            self.obj_sdrs -= self.lr * grad

        Returns:
            The summed distance for logging.
        """
        # Initialize the gradient array
        grad = np.zeros_like(self.obj_sdrs)

        # Track the summed distance for logging, i.e. to be able to visualize that
        # it is decreasing during each update
        summed_distance = 0

        # Calculate the gradient for each pair of objects
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if mask[i, j]:
                    # As we're optimizing the L2 norm of the difference between
                    # the two SDRs, the gradient is the difference between the
                    # two non-binarized (dense) representations.
                    diff = self.obj_sdrs[i] - self.obj_sdrs[j]
                    if np.sum(np.abs(diff)) > 0:
                        summed_distance += overlap_error[i, j] * np.sum(np.abs(diff))
                        # Multiply the gradient by 2 to exactly match the
                        # derivative of the L2 norm
                        grad[i] += overlap_error[i, j] * 2 * diff
                        grad[j] -= overlap_error[i, j] * 2 * diff

        # Update the SDRs using the gradient
        self.obj_sdrs -= self.lr * grad

        return summed_distance

    def add_objects(self, n_objects):
        """Adds more objects to the available objects and re-initializes the optimizer.

        We keep track of the stable representation ids (old objects) when
        adding new objects.

        Args:
            n_objects: Number of objects to add

        """
        if n_objects == 0:
            return

        # store stable data and ids
        stable_data = self.obj_sdrs.copy()
        self.stable_ids = np.arange(stable_data.shape[0])

        new_obj_sdrs = np.random.randn(
            stable_data.shape[0] + n_objects, self.sdr_length
        )

        new_obj_sdrs[: stable_data.shape[0]] = stable_data
        self.obj_sdrs = new_obj_sdrs

    def train_sdrs(self, target_overlaps, log_epoch_every=10):
        """Main SDR training function.

        This function receives a copy of the average target overlap 2D tensor
        and trains the sdr representations for `n_epochs` to achieve these target
        overlap scores.

        We use the overlap target as a learning signal to move the dense representations
        towards or away from each other. The magnitude of the overlap error controls the
        strength of moving dense representations. Also the sign of the overlap error
        controls whether the representations will be moving towards or away from each
        other.

        We want to limit the amount by which trained representation change relative
        to untrained object representations such that higher-level LMs would not suffer
        from significant changes in lower-level representations that were used to build
        higher-level graphs.

        When adding new representations, we keep track of the ids of the older
        representations (i.e., `self.stable_ids`). This allows us to control by how much
        the older representations move relative to the newer ones during training. This
        behavior is controlled by the stability value. During each training iteration,
        we update these older representations with an average of the optimizer output
        and the original representation (weighted by the stability value). Note that
        too much stability restricts the SDRs from adapting to desired changes in the
        target overlaps caused by normalization or distribution shift, affecting the
        overall encoding performance.

        Consider two dense representations, A_dense and B_dense. We apply top-k
        operation on both to convert them to A_sdr and B_sdr, then calculate their
        overlaps. If the overlap is less than the target overlap, we move dense
        representations (A_dense and B_dense) closer to each other with strength
        proportional to the error in overlaps. We move them apart if they have more
        overlap than the target.

        Note:
            The `distance_matrix` variable is calculated using the cdist function
            and it denotes the pairwise euclidean distances between *dense*
            representations. The term "overlap" always refers to the overlap in bits
            between SDRs.

        Note:
            The overlap_error is only used to weight the distance_matrix for
            each pair of objects, and gradients *do not* flow through the sparse
            overlap calculations.

        Returns:
            The stats dictionary for logging.
        """
        # return if no target provided
        stats = {}
        if np.all(np.isnan(target_overlaps)):
            logger.warning("Empty overlap targets. No training needed.")
            return stats

        if np.all(np.array(target_overlaps.shape) > self.n_objects):
            logger.warning(
                f"Overlap targets have larger size than "
                f"{(self.n_objects, self.n_objects)}"
            )
            target_overlaps = target_overlaps[: self.n_objects, : self.n_objects]

        # Calculate the training mask and target
        # The mask determines the valid entries with value:
        #     - 1: where overlap target exists
        #     - 0: where overlap target does not exist as of this episode.
        mask = ~np.isnan(target_overlaps)
        overlaps = np.nan_to_num(target_overlaps, nan=0)

        # logging details
        if self.log_flag:
            stats["mask"] = mask
            stats["target_overlap"] = overlaps
            stats["training"] = {}

        for epoch in tqdm(range(self.n_epochs)):
            # These values are used to pull back the representations from moving
            # too far during training. Notice this is only applied on self.stable_ids.
            sdrs_stable_before = self.obj_sdrs[self.stable_ids].copy()

            # calculate predicted overlaps from existing representations
            reps = self.obj_sdrs
            bins = self.binarize(reps)
            pred_overlaps = bins @ bins.T

            # calculate error and optimize
            overlap_error = overlaps - pred_overlaps

            summed_distance = self.optimize(overlap_error, mask)

            # stabilize the SDRs at `self.stable_ids` by pulling them back towards
            # sdrs_stable_before
            sdrs_stable_after = self.obj_sdrs[self.stable_ids].copy()
            self.obj_sdrs[self.stable_ids] = (self.stability * sdrs_stable_before) + (
                (1 - self.stability) * sdrs_stable_after
            )

            # logging details
            if self.log_flag and epoch % log_epoch_every == 0:
                stats["training"][epoch] = {}
                stats["training"][epoch]["obj_dense"] = self.obj_sdrs.copy()
                stats["training"][epoch]["obj_sdr"] = bins.copy()
                stats["training"][epoch]["overlap_error"] = overlap_error
                stats["training"][epoch]["summed_distance"] = summed_distance

        # Reset stable ids.
        # Stability training only used after adding new objects
        self.stable_ids = np.array([]).astype(int)
        return stats

    def binarize(self, emb):
        """Convert dense representations to SDRs (0s and 1s) using Top-k function.

        Returns:
            The SDRs.
        """
        topk_indices = np.argsort(emb, axis=1)[:, -self.sdr_on_bits :]
        mask = np.zeros_like(emb)
        np.put_along_axis(mask, topk_indices, 1, axis=1)
        return mask


class EvidenceSDRTargetOverlaps:
    """Keep track of the running average of target overlaps for each episode.

    The target overlaps is implemented as a 2D tensor where the indices
    of the tensor represent the ids of the objects, and the values of the
    tensor represent a running average of the overlap target.

    To achieve this, we implement functions for expanding the size of the
    overlap target tensor, linear mapping for normalization, and updating
    the overlap tensor (i.e., running average).

    Note:
        We are averaging over multiple overlap targets.
        Multiple targets can happen for different reasons:
            - Asymmetric evidences: the target overlap for object 1 w.r.t object 2
                ([2,1]) is averaged with object 2 w.r.t object 1 ([1,2]). This is
                possible because we sort the ids when we add the evidences to
                overlaps. Both evidences get added to the location [1,2].
            - Additional episodes with similar MLO (most-likely object): More episodes
                can accumulate additional evidences on to the same key if the MLO
                is similar to previous MLO of another episode.
    """

    def __init__(self):
        """Initialize with overlap tensor.

        Initialize the class with overlap tensor to store the running average of the
        target scores. Additionally we store the counts to easily calculate the running
        average.
        """
        self._overlaps = np.full((0, 0), np.nan)
        self._counts = np.zeros_like(self._overlaps)

    @property
    def overlaps(self):
        """Returns the target overlap values rounded to the nearest integer."""
        # TODO: Experiment without rounding. Shouldn't make much of a difference
        # since this is only used to weight the encoding distances.
        return np.round(self._overlaps)

    def state_dict(self):
        """Return the running-overlap state."""
        return dict(
            overlaps=self._overlaps.copy(),
            counts=self._counts.copy(),
        )

    def load_state_dict(self, state_dict):
        """Restore the running-overlap state."""
        self._overlaps = state_dict["overlaps"].copy()
        self._counts = state_dict["counts"].copy()

    def add_objects(self, new_size):
        """Expands the overlaps and the counts 2D tensors to accommodate new objects."""
        # expand the overlaps tensor to the new size
        new_overlaps = np.full((new_size, new_size), np.nan)
        new_overlaps[: self._overlaps.shape[0], : self._overlaps.shape[0]] = (
            self._overlaps
        )
        self._overlaps = new_overlaps

        # expand the counts tensor to the new size
        new_counts = np.zeros((new_size, new_size))
        new_counts[: self._counts.shape[0], : self._counts.shape[0]] = self._counts
        self._counts = new_counts

    def map_to_overlaps(self, evidence, output_range):
        """Linear mapping of values from input range to output range.

        Only applies to real values (i.e., ignores nan values).

        Returns:
            ?
        """
        valid_ix = ~np.isnan(evidence)
        if not np.any(valid_ix):
            return evidence

        min_evidence = np.nanmin(evidence[valid_ix])
        max_evidence = np.nanmax(evidence[valid_ix])
        input_range = [min_evidence, max_evidence]

        output_range_diff = output_range[1] - output_range[0]
        input_range_diff = input_range[1] - input_range[0]

        if np.isclose(input_range_diff, 0.0):
            evidence[valid_ix] = output_range[1]
            return evidence

        evidence[valid_ix] = (evidence[valid_ix] - input_range[0]) * (
            output_range_diff
        ) / input_range_diff + output_range[0]

        return evidence

    def add_overlaps(self, mapped_overlaps):
        """Main function for updating the running average with overlaps.

        The running average equation we use is:
        new_average = ((old_average * counts) + (new_val * 1))/ (counts + 1)

        This calculates equally-weighted average, assuming that we keep track
        of the counts and increment them every time we add a new value to the
        average.
        """
        # calculate the mask of indices for existing avg overlaps and
        # new overlaps. The mask should be True where both values are
        # not nan
        mask_avg = np.logical_and(~np.isnan(self._overlaps), ~np.isnan(mapped_overlaps))

        # apply the running average equation explained in the docstring
        self._overlaps[mask_avg] = (
            (self._overlaps[mask_avg] * self._counts[mask_avg])
            + mapped_overlaps[mask_avg]
        ) / (self._counts[mask_avg] + 1)

        # calculate the mask of indices with True values where existing overlaps
        # are nan and new overlaps are not nan.
        mask_overwrite = np.logical_and(
            np.isnan(self._overlaps), ~np.isnan(mapped_overlaps)
        )

        # overlap existing nan values in `self._overlaps` with new overlaps values
        # in `mapped_overlaps`
        self._overlaps[mask_overwrite] = mapped_overlaps[mask_overwrite]

        # update counts of all new entries
        self._counts[np.logical_or(mask_avg, mask_overwrite)] += 1

    def add_evidence(self, evidence, mapping_output_range):
        """Main function for updating the running average with evidence.

        This function receives as input the relative evidence scores and
        maps them to overlaps in the `mapping_output_range`. The mapped
        overlaps are added to the running average in the function `add_overlaps`.
        """
        # map relative evidences of the current episode to the output range
        mapped_overlaps = self.map_to_overlaps(evidence, mapping_output_range)

        # add overlaps to running average
        self.add_overlaps(mapped_overlaps)


class EvidenceSDRLMMixin:
    """This Mixin adds training of SDR representations to the EvidenceGraphLM.

    It overrides the __init__ and post_episode functions of the LM

    To use this Mixin, pass the EvidenceSDRGraphLM class as the `learning_module_class`
    in the `learning_module_configs`.

    Additionally pass the `sdr_args` dictionary as an additional key
    in the `learning_module_args`.

    The sdr_args dictionary should contain:
        - `log_path` (string): A string that points to a temporary location for saving
                experiment logs. "None" means don't save to file
        - `sdr_length` (int): The size of the SDR to be used for encoding
        - `sdr_on_bits` (int): The number of active bits to be used with these SDRs
        - `sdr_lr` (float): The learning rate of the encoding algorithm
        - `n_sdr_epochs` (int): The number of epochs to train the encoding algorithm
        - `stability` (float): Stability of older object SDRs.
                Value range is [0.0, 1.0], where 0.0 is no stability
                applied and 1.0 is fixed SDRs.
        - `sdr_log_flag` (bool): Flag indicating whether to log the results or not
        - `upward_top_k` (int, optional): Number of top child hypotheses to union
            into the upward `object_id` carrier.
        - `upward_support_top_k` (int, optional): Number of top child hypotheses to
            encode in the upward `object_support` vector. Defaults to
            `upward_top_k`.
        - `upward_support_temperature` (float, optional): Softmax temperature used
            when converting top-k child evidences into the `object_support`
            vector. Defaults to `1.0`.
        - `upward_support_evidence_normalization` (string, optional): Optional
            normalization applied to the top-k child evidence gaps before the
            support softmax. `none` preserves the raw evidence scale; `range`
            divides the relative logits by `max(ptp(top_k_evidences), 1.0)` so
            support does not collapse just because absolute evidence magnitudes
            grow over time. Defaults to `none`.
        - `upward_packet_weight_features` (bool, optional): If True, materialize
            each packet rank slot's support weight into a fixed scalar feature on
            the receiver side.
        - `upward_packet_weight_tolerance` (float, optional): Similarity tolerance
            for packet rank-weight features. Defaults to `0.75`.
        - `upward_packet_weight_feature_weight` (float, optional): Feature weight
            assigned to packet rank-weight features when auto-registering them in
            LM input tolerances. Defaults to `0.1`.
        - `temporal_support_accumulator` (bool, optional): If True, accumulate
            incoming LM `object_support` evidence across episode steps before
            parent matching.
        - `temporal_support_decay` (float, optional): Decay factor in [0, 1] used
            for the temporal accumulator. Higher values preserve past support more
            strongly.
        - `temporal_support_object_id` (bool, optional): If True, apply the same
            temporal smoothing to incoming LM `object_id` carriers.
        - `graph_support_objective` (bool, optional): If True, add a graph-level
            evidence bonus based on similarity between accumulated LM
            `object_support` and each parent graph's stored support prototype.
        - `graph_support_objective_weight` (float, optional): Weight applied to the
            graph-level support bonus.
        - `graph_support_objective_tolerance` (float, optional): Minimum cosine
            similarity required before the graph-level support bonus becomes
            positive.
        - `graph_support_competitive_objective` (bool, optional): If True,
            convert graph-support matching into a relative objective that only
            rewards a graph when its support similarity beats the nearest stored
            competitor.
        - `graph_support_competitive_margin` (float, optional): Required margin
            over the nearest competitor before the competitive support bonus
            becomes positive.
        - `graph_support_competitive_scope` (str, optional): Scope used to pick
            competitors for the competitive support objective. Supported values
            are `all`, `top_k_evidence`, `possible_matches`, and
            `top_k_possible_matches`.
        - `graph_support_competitive_same_peak_only` (bool, optional): If True,
            restrict the competitive rival to graphs whose best-matching support
            row shares the same dominant support slot as the candidate graph,
            falling back to the unrestricted rival if no same-peak competitor is
            available.
        - `graph_support_competitive_packet_context_floor` (float, optional):
            When packet-context scoring is active, require the chosen rival to
            have packet-context similarity at least this fraction of the
            candidate graph's packet-context similarity, falling back to the
            unrestricted rival if no such competitor is available.
        - `graph_support_competitive_packet_context_strict` (bool, optional):
            If True, and no rival clears
            `graph_support_competitive_packet_context_floor`, skip competitive
            subtraction for that update instead of falling back to the
            unrestricted rival.
        - `graph_support_competitive_packet_context_strict_ceiling` (float,
            optional): Maximum candidate packet-context similarity for which
            strict packet-context rival filtering applies. Candidates above this
            ceiling keep the unrestricted fallback even when
            `graph_support_competitive_packet_context_strict` is enabled.
        - `graph_support_competitive_top_k` (int, optional): If greater than
            zero, restrict competitive support scoring to the top-K current
            evidence candidates instead of comparing against every stored graph.
        - `graph_support_num_prototypes` (int, optional): Number of stored
            support prototypes to compare against per graph/input channel. `1`
            preserves the original mean-prototype behavior. Values greater than
            `1` keep the mean prototype and add diverse stored support rows,
            using the best-matching prototype for scoring.
        - `graph_support_packet_context` (bool, optional): If True, score
            graph-support rows in the context of aligned packet rank features
            such as `object_id_rank_k` and `object_rank_weight_k`.

    See the `monty_lab` repo for reference. Specifically,
    `experiments/configs/evidence_sdr_evaluation.py`

    TODO: This mixin adds state to the instance it is being mixed with. As such, it
    is attempting to reuse some common functionality of EvidenceGraphLM while being
    a different thing. The likely refactor is to extract the reusable EvidenceGraphLM
    functionality into a component (since it probably requires its own state), use
    that component as the default in EvidenceGraphLM, and then reuse that component in
    a new EvidenceSDRGraphLM class instead of inheriting from EvidenceGraphLM.
    """

    def __init__(self, *args, **kwargs):
        """The mixin overrides the `__init__` function of the Learning Module.

        The encoding algorithm is initialized here and it stores the actual SDRs.
        Also, a temporary logging function is initialized here.

        """
        self.sdr_args = kwargs.pop("sdr_args")
        self.train_sdr_on_eval = self.sdr_args.pop("train_sdr_on_eval", False)
        self.upward_top_k = max(int(self.sdr_args.pop("upward_top_k", 1)), 1)
        self.upward_support_top_k = max(
            int(self.sdr_args.pop("upward_support_top_k", self.upward_top_k)),
            1,
        )
        self.upward_support_temperature = float(
            self.sdr_args.pop("upward_support_temperature", 1.0)
        )
        self.upward_support_evidence_normalization = str(
            self.sdr_args.pop("upward_support_evidence_normalization", "none")
        ).lower()
        self.upward_packet_top_k = max(
            int(self.sdr_args.pop("upward_packet_top_k", 0)),
            0,
        )
        self.upward_packet_weight_features = bool(
            self.sdr_args.pop("upward_packet_weight_features", False)
        )
        self.upward_packet_weight_tolerance = float(
            self.sdr_args.pop("upward_packet_weight_tolerance", 0.75)
        )
        self.upward_packet_weight_feature_weight = float(
            self.sdr_args.pop("upward_packet_weight_feature_weight", 0.1)
        )
        self.temporal_support_accumulator = bool(
            self.sdr_args.pop("temporal_support_accumulator", False)
        )
        self.temporal_support_decay = float(
            self.sdr_args.pop("temporal_support_decay", 1.0)
        )
        self.temporal_support_object_id = bool(
            self.sdr_args.pop("temporal_support_object_id", True)
        )
        self.graph_support_objective = bool(
            self.sdr_args.pop("graph_support_objective", False)
        )
        self.graph_support_objective_weight = float(
            self.sdr_args.pop("graph_support_objective_weight", 1.0)
        )
        self.graph_support_objective_tolerance = float(
            self.sdr_args.pop("graph_support_objective_tolerance", 0.0)
        )
        self.graph_support_competitive_objective = bool(
            self.sdr_args.pop("graph_support_competitive_objective", False)
        )
        self.graph_support_competitive_margin = float(
            self.sdr_args.pop("graph_support_competitive_margin", 0.0)
        )
        self.graph_support_competitive_penalty = float(
            self.sdr_args.pop("graph_support_competitive_penalty", 1.0)
        )
        self.graph_support_competitive_same_peak_only = bool(
            self.sdr_args.pop("graph_support_competitive_same_peak_only", False)
        )
        self.graph_support_competitive_packet_context_floor = float(
            self.sdr_args.pop("graph_support_competitive_packet_context_floor", 0.0)
        )
        self.graph_support_competitive_packet_context_strict = bool(
            self.sdr_args.pop("graph_support_competitive_packet_context_strict", False)
        )
        self.graph_support_competitive_packet_context_strict_ceiling = float(
            self.sdr_args.pop(
                "graph_support_competitive_packet_context_strict_ceiling",
                1.0,
            )
        )
        self.graph_support_competitive_scope = str(
            self.sdr_args.pop("graph_support_competitive_scope", "all")
        ).lower()
        self.graph_support_competitive_top_k = max(
            int(self.sdr_args.pop("graph_support_competitive_top_k", 0)),
            0,
        )
        self.graph_support_num_prototypes = max(
            int(self.sdr_args.pop("graph_support_num_prototypes", 1)),
            1,
        )
        self.graph_support_packet_context = bool(
            self.sdr_args.pop("graph_support_packet_context", False)
        )
        self.graph_support_object_id_context = bool(
            self.sdr_args.pop("graph_support_object_id_context", False)
        )
        self.graph_support_debug_top_k = max(
            int(self.sdr_args.pop("graph_support_debug_top_k", 10)),
            1,
        )
        if self.upward_support_temperature <= 0:
            logger.warning(
                "Invalid upward_support_temperature %.3f; clamping to 1.0",
                self.upward_support_temperature,
            )
            self.upward_support_temperature = 1.0
        if self.upward_support_evidence_normalization not in {"none", "range"}:
            logger.warning(
                "Invalid upward_support_evidence_normalization '%s'; defaulting to 'none'",
                self.upward_support_evidence_normalization,
            )
            self.upward_support_evidence_normalization = "none"
        if self.graph_support_competitive_scope not in {
            "all",
            "top_k_evidence",
            "possible_matches",
            "top_k_possible_matches",
        }:
            logger.warning(
                "Invalid graph_support_competitive_scope '%s'; defaulting to 'all'",
                self.graph_support_competitive_scope,
            )
            self.graph_support_competitive_scope = "all"
        if self.upward_packet_weight_tolerance < 0.0 or self.upward_packet_weight_tolerance > 1.0:
            logger.warning(
                "Invalid upward_packet_weight_tolerance %.3f; clamping to [0, 1]",
                self.upward_packet_weight_tolerance,
            )
            self.upward_packet_weight_tolerance = np.clip(
                self.upward_packet_weight_tolerance,
                0.0,
                1.0,
            )
        if self.upward_packet_weight_feature_weight < 0.0:
            logger.warning(
                "Invalid upward_packet_weight_feature_weight %.3f; clamping to 0.0",
                self.upward_packet_weight_feature_weight,
            )
            self.upward_packet_weight_feature_weight = 0.0
        kwargs["tolerances"], kwargs["feature_weights"] = (
            self._augment_packet_weight_feature_config(
                kwargs.get("tolerances"),
                kwargs.get("feature_weights"),
                self.upward_packet_top_k,
                self.upward_packet_weight_features,
                self.upward_packet_weight_tolerance,
                self.upward_packet_weight_feature_weight,
            )
        )
        if self.temporal_support_decay < 0.0 or self.temporal_support_decay > 1.0:
            logger.warning(
                "Invalid temporal_support_decay %.3f; clamping to [0, 1]",
                self.temporal_support_decay,
            )
            self.temporal_support_decay = np.clip(
                self.temporal_support_decay,
                0.0,
                1.0,
            )
        if self.graph_support_objective_tolerance < 0.0:
            logger.warning(
                "Invalid graph_support_objective_tolerance %.3f; clamping to 0.0",
                self.graph_support_objective_tolerance,
            )
            self.graph_support_objective_tolerance = 0.0
        if self.graph_support_competitive_margin < 0.0:
            logger.warning(
                "Invalid graph_support_competitive_margin %.3f; clamping to 0.0",
                self.graph_support_competitive_margin,
            )
            self.graph_support_competitive_margin = 0.0
        if self.graph_support_competitive_margin > 1.0:
            logger.warning(
                "Invalid graph_support_competitive_margin %.3f; clamping to 1.0",
                self.graph_support_competitive_margin,
            )
            self.graph_support_competitive_margin = 1.0
        if self.graph_support_competitive_penalty < 0.0:
            logger.warning(
                "Invalid graph_support_competitive_penalty %.3f; clamping to 0.0",
                self.graph_support_competitive_penalty,
            )
            self.graph_support_competitive_penalty = 0.0
        if self.graph_support_competitive_penalty > 1.0:
            logger.warning(
                "Invalid graph_support_competitive_penalty %.3f; clamping to 1.0",
                self.graph_support_competitive_penalty,
            )
            self.graph_support_competitive_penalty = 1.0
        if self.graph_support_competitive_packet_context_floor < 0.0:
            logger.warning(
                "Invalid graph_support_competitive_packet_context_floor %.3f; clamping to 0.0",
                self.graph_support_competitive_packet_context_floor,
            )
            self.graph_support_competitive_packet_context_floor = 0.0
        if self.graph_support_competitive_packet_context_floor > 1.0:
            logger.warning(
                "Invalid graph_support_competitive_packet_context_floor %.3f; clamping to 1.0",
                self.graph_support_competitive_packet_context_floor,
            )
            self.graph_support_competitive_packet_context_floor = 1.0
        if self.graph_support_competitive_packet_context_strict_ceiling < 0.0:
            logger.warning(
                "Invalid graph_support_competitive_packet_context_strict_ceiling %.3f; clamping to 0.0",
                self.graph_support_competitive_packet_context_strict_ceiling,
            )
            self.graph_support_competitive_packet_context_strict_ceiling = 0.0
        if self.graph_support_competitive_packet_context_strict_ceiling > 1.0:
            logger.warning(
                "Invalid graph_support_competitive_packet_context_strict_ceiling %.3f; clamping to 1.0",
                self.graph_support_competitive_packet_context_strict_ceiling,
            )
            self.graph_support_competitive_packet_context_strict_ceiling = 1.0
        # Configure the evidence updater with SDRFeatureEvidenceCalculator and
        # AllFeaturesForMatchingSelector by default.
        updater_args = kwargs.get("hypotheses_updater_args", {})
        if not hasattr(updater_args, "feature_evidence_calculator"):
            updater_args["feature_evidence_calculator"] = SDRFeatureEvidenceCalculator
        if not hasattr(updater_args, "features_for_matching_selector"):
            updater_args["features_for_matching_selector"] = (
                AllFeaturesForMatchingSelector
            )
        kwargs["hypotheses_updater_args"] = updater_args
        super().__init__(*args, **kwargs)

        # keeps track of the Graph objects and their ids
        self.obj2id = {}
        self.id2obj = {}

        # keeps track of overlap running average values
        self.target_overlaps = EvidenceSDRTargetOverlaps()

        # initialize the encoding algorithm
        self.sdr_encoder = EncoderSDR(
            sdr_length=self.sdr_args["sdr_length"],
            sdr_on_bits=self.sdr_args["sdr_on_bits"],
            lr=self.sdr_args["sdr_lr"],
            n_epochs=self.sdr_args["n_sdr_epochs"],
            log_flag=self.sdr_args["sdr_log_flag"],
        )

        # TODO: remove this logger and merge with the Monty Loggers after
        # issue #328 is fixed.
        if self.sdr_args["sdr_log_flag"]:
            self.tmp_logger = LoggerSDR(self.sdr_args["log_path"])

        self._reset_temporal_support_accumulators()
        self._reset_graph_support_debug_stats()

    def reset(self):
        """Reset LM state and any episode-local temporal support accumulators."""
        super().reset()
        self._reset_temporal_support_accumulators()
        self._reset_graph_support_debug_stats()

    def matching_step(self, ctx, observations):
        """Preprocess LM inputs with temporal accumulation before matching."""
        observations = self._preprocess_temporal_lm_observations(observations)
        super().matching_step(ctx, observations)

    def exploratory_step(self, ctx, observations):
        """Preprocess LM inputs with temporal accumulation during training too."""
        observations = self._preprocess_temporal_lm_observations(observations)
        super().exploratory_step(ctx, observations)

    def _update_evidence(self, features, displacements, graph_id):
        """Update evidence and optionally add a graph-level child-support bonus."""
        super()._update_evidence(features, displacements, graph_id)

        if not self.graph_support_objective:
            return
        if graph_id not in self.evidence or len(self.evidence[graph_id]) == 0:
            return

        base_evidence = np.asarray(self.evidence[graph_id], dtype=np.float64)
        graph_support_bonus, channel_details = (
            self._calculate_graph_support_objective_bonus_details(
                features,
                graph_id,
            )
        )
        if graph_support_bonus > 0.0:
            self.evidence[graph_id] = self.evidence[graph_id] + graph_support_bonus

        if self.sdr_args["sdr_log_flag"]:
            updated_evidence = np.asarray(self.evidence[graph_id], dtype=np.float64)
            self._record_graph_support_debug_update(
                graph_id=graph_id,
                graph_support_bonus=graph_support_bonus,
                base_evidence=base_evidence,
                updated_evidence=updated_evidence,
                channel_details=channel_details,
            )

    def _reset_graph_support_debug_stats(self):
        """Clear per-episode debug summaries for LM1 graph-support scoring."""
        self.graph_support_debug_stats = {
            "update_count": 0,
            "positive_bonus_count": 0,
            "graphs": {},
            "top_updates": [],
        }

    def _record_graph_support_debug_update(
        self,
        graph_id,
        graph_support_bonus,
        base_evidence,
        updated_evidence,
        channel_details,
    ):
        """Accumulate compact per-episode summaries of graph-support updates."""
        if not hasattr(self, "graph_support_debug_stats"):
            self._reset_graph_support_debug_stats()

        bonus = float(graph_support_bonus)
        base_mean = float(np.mean(base_evidence))
        base_max = float(np.max(base_evidence))
        updated_mean = float(np.mean(updated_evidence))
        updated_max = float(np.max(updated_evidence))
        mean_similarity = 0.0
        if len(channel_details) > 0:
            mean_similarity = float(
                np.mean([detail["support_similarity"] for detail in channel_details])
            )

        stats = self.graph_support_debug_stats
        stats["update_count"] += 1
        if bonus > 0.0:
            stats["positive_bonus_count"] += 1

        graph_stats = stats["graphs"].setdefault(
            graph_id,
            {
                "update_count": 0,
                "positive_bonus_count": 0,
                "bonus_sum": 0.0,
                "bonus_max": 0.0,
                "base_max_sum": 0.0,
                "updated_max_sum": 0.0,
                "mean_similarity_sum": 0.0,
            },
        )
        graph_stats["update_count"] += 1
        if bonus > 0.0:
            graph_stats["positive_bonus_count"] += 1
        graph_stats["bonus_sum"] += bonus
        graph_stats["bonus_max"] = max(graph_stats["bonus_max"], bonus)
        graph_stats["base_max_sum"] += base_max
        graph_stats["updated_max_sum"] += updated_max
        graph_stats["mean_similarity_sum"] += mean_similarity

        stats["top_updates"].append(
            {
                "graph_id": graph_id,
                "bonus": bonus,
                "base_mean_evidence": base_mean,
                "base_max_evidence": base_max,
                "updated_mean_evidence": updated_mean,
                "updated_max_evidence": updated_max,
                "mean_support_similarity": mean_similarity,
                "channel_details": channel_details,
            }
        )
        stats["top_updates"].sort(
            key=lambda update: (
                update["bonus"],
                update["updated_max_evidence"],
            ),
            reverse=True,
        )
        del stats["top_updates"][self.graph_support_debug_top_k :]

    def _get_graph_support_debug_payload(self):
        """Return a compact, serializable episode summary for graph-support updates."""
        competitive_objective = bool(
            getattr(self, "graph_support_competitive_objective", False)
        )
        competitive_margin = float(
            getattr(self, "graph_support_competitive_margin", 0.0)
        )
        competitive_penalty = float(
            getattr(self, "graph_support_competitive_penalty", 1.0)
        )
        payload = {
            "update_count": int(self.graph_support_debug_stats["update_count"]),
            "positive_bonus_count": int(
                self.graph_support_debug_stats["positive_bonus_count"]
            ),
            "graph_support_objective_weight": float(
                self.graph_support_objective_weight
            ),
            "graph_support_objective_tolerance": float(
                self.graph_support_objective_tolerance
            ),
            "graph_support_competitive_objective": competitive_objective,
            "graph_support_competitive_margin": competitive_margin,
            "graph_support_competitive_penalty": competitive_penalty,
            "graph_support_competitive_same_peak_only": bool(
                getattr(self, "graph_support_competitive_same_peak_only", False)
            ),
            "graph_support_competitive_packet_context_floor": float(
                getattr(self, "graph_support_competitive_packet_context_floor", 0.0)
            ),
            "graph_support_competitive_packet_context_strict": bool(
                getattr(self, "graph_support_competitive_packet_context_strict", False)
            ),
            "graph_support_competitive_packet_context_strict_ceiling": float(
                getattr(
                    self,
                    "graph_support_competitive_packet_context_strict_ceiling",
                    1.0,
                )
            ),
            "graph_support_competitive_scope": str(
                getattr(self, "graph_support_competitive_scope", "all")
            ),
            "graph_support_competitive_top_k": int(
                getattr(self, "graph_support_competitive_top_k", 0)
            ),
            "graph_support_packet_context": bool(
                getattr(self, "graph_support_packet_context", False)
            ),
            "graph_support_object_id_context": bool(
                getattr(self, "graph_support_object_id_context", False)
            ),
            "graphs": {},
            "top_updates": list(self.graph_support_debug_stats["top_updates"]),
        }

        for graph_id, graph_stats in self.graph_support_debug_stats["graphs"].items():
            update_count = max(int(graph_stats["update_count"]), 1)
            payload["graphs"][graph_id] = {
                "update_count": int(graph_stats["update_count"]),
                "positive_bonus_count": int(graph_stats["positive_bonus_count"]),
                "mean_bonus": float(graph_stats["bonus_sum"] / update_count),
                "max_bonus": float(graph_stats["bonus_max"]),
                "mean_base_max_evidence": float(
                    graph_stats["base_max_sum"] / update_count
                ),
                "mean_updated_max_evidence": float(
                    graph_stats["updated_max_sum"] / update_count
                ),
                "mean_support_similarity": float(
                    graph_stats["mean_similarity_sum"] / update_count
                ),
            }

        return payload

    def _calculate_graph_support_objective_bonus(
        self,
        features,
        graph_id,
    ):
        """Return only the scalar graph-support bonus for compatibility."""
        graph_support_bonus, _ = self._calculate_graph_support_objective_bonus_details(
            features,
            graph_id,
        )
        return graph_support_bonus

    def _reset_temporal_support_accumulators(self):
        """Clear per-sender temporal LM support state for the current episode."""
        self.temporal_support_accumulators = {}

    def _calculate_graph_support_objective_bonus_details(self, features, graph_id):
        """Return the graph-support bonus together with per-channel debug details."""
        support_similarities = []
        channel_details = []
        competitive_objective = bool(
            getattr(self, "graph_support_competitive_objective", False)
        )
        competitive_margin = float(
            getattr(self, "graph_support_competitive_margin", 0.0)
        )
        competitive_penalty = float(
            getattr(self, "graph_support_competitive_penalty", 1.0)
        )
        competitive_same_peak_only = bool(
            getattr(self, "graph_support_competitive_same_peak_only", False)
        )
        competitive_packet_context_floor = float(
            getattr(self, "graph_support_competitive_packet_context_floor", 0.0)
        )
        competitive_packet_context_strict = bool(
            getattr(self, "graph_support_competitive_packet_context_strict", False)
        )
        competitive_packet_context_strict_ceiling = float(
            getattr(
                self,
                "graph_support_competitive_packet_context_strict_ceiling",
                1.0,
            )
        )
        for input_channel, channel_features in features.items():
            if not input_channel.startswith("learning_module"):
                continue
            if "object_support" not in channel_features:
                continue

            query_support = np.asarray(
                channel_features["object_support"],
                dtype=np.float64,
            ).reshape(-1)
            query_object_id = None
            if "object_id" in channel_features:
                query_object_id = np.asarray(
                    channel_features["object_id"],
                    dtype=np.float64,
                ).reshape(-1)
            query_packet_context = (
                EvidenceSDRLMMixin._get_query_graph_support_packet_context(
                    self,
                    channel_features,
                )
            )
            if query_support.size == 0:
                continue

            support_detail = self._get_graph_support_similarity_details(
                graph_id,
                input_channel,
                query_support,
                query_packet_context=query_packet_context,
                query_object_id=query_object_id,
            )
            if support_detail is None:
                continue

            raw_support_similarity = float(support_detail["support_similarity"])
            support_similarity = raw_support_similarity
            support_peak_index = support_detail.get("support_peak_index")
            support_packet_context_similarity = float(
                support_detail.get("packet_context_similarity", 1.0)
            )
            best_competitor_graph_id = None
            best_competitor_similarity = 0.0
            best_competitor_prototype_index = None
            best_competitor_support_peak_index = None
            best_competitor_packet_context_similarity = None
            best_same_peak_competitor_graph_id = None
            best_same_peak_competitor_similarity = 0.0
            best_same_peak_competitor_prototype_index = None
            best_same_peak_competitor_support_peak_index = None
            best_context_floor_competitor_graph_id = None
            best_context_floor_competitor_similarity = 0.0
            best_context_floor_competitor_prototype_index = None
            best_context_floor_competitor_support_peak_index = None
            best_context_floor_competitor_packet_context_similarity = None

            if competitive_objective:
                for competitor_graph_id in self._get_graph_support_competitor_ids(
                    graph_id
                ):
                    competitor_detail = self._get_graph_support_similarity_details(
                        competitor_graph_id,
                        input_channel,
                        query_support,
                        query_packet_context=query_packet_context,
                        query_object_id=query_object_id,
                    )
                    if competitor_detail is None:
                        continue
                    competitor_similarity = float(
                        competitor_detail["support_similarity"]
                    )
                    competitor_packet_context_similarity = float(
                        competitor_detail.get("packet_context_similarity", 1.0)
                    )
                    if competitor_similarity > best_competitor_similarity:
                        best_competitor_similarity = competitor_similarity
                        best_competitor_graph_id = competitor_graph_id
                        best_competitor_prototype_index = int(
                            competitor_detail["best_support_prototype_index"]
                        )
                        best_competitor_support_peak_index = competitor_detail.get(
                            "support_peak_index"
                        )
                        best_competitor_packet_context_similarity = (
                            competitor_packet_context_similarity
                        )

                    if (
                        support_peak_index is not None
                        and competitor_detail.get("support_peak_index")
                        == support_peak_index
                        and competitor_similarity > best_same_peak_competitor_similarity
                    ):
                        best_same_peak_competitor_similarity = competitor_similarity
                        best_same_peak_competitor_graph_id = competitor_graph_id
                        best_same_peak_competitor_prototype_index = int(
                            competitor_detail["best_support_prototype_index"]
                        )
                        best_same_peak_competitor_support_peak_index = (
                            competitor_detail.get("support_peak_index")
                        )

                    if (
                        competitive_packet_context_floor > 0.0
                        and competitor_packet_context_similarity
                        >= support_packet_context_similarity
                        * competitive_packet_context_floor
                        and competitor_similarity
                        > best_context_floor_competitor_similarity
                    ):
                        best_context_floor_competitor_similarity = competitor_similarity
                        best_context_floor_competitor_graph_id = competitor_graph_id
                        best_context_floor_competitor_prototype_index = int(
                            competitor_detail["best_support_prototype_index"]
                        )
                        best_context_floor_competitor_support_peak_index = (
                            competitor_detail.get("support_peak_index")
                        )
                        best_context_floor_competitor_packet_context_similarity = (
                            competitor_packet_context_similarity
                        )

                if (
                    competitive_same_peak_only
                    and best_same_peak_competitor_graph_id is not None
                ):
                    best_competitor_similarity = best_same_peak_competitor_similarity
                    best_competitor_graph_id = best_same_peak_competitor_graph_id
                    best_competitor_prototype_index = (
                        best_same_peak_competitor_prototype_index
                    )
                    best_competitor_support_peak_index = (
                        best_same_peak_competitor_support_peak_index
                    )
                elif best_context_floor_competitor_graph_id is not None:
                    best_competitor_similarity = best_context_floor_competitor_similarity
                    best_competitor_graph_id = best_context_floor_competitor_graph_id
                    best_competitor_prototype_index = (
                        best_context_floor_competitor_prototype_index
                    )
                    best_competitor_support_peak_index = (
                        best_context_floor_competitor_support_peak_index
                    )
                    best_competitor_packet_context_similarity = (
                        best_context_floor_competitor_packet_context_similarity
                    )
                elif (
                    competitive_packet_context_floor > 0.0
                    and competitive_packet_context_strict
                    and support_packet_context_similarity
                    <= competitive_packet_context_strict_ceiling
                ):
                    best_competitor_similarity = 0.0
                    best_competitor_graph_id = None
                    best_competitor_prototype_index = None
                    best_competitor_support_peak_index = None
                    best_competitor_packet_context_similarity = None

                support_similarity = self._calculate_competitive_support_similarity(
                    raw_support_similarity,
                    best_competitor_similarity,
                    competitive_margin,
                    competitive_penalty,
                )

            support_similarities.append(support_similarity)
            channel_details.append(
                {
                    "input_channel": input_channel,
                    "support_similarity": float(support_similarity),
                    "support_similarity_raw": float(raw_support_similarity),
                    "graph_support_norm": float(support_detail["graph_support_norm"]),
                    "query_support_norm": float(np.linalg.norm(query_support)),
                    "support_prototype_count": int(
                        support_detail["support_prototype_count"]
                    ),
                    "best_support_prototype_index": int(
                        support_detail["best_support_prototype_index"]
                    ),
                    "support_peak_index": support_peak_index,
                    "object_id_context_similarity": float(
                        support_detail.get("object_id_context_similarity", 1.0)
                    ),
                    "packet_context_similarity": float(
                        support_detail.get("packet_context_similarity", 1.0)
                    ),
                    "best_competitor_graph_id": best_competitor_graph_id,
                    "best_competitor_support_similarity": float(
                        best_competitor_similarity
                    ),
                    "best_competitor_packet_context_similarity": (
                        None
                        if best_competitor_packet_context_similarity is None
                        else float(best_competitor_packet_context_similarity)
                    ),
                    "best_competitor_penalized_support_similarity": float(
                        best_competitor_similarity * competitive_penalty
                    ),
                    "best_competitor_prototype_index": best_competitor_prototype_index,
                    "best_competitor_support_peak_index": best_competitor_support_peak_index,
                }
            )

        if len(support_similarities) == 0:
            return 0.0, channel_details

        return (
            float(np.mean(support_similarities)) * self.graph_support_objective_weight,
            channel_details,
        )

    def _get_graph_support_competitor_ids(self, graph_id):
        """Return competitor graph ids for competitive support scoring."""
        scope = str(getattr(self, "graph_support_competitive_scope", "all")).lower()
        possible_matches = None
        if hasattr(self, "get_possible_matches"):
            possible_matches = [
                competitor_graph_id
                for competitor_graph_id in self.get_possible_matches()
                if competitor_graph_id != graph_id
            ]
            if scope == "possible_matches" and len(possible_matches) > 0:
                return possible_matches

        top_k = int(getattr(self, "graph_support_competitive_top_k", 0))
        if (
            scope in {"top_k_evidence", "top_k_possible_matches"}
            and top_k > 0
            and hasattr(self, "evidence")
        ):
            graph_memory_ids = set(self.graph_memory.get_memory_ids())
            allowed_graph_ids = graph_memory_ids
            if scope == "top_k_possible_matches" and possible_matches is not None:
                allowed_graph_ids = set(possible_matches) & graph_memory_ids
                if len(allowed_graph_ids) == 0:
                    allowed_graph_ids = graph_memory_ids

            ranked_competitors = []
            for competitor_graph_id, competitor_evidence in self.evidence.items():
                if competitor_graph_id == graph_id or competitor_graph_id not in allowed_graph_ids:
                    continue
                competitor_evidence = np.asarray(competitor_evidence, dtype=np.float64)
                if competitor_evidence.size == 0:
                    continue
                ranked_competitors.append(
                    (float(np.max(competitor_evidence)), competitor_graph_id)
                )

            ranked_competitors.sort(reverse=True)
            competitor_ids = [
                competitor_graph_id
                for _, competitor_graph_id in ranked_competitors[:top_k]
            ]
            if len(competitor_ids) > 0:
                return competitor_ids

        return [
            competitor_graph_id
            for competitor_graph_id in self.graph_memory.get_memory_ids()
            if competitor_graph_id != graph_id
        ]

    def _get_graph_support_similarity_details(
        self,
        graph_id,
        input_channel,
        query_support,
        query_packet_context=None,
        query_object_id=None,
    ):
        """Return the best support-similarity match details for one graph/channel."""
        if bool(getattr(self, "graph_support_packet_context", False)):
            row_features = self._get_graph_support_packet_row_features(
                graph_id,
                input_channel,
            )
            if row_features is not None and query_packet_context is not None:
                support_rows, packet_context_rows = row_features
                row_scores = []
                for row_index, (support_row, packet_context_row) in enumerate(
                    zip(support_rows, packet_context_rows)
                ):
                    if support_row.shape != query_support.shape:
                        continue
                    support_similarity = self._calculate_support_similarity(
                        support_row,
                        query_support,
                        self.graph_support_objective_tolerance,
                    )
                    packet_context_similarity = (
                        self._calculate_graph_support_packet_context_similarity(
                            packet_context_row,
                            query_packet_context,
                        )
                    )
                    row_scores.append(
                        (
                            support_similarity * packet_context_similarity,
                            row_index,
                            support_row,
                            packet_context_similarity,
                        )
                    )

                if len(row_scores) > 0:
                    (
                        support_similarity,
                        best_prototype_index,
                        best_graph_support,
                        packet_context_similarity,
                    ) = max(row_scores, key=lambda item: item[0])
                    return {
                        "support_similarity": float(support_similarity),
                        "best_support_prototype_index": int(best_prototype_index),
                        "support_prototype_count": int(len(support_rows)),
                        "graph_support_norm": float(
                            np.linalg.norm(best_graph_support)
                        ),
                        "support_peak_index": int(np.argmax(best_graph_support)),
                        "packet_context_similarity": float(
                            packet_context_similarity
                        ),
                        "object_id_context_similarity": 1.0,
                    }

        if bool(getattr(self, "graph_support_object_id_context", False)):
            row_features = self._get_graph_support_row_features(graph_id, input_channel)
            if row_features is not None:
                support_rows, object_id_rows = row_features
                if object_id_rows is not None and query_object_id is not None:
                    row_scores = []
                    for row_index, (support_row, object_id_row) in enumerate(
                        zip(support_rows, object_id_rows)
                    ):
                        if support_row.shape != query_support.shape:
                            continue
                        support_similarity = self._calculate_support_similarity(
                            support_row,
                            query_support,
                            self.graph_support_objective_tolerance,
                        )
                        object_id_context_similarity = (
                            self._calculate_graph_support_object_id_context_similarity(
                                object_id_row,
                                query_object_id,
                            )
                        )
                        row_scores.append(
                            (
                                support_similarity * object_id_context_similarity,
                                row_index,
                                support_row,
                                object_id_context_similarity,
                            )
                        )

                    if len(row_scores) > 0:
                        (
                            support_similarity,
                            best_prototype_index,
                            best_graph_support,
                            object_id_context_similarity,
                        ) = max(row_scores, key=lambda item: item[0])
                        return {
                            "support_similarity": float(support_similarity),
                            "best_support_prototype_index": int(best_prototype_index),
                            "support_prototype_count": int(len(support_rows)),
                            "graph_support_norm": float(
                                np.linalg.norm(best_graph_support)
                            ),
                            "support_peak_index": int(np.argmax(best_graph_support)),
                            "packet_context_similarity": 1.0,
                            "object_id_context_similarity": float(
                                object_id_context_similarity
                            ),
                        }

        graph_support_prototypes = self._get_graph_support_prototypes(
            graph_id,
            input_channel,
        )
        if len(graph_support_prototypes) == 0:
            return None

        prototype_similarities = []
        for prototype_index, graph_support in enumerate(graph_support_prototypes):
            if graph_support.shape != query_support.shape:
                continue
            support_similarity = self._calculate_support_similarity(
                graph_support,
                query_support,
                self.graph_support_objective_tolerance,
            )
            prototype_similarities.append(
                (support_similarity, prototype_index, graph_support)
            )

        if len(prototype_similarities) == 0:
            return None

        support_similarity, best_prototype_index, best_graph_support = max(
            prototype_similarities,
            key=lambda item: item[0],
        )
        return {
            "support_similarity": float(support_similarity),
            "best_support_prototype_index": int(best_prototype_index),
            "support_prototype_count": int(len(graph_support_prototypes)),
            "graph_support_norm": float(np.linalg.norm(best_graph_support)),
            "support_peak_index": int(np.argmax(best_graph_support)),
            "packet_context_similarity": 1.0,
            "object_id_context_similarity": 1.0,
        }

    def _get_graph_support_row_features(self, graph_id, input_channel):
        """Return normalized support rows and aligned object-id rows for a graph."""
        graph_feature_array = self.graph_memory.get_feature_array(graph_id)
        graph_feature_order = self.graph_memory.get_feature_order(graph_id)
        if input_channel not in graph_feature_array or input_channel not in graph_feature_order:
            return None

        support_features = self._extract_feature_block_from_array(
            graph_id,
            input_channel,
            graph_feature_array[input_channel],
            graph_feature_order[input_channel],
            "object_support",
        )
        if support_features is None or support_features.size == 0:
            return None

        object_id_features = self._extract_feature_block_from_array(
            graph_id,
            input_channel,
            graph_feature_array[input_channel],
            graph_feature_order[input_channel],
            "object_id",
        )

        valid_rows = np.linalg.norm(support_features, axis=1) > 0
        if not np.any(valid_rows):
            return None

        normalized_support_rows = np.asarray(
            support_features[valid_rows],
            dtype=np.float64,
        )
        row_sums = np.sum(normalized_support_rows, axis=1, keepdims=True)
        nonzero_rows = row_sums.reshape(-1) > 0
        if not np.any(nonzero_rows):
            return None

        normalized_support_rows = (
            normalized_support_rows[nonzero_rows] / row_sums[nonzero_rows]
        )
        if object_id_features is None or object_id_features.size == 0:
            return normalized_support_rows, None

        object_id_rows = np.asarray(object_id_features[valid_rows], dtype=np.float64)
        object_id_rows = object_id_rows[nonzero_rows]
        return normalized_support_rows, object_id_rows

    def _get_query_graph_support_packet_context(self, channel_features):
        """Return the concatenated query packet context vector when available."""
        slot_indices = EvidenceSDRLMMixin._get_packet_context_slot_indices(
            channel_features.keys()
        )
        if len(slot_indices) == 0:
            return None

        packet_parts = []
        for slot_index in slot_indices:
            object_id_key = f"object_id_rank_{slot_index}"
            if object_id_key not in channel_features:
                continue
            packet_parts.append(
                np.asarray(channel_features[object_id_key], dtype=np.float64).reshape(-1)
            )
            weight_key = f"object_rank_weight_{slot_index}"
            if weight_key in channel_features:
                packet_parts.append(
                    np.asarray(channel_features[weight_key], dtype=np.float64).reshape(-1)
                )

        if len(packet_parts) == 0:
            return None

        packet_context = np.concatenate(packet_parts)
        if np.isclose(np.linalg.norm(packet_context), 0.0):
            return None
        return packet_context

    def _get_graph_support_packet_row_features(
        self,
        graph_id,
        input_channel,
    ):
        """Return normalized support rows and aligned packet context rows."""
        graph_feature_array = self.graph_memory.get_feature_array(graph_id)
        graph_feature_order = self.graph_memory.get_feature_order(graph_id)
        if input_channel not in graph_feature_array or input_channel not in graph_feature_order:
            return None

        support_features = self._extract_feature_block_from_array(
            graph_id,
            input_channel,
            graph_feature_array[input_channel],
            graph_feature_order[input_channel],
            "object_support",
        )
        if support_features is None or support_features.size == 0:
            return None

        valid_rows = np.linalg.norm(support_features, axis=1) > 0
        if not np.any(valid_rows):
            return None

        normalized_support_rows = np.asarray(
            support_features[valid_rows],
            dtype=np.float64,
        )
        row_sums = np.sum(normalized_support_rows, axis=1, keepdims=True)
        nonzero_rows = row_sums.reshape(-1) > 0
        if not np.any(nonzero_rows):
            return None

        normalized_support_rows = (
            normalized_support_rows[nonzero_rows] / row_sums[nonzero_rows]
        )

        slot_indices = self._get_packet_context_slot_indices(
            graph_feature_order[input_channel]
        )
        if len(slot_indices) == 0:
            return None

        packet_parts = []
        for slot_index in slot_indices:
            object_id_features = self._extract_feature_block_from_array(
                graph_id,
                input_channel,
                graph_feature_array[input_channel],
                graph_feature_order[input_channel],
                f"object_id_rank_{slot_index}",
            )
            if object_id_features is None or object_id_features.size == 0:
                continue
            packet_parts.append(
                np.asarray(object_id_features[valid_rows], dtype=np.float64)[nonzero_rows]
            )

            weight_features = self._extract_feature_block_from_array(
                graph_id,
                input_channel,
                graph_feature_array[input_channel],
                graph_feature_order[input_channel],
                f"object_rank_weight_{slot_index}",
            )
            if weight_features is not None and weight_features.size > 0:
                packet_parts.append(
                    np.asarray(weight_features[valid_rows], dtype=np.float64)[nonzero_rows]
                )

        if len(packet_parts) == 0:
            return None

        packet_context_rows = np.concatenate(packet_parts, axis=1)
        nonzero_packet_rows = np.linalg.norm(packet_context_rows, axis=1) > 0
        if not np.any(nonzero_packet_rows):
            return None

        return (
            normalized_support_rows[nonzero_packet_rows],
            packet_context_rows[nonzero_packet_rows],
        )

    @staticmethod
    def _get_packet_context_slot_indices(feature_names):
        """Return sorted packet slot indices present in a feature list."""
        slot_indices = set()
        for feature_name in feature_names:
            if not str(feature_name).startswith("object_id_rank_"):
                continue
            try:
                slot_indices.add(int(str(feature_name).rsplit("_", 1)[-1]))
            except ValueError:
                continue
        return sorted(slot_indices)

    @staticmethod
    def _calculate_competitive_support_similarity(
        support_similarity,
        competitor_similarity,
        margin,
        penalty=1.0,
    ):
        """Return support similarity only when it beats the nearest competitor."""
        denominator = max(1.0 - margin, np.finfo(np.float64).eps)
        relative_advantage = (
            float(support_similarity)
            - (float(competitor_similarity) * float(penalty))
            - margin
        )
        return float(np.clip(relative_advantage / denominator, 0.0, 1.0))

    @staticmethod
    def _calculate_graph_support_object_id_context_similarity(
        stored_object_id,
        query_object_id,
    ):
        """Return a non-negative cosine similarity between aligned object-id rows."""
        stored_norm = float(np.linalg.norm(stored_object_id))
        query_norm = float(np.linalg.norm(query_object_id))
        if np.isclose(stored_norm, 0.0) or np.isclose(query_norm, 0.0):
            return 0.0

        similarity = float(
            (stored_object_id @ query_object_id) / (stored_norm * query_norm)
        )
        return float(np.clip(similarity, 0.0, 1.0))

    @staticmethod
    def _calculate_graph_support_packet_context_similarity(
        stored_packet_context,
        query_packet_context,
    ):
        """Return a non-negative cosine similarity between aligned packet rows."""
        stored_norm = float(np.linalg.norm(stored_packet_context))
        query_norm = float(np.linalg.norm(query_packet_context))
        if np.isclose(stored_norm, 0.0) or np.isclose(query_norm, 0.0):
            return 0.0

        similarity = float(
            (stored_packet_context @ query_packet_context) / (stored_norm * query_norm)
        )
        return float(np.clip(similarity, 0.0, 1.0))

    def _get_graph_support_prototype(self, graph_id, input_channel):
        """Return the mean stored object-support vector for a graph/input channel."""
        prototypes = self._get_graph_support_prototypes(graph_id, input_channel)
        if len(prototypes) == 0:
            return None
        return prototypes[0]

    def _get_graph_support_prototypes(self, graph_id, input_channel):
        """Return one or more stored object-support prototypes for a graph."""
        graph_feature_array = self.graph_memory.get_feature_array(graph_id)
        graph_feature_order = self.graph_memory.get_feature_order(graph_id)
        if input_channel not in graph_feature_array or input_channel not in graph_feature_order:
            return []

        support_features = self._extract_feature_block_from_array(
            graph_id,
            input_channel,
            graph_feature_array[input_channel],
            graph_feature_order[input_channel],
            "object_support",
        )
        if support_features is None or support_features.size == 0:
            return []

        valid_rows = np.linalg.norm(support_features, axis=1) > 0
        if not np.any(valid_rows):
            return []

        normalized_rows = np.asarray(support_features[valid_rows], dtype=np.float64)
        row_sums = np.sum(normalized_rows, axis=1, keepdims=True)
        nonzero_rows = row_sums.reshape(-1) > 0
        if not np.any(nonzero_rows):
            return []
        normalized_rows = normalized_rows[nonzero_rows] / row_sums[nonzero_rows]

        prototype = np.mean(normalized_rows, axis=0)
        prototype_sum = float(np.sum(prototype))
        if np.isclose(prototype_sum, 0.0):
            return []

        prototypes = [prototype / prototype_sum]
        extra_prototype_count = self.graph_support_num_prototypes - 1
        if extra_prototype_count <= 0 or normalized_rows.shape[0] <= 1:
            return prototypes

        extra_prototypes = self._select_diverse_support_rows(
            normalized_rows,
            extra_prototype_count,
        )
        prototypes.extend(extra_prototypes)
        return prototypes

    @staticmethod
    def _select_diverse_support_rows(support_rows, num_rows):
        """Return a small diverse subset of normalized support rows."""
        if num_rows <= 0:
            return []

        rounded_rows = np.round(np.asarray(support_rows, dtype=np.float64), decimals=6)
        _, unique_indices = np.unique(rounded_rows, axis=0, return_index=True)
        unique_rows = np.asarray(support_rows[np.sort(unique_indices)], dtype=np.float64)
        if unique_rows.size == 0:
            return []
        if unique_rows.shape[0] <= num_rows:
            return [row for row in unique_rows]

        selected_indices = []
        mean_row = np.mean(unique_rows, axis=0)
        first_index = int(np.argmax(np.linalg.norm(unique_rows - mean_row, axis=1)))
        selected_indices.append(first_index)

        while len(selected_indices) < num_rows:
            remaining_indices = [
                index for index in range(unique_rows.shape[0]) if index not in selected_indices
            ]
            if len(remaining_indices) == 0:
                break

            candidate_distances = []
            for index in remaining_indices:
                min_distance = min(
                    np.linalg.norm(unique_rows[index] - unique_rows[selected_index])
                    for selected_index in selected_indices
                )
                candidate_distances.append(min_distance)

            best_remaining_offset = int(np.argmax(candidate_distances))
            if np.isclose(candidate_distances[best_remaining_offset], 0.0):
                break
            selected_indices.append(remaining_indices[best_remaining_offset])

        return [unique_rows[index] for index in selected_indices]

    def _extract_feature_block_from_array(
        self,
        graph_id,
        input_channel,
        feature_array,
        feature_order,
        feature_name,
    ):
        """Extract one named feature block from a packed graph feature array."""
        start_idx = 0
        for feature in feature_order:
            feature_len = self._get_graph_feature_length(graph_id, input_channel, feature)
            end_idx = start_idx + feature_len
            if feature == feature_name:
                return feature_array[:, start_idx:end_idx]
            start_idx = end_idx
        return None

    def _get_graph_feature_length(self, graph_id, input_channel, feature_name):
        """Return the length of a stored non-pose feature for a graph channel."""
        node_features = self.graph_memory.get_features_at_node(
            graph_id,
            input_channel,
            node_id=0,
            feature_keys=[feature_name],
        )
        if feature_name not in node_features:
            return 0
        return len(np.asarray(node_features[feature_name]).reshape(-1))

    @staticmethod
    def _calculate_support_similarity(stored_support, query_support, tolerance):
        """Return normalized cosine similarity for graph-level support matching."""
        stored_norm = float(np.linalg.norm(stored_support))
        query_norm = float(np.linalg.norm(query_support))
        if np.isclose(stored_norm, 0.0) or np.isclose(query_norm, 0.0):
            return 0.0

        similarity = float((stored_support @ query_support) / (stored_norm * query_norm))
        denominator = max(1.0 - tolerance, np.finfo(np.float64).eps)
        return float(np.clip((similarity - tolerance) / denominator, 0.0, 1.0))

    def _preprocess_temporal_lm_observations(self, observations):
        """Return LM observations with accumulated and packet-expanded features."""
        processed_observations = []
        for observation in observations:
            processed_features = observation.non_morphological_features
            if (
                observation.sender_type == "LM"
                and observation.use_state
                and observation.non_morphological_features is not None
            ):
                if self.temporal_support_accumulator:
                    processed_features = self._accumulate_temporal_lm_features(
                        observation.sender_id,
                        processed_features,
                    )
                processed_features = (
                    EvidenceSDRLMMixin._materialize_upward_hypothesis_packet_features(
                        self,
                        processed_features,
                    )
                )
                if processed_features is not observation.non_morphological_features:
                    observation = self._clone_state_with_features(
                        observation,
                        processed_features,
                    )
            processed_observations.append(observation)

        return processed_observations

    def _accumulate_temporal_lm_features(self, sender_id, non_morphological_features):
        """Accumulate LM support features over time for a specific sender."""
        if "object_support" not in non_morphological_features:
            return non_morphological_features

        current_support = np.asarray(
            non_morphological_features["object_support"],
            dtype=np.float64,
        ).reshape(-1)
        if current_support.size == 0:
            return non_morphological_features

        accumulated = self.temporal_support_accumulators.get(sender_id, {})
        previous_support = accumulated.get("object_support")
        if previous_support is None or previous_support.shape != current_support.shape:
            accumulated_support = current_support.copy()
        else:
            accumulated_support = (
                self.temporal_support_decay * previous_support + current_support
            )

        accumulated["object_support"] = accumulated_support
        transformed_features = {
            feature_name: EvidenceSDRLMMixin._copy_feature_value(feature_value)
            for feature_name, feature_value in non_morphological_features.items()
        }
        support_sum = float(np.sum(accumulated_support))
        if np.isclose(support_sum, 0.0):
            transformed_features["object_support"] = np.zeros_like(accumulated_support)
        else:
            transformed_features["object_support"] = accumulated_support / support_sum

        if self.temporal_support_object_id and "object_id" in non_morphological_features:
            current_object_id = np.asarray(
                non_morphological_features["object_id"],
                dtype=np.float64,
            ).reshape(-1)
            previous_object_id = accumulated.get("object_id")
            if previous_object_id is None or previous_object_id.shape != current_object_id.shape:
                accumulated_object_id = current_object_id.copy()
            else:
                accumulated_object_id = np.maximum(
                    self.temporal_support_decay * previous_object_id,
                    current_object_id,
                )

            accumulated["object_id"] = accumulated_object_id
            object_id_scale = float(np.max(accumulated_object_id))
            if np.isclose(object_id_scale, 0.0):
                transformed_features["object_id"] = np.zeros_like(accumulated_object_id)
            else:
                transformed_features["object_id"] = (
                    accumulated_object_id / object_id_scale
                )

        self.temporal_support_accumulators[sender_id] = accumulated
        return transformed_features

    def _materialize_upward_hypothesis_packet_features(self, non_morphological_features):
        """Expand an LM packet into fixed numeric features and drop the raw packet."""
        packet = non_morphological_features.get("upward_hypothesis_packet")
        if not isinstance(packet, dict):
            return non_morphological_features

        transformed_features = {
            feature_name: EvidenceSDRLMMixin._copy_feature_value(feature_value)
            for feature_name, feature_value in non_morphological_features.items()
            if feature_name != "upward_hypothesis_packet"
        }

        slot_count = self.upward_packet_top_k
        if slot_count <= 0:
            return transformed_features

        packet_object_ids = packet.get("object_ids", [])
        packet_support_weights = packet.get("support_weights", [])
        sdr_length = int(self.sdr_args["sdr_length"])
        zero_object_id = np.zeros(sdr_length, dtype=np.float64)

        for slot_index in range(slot_count):
            if slot_index < len(packet_object_ids):
                slot_object_id = np.asarray(
                    packet_object_ids[slot_index],
                    dtype=np.float64,
                ).reshape(-1)
                if slot_object_id.shape != zero_object_id.shape:
                    slot_object_id = zero_object_id.copy()
            else:
                slot_object_id = zero_object_id.copy()

            transformed_features[f"object_id_rank_{slot_index}"] = slot_object_id
            if getattr(self, "upward_packet_weight_features", False):
                if slot_index < len(packet_support_weights):
                    slot_weight = np.array(
                        [float(packet_support_weights[slot_index])],
                        dtype=np.float64,
                    )
                else:
                    slot_weight = np.zeros(1, dtype=np.float64)
                transformed_features[f"object_rank_weight_{slot_index}"] = slot_weight

        return transformed_features

    @staticmethod
    def _augment_packet_weight_feature_config(
        tolerances,
        feature_weights,
        slot_count,
        enabled,
        default_tolerance,
        default_weight,
    ):
        """Register optional packet rank-weight features in LM matcher config."""
        if not enabled or slot_count <= 0 or tolerances is None:
            return tolerances, feature_weights

        updated_tolerances = dict(tolerances)
        updated_feature_weights = dict(feature_weights or {})

        for input_channel, channel_tolerances in updated_tolerances.items():
            if not str(input_channel).startswith("learning_module"):
                continue

            updated_channel_tolerances = dict(channel_tolerances)
            updated_channel_weights = dict(updated_feature_weights.get(input_channel, {}))
            for slot_index in range(slot_count):
                feature_name = f"object_rank_weight_{slot_index}"
                if feature_name not in updated_channel_tolerances:
                    updated_channel_tolerances[feature_name] = default_tolerance
                if feature_name not in updated_channel_weights:
                    updated_channel_weights[feature_name] = default_weight

            updated_tolerances[input_channel] = updated_channel_tolerances
            updated_feature_weights[input_channel] = updated_channel_weights

        return updated_tolerances, updated_feature_weights

    @staticmethod
    def _copy_feature_value(feature_value):
        """Recursively copy feature payloads used in LM-to-LM messaging."""
        if isinstance(feature_value, np.ndarray):
            return np.array(feature_value, copy=True)
        if isinstance(feature_value, dict):
            return {
                key: EvidenceSDRLMMixin._copy_feature_value(value)
                for key, value in feature_value.items()
            }
        if isinstance(feature_value, list):
            return [
                EvidenceSDRLMMixin._copy_feature_value(value)
                for value in feature_value
            ]
        if isinstance(feature_value, tuple):
            return tuple(
                EvidenceSDRLMMixin._copy_feature_value(value)
                for value in feature_value
            )
        return feature_value

    @staticmethod
    def _clone_state_with_features(observation, non_morphological_features):
        """Clone a state so temporal preprocessing doesn't mutate shared LM output."""
        morphological_features = None
        if observation.morphological_features is not None:
            morphological_features = {
                feature_name: EvidenceSDRLMMixin._copy_feature_value(feature_value)
                for feature_name, feature_value in observation.morphological_features.items()
            }

        cloned_state = State(
            location=np.array(observation.location, copy=True),
            morphological_features=morphological_features,
            non_morphological_features=non_morphological_features,
            confidence=observation.confidence,
            use_state=observation.use_state,
            sender_id=observation.sender_id,
            sender_type=observation.sender_type,
        )
        if hasattr(observation, "displacement"):
            cloned_state.displacement = observation.displacement.copy()
        return cloned_state

    def collect_evidences(self):
        """Collect evidence scores from the Learning Module.

        We do this in three steps:
            - Step 1: We use the number of objects in the LM
                to update the sdr_encoder and id <-> obj tracking
                dictionaries, as well as the target overlap tensor
            - Step 2: We collect evidences relative to the current
                most likely hypothesis (mlh). Evidences are stored
                in a 2d tensor.
            - Step 3: We use the stored evidences to update the target overlap
                which stores the running average.
                Refer to `EvidenceSDRTargetOverlaps` for more details.

        **Note:** We sort the ids in step 2 because the overlap values are
        supposed to be symmetric (e.g., "2,5" = "5,2"). This way the target
        overlaps for the ids "x,y" and "y,x" will be averaged together in the
        `EvidenceSDRTargetOverlaps` class.

        """
        # TODO: add more sophisticated logic to sync the SDR representations
        # with available objects in the graph memory. This should facilitate
        # merging or removing objects. The SDR representations should always
        # be in sync with graphs in memory

        # Step 1: add new objects if needed. Useful in learning from scratch experiments
        EvidenceSDRLMMixin._sync_object_registry_with_graph_memory(self)

        # Step 2: collect evidences
        mlh_object = self.get_current_mlh()["graph_id"]
        if mlh_object == "no_observations_yet" or self.sdr_encoder.n_objects == 1:
            return
        valid_evidences = {
            obj: np.asarray(evidence)
            for obj, evidence in self.evidence.items()
            if np.asarray(evidence).size > 0
        }
        if mlh_object not in valid_evidences or len(valid_evidences) <= 1:
            return
        mlh_object_id = self.obj2id[mlh_object]
        mlh_evidence = np.max(valid_evidences[mlh_object])

        relative_evidences = np.full_like(self.target_overlaps.overlaps, np.nan)
        for obj, evidence in valid_evidences.items():
            ids = sorted([mlh_object_id, self.obj2id[obj]])
            ev = np.max(evidence) - mlh_evidence
            relative_evidences[ids[0], ids[1]] = ev

        # Step 3: update running average with new evidence scores
        self.target_overlaps.add_evidence(
            relative_evidences, [0, self.sdr_args["sdr_on_bits"]]
        )

    @staticmethod
    def _has_pairwise_overlap_targets(target_overlaps):
        """Return whether overlap targets contain any off-diagonal training signal."""
        if target_overlaps.size == 0:
            return False

        pairwise_targets = np.isfinite(target_overlaps).copy()
        np.fill_diagonal(pairwise_targets, False)
        return pairwise_targets.any()

    @staticmethod
    def _should_train_sdrs(mode, train_sdr_on_eval):
        """Return whether the LM should update SDRs after an episode."""
        return mode is ExperimentMode.TRAIN or train_sdr_on_eval

    def post_episode(self, *args, **kwargs):
        """Overrides the LM post_episode function.

        This function collects evidences, trains SDRs and logs the output.
        """
        super().post_episode(*args, **kwargs)

        stats = None

        if self._should_train_sdrs(self.mode, self.train_sdr_on_eval):
            # collect the evidences from Learning Module
            self.collect_evidences()

            if self._has_pairwise_overlap_targets(self.target_overlaps.overlaps):
                # Train the SDR Encoder based on overlap targets
                stats = self.sdr_encoder.train_sdrs(self.target_overlaps.overlaps)

        # logging episode information if flag set to True
        if self.sdr_args["sdr_log_flag"]:
            log_payload = {}
            if stats is not None:
                log_payload.update(stats)

            upward_hypothesis_packet = self._get_upward_hypothesis_packet()
            if upward_hypothesis_packet["graph_ids"]:
                log_payload["upward_hypothesis_packet"] = upward_hypothesis_packet

            if self.graph_support_objective and self.graph_support_debug_stats["update_count"] > 0:
                log_payload["graph_support_debug"] = (
                    self._get_graph_support_debug_payload()
                )

            if len(log_payload) > 0:
                log_payload.update(
                    {
                        "obj2id": self.obj2id,
                        "id2obj": self.id2obj,
                        "mode": self.mode.name if hasattr(self.mode, "name") else str(self.mode),
                    }
                )
                self.tmp_logger.log_episode(log_payload)

    def state_dict(self):
        """Return LM state including SDR-specific encoder data."""
        state = super().state_dict()
        state.update(
            obj2id=self.obj2id.copy(),
            id2obj=self.id2obj.copy(),
            target_overlaps=self.target_overlaps.state_dict(),
            sdr_encoder=self.sdr_encoder.state_dict(),
        )
        return state

    def load_state_dict(self, state_dict):
        """Restore LM state including SDR-specific encoder data."""
        super().load_state_dict(state_dict)
        self.obj2id = state_dict.get("obj2id", {}).copy()
        self.id2obj = state_dict.get("id2obj", {}).copy()

        target_overlaps_state = state_dict.get("target_overlaps")
        if target_overlaps_state is not None:
            self.target_overlaps.load_state_dict(target_overlaps_state)

        sdr_encoder_state = state_dict.get("sdr_encoder")
        if sdr_encoder_state is not None:
            self.sdr_encoder.load_state_dict(sdr_encoder_state)

        EvidenceSDRLMMixin._sync_object_registry_with_graph_memory(self)

    def _sync_object_registry_with_graph_memory(self):
        """Ensure SDR object ids exist for every graph currently stored in memory."""
        if not hasattr(self, "get_all_known_object_ids"):
            return
        if not hasattr(self, "obj2id") or not hasattr(self, "id2obj"):
            return
        if not hasattr(self, "sdr_encoder") or not hasattr(self, "target_overlaps"):
            return

        available_objects = self.get_all_known_object_ids()

        for obj in available_objects:
            if obj not in self.obj2id:
                self.obj2id[obj] = len(self.obj2id)
                self.id2obj[len(self.id2obj)] = obj
        self.sdr_encoder.add_objects(len(self.id2obj) - self.sdr_encoder.n_objects)
        self.target_overlaps.add_objects(len(self.id2obj))

    def _object_id_to_features(self, object_id):
        """Retrieves the trained SDR corresponding to the object ID.

        Returns:
            The trained SDR corresponding to the object ID.
        """
        if object_id in self.obj2id:
            return self.sdr_encoder.get_sdr(self.obj2id[object_id])

        return np.zeros(self.sdr_args["sdr_length"])

    def _safe_get_evidence_for_each_graph(self):
        """Return graph evidence, tolerating the startup state with no graph ids."""
        try:
            return self.get_evidence_for_each_graph()
        except IndexError:
            return [], np.array([], dtype=np.float64)

    def _get_upward_object_id_features(self):
        """Return an upward object carrier that preserves top-k ambiguity.

        When configured with `upward_top_k > 1`, the LM sends the union of the top-k
        object SDRs instead of collapsing to a single winning object. This preserves
        lower-level ambiguity as a sparse distributed code while keeping the message
        shape compatible with existing stacked LM plumbing.
        """
        mlh = self.get_current_mlh()
        if self.upward_top_k <= 1 or mlh["graph_id"] == "no_observations_yet":
            return self._object_id_to_features(mlh["graph_id"])

        graph_ids, graph_evidences = EvidenceSDRLMMixin._safe_get_evidence_for_each_graph(
            self
        )
        if len(graph_ids) == 0 or graph_ids == ["patch_off_object"]:
            return self._object_id_to_features(mlh["graph_id"])

        top_k = min(self.upward_top_k, len(graph_ids))
        top_indices = np.argsort(graph_evidences)[-top_k:][::-1]
        top_graph_ids = [graph_ids[index] for index in top_indices]

        object_id_features = np.zeros(self.sdr_args["sdr_length"], dtype=np.float64)
        for graph_id in top_graph_ids:
            object_id_features = np.maximum(
                object_id_features,
                self._object_id_to_features(graph_id),
            )

        return object_id_features

    def _get_upward_object_support_features(self):
        """Return a normalized child-support vector over the known child objects."""
        support_length = max(self.sdr_encoder.n_objects, len(self.id2obj))
        object_support = np.zeros(support_length, dtype=np.float64)
        if support_length == 0:
            return object_support

        graph_ids, graph_evidences = EvidenceSDRLMMixin._safe_get_evidence_for_each_graph(
            self
        )
        if len(graph_ids) == 0 or graph_ids == ["patch_off_object"]:
            return object_support

        top_k = min(self.upward_support_top_k, len(graph_ids))
        top_indices = np.argsort(graph_evidences)[-top_k:][::-1]
        top_graph_ids = [graph_ids[index] for index in top_indices]
        top_graph_evidences = graph_evidences[top_indices].astype(np.float64)

        weights = EvidenceSDRLMMixin._calculate_upward_support_weights(
            self,
            top_graph_evidences,
        )
        if weights.size == 0:
            return object_support

        for graph_id, weight in zip(top_graph_ids, weights):
            object_index = self.obj2id.get(graph_id)
            if object_index is not None:
                object_support[object_index] = weight

        return object_support

    def _get_upward_hypothesis_packet(self):
        """Return the ranked child hypotheses currently compressed into upward features.

        This packet is not yet consumed by LM1 for scoring. It exists so the stacked
        sender path can expose exactly which child hypotheses and weights are being
        collapsed into the current `object_id` and `object_support` carriers.
        """
        mlh = self.get_current_mlh()
        graph_ids, graph_evidences = EvidenceSDRLMMixin._safe_get_evidence_for_each_graph(
            self
        )
        if (
            mlh["graph_id"] == "no_observations_yet"
            or len(graph_ids) == 0
            or graph_ids == ["patch_off_object"]
        ):
            return {
                "mlh_graph_id": mlh["graph_id"],
                "top_k": 0,
                "graph_ids": [],
                "evidences": [],
                "support_weights": [],
                "object_indices": [],
            }

        packet_top_k = min(
            max(self.upward_top_k, self.upward_support_top_k),
            len(graph_ids),
        )
        top_indices = np.argsort(graph_evidences)[-packet_top_k:][::-1]
        top_graph_ids = [graph_ids[index] for index in top_indices]
        top_graph_evidences = graph_evidences[top_indices].astype(np.float64)

        support_weights = EvidenceSDRLMMixin._calculate_upward_support_weights(
            self,
            top_graph_evidences,
        )
        if support_weights.size == 0:
            support_weights = np.zeros_like(top_graph_evidences)

        return {
            "mlh_graph_id": mlh["graph_id"],
            "top_k": int(packet_top_k),
            "graph_ids": top_graph_ids,
            "object_ids": [
                self._object_id_to_features(graph_id).astype(np.float64).tolist()
                for graph_id in top_graph_ids
            ],
            "evidences": top_graph_evidences.tolist(),
            "support_weights": support_weights.tolist(),
            "object_indices": [self.obj2id.get(graph_id) for graph_id in top_graph_ids],
        }

    def _calculate_upward_support_weights(self, top_graph_evidences):
        """Convert top-k child evidences into normalized upward support weights."""
        if top_graph_evidences is None:
            return np.array([], dtype=np.float64)

        logits = np.asarray(top_graph_evidences, dtype=np.float64).reshape(-1)
        if logits.size == 0:
            return logits

        logits -= np.max(logits)
        if getattr(self, "upward_support_evidence_normalization", "none") == "range":
            evidence_range = float(np.ptp(top_graph_evidences))
            if evidence_range > 1.0:
                logits /= evidence_range

        logits /= getattr(self, "upward_support_temperature", 1.0)
        weights = np.exp(logits)
        weights_sum = weights.sum()
        if np.isclose(weights_sum, 0.0):
            return np.array([], dtype=np.float64)

        return weights / weights_sum

    def get_output(self):
        """Return LM output with a distributed upward object carrier when enabled."""
        state = super().get_output()
        if state.non_morphological_features is not None:
            EvidenceSDRLMMixin._sync_object_registry_with_graph_memory(self)
            state.non_morphological_features["object_id"] = (
                self._get_upward_object_id_features()
            )
            object_support = self._get_upward_object_support_features()
            if object_support.size > 0:
                state.non_morphological_features["object_support"] = object_support
            upward_hypothesis_packet = self._get_upward_hypothesis_packet()
            if upward_hypothesis_packet["graph_ids"]:
                state.non_morphological_features["upward_hypothesis_packet"] = (
                    upward_hypothesis_packet
                )
        return state


class EvidenceSDRGraphLM(EvidenceSDRLMMixin, EvidenceGraphLM):
    """Class that incorporates the EvidenceSDR Mixin with the EvidenceGraphLM."""

    pass
