"""Within-episode temporal memory using SDR encoding and Hebbian learning.

Biological basis:
- Raw sensory observations encoded as Sparse Distributed Representations via
  locality-sensitive random projection — analogous to receptive field encoding
  in sensory cortex where similar stimuli activate overlapping neuron
  populations.

- Consecutive SDRs associated via outer-product Hebbian rule — analogous to
  STDP (spike-timing dependent plasticity) where neurons that fire in sequence
  strengthen their synaptic connections.

- Temporal prediction via spreading activation — analogous to dendritic
  prediction in cortical pyramidal cells where distal synapses depolarize
  predicted cells before the next input arrives.

- Surprise as prediction error — analogous to burst firing in cortical columns
  when predictions fail (all cells in a minicolumn activate instead of just
  the predicted subset).

Same computational motif (SDR + Hebbian + spreading activation) as HPC
cross-episode learning, applied here within episodes to raw sensory features.
The brain reuses this algorithm across cortical hierarchy levels.
"""

import numpy as np

from tbp.monty.frameworks.models.states import State


class TemporalMemory:
    """Within-episode temporal sequence learning on raw sensory data.

    Encodes each observation as an SDR via locality-sensitive hashing, learns
    temporal transitions via Hebbian outer-product rule, and predicts upcoming
    observations via spreading activation. Operates entirely on raw features —
    no object labels or categories.

    Parameters
    ----------
    sdr_dim : int
        Dimensionality of sparse distributed representations.
    sdr_sparsity : float
        Fraction of bits active in each SDR.
    learning_rate : float
        Hebbian learning rate for temporal associations.
    projection_seed : int
        Seed for the random projection matrix (ensures reproducibility).
    include_location : bool
        Whether to include spatial location in the feature encoding.
    """

    def __init__(
        self,
        sdr_dim=2048,
        sdr_sparsity=0.02,
        learning_rate=0.1,
        projection_seed=42,
        include_location=True,
        surprise_learning_boost=0.0,
        low_surprise_decay_rate=0.0,
        low_surprise_decay_threshold=0.2,
    ):
        self.sdr_dim = sdr_dim
        self.n_active = max(1, int(sdr_dim * sdr_sparsity))
        self.learning_rate = learning_rate
        self._projection_seed = projection_seed
        self.include_location = include_location

        # T6.1: Surprise-modulated Hebbian learning rate.
        # Modulated LR = learning_rate * (1 + surprise_learning_boost * surprise).
        # When surprise is high (novel transition), learning is amplified.
        # When surprise is low (predicted transition), learning rate stays at base.
        # Default 0.0 = no modulation, backward compatible.
        self.surprise_learning_boost = surprise_learning_boost

        # T6.2: Surprise-gated weight decay for well-predicted transitions.
        # When surprise < threshold, participating weights decay slightly,
        # preventing unbounded growth while preserving novel transitions.
        # Biological analog: synaptic depression for already-learned pathways.
        self.low_surprise_decay_rate = low_surprise_decay_rate
        self.low_surprise_decay_threshold = low_surprise_decay_threshold

        # Random projection matrix — initialized lazily on first observation
        # when we know the feature dimensionality.
        self._projection = None
        self._n_features = None

        # Hebbian temporal association matrix.
        # W[i,j] encodes how strongly activation of bit j at time t
        # predicts activation of bit i at time t+1.
        self._W = np.zeros((sdr_dim, sdr_dim), dtype=np.float64)

        # Episode state
        self._history = []  # list of dicts: step, sdr, features, surprise
        self._prev_sdr = None
        self._step_count = 0

        # Prediction tracking (for accuracy measurement)
        self._prediction_log = []

        # Named behavioral sequences (learned prototypes for recognition)
        self._known_behaviors = {}  # name → list of SDRs

    def _init_projection(self, n_features):
        """Initialize random projection matrix for feature → SDR encoding.

        Each row of the projection matrix acts like a neuron's receptive field
        — a random linear combination of input features. The winner-take-all
        step (in encode) acts like cortical lateral inhibition.
        """
        self._n_features = n_features
        rng = np.random.RandomState(self._projection_seed)
        self._projection = rng.randn(n_features, self.sdr_dim).astype(np.float64)
        self._projection /= np.sqrt(max(n_features, 1))

    def extract_features(self, state):
        """Extract a flat feature vector from a State observation.

        Concatenates all continuous sensory features in a deterministic order.
        This is the raw sensory representation — no labels, no categories.
        """
        features = []

        # Spatial location (3D)
        if self.include_location:
            loc = state.location
            if hasattr(loc, "tolist"):
                features.extend(loc.tolist())
            else:
                features.extend(list(loc))

        # Pose vectors (3x3 rotation matrix, flattened to 9 values)
        if hasattr(state, "morphological_features") and state.morphological_features:
            pv = state.morphological_features.get("pose_vectors")
            if pv is not None:
                if hasattr(pv, "flatten"):
                    features.extend(pv.flatten().tolist())
                elif hasattr(pv, "__iter__"):
                    for row in pv:
                        if hasattr(row, "tolist"):
                            features.extend(row.tolist())
                        elif hasattr(row, "__iter__"):
                            features.extend(list(row))
                        else:
                            features.append(float(row))

        # Non-morphological features (curvatures, color, etc.)
        # Sorted by key name for deterministic ordering.
        if (
            hasattr(state, "non_morphological_features")
            and state.non_morphological_features
        ):
            for key in sorted(state.non_morphological_features.keys()):
                val = state.non_morphological_features[key]
                if isinstance(val, (list, tuple)):
                    features.extend([float(v) for v in val if _is_numeric(v)])
                elif isinstance(val, np.ndarray):
                    features.extend(val.flatten().tolist())
                elif _is_numeric(val):
                    features.append(float(val))
                # Skip non-numeric features (strings like graph_id)

        return np.array(features, dtype=np.float64)

    def encode(self, state_or_features):
        """Encode an observation as a sparse distributed representation.

        Uses locality-sensitive hashing via random projection:
        1. Project feature vector through random matrix
        2. Winner-take-all: keep top n_active bits

        Similar feature vectors → similar SDRs (high bit overlap).
        Dissimilar features → low overlap.

        This is analogous to how sensory cortex creates distributed
        representations where similar stimuli activate overlapping neuron
        populations due to overlapping receptive fields.
        """
        if isinstance(state_or_features, State):
            features = self.extract_features(state_or_features)
        else:
            features = np.asarray(state_or_features, dtype=np.float64)

        if self._projection is None:
            self._init_projection(len(features))

        # Handle dimension mismatch (pad or truncate)
        if len(features) != self._n_features:
            padded = np.zeros(self._n_features, dtype=np.float64)
            n = min(len(features), self._n_features)
            padded[:n] = features[:n]
            features = padded

        # Random projection: features → activations
        activations = features @ self._projection  # shape: (sdr_dim,)

        # Winner-take-all (cortical inhibition): keep only top n_active bits
        top_indices = np.argsort(activations)[-self.n_active :]
        sdr = np.zeros(self.sdr_dim, dtype=np.float64)
        sdr[top_indices] = 1.0

        return sdr

    def step(self, state, learn=True):
        """Process one observation. Learn transition from previous if enabled.

        This is the main per-timestep method. On each call:
        1. Encode the observation as an SDR
        2. Generate prediction from previous SDR (if any)
        3. Compute surprise (prediction error)
        4. Learn the transition via Hebbian update (if learn=True)

        Parameters
        ----------
        state : State
            Current sensory observation.
        learn : bool
            Whether to update temporal associations.

        Returns
        -------
        dict with keys:
            sdr : np.ndarray — the observation's SDR encoding
            prediction : np.ndarray or None — predicted SDR before this step
            surprise : float — prediction error [0.0, 1.0]
        """
        sdr = self.encode(state)
        features = self.extract_features(state)

        # Generate prediction BEFORE learning this transition
        prediction = self.predict_next() if self._prev_sdr is not None else None
        surprise = self.compute_surprise(sdr, prediction)

        # Record in history
        self._history.append({
            "step": self._step_count,
            "sdr": sdr.copy(),
            "features": features.copy(),
            "surprise": surprise,
        })
        self._step_count += 1

        # Hebbian learning: associate current SDR with previous SDR.
        # T6.1: Learning rate modulated by surprise — novel transitions
        # (high surprise) get amplified learning, predicted transitions
        # (low surprise) use base rate. Analogous to STDP modulation by
        # neuromodulators (dopamine/norepinephrine) released on prediction
        # error.
        if learn and self._prev_sdr is not None:
            modulated_lr = self.learning_rate * (
                1.0 + self.surprise_learning_boost * surprise
            )
            outer = np.outer(sdr, self._prev_sdr)
            self._W += modulated_lr * outer

            # T6.2: Surprise-gated decay for well-predicted transitions.
            # When surprise is below threshold, mildly decay the weights
            # that participated in this transition. Prevents unbounded
            # weight growth on repeatedly-observed transitions while
            # preserving novel associations. Biological analog: synaptic
            # depression / homeostatic scaling on stable pathways.
            if (
                self.low_surprise_decay_rate > 0
                and surprise < self.low_surprise_decay_threshold
            ):
                mask = outer > 0
                self._W[mask] *= (1.0 - self.low_surprise_decay_rate)

        # Track prediction for accuracy measurement
        if prediction is not None:
            self._prediction_log.append({
                "predicted": prediction,
                "observed": sdr.copy(),
                "surprise": surprise,
                "step": self._step_count - 1,
            })

        self._prev_sdr = sdr.copy()

        return {
            "sdr": sdr,
            "prediction": prediction,
            "surprise": surprise,
        }

    def predict_next(self, from_sdr=None):
        """Predict next observation SDR via spreading activation.

        Multiplies the association matrix by the source SDR and applies
        winner-take-all to produce a predicted SDR pattern.

        This is analogous to dendritic prediction in cortical pyramidal cells:
        the temporal association matrix represents distal synaptic connections
        that depolarize predicted cells before the next input arrives.

        Parameters
        ----------
        from_sdr : np.ndarray or None
            Source SDR to predict from. Defaults to the last observed SDR.

        Returns
        -------
        np.ndarray or None
            Predicted SDR pattern, or None if no prediction can be made.
        """
        source = from_sdr if from_sdr is not None else self._prev_sdr
        if source is None:
            return None

        # Spreading activation through temporal association matrix
        activation = self._W @ source

        if np.max(activation) <= 0:
            return None

        # Winner-take-all: select the n_active most activated bits
        top_indices = np.argsort(activation)[-self.n_active :]
        predicted = np.zeros(self.sdr_dim, dtype=np.float64)
        predicted[top_indices] = 1.0

        return predicted

    def compute_surprise(self, observed_sdr, predicted_sdr):
        """Compute surprise as prediction error.

        surprise = 1 - (overlap / n_active)

        - 0.0 = perfect prediction (all active bits predicted correctly)
        - 1.0 = maximum surprise (no prediction or zero overlap)

        Analogous to burst firing in cortical columns: when predicted cells
        match the actual input, only those cells fire (sparse). When prediction
        fails, all cells in the minicolumn fire (burst = surprise).
        """
        if predicted_sdr is None:
            return 1.0

        overlap = float(np.dot(observed_sdr, predicted_sdr))
        if self.n_active == 0:
            return 1.0

        return max(0.0, 1.0 - overlap / self.n_active)

    def get_surprise_history(self):
        """Return surprise values for all predictions in this episode."""
        return [entry["surprise"] for entry in self._prediction_log]

    def get_mean_surprise(self, last_n=None):
        """Return mean surprise over recent predictions."""
        surprises = self.get_surprise_history()
        if not surprises:
            return 1.0
        if last_n is not None:
            surprises = surprises[-last_n:]
        return float(np.mean(surprises))

    def reset_episode(self):
        """Reset episode state but keep learned associations (W matrix).

        Called between episodes to clear temporal context while preserving
        everything the memory has learned.
        """
        self._history = []
        self._prev_sdr = None
        self._step_count = 0
        self._prediction_log = []

    def learn_behavior(self, name, states, n_repetitions=1):
        """Learn a named behavioral sequence from a list of States.

        Processes the sequence through the temporal memory, learning
        temporal associations via Hebbian rule. Stores the SDR sequence
        as a prototype for later recognition.

        Parameters
        ----------
        name : str
            Name of the behavior (e.g., "walking", "stapler_press").
        states : list[State]
            Observation sequence representing the behavior.
        n_repetitions : int
            How many times to replay for stronger learning.
        """
        self.reset_episode()

        sdrs = []
        for state in states:
            result = self.step(state, learn=True)
            sdrs.append(result["sdr"])

        # Store prototype SDR sequence
        self._known_behaviors[name] = sdrs

        # Replay for consolidation
        if n_repetitions > 1:
            self.replay_episode(n_replays=n_repetitions - 1)

        self.reset_episode()

    def recognize_behavior(self, observed_states, min_overlap=0.3):
        """Recognize which learned behavior best matches observed sequence.

        Compares the SDR sequence of observed states against stored
        behavior prototypes using average SDR overlap.

        Parameters
        ----------
        observed_states : list[State]
            Partial or full observation sequence.
        min_overlap : float
            Minimum average overlap to count as a match.

        Returns
        -------
        tuple of (name, score) or (None, best_score)
        """
        if not self._known_behaviors:
            return None, 0.0

        observed_sdrs = [self.encode(s) for s in observed_states]

        best_name = None
        best_score = 0.0

        for name, prototype_sdrs in self._known_behaviors.items():
            n = min(len(observed_sdrs), len(prototype_sdrs))
            if n == 0:
                continue

            overlaps = []
            for i in range(n):
                overlap = float(np.dot(observed_sdrs[i], prototype_sdrs[i]))
                overlaps.append(
                    overlap / self.n_active if self.n_active > 0 else 0
                )

            score = float(np.mean(overlaps))
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= min_overlap:
            return best_name, best_score
        return None, best_score

    def replay_episode(self, n_replays=3, learning_rate_scale=0.5):
        """Replay current episode to strengthen temporal associations.

        Re-applies Hebbian updates on the stored observation sequence.
        Biological analog: hippocampal replay during rest/sleep, where
        recently experienced sequences are replayed to consolidate
        neocortical temporal associations.

        Parameters
        ----------
        n_replays : int
            Number of replay passes through the episode.
        learning_rate_scale : float
            Scale factor for learning rate during replay (typically < 1).

        Returns
        -------
        int : number of Hebbian updates applied.
        """
        if len(self._history) < 2:
            return 0

        lr = self.learning_rate * learning_rate_scale
        updates = 0

        for _ in range(n_replays):
            for i in range(1, len(self._history)):
                prev_sdr = self._history[i - 1]["sdr"]
                curr_sdr = self._history[i]["sdr"]
                self._W += lr * np.outer(curr_sdr, prev_sdr)
                updates += 1

        return updates

    def predict_sequence(self, start_state, n_steps):
        """Predict a sequence of future observations from a starting state.

        Chains spreading activation predictions: each predicted SDR becomes
        the source for the next prediction. This is analogous to mental
        simulation / imagination — activating learned temporal sequences
        without actual sensory input.

        Parameters
        ----------
        start_state : State
            Starting observation.
        n_steps : int
            Number of future steps to predict.

        Returns
        -------
        list of np.ndarray : predicted SDRs for each future step.
        """
        current_sdr = self.encode(start_state)
        predictions = []

        for _ in range(n_steps):
            next_sdr = self.predict_next(from_sdr=current_sdr)
            if next_sdr is None:
                break
            predictions.append(next_sdr)
            current_sdr = next_sdr

        return predictions

    def get_temporal_context(self):
        """Return current temporal context for downstream consumers.

        Provides the prediction and surprise signal that can be used by
        other components (e.g., LM evidence modulation, attention allocation).

        Returns
        -------
        dict or None
        """
        if self._prev_sdr is None:
            return None

        prediction = self.predict_next()
        return {
            "predicted_sdr": prediction,
            "last_sdr": self._prev_sdr.copy(),
            "mean_surprise": self.get_mean_surprise(last_n=5),
            "step_count": self._step_count,
            "surprise_history": self.get_surprise_history()[-10:],
        }

    def state_dict(self):
        """Serialize temporal memory state for persistence."""
        return {
            "sdr_dim": self.sdr_dim,
            "n_active": self.n_active,
            "learning_rate": self.learning_rate,
            "projection_seed": self._projection_seed,
            "include_location": self.include_location,
            "surprise_learning_boost": self.surprise_learning_boost,
            "low_surprise_decay_rate": self.low_surprise_decay_rate,
            "low_surprise_decay_threshold": self.low_surprise_decay_threshold,
            "W": self._W.copy(),
            "projection": (
                self._projection.copy() if self._projection is not None else None
            ),
            "n_features": self._n_features,
            "known_behaviors": {
                name: [sdr.copy() for sdr in sdrs]
                for name, sdrs in self._known_behaviors.items()
            },
        }

    def load_state_dict(self, state):
        """Load temporal memory state from a saved dict."""
        self.sdr_dim = state["sdr_dim"]
        self.n_active = state["n_active"]
        self.learning_rate = state["learning_rate"]
        self._projection_seed = state["projection_seed"]
        self.include_location = state["include_location"]
        self.surprise_learning_boost = state.get("surprise_learning_boost", 0.0)
        self.low_surprise_decay_rate = state.get("low_surprise_decay_rate", 0.0)
        self.low_surprise_decay_threshold = state.get(
            "low_surprise_decay_threshold", 0.2
        )
        self._W = state["W"].copy()
        self._projection = (
            state["projection"].copy() if state["projection"] is not None else None
        )
        self._n_features = state["n_features"]
        self._known_behaviors = {
            name: [sdr.copy() for sdr in sdrs]
            for name, sdrs in state.get("known_behaviors", {}).items()
        }
        self.reset_episode()


def _is_numeric(val):
    """Check if a value is numeric (int or float)."""
    return isinstance(val, (int, float, np.integer, np.floating))
