from __future__ import annotations

from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)

__all__ = ["MontyEvidenceSDRFittingExperiment"]


class MontyEvidenceSDRFittingExperiment(MontyObjectRecognitionExperiment):
    """Run evidence-bearing episodes and save the fitted SDR state after evaluation.

    This keeps graph memory fixed while allowing `EvidenceSDRGraphLM` to accumulate
    overlap targets and update its SDR encoder during `post_episode()`.
    """

    def evaluate(self):
        super().evaluate()
        self.save_state_dict(output_dir=self.output_dir)
