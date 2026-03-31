#!/usr/bin/env python
"""Check Hopfield pattern counts and similarity distribution."""
import sys
import torch
from tbp.monty.simulators.panda3d.cortical_column_torch_evaluation import (
    CorticalColumnTorchEvalHarness,
)

no_gate = "--no-gate" in sys.argv
novelty = None if no_gate else 0.7
label = "WITHOUT" if no_gate else "WITH (threshold=%.1f)" % novelty

col_kw = {
    "n_minicolumns": 2048,
    "n_cells_per_minicolumn": 8,
    "sparsity": 0.03,
    "beta": 12.0,
    "max_settle_iters": 10,
    "evidence_decay": 0.01,
    "hopfield_kwargs": {"novelty_threshold": novelty},
}

harness = CorticalColumnTorchEvalHarness(
    object_names=[
        "011_banana", "025_mug", "003_cracker_box",
        "013_apple", "035_power_drill",
    ],
    eval_rotations=[(0, 0, 0)],
    train_steps=120,
    eval_steps=60,
    train_episodes=2,
    resolution=(64, 64),
    fov=90.0,
    orbit_radius=0.5,
    column_kwargs=col_kw,
    sm_features=["on_object", "hsv", "principal_curvatures_log"],
    evidence_threshold=2.0,
    seed=42,
)
harness._setup()
for obj in harness._object_names:
    for ep in range(harness._train_episodes):
        harness._train_object(obj)

col = harness._column
n_attractors = col._hopfield.n_stored
n_episodes = col._episodic_memory.n_stored
print("=" * 50)
print("CORTICAL ATTRACTORS + EPISODIC MEMORY")
print("=" * 50)
print("Cortical attractors (L2/3):", n_attractors)
print("Episodic episodes (HPC):", n_episodes)

am = col._associative_memory
for obj in am.known_objects:
    print("  %s: %d observations" % (obj, am._proto_counts.get(obj, 0)))
print("Total observations:", sum(am._proto_counts.values()))

if n_attractors > 1 and n_attractors <= 1000:
    pats = col._hopfield._patterns[:n_attractors]
    norms = pats.norm(dim=1, keepdim=True) + 1e-8
    pats_normed = pats / norms
    sims = torch.mm(pats_normed, pats_normed.t())
    sims.fill_diagonal_(0)
    mask = sims.abs() > 0
    print()
    print("Inter-attractor cosine similarities:")
    print("  Max:  %.4f" % sims.max().item())
    print("  Mean: %.4f" % (sims.sum() / (n_attractors * (n_attractors - 1))).item())
    if mask.any():
        print("  Min:  %.4f" % sims[mask].min().item())

harness.close()
