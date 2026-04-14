import os

from tbp.monty.frameworks.models.predictive_hypothesis_torch.sparse_recurrent_memory import FixedSparseRecurrentMemory

print('before', flush=True)
mem = FixedSparseRecurrentMemory(embedding_dim=84, seed=42, device='cpu')
os.write(1, b'after\n')
os._exit(0)
