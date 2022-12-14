"""Config for training."""

from typing import Any, Sequence
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class T5Config:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  # Activation dtypes.
  dtype: Any = jnp.float32
  emb_dim: int = 128
  num_heads: int = 8
  head_dim: int = 16
  mlp_dim: int = 512
  num_decoder_layers: int = 2

  # Activation functions are retrieved from Flax.
  mlp_activations: Sequence[str] = ('relu',)
  dropout_rate: float = 0
  # If `True`, the embedding weights are used in the decoder output layer.
  logits_via_embedding: bool = True
  # minimal, full, or none
  remat_policy: str = 'none'

  # Parallelism
  scan_layers: bool = False
  param_scan_axis: int = 1

  # Dataset
  vocab_size: int = 10000
  dataset_name: str = 'lm1b'
  eval_dataset_name: str = 'lm1b'
  eval_split: str = 'test'
  per_device_batch_size: int = 32
  eval_per_device_batch_size: int = 0
  max_corpus_chars: int = 10**7  # for tokenization

  # Copied from lm1b example - no idea why their lengths are different
  # Maximum length cutoff for training examples.
  max_target_length: int = 128
  # Maximum length cutoff for eval examples.
  max_eval_target_length: int = 512
  # Maximum length cutoff for predicted tokens.
  max_predict_length: int = 50

  # Training loop
  learning_rate: float = 1e-3
  warmup_steps: int = 1000

  save_checkpoints: bool = True
  restore_checkpoints: bool = True
