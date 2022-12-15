"""Training loop and Decoding of the model."""

import functools
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from absl import app
import flax
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import checkpoints
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax import lax
from jax import random
import numpy as np
import optax

from jax.experimental.pjit import pjit
from jax.experimental.pjit import PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.maps import Mesh

from layers import Transformer
from config import T5Config
from input_pipeline import get_datasets
import temperature_sampler

# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

Array = jnp.ndarray
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]
InitializerAxis = Union[int, Tuple[int, ...]]
NdInitializer = Callable[
    [PRNGKey, Shape, DType, InitializerAxis, InitializerAxis], Array]


# -----------------------------------------------------------------------------
# Flax TrainState with mutable variables field
# -----------------------------------------------------------------------------
# TODO(levskaya): upstream this field to the main TrainState.
# ...is this even needed here?  probably not for kv-cache.
class MutableTrainState(train_state.TrainState):
    mutables: Optional[flax.core.FrozenDict[str, Any]]


# -----------------------------------------------------------------------------
# Data Iterators
# -----------------------------------------------------------------------------

# A generator should return (input, target, segments, positions)
def shift_right(x, axis=1):
  """Shift the input to the right by padding and slicing on axis."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  slices = [slice(None),] * x.ndim
  slices[axis] = slice(0, -1)
  padded = np.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[tuple(slices)]

def shift_inputs(x, segment_ids=None, axis=1):
  """Shift inputs and replace EOS by 0 for packed inputs."""
  shifted = shift_right(x, axis=axis)
  # For packed targets, the first shifted token of a new sequence is made
  # 0, rather than being the EOS token for the last sequence.
  if segment_ids is not None:
    shifted *= (segment_ids == shift_right(segment_ids, axis=axis))
  return shifted

def data_generator(train_ds):
  train_iter = iter(train_ds)
  while True:
    data = jax.tree_util.tree_map(np.asarray, next(train_iter))
    yield (
      shift_inputs(data['inputs'], data['inputs_segmentation']),
      data['inputs'],
      data['inputs_segmentation'],
      data['inputs_position']
    )


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

# learning rate scheduling
def rsqrt_schedule(init_value: float, shift: int = 0):
  def schedule(count):
    return init_value * (count + shift)**-.5 * shift**.5
  return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules([
      optax.linear_schedule(
          init_value=0,
          end_value=learning_rate,
          transition_steps=warmup_steps
          ),
      rsqrt_schedule(
        init_value=learning_rate,
        shift=warmup_steps),
    ],
    boundaries=[warmup_steps])


def init_train_state(model, tx, config, key):
  input_shape = (
      len(jax.devices()) * config.per_device_batch_size,
      config.max_target_length
  )
  model_vars = model.init({'params': key, 'dropout': key},
                          jnp.ones(input_shape),
                          jnp.ones(input_shape))
  state = MutableTrainState.create(
      apply_fn=model.apply,
      params=model_vars['params'],
      tx=tx,
      mutables=None)
  return state


def train_step(model, state, apply_args, dropout_rng):
  inputs, targets, segments, positions = apply_args
  rng1, rng2 = jax.random.split(dropout_rng)

  def loss_fn(params):
    logits = model.apply({'params': params},
                         inputs,
                         targets,
                         segments,
                         positions,
                         rngs={'dropout': rng1})
    # TODO: is optax xent as good as custom T5X one?
    xent = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    # Mask out paddings at the end of each example.
    xent = xent * (segments != 0)
    # TODO: mask out the prompt if training prefix-LM
    return jnp.sum(xent), logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, _), grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)

  metrics = {'loss': loss}

  return new_state, metrics, rng2


def train_loop(
  config,
  steps,
  data_gen,
  state=None,
  ckpt_path='~/flaxformer/lm1b'):

  # Initial PRNG Keys
  init_rng, nextrng = random.split(random.PRNGKey(0), 2)

  # Model and Optimizer definition
  model = Transformer(config)
  tx = optax.adam(
    create_learning_rate_schedule(
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps))

  # Mesh definition
  mesh = Mesh(mesh_utils.create_device_mesh(config.mesh_shape), config.mesh_axes)

  # Abstract initialization
  init_fn = functools.partial(init_train_state, model, tx, config)
  abstract_state = jax.eval_shape(init_fn, init_rng)
  state_logical_annotations = nn.get_partition_spec(abstract_state)

  # Initialization
  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state_mesh_annotations = nn.logical_to_mesh(state_logical_annotations)
      state = pjit(
          init_fn,
          in_axis_resources=None,
          out_axis_resources=state_mesh_annotations
      )(init_rng)

  # Checkpoint Restoration
  # TODO: we shouldn't run full init compilation when we need to load ckpt.
  if config.restore_checkpoints:
    state = checkpoints.restore_checkpoint(ckpt_path, state)

  data_pspec = nn.logical_to_mesh(P('batch'))

  p_train_step = pjit(
    train_step,
    in_axis_resources=(state_mesh_annotations,
                       data_pspec,
                       None),
    out_axis_resources=(state_mesh_annotations, None, None),
    static_argnums=(0,))


  # Main Loop
  # ---------------------------------------------------------------------------

  for step in np.arange(steps):
    # shifted_input, targets, segments, positions = next(data_gen)
    example_batch = next(data_gen)
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      state, metrics, nextrng = p_train_step(model, state, example_batch, nextrng)
    if step % 400 == 0:
      print(step, metrics)
    if step % 2000 == 0:
      if config.save_checkpoints and step != 0:
        checkpoints.save_checkpoint(
            ckpt_path, state, step=step, keep=1000, overwrite=True)

  return state, model


# Decodeing data without KV Cache
def decode_without_kvc(state, model, shifted, data, config):
  shifted_in = shifted + 0.0
  shifted_in[:, 17:] = 0
  for i in range(15):
    output = model.apply({'params': state.params},
                         shifted_in,
                         data,
                         rngs={'dropout': jax.random.PRNGKey(0)})
    updates = jnp.argmax(jax.nn.softmax(output), axis=2)
    shifted_in[:, 17 + i] = updates[:, 16 + i]  # greedy sampling

  print(shifted_in.shape)
  return shifted_in[0]


# Decoding with KV cache
def decode_with_kvc(state, model, shifted, data, config):
  output, updated_vars = model.apply({'params': state.params},
                                     data,
                                     data,
                                     decode=True,
                                     max_decode_length=config.max_target_length,
                                     rngs={'dropout': jax.random.PRNGKey(0)},
                                     mutable=('cache',))

  cache = updated_vars['cache']
  cache = jax.tree_map(jnp.zeros_like, cache)
  jax.tree_map(jnp.shape, cache)

  print(data[0])
  print(shifted[0])
  data_in = shifted + 0.0
  data_in[:, 17:] = 0

  # sampling
  for i in range(31):
    output, updated_vars = model.apply(
        {
            'params': state.params,
            'cache': cache
        },
        data_in[:, i:i + 1],
        data[:, i:i + 1],
        rngs={},
        decode=True,
        max_decode_length=config.max_target_length,
        mutable=('cache',))
    if i == 0:
      print(data_in[:, i:i + 1].shape)
      print(output.shape)
    updates = jnp.argmax(jax.nn.softmax(output), axis=2)
    # print(i, data_in[0, i:i+1], updates[0])
    cache = updated_vars['cache']
    if i >= 16:
      data_in[:, i + 1:i + 2] = updates[:]  # greedy sampling

  return data_in[0]



def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config = T5Config()

  # Load data
  train_ds, eval_ds, predict_ds, sp_tokenizer = get_datasets(
      n_devices=jax.local_device_count(), config=config, vocab_path=None)
  lm1b_gen = data_generator(train_ds)

  state, model = train_loop(config, 1000, lm1b_gen)


if __name__ == '__main__':
  app.run(main)
