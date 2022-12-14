"""Training loop and Decoding of the model."""

import functools
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from absl import app
from config import T5Config
from flax.linen import partitioning as nn_partitioning
from flax.training import checkpoints
from flax.training import train_state
from input_pipeline import get_datasets
import jax
import jax.numpy as jnp
from layers import Transformer

import numpy as np
import optax


logical_to_mesh_axes = nn_partitioning.logical_to_mesh_axes
logical_to_mesh = nn_partitioning.logical_to_mesh


# Type annotations
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
# Data Iterators
# -----------------------------------------------------------------------------

# A generator should return (input, target, segments, positions)
# TODO: not correct, grab _old_ t5x def. that handles packing correctly
def right_shift(arr):
  result = np.zeros_like(arr)
  result[:, 1:] = arr[:, :-1]
  return result


def make_lm1b_gen(train_ds):
  train_iter = iter(train_ds)
  while True:
    step_data = jax.tree_util.tree_map(np.asarray, next(train_iter))
    tokens = step_data['inputs']
    shifted = right_shift(tokens)
    yield shifted, tokens, step_data['inputs_segmentation'], step_data[
        'inputs_position']


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------


# learning rate scheduling
def rsqrt_schedule(
    init_value: float,
    shift: int = 0,
):

  def schedule(count):
    return init_value * (count + shift)**-.5 * shift**.5

  return schedule


def create_learning_rate_schedule(learning_rate: float, warmup_steps: int):
  """Creates a rsqrt schedule with linear warmup."""
  return optax.join_schedules([
      optax.linear_schedule(
          init_value=0, end_value=learning_rate, transition_steps=warmup_steps),
      rsqrt_schedule(init_value=learning_rate, shift=warmup_steps),
  ],
                              boundaries=[warmup_steps])


# @functools.partial(
#     pjit,
#     in_axis_resources=(param_sharding, (pspec, pspec, pspec, pspec), None, None,
#                        None),
#     out_axis_resources=(None, pspec),
#     static_argnums=(2, 4))
@functools.partial(jax.jit, static_argnums=(2,))
def train_step(state, apply_args, model, dropout_rng):
  inputs, targets, segments, positions = apply_args
  rng1, rng2 = jax.random.split(dropout_rng)

  def loss_fn(params):
    # with mesh, nn_partitioning.axis_rules(logical_axis_rules):
    logits = model.apply({'params': params},
                         inputs,
                         targets,
                         segments,
                         positions,
                         rngs={'dropout': rng1})
    entropy = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    # Mask out paddings at the end of each example.
    # jax.debug.print(f'{entropy.shape}, {segments.shape}')
    entropy = entropy * (segments != 0)
    # TODO: mask out the prompt if training example has one (lm1b doesn't)
    return jnp.sum(entropy), logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, _), grads = grad_fn(state.params)
  new_state = state.apply_gradients(grads=grads)

  metrics = {'loss': loss}

  return new_state, metrics, rng2


def train_loop(config, steps, data_gen, state=None,
               ckpt_path='~/flaxformer/lm1b'):
  nextrng = jax.random.PRNGKey(0)
  model = Transformer(config)
  # with mesh, nn_partitioning.axis_rules(logical_axis_rules):
  #   vars = p_init({'params': jax.random.PRNGKey(0), 'dropout': jax.random.PRNGKey(0)},
  #                 jnp.ones((16, 128)), jnp.ones((16, 128)))
  vars = model.init({
      'params': jax.random.PRNGKey(0),
      'dropout': jax.random.PRNGKey(0)
  }, jnp.ones((16, 128)), jnp.ones((16, 128)))
  tx = optax.adam(
      create_learning_rate_schedule(
          learning_rate=config.learning_rate, warmup_steps=config.warmup_steps))

  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=vars['params'],
      tx=tx,
  )
  if config.restore_checkpoints:
    state = checkpoints.restore_checkpoint(ckpt_path, state)

  for step in np.arange(steps):
    # shifted_input, targets, segments, positions = next(data_gen)
    apply_args = next(data_gen)
    state, metrics, nextrng = train_step(state, apply_args, model, nextrng)
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

  # Load lm1b data
  train_ds, eval_ds, predict_ds, sp_tokenizer = get_datasets(
      n_devices=jax.local_device_count(), config=config, vocab_path=None)
  lm1b_gen = make_lm1b_gen(train_ds)

  state, model = train_loop(config, 1000, lm1b_gen)


if __name__ == '__main__':
  app.run(main)
