import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from flax import linen as nn, struct
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state, checkpoints

import jax
from jax import lax
from jax import random
import jax.numpy as jnp

from jax.experimental import mesh_utils
from jax.experimental import maps
from jax.experimental.pjit import pjit
from jax.experimental import PartitionSpec
from jax.sharding import MeshPspecSharding

import numpy as np
import optax

param_with_axes = nn_partitioning.param_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint
scan_with_axes = nn_partitioning.scan_with_axes
remat = nn_partitioning.remat
ScanIn = nn_partitioning.ScanIn


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

default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)


#------------------------------------------------------------------------------
# Dot product attention layer.
#------------------------------------------------------------------------------

def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: DType = jnp.float32,
                          float32_logits: bool = False):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  # `attn_weights`: [batch, num_heads, q_length, kv_length]
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)

  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    keep_prob = 1.0 - dropout_rate
    # T5 broadcasts along the "length" dim, but unclear which one that
    # corresponds to in positional dimensions here, assuming query dim.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # Take the linear combination of `value`.
  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


#------------------------------------------------------------------------------
# DenseGeneral for attention layers.
#------------------------------------------------------------------------------

def nd_dense_init(scale, mode, distribution):
  """Initializer with in_axis, out_axis set at call time."""
  def init_fn(key, shape, dtype, in_axis, out_axis):
    fn = jax.nn.initializers.variance_scaling(
        scale, mode, distribution, in_axis, out_axis)
    return fn(key, shape, dtype)
  return init_fn

def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


class DenseGeneral(nn.Module):
  """A linear transformation (without bias) with flexible axes.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
  """
  features: Union[Iterable[int], int]
  axis: Union[Iterable[int], int] = -1
  dtype: DType = jnp.float32
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
  kernel_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    kernel_in_axis = np.arange(len(axis))
    kernel_out_axis = np.arange(len(axis), len(axis) + len(features))
    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        kernel_shape,
        jnp.float32,
        kernel_in_axis,
        kernel_out_axis,
        axes=self.kernel_axes)
    kernel = jnp.asarray(kernel, self.dtype)

    contract_ind = tuple(range(0, len(axis)))
    return lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))


def _convert_to_activation_function(
    fn_or_string: Union[str, Callable]) -> Callable:
  """Convert a string to an activation function."""
  if fn_or_string == 'linear':
    return lambda x: x
  elif isinstance(fn_or_string, str):
    return getattr(nn, fn_or_string)
  elif callable(fn_or_string):
    return fn_or_string
  else:
    raise ValueError("don't know how to convert %s to an activation function" %
                     (fn_or_string,))


class MultiHeadDotProductAttention(nn.Module):
  """Multi-head dot-product attention.

    Attributes:
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      head_dim: dimension of each head.
      dtype: the dtype of the computation.
      dropout_rate: dropout rate
      kernel_init: initializer for the kernel of the Dense layers.
      float32_logits: bool, if True then compute logits in float32 to avoid
        numerical issues with bfloat16.
  """

  num_heads: int
  head_dim: int
  dtype: DType = jnp.float32
  dropout_rate: float = 0.
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'normal')
  float32_logits: bool = False  # computes logits in float32 for stability.

  @nn.compact
  def __call__(self,
               inputs_q: Array,
               inputs_kv: Array,
               mask: Optional[Array] = None,
               bias: Optional[Array] = None,
               *,
               decode: bool = False,
               deterministic: bool = False) -> Array:
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are two modes: decoding and non-decoding (e.g., training). The mode is
    determined by `decode` argument. For decoding, this method is called twice,
    first to initialize the cache and then for an actual decoding process. The
    two calls are differentiated by the presence of 'cached_key' in the variable
    dict. In the cache initialization stage, the cache variables are initialized
    as zeros and will be filled in the subsequent decoding process.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      mask: attention mask of shape `[batch, num_heads, q_length, kv_length]`.
      bias: attention bias of shape `[batch, num_heads, q_length, kv_length]`.
      decode: Whether to prepare and use an autoregressive cache.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    projection = functools.partial(
        DenseGeneral,
        axis=-1,
        features=(self.num_heads, self.head_dim),
        kernel_axes=('embed', 'heads', 'kv'),
        dtype=self.dtype)

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query_init = lambda *args: self.kernel_init(*args) / depth_scaling

    # Project inputs_q to multi-headed q/k/v
    # dimensions are then [batch, length, num_heads, head_dim]
    query = projection(kernel_init=query_init, name='query')(inputs_q)
    key = projection(kernel_init=self.kernel_init, name='key')(inputs_kv)
    value = projection(kernel_init=self.kernel_init, name='value')(inputs_kv)

    query = with_sharding_constraint(query, ('batch', 'length', 'heads', 'kv'))
    key = with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
    value = with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

    if decode:
      # Detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      # The key and value have dimension [batch, length, num_heads, head_dim],
      # but we cache them as [batch, num_heads, head_dim, length] as a TPU
      # fusion optimization. This also enables the "scatter via one-hot
      # broadcast" trick, which means we do a one-hot broadcast instead of a
      # scatter/gather operations, resulting in a 3-4x speedup in practice.
      swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])
      cached_key = self.variable('cache', 'cached_key', jnp.zeros,
                                 swap_dims(key.shape), key.dtype)
      cached_value = self.variable('cache', 'cached_value', jnp.zeros,
                                   swap_dims(value.shape), value.dtype)
      cache_index = self.variable('cache', 'cache_index',
                                  lambda: jnp.array(0, dtype=jnp.int32))
      if is_initialized:
        batch, num_heads, head_dim, length = (cached_key.value.shape)
        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        # Sanity shape check of cached key against input query.
        expected_shape = (batch, 1, num_heads, head_dim)
        if expected_shape != query.shape:
          raise ValueError('Autoregressive cache shape error, '
                           'expected query shape %s instead got %s.' %
                           (expected_shape, query.shape))

        # Create a OHE of the current index. NOTE: the index is increased below.
        cur_index = cache_index.value
        one_hot_indices = jax.nn.one_hot(cur_index, length, dtype=key.dtype)
        # In order to update the key, value caches with the current key and
        # value, we move the length axis to the back, similar to what we did for
        # the cached ones above.
        # Note these are currently the key and value of a single position, since
        # we feed one position at a time.
        one_token_key = jnp.moveaxis(key, -3, -1)
        one_token_value = jnp.moveaxis(value, -3, -1)
        # Update key, value caches with our new 1d spatial slices.
        # We implement an efficient scatter into the cache via one-hot
        # broadcast and addition.
        key = cached_key.value + one_token_key * one_hot_indices
        value = cached_value.value + one_token_value * one_hot_indices
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # Move the keys and values back to their original shapes.
        key = jnp.moveaxis(key, -1, -3)
        value = jnp.moveaxis(value, -1, -3)

        # Causal mask for cached decoder self-attention: our single query
        # position should only attend to those key positions that have already
        # been generated and cached, not the remaining zero elements.
        mask = combine_masks(
            mask,
            jnp.broadcast_to(
                jnp.arange(length) <= cur_index,
                # (1, 1, length) represent (head dim, query length, key length)
                # query length is 1 because during decoding we deal with one
                # index.
                # The same mask is applied to all batch elements and heads.
                (batch, 1, 1, length)))

        # Grab the correct relative attention bias during decoding. This is
        # only required during single step decoding.
        if bias is not None:
          # The bias is a full attention matrix, but during decoding we only
          # have to take a slice of it.
          # This is equivalent to bias[..., cur_index:cur_index+1, :].
          bias = dynamic_vector_slice_in_dim(
              jnp.squeeze(bias, axis=0), jnp.reshape(cur_index, (-1)), 1, -2)

    # Convert the boolean attention mask to an attention bias.
    if mask is not None:
      # attention mask in the form of attention bias
      attention_bias = lax.select(
          mask > 0,
          jnp.full(mask.shape, 0.).astype(self.dtype),
          jnp.full(mask.shape, -1e10).astype(self.dtype))
    else:
      attention_bias = None

    # Add provided bias term (e.g. relative position embedding).
    if bias is not None:
      attention_bias = combine_biases(attention_bias, bias)

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = dot_product_attention(
        query,
        key,
        value,
        bias=attention_bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
        dtype=self.dtype,
        float32_logits=self.float32_logits)

    # Back to the original inputs dimensions.
    out = DenseGeneral(
        features=inputs_q.shape[-1],  # output dim is set to the input dim.
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=('heads', 'kv', 'embed'),
        dtype=self.dtype,
        name='out')(
            x)
    return out





class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block.

  Attributes:
    intermediate_dim: Shared dimension of hidden layers.
    activations: Type of activations for each layer.  Each element is either
      'linear', a string function name in flax.linen, or a function.
    kernel_init: Kernel function, passed to the dense layers.
    deterministic: Whether the dropout layers should be deterministic.
    intermediate_dropout_rate: Dropout rate used after the intermediate layers.
    dtype: Type for the dense layer.
  """
  intermediate_dim: int = 2048
  activations: Sequence[Union[str, Callable]] = ('relu',)
  kernel_init: NdInitializer = nd_dense_init(1.0, 'fan_in', 'truncated_normal')
  intermediate_dropout_rate: float = 0.1
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs, decode: bool = False, deterministic: bool = False):
    """Applies Transformer MlpBlock module."""
    # Iterate over specified MLP input activation functions.
    # e.g. ('relu',) or ('gelu', 'linear') for gated-gelu.
    activations = []
    for idx, act_fn in enumerate(self.activations):
      dense_name = 'wi' if len(self.activations) == 1 else f'wi_{idx}'
      x = DenseGeneral(
          self.intermediate_dim,
          dtype=self.dtype,
          kernel_init=self.kernel_init,
          kernel_axes=('embed', 'mlp'),
          name=dense_name)(
              inputs)
      x = _convert_to_activation_function(act_fn)(x)
      activations.append(x)

    # Take elementwise product of above intermediate activations.
    x = functools.reduce(operator.mul, activations)
    # Apply dropout and final dense output projection.
    x = nn.Dropout(
        rate=self.intermediate_dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)  # Broadcast along length.
    x = with_sharding_constraint(x, ('batch', 'length', 'mlp'))
    output = DenseGeneral(
        inputs.shape[-1],
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        kernel_axes=('mlp', 'embed'),
        name='wo')(
            x)
    return output



#------------------------------------------------------------------------------
# T5 Layernorm - no subtraction of mean or bias.
#------------------------------------------------------------------------------

class LayerNorm(nn.Module):
  """T5 Layer normalization operating on the last axis of the input data."""
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  scale_init: Initializer = nn.initializers.ones

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    y = jnp.asarray(x * lax.rsqrt(mean2 + self.epsilon), self.dtype)
    scale = param_with_axes(
        'scale', self.scale_init, (features,), jnp.float32, axes=('embed',))

    scale = jnp.asarray(scale, self.dtype)
    return y * scale


#------------------------------------------------------------------------------
# Utils
#------------------------------------------------------------------------------

class Embed(nn.Module):
  """A parameterized function from integers [0, n) to d-dimensional vectors.

  Attributes:
    num_embeddings: number of embeddings.
    features: number of feature dimensions for each embedding.
    dtype: the dtype of the embedding vectors (default: float32).
    embedding_init: embedding initializer.
    one_hot: performs the gather with a one-hot contraction rather than a true
      gather. This is currently needed for SPMD partitioning.
  """
  num_embeddings: int
  features: int
  cast_input_dtype: Optional[DType] = None
  dtype: DType = jnp.float32
  attend_dtype: Optional[DType] = None
  embedding_init: Initializer = default_embed_init
  one_hot: bool = False
  embedding: Array = dataclasses.field(init=False)

  def setup(self):
    self.embedding = param_with_axes(
        'embedding',
        self.embedding_init, (self.num_embeddings, self.features),
        jnp.float32,
        axes=('vocab', 'embed'))

  def __call__(self, inputs: Array) -> Array:
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    if self.cast_input_dtype:
      inputs = inputs.astype(self.cast_input_dtype)
    if not jnp.issubdtype(inputs.dtype, jnp.integer):
      raise ValueError('Input type must be an integer or unsigned integer.')
    if self.one_hot:
      iota = lax.iota(jnp.int32, self.num_embeddings)
      one_hot = jnp.array(inputs[..., jnp.newaxis] == iota, dtype=self.dtype)
      output = jnp.dot(one_hot, jnp.asarray(self.embedding, self.dtype))
    else:
      output = jnp.asarray(self.embedding, self.dtype)[inputs]
      output = with_sharding_constraint(output, ('batch', 'length', 'embed'))
    return output

  def attend(self, query: Array) -> Array:
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    dtype = self.attend_dtype if self.attend_dtype is not None else self.dtype
    return jnp.dot(query, jnp.asarray(self.embedding, dtype).T)


class RelativePositionBiases(nn.Module):
  """Adds T5-style relative positional embeddings to the attention logits.

  Attributes:
    num_buckets: Number of buckets to bucket distances between key and query
      positions into.
    max_distance: Maximum distance before everything is lumped into the last
      distance bucket.
    num_heads: Number of heads in the attention layer. Each head will get a
      different relative position weighting.
    dtype: Type of arrays through this module.
    embedding_init: initializer for relative embedding table.
  """
  num_buckets: int
  max_distance: int
  num_heads: int
  dtype: Any
  embedding_init: Callable[..., Array] = nn.linear.default_embed_init

  @staticmethod
  def _relative_position_bucket(relative_position,
                                bidirectional=True,
                                num_buckets=32,
                                max_distance=128):
    """Translate relative position to a bucket number for relative attention.

    The relative position is defined as memory_position - query_position, i.e.
    the distance in tokens from the attending position to the attended-to
    position.  If bidirectional=False, then positive relative positions are
    invalid.
    We use smaller buckets for small absolute relative_position and larger
    buckets for larger absolute relative_positions.  All relative
    positions >=max_distance  map to the same bucket.  All relative
    positions <=-max_distance map to the same bucket.  This should allow for
    more graceful generalization to longer sequences than the model has been
    trained on.

    Args:
      relative_position: an int32 array
      bidirectional: a boolean - whether the attention is bidirectional
      num_buckets: an integer
      max_distance: an integer

    Returns:
      a Tensor with the same shape as relative_position, containing int32
        values in the range [0, num_buckets)
    """
    ret = 0
    n = -relative_position
    if bidirectional:
      num_buckets //= 2
      ret += (n < 0).astype(np.int32) * num_buckets
      n = np.abs(n)
    else:
      n = np.maximum(n, 0)
    # now n is in the range [0, inf)
    max_exact = num_buckets // 2
    is_small = (n < max_exact)
    val_if_large = max_exact + (
        np.log(n.astype(np.float32) / max_exact + np.finfo(np.float32).eps) /
        np.log(max_distance / max_exact) *
        (num_buckets - max_exact)).astype(np.int32)
    val_if_large = np.minimum(val_if_large, num_buckets - 1)
    ret += np.where(is_small, n, val_if_large)
    return ret

  @nn.compact
  def __call__(self, qlen, klen, bidirectional=True):
    """Produce relative position embedding attention biases.

    Args:
      qlen: attention query length.
      klen: attention key length.
      bidirectional: whether to allow positive memory-query relative position
        embeddings.

    Returns:
      output: `(1, len, q_len, k_len)` attention bias
    """
    # TODO(levskaya): should we be computing this w. numpy as a program
    # constant?
    context_position = np.arange(qlen, dtype=jnp.int32)[:, None]
    memory_position = np.arange(klen, dtype=jnp.int32)[None, :]
    relative_position = memory_position - context_position  # shape (qlen, klen)
    rp_bucket = self._relative_position_bucket(
        relative_position,
        bidirectional=bidirectional,
        num_buckets=self.num_buckets,
        max_distance=self.max_distance)
    relative_attention_bias = param_with_axes(
        'rel_embedding',
        self.embedding_init, (self.num_heads, self.num_buckets),
        jnp.float32,
        axes=('heads', 'relpos_buckets'))

    relative_attention_bias = jnp.asarray(relative_attention_bias, self.dtype)
    # Instead of using a slow gather, we create a leading-dimension one-hot
    # array from rp_bucket and use it to perform the gather-equivalent via a
    # contraction, i.e.:
    # (num_head, num_buckets) x (num_buckets one-hot, qlen, klen).
    # This is equivalent to relative_attention_bias[:, rp_bucket]
    bcast_iota = lax.broadcasted_iota(jnp.int32, (self.num_buckets, 1, 1), 0)
    rp_bucket_one_hot = jnp.array(
        rp_bucket[jnp.newaxis, ...] == bcast_iota, dtype=self.dtype)
    # --> shape (qlen, klen, num_heads)
    values = lax.dot_general(
        relative_attention_bias,
        rp_bucket_one_hot,
        (
            ((1,), (0,)),  # rhs, lhs contracting dims
            ((), ())))  # no batched dims
    # Add a singleton batch dimension.
    # --> shape (1, num_heads, qlen, klen)
    return values[jnp.newaxis, ...]


#------------------------------------------------------------------------------
# Mask-making utility functions.
#------------------------------------------------------------------------------
def make_attention_mask(query_input: Array,
                        key_input: Array,
                        pairwise_fn: Callable = jnp.multiply,
                        extra_batch_dims: int = 0,
                        dtype: DType = jnp.float32) -> Array:
  """Mask-making helper for attention weights.

  In case of 1d inputs (i.e., `[batch, len_q]`, `[batch, len_kv]`, the
  attention weights will be `[batch, heads, len_q, len_kv]` and this
  function will produce `[batch, 1, len_q, len_kv]`.

  Args:
    query_input: a batched, flat input of query_length size
    key_input: a batched, flat input of key_length size
    pairwise_fn: broadcasting elementwise comparison function
    extra_batch_dims: number of extra batch dims to add singleton axes for, none
      by default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len_q, len_kv]` shaped mask for 1d attention.
  """
  # [batch, len_q, len_kv]
  mask = pairwise_fn(
      # [batch, len_q] -> [batch, len_q, 1]
      jnp.expand_dims(query_input, axis=-1),
      # [batch, len_q] -> [batch, 1, len_kv]
      jnp.expand_dims(key_input, axis=-2))

  # [batch, 1, len_q, len_kv]. This creates the head dim.
  mask = jnp.expand_dims(mask, axis=-3)
  mask = jnp.expand_dims(mask, axis=tuple(range(extra_batch_dims)))
  return mask.astype(dtype)


def make_causal_mask(x: Array,
                     extra_batch_dims: int = 0,
                     dtype: DType = jnp.float32) -> Array:
  """Make a causal mask for self-attention.

  In case of 1d inputs (i.e., `[batch, len]`, the self-attention weights
  will be `[batch, heads, len, len]` and this function will produce a
  causal mask of shape `[batch, 1, len, len]`.

  Note that a causal mask does not depend on the values of x; it only depends on
  the shape. If x has padding elements, they will not be treated in a special
  manner.

  Args:
    x: input array of shape `[batch, len]`
    extra_batch_dims: number of batch dims to add singleton axes for, none by
      default
    dtype: mask return dtype

  Returns:
    A `[batch, 1, len, len]` shaped causal mask for 1d attention.
  """
  idxs = jnp.broadcast_to(jnp.arange(x.shape[-1], dtype=jnp.int32), x.shape)
  return make_attention_mask(
      idxs,
      idxs,
      jnp.greater_equal,
      extra_batch_dims=extra_batch_dims,
      dtype=dtype)


def combine_masks(*masks: Optional[Array], dtype: DType = jnp.float32):
  """Combine attention masks.

  Args:
    *masks: set of attention mask arguments to combine, some can be None.
    dtype: final mask dtype

  Returns:
    Combined mask, reduced by logical and, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = jnp.logical_and(mask, other_mask)
  return mask.astype(dtype)


def combine_biases(*masks: Optional[Array]):
  """Combine attention biases.

  Args:
    *masks: set of attention bias arguments to combine, some can be None.

  Returns:
    Combined mask, reduced by summation, returns None if no masks given.
  """
  masks = [m for m in masks if m is not None]
  if not masks:
    return None
  assert all(map(lambda x: x.ndim == masks[0].ndim, masks)), (
      f'masks must have same rank: {tuple(map(lambda x: x.ndim, masks))}')
  mask, *other_masks = masks
  for other_mask in other_masks:
    mask = mask + other_mask
  return mask


def make_decoder_mask(decoder_target_tokens: Array,
                      dtype: DType,
                      decoder_causal_attention: Optional[Array] = None,
                      decoder_segment_ids: Optional[Array] = None) -> Array:
  """Compute the self-attention mask for a decoder.

  Decoder mask is formed by combining a causal mask, a padding mask and an
  optional packing mask. If decoder_causal_attention is passed, it makes the
  masking non-causal for positions that have value of 1.

  A prefix LM is applied to a dataset which has a notion of "inputs" and
  "targets", e.g., a machine translation task. The inputs and targets are
  concatenated to form a new target. `decoder_target_tokens` is the concatenated
  decoder output tokens.

  The "inputs" portion of the concatenated sequence can attend to other "inputs"
  tokens even for those at a later time steps. In order to control this
  behavior, `decoder_causal_attention` is necessary. This is a binary mask with
  a value of 1 indicating that the position belonged to "inputs" portion of the
  original dataset.

  Example:

    Suppose we have a dataset with two examples.

    ds = [{"inputs": [6, 7], "targets": [8]},
          {"inputs": [3, 4], "targets": [5]}]

    After the data preprocessing with packing, the two examples are packed into
    one example with the following three fields (some fields are skipped for
    simplicity).

       decoder_target_tokens = [[6, 7, 8, 3, 4, 5, 0]]
         decoder_segment_ids = [[1, 1, 1, 2, 2, 2, 0]]
    decoder_causal_attention = [[1, 1, 0, 1, 1, 0, 0]]

    where each array has [batch, length] shape with batch size being 1. Then,
    this function computes the following mask.

                      mask = [[[[1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 0, 0, 0, 0, 0],
                                [1, 1, 1, 0, 0, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 0, 0],
                                [0, 0, 0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0, 0, 0]]]]

    mask[b, 1, :, :] represents the mask for the example `b` in the batch.
    Because mask is for a self-attention layer, the mask's shape is a square of
    shape [query length, key length].

    mask[b, 1, i, j] = 1 means that the query token at position i can attend to
    the key token at position j.

  Args:
    decoder_target_tokens: decoder output tokens. [batch, length]
    dtype: dtype of the output mask.
    decoder_causal_attention: a binary mask indicating which position should
      only attend to earlier positions in the sequence. Others will attend
      bidirectionally. [batch, length]
    decoder_segment_ids: decoder segmentation info for packed examples. [batch,
      length]

  Returns:
    the combined decoder mask.
  """
  masks = []
  # The same mask is applied to all attention heads. So the head dimension is 1,
  # i.e., the mask will be broadcast along the heads dim.
  # [batch, 1, length, length]
  causal_mask = make_causal_mask(decoder_target_tokens, dtype=dtype)

  # Positions with value 1 in `decoder_causal_attneition` can attend
  # bidirectionally.
  if decoder_causal_attention is not None:
    # [batch, 1, length, length]
    inputs_mask = make_attention_mask(
        decoder_causal_attention,
        decoder_causal_attention,
        jnp.logical_and,
        dtype=dtype)
    masks.append(jnp.logical_or(causal_mask, inputs_mask).astype(dtype))
  else:
    masks.append(causal_mask)

  # Padding mask.
  masks.append(
      make_attention_mask(
          decoder_target_tokens > 0, decoder_target_tokens > 0, dtype=dtype))

  # Packing mask
  if decoder_segment_ids is not None:
    masks.append(
        make_attention_mask(
            decoder_segment_ids, decoder_segment_ids, jnp.equal, dtype=dtype))

  return combine_masks(*masks, dtype=dtype)



#------------------------------------------------------------------------------
# The network: Decoder & Transformer Definitions
#------------------------------------------------------------------------------

class T5Config:  # placeholder
  pass


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: T5Config

  @nn.compact
  def __call__(self,
               inputs,
               decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None):
    cfg = self.config

    # Relative position embedding as attention biases.
    l = max_decode_length if decode and max_decode_length else inputs.shape[-2]
    decoder_bias = RelativePositionBiases(
        num_buckets=32,
        max_distance=128,
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        embedding_init=nn.initializers.variance_scaling(1.0, 'fan_avg',
                                                        'uniform'),
        name='relpos_bias')(l, l, False)

    inputs = with_sharding_constraint(inputs, ('batch', 'length', 'embed'))

    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    x = LayerNorm(
        dtype=cfg.dtype, name='pre_self_attention_layer_norm')(
            inputs)
    x = with_sharding_constraint(x, ('batch', 'length', 'embed'))

    # Self-attention block
    x = MultiHeadDotProductAttention(
        num_heads=cfg.num_heads,
        dtype=cfg.dtype,
        head_dim=cfg.head_dim,
        dropout_rate=cfg.dropout_rate,
        name='self_attention')(
            x,
            x,
            decoder_mask,
            decoder_bias,
            deterministic=deterministic,
            decode=decode)
    x = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            x, deterministic=deterministic)
    x = x + inputs
    x = with_sharding_constraint(x, ('batch', 'length', 'embed'))

    # MLP block.
    z = LayerNorm(dtype=cfg.dtype, name='pre_mlp_layer_norm')(x)
    z = with_sharding_constraint(z, ('batch', 'length', 'embed'))
    z = MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        name='mlp',
    )(z, deterministic=deterministic)
    z = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            z, deterministic=deterministic)
    z = z + x
    z = with_sharding_constraint(z, ('batch', 'length', 'embed'))

    if cfg.scan_layers:
      return z, None
    else:
      return z



class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""
  config: T5Config
  shared_embedding: nn.Module

  @nn.compact
  def __call__(self,
               decoder_input_tokens,
               decoder_positions=None,
               decoder_mask=None,
               deterministic=False,
               decode=False,
               max_decode_length=None):
    cfg = self.config
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype('int32'))
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    BlockLayer = DecoderLayer

    if cfg.remat_policy not in (None, 'none'):
      if cfg.remat_policy == 'minimal':
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      else:
        policy = None
      BlockLayer = remat(  # pylint: disable=invalid-name
          BlockLayer,
          prevent_cse=not cfg.scan_layers,
          policy=policy,
          static_argnums=(4, 5, 6))
    if cfg.scan_layers:
      initializing = self.is_mutable_collection('params')
      params_spec = (
          cfg.param_scan_axis if initializing else ScanIn(cfg.param_scan_axis))
      cache_spec = 0
      y, _ = scan_with_axes(
          BlockLayer,
          variable_axes={
              'params': params_spec,
              'cache': cache_spec
          },
          split_rngs={
              'params': True,
              'dropout': True
          },
          in_axes=(nn.broadcast, nn.broadcast, nn.broadcast,
                   nn.broadcast, nn.broadcast),
          length=cfg.num_decoder_layers,
          axis_name='layers')(
              config=cfg,
              name='decoder')(y, decoder_mask,
                              deterministic, decode, max_decode_length)
    else:
      for lyr in range(cfg.num_decoder_layers):
        # [batch, length, emb_dim] -> [batch, length, emb_dim]
        y = BlockLayer(
            config=cfg, name=f'layers_{lyr}')(
                y,
                decoder_mask=decoder_mask,
                deterministic=deterministic,
                decode=decode,
                max_decode_length=max_decode_length)

    y = LayerNorm(dtype=cfg.dtype, name='decoder_norm')(y)
    y = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            y, deterministic=deterministic)

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = DenseGeneral(
          cfg.vocab_size,
          dtype=jnp.float32,  # Use float32 for stabiliity.
          kernel_axes=('embed', 'vocab'),
          name='logits_dense')(
              y)
    return logits


class Transformer(nn.Module):
  """An encoder-decoder Transformer model."""
  config: T5Config

  def setup(self):
    cfg = self.config
    self.shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32,  # for logit training stability
        embedding_init=nn.initializers.normal(stddev=1.0),
        one_hot=True,
        name='token_embedder')

    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding)

  def __call__(
      self,
      decoder_input_tokens,
      decoder_target_tokens,
      decoder_segment_ids=None,
      decoder_positions=None,
      enable_dropout=True,
      decode=False,
      max_decode_length=None):
    """Applies Transformer decoder-branch on encoded-input and target."""
    cfg = self.config

    # Make padding attention masks.
    if decode:
      # Do not mask decoder attention based on targets padding at
      # decoding/inference time.
      decoder_mask = None
    else:
      decoder_mask = make_decoder_mask(
          decoder_target_tokens=decoder_target_tokens,
          dtype=cfg.dtype,
          decoder_segment_ids=decoder_segment_ids)

    # Add segmentation block-diagonal attention masks if using segmented data.
    if decoder_segment_ids is not None:
      if decode:
        raise ValueError(
            'During decoding, packing should not be used but '
            '`encoder_segment_ids` was passed to `Transformer.decode`.')

    logits = self.decoder(
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_mask=decoder_mask,
        deterministic=not enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length)
    return logits


#------------------------------------------------------------------------------
# Config for training
#------------------------------------------------------------------------------

import ml_collections

@struct.dataclass
class T5Config:
  """Global hyperparameters used to minimize obnoxious kwarg plumbing."""

  # Activation dtypes.
  dtype: Any = jnp.float32
  emb_dim: int = 128
  num_heads: int = 8
  head_dim: int = 16
  mlp_dim: int = 512
  num_decoder_layers: int = 8

  # Activation functions are retrieved from Flax.
  mlp_activations: Sequence[str] = ('relu',)
  dropout_rate: float = 0
  # If `True`, the embedding weights are used in the decoder output layer.
  logits_via_embedding: bool = False
  # minimal, full, or none
  remat_policy: str = 'none'

  # Parallelism
  scan_layers: bool = False
  param_scan_axis: int = 1

  # Dataset
  vocab_size: int = 30000
  dataset_name: str = 'lm1b'
  eval_dataset_name: str = 'lm1b'
  eval_split: str = "test"
  per_device_batch_size: int = 32
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


config = T5Config()



#------------------------------------------------------------------------------
# TFDS loading and tokenization
#------------------------------------------------------------------------------
"""Provides op for tokenizing a dataset."""

import os
import tempfile
import time
from typing import Any, Dict, Iterable, Tuple

from absl import logging
import dataclasses
import jax
from sentencepiece import SentencePieceTrainer
import tensorflow as tf
import tensorflow_text as tftxt

Features = Dict[str, tf.Tensor]


def _dump_chars_to_textfile(
    dataset: tf.data.Dataset,
    maxchars: int = int(1e7),
    data_keys=('inputs', 'targets')
) -> Tuple[str, int]:
  """Write part of a TFDS sentence dataset to lines in a text file.

  Args:
    dataset: tf.dataset containing string-data.
    maxchars: int: approximate number of characters to save from dataset.
    data_keys: Tuple[str]: what keys in dataset to dump from.

  Returns:
    name of temp file with dataset bytes, exact number of characters dumped.
  """
  char_count = 0
  ds_iter = dataset.as_numpy_iterator()
  with tempfile.NamedTemporaryFile(
      delete=False, prefix='/tmp/ds_chars') as outfp:
    while char_count < maxchars:
      example = next(ds_iter)
      for k in data_keys:
        line = example[k] + b'\n'
        char_count += len(line)
        outfp.write(line)
  return outfp.name, char_count


def _train_sentencepiece(dataset: tf.data.Dataset,
                         *,
                         vocab_size: int,
                         maxchars: int = int(1e7),
                         model_path: str,
                         model_type: str = 'unigram',
                         character_coverage: float = 1.0,
                         data_keys=('inputs', 'targets')):
  """Train SentencePiece tokenizer from subset of tf dataset.

  Args:
    dataset: tf.dataset
    vocab_size: int: size of vocab tokens to train.
    maxchars: int: number of characters to use for sentencepiece training.
    model_path: str: path of model file to save vocab model to.
    model_type: str: type of sentencepiece vocab to train.
    character_coverage: amount of characters covered by the model, good defaults
      are 0.9995 for languages with rich character set like Japanese or Chinese
      and 1.0 for other languages with small character set.
    data_keys: Tuple[str]: keys of dataset to use for training.

  Returns:
    path to the trained sentencepiece vocabulary model.
  """
  if model_path.startswith('gs://'):
    abs_model_path = model_path
  else:
    abs_model_path = os.path.abspath(os.path.expanduser(model_path))
  fname, _ = _dump_chars_to_textfile(
      dataset, maxchars=maxchars, data_keys=data_keys)
  with tempfile.NamedTemporaryFile(
      delete=False, prefix='/tmp/sp_tmp') as model_fp:
    pass  # we just want a prefix'd tmp-filename
  argstr = ' '.join([
      f'--input={fname}', f'--vocab_size={vocab_size}',
      f'--character_coverage={character_coverage}',
      f'--model_prefix={model_fp.name}', f'--model_type={model_type}'
  ])
  SentencePieceTrainer.Train(argstr)
  if jax.process_index() == 0:
    # Use an intermediate filename that is renamed to the target name to address
    # create and fill delays.
    copy_rename_path = abs_model_path + '.rntmp'
    tf.io.gfile.copy(model_fp.name + '.model', copy_rename_path, overwrite=True)
    tf.io.gfile.rename(copy_rename_path, abs_model_path, overwrite=True)
    logging.info('copied %s to %s', model_fp.name + '.model', abs_model_path)
  else:
    while not tf.io.gfile.exists(abs_model_path):
      time.sleep(1)
    time.sleep(1)
  return abs_model_path


def _load_sentencepiece_tokenizer(model_path: str,
                                  add_bos: bool = False,
                                  add_eos: bool = True,
                                  reverse: bool = False):
  """Load a tf-text SentencePiece tokenizer from given model filepath."""
  with tf.io.gfile.GFile(model_path, 'rb') as model_fp:
    sp_model = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(
      model=sp_model, add_bos=add_bos, add_eos=add_eos, reverse=reverse)
  return sp_tokenizer


def load_or_train_tokenizer(dataset: tf.data.Dataset,
                            *,
                            vocab_path: str,
                            vocab_size: int,
                            max_corpus_chars: int,
                            data_keys: Tuple[str, str] = ('inputs', 'targets')):
  """Loads the tokenizer at `vocab_path` or trains a one from `dataset`."""
  try:
    return _load_sentencepiece_tokenizer(vocab_path)
  except tf.errors.NotFoundError:
    logging.info('SentencePiece vocab not found, building one from data.')
    vocab_path = _train_sentencepiece(
        dataset,
        vocab_size=vocab_size,
        maxchars=max_corpus_chars,
        model_path=vocab_path,
        data_keys=data_keys)
    return _load_sentencepiece_tokenizer(vocab_path)


@dataclasses.dataclass
class TokenizeOp:

  sp_tokenizer: Any
  data_keys: Iterable[str] = ('inputs', 'targets')

  def __call__(self, features: Features) -> Features:
    for k in self.data_keys:
      features[k] = self.sp_tokenizer.tokenize(features[k])
    return features


###### Load TF Data ######


import os
from typing import Dict, Optional, List, Union

from clu import deterministic_data
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.experimental.AUTOTUNE
Features = Dict[str, tf.Tensor]


class NormalizeFeatureNamesOp:
  """Normalizes feature names to 'inputs' and 'targets'."""

  def __init__(self, ds_info: tfds.core.DatasetInfo):
    self.ds_info = ds_info

  def __call__(self, features: Features) -> Features:
    features['inputs'] = features.pop('text')
    # Unnecessary step used for uniformizing with examples/wmt.
    features['targets'] = features['inputs']
    return features


def get_raw_dataset(dataset_builder: tfds.core.DatasetBuilder,
                    split: str) -> tf.data.Dataset:
  """Loads a raw text dataset and normalizes feature keys.

  Args:
    dataset_builder: TFDS dataset builder that can build `split`.
    split: Split to use. This must be the full split. We shard the split across
      multiple hosts and currently don't support sharding subsplits.

  Returns:
    Dataset with source and target language features mapped to 'inputs' and
    'targets'.
  """
  num_examples = dataset_builder.info.splits[split].num_examples
  per_host_split = deterministic_data.get_read_instruction_for_host(
      split, num_examples, drop_remainder=False)
  ds = dataset_builder.as_dataset(split=per_host_split, shuffle_files=False)
  ds = ds.map(
      NormalizeFeatureNamesOp(dataset_builder.info),
      num_parallel_calls=AUTOTUNE)
  return ds


def pack_dataset(dataset: tf.data.Dataset,
                 key2length: Union[int, Dict[str, int]],
                 keys: Optional[List[str]] = None) -> tf.data.Dataset:
  """Creates a 'packed' version of a dataset on-the-fly.

  Adapted from the mesh-tf implementation.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.
  Each example in the output dataset represents several examples in the
  input dataset.
  For each key in the input dataset, two additional keys are created:
  <key>_segmentation: an int32 tensor identifying the parts
     representing the original example.
  <key>_position: an int32 tensor identifying the position within the original
     example.
  Example:
  Two input examples get combined to form an output example.
  The input examples are:
  {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0]}
  {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1]}
  The output example is:
  {
                 "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
    "inputs_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
        "inputs_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
   "targets_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
       "targets_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
  }
  0 represents padding in both the inputs and the outputs.
  Sequences in the incoming examples are truncated to length "length", and the
  sequences in the output examples all have fixed (padded) length "length".

  Args:
    dataset: a tf.data.Dataset
    key2length: an integer, or a dict from feature-key to integer
    keys: a list of strings (e.g. ["inputs", "targets"])

  Returns:
    a tf.data.Dataset
  """
  shapes = tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)
  if keys is None:
    keys = list(shapes.keys())
  for k in keys:
    if k not in shapes:
      raise ValueError('Key %s not found in dataset.  Available keys are %s' %
                       (k, shapes.keys()))
    if not shapes[k].is_compatible_with(tf.TensorShape([None])):
      raise ValueError('Tensors to be packed must be one-dimensional.')
  # make sure that the length dictionary contains all keys as well as the
  # keys suffixed by "_segmentation" and "_position"
  if isinstance(key2length, int):
    key2length = {k: key2length for k in keys}
  for k in keys:
    for suffix in ['_segmentation', '_position']:
      key2length[k + suffix] = key2length[k]

  # trim to length
  dataset = dataset.map(
      lambda x: {k: x[k][:key2length[k]] for k in keys},
      num_parallel_calls=AUTOTUNE)
  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = max(key2length.values())
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={k: [-1] for k in keys})
  dataset = _pack_with_tf_ops(dataset, keys, key2length)

  # Set the Tensor shapes correctly since they get lost in the process.
  def my_fn(x):
    return {k: tf.reshape(v, [key2length[k]]) for k, v in x.items()}

  return dataset.map(my_fn, num_parallel_calls=AUTOTUNE)


def _pack_with_tf_ops(dataset: tf.data.Dataset, keys: List[str],
                      key2length: Dict[str, int]) -> tf.data.Dataset:
  """Helper-function for packing a dataset which has already been batched.

  Helper for pack_dataset()  Uses tf.while_loop.

  Args:
    dataset: a dataset containing padded batches of examples.
    keys: a list of strings
    key2length: an dict from feature-key to integer

  Returns:
    a dataset.
  """
  empty_example = {}
  for k in keys:
    empty_example[k] = tf.zeros([0], dtype=tf.int32)
    empty_example[k + '_position'] = tf.zeros([0], dtype=tf.int32)
  keys_etc = empty_example.keys()

  def write_packed_example(partial, outputs):
    new_partial = empty_example.copy()
    new_outputs = {}
    for k in keys_etc:
      new_outputs[k] = outputs[k].write(
          outputs[k].size(),
          tf.pad(partial[k], [[0, key2length[k] - tf.size(partial[k])]]))
    return new_partial, new_outputs

  def map_fn(x):
    """Internal function to flat_map over.

    Consumes a batch of input examples and produces a variable number of output
    examples.
    Args:
      x: a single example

    Returns:
      a tf.data.Dataset
    """
    partial = empty_example.copy()
    i = tf.zeros([], dtype=tf.int32)
    dynamic_batch_size = tf.shape(x[keys[0]])[0]
    outputs = {}
    for k in keys:
      outputs[k] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])
      outputs[k + '_position'] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[key2length[k]])

    def body_fn(i, partial, outputs):
      """Body function for while_loop.

      Args:
        i: integer scalar
        partial: dictionary of Tensor (partially-constructed example)
        outputs: dictionary of TensorArray

      Returns:
        A triple containing the new values of the inputs.
      """
      can_append = True
      one_example = {}
      for k in keys:
        val = tf.cast(x[k][i], tf.int32)
        val = val[:tf.reduce_sum(tf.cast(tf.not_equal(val, 0), tf.int32))]
        one_example[k] = val
      for k in keys:
        can_append = tf.logical_and(
            can_append,
            tf.less_equal(
                tf.size(partial[k]) + tf.size(one_example[k]), key2length[k]))

      def false_fn():
        return write_packed_example(partial, outputs)

      def true_fn():
        return partial, outputs

      partial, outputs = tf.cond(can_append, true_fn, false_fn)
      new_partial = {}
      for k in keys:
        new_seq = one_example[k][:key2length[k]]
        new_seq_len = tf.size(new_seq)
        new_partial[k] = tf.concat([partial[k], new_seq], 0)
        new_partial[k + '_position'] = tf.concat(
            [partial[k + '_position'],
             tf.range(new_seq_len)], 0)
      partial = new_partial
      return i + 1, partial, outputs

    # For loop over all examples in the batch.
    i, partial, outputs = tf.while_loop(
        cond=lambda *_: True,
        body=body_fn,
        loop_vars=(i, partial, outputs),
        shape_invariants=(
            tf.TensorShape([]),
            {k: tf.TensorShape([None]) for k in keys_etc},
            {k: tf.TensorShape(None) for k in keys_etc},
        ),
        maximum_iterations=dynamic_batch_size)
    _, outputs = write_packed_example(partial, outputs)
    packed = {k: outputs[k].stack() for k in keys_etc}
    for k in keys:
      packed[k + '_segmentation'] = (
          tf.cumsum(
              tf.cast(tf.equal(packed[k + '_position'], 0), tf.int32), axis=1) *
          tf.cast(tf.not_equal(packed[k], 0), tf.int32))
    return packed

  dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
  return dataset.unbatch()

# -----------------------------------------------------------------------------
# Main dataset prep routines.
# -----------------------------------------------------------------------------
def preprocess_data(dataset,
                    shuffle: bool,
                    num_epochs: Optional[int] = 1,
                    pack_examples: bool = True,
                    shuffle_buffer_size: int = 1024,
                    max_length: int = 512,
                    batch_size: int = 256,
                    drop_remainder: bool = True,
                    prefetch_size: int = AUTOTUNE):
  """Shuffle and batch/pack the given dataset."""

  def length_filter(max_len):

    def filter_fn(x):
      source, target = x['inputs'], x['targets']
      l = tf.maximum(tf.shape(source)[0], tf.shape(target)[0])
      return tf.less(l, max_len + 1)

    return filter_fn

  if max_length > 0:
    dataset = dataset.filter(length_filter(max_length))

  if shuffle:
    dataset = dataset.shuffle(shuffle_buffer_size)
  dataset = dataset.repeat(num_epochs)

  if pack_examples:
    dataset = pack_dataset(dataset, max_length)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  else:  # simple (static-shape) padded batching
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes={
            'inputs': max_length,
            'targets': max_length
        },
        padding_values={
            'inputs': 0,
            'targets': 0
        },
        drop_remainder=drop_remainder)

  if prefetch_size:
    dataset = dataset.prefetch(prefetch_size)

  return dataset


def get_datasets(config: ml_collections.ConfigDict,
                 *,
                 n_devices: int,
                 vocab_path: Optional[str] = None):
  """Load and return dataset of batched examples for use during training."""
  if vocab_path is None:
    vocab_path = os.path.expanduser('~/' + config.dataset_name +
                                    '_sentencepiece_model')

  train_ds_builder = tfds.builder(config.dataset_name)
  train_data = get_raw_dataset(train_ds_builder, 'train')

  if config.eval_dataset_name:
    eval_ds_builder = tfds.builder(config.eval_dataset_name)
  else:
    eval_ds_builder = train_ds_builder
  eval_data = get_raw_dataset(eval_ds_builder, config.eval_split)

  # Tokenize data.
  sp_tokenizer = load_or_train_tokenizer(
      train_data,
      vocab_path=vocab_path,
      vocab_size=config.vocab_size,
      max_corpus_chars=config.max_corpus_chars)
  train_data = train_data.map(
      TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)
  eval_data = eval_data.map(
      TokenizeOp(sp_tokenizer), num_parallel_calls=AUTOTUNE)

  batch_size = config.per_device_batch_size * n_devices
  eval_batch_size = batch_size

  train_ds = preprocess_data(
      train_data,
      shuffle=True,
      num_epochs=None,
      pack_examples=True,
      batch_size=batch_size,
      max_length=config.max_target_length)

  eval_ds = preprocess_data(
      eval_data,
      shuffle=False,
      pack_examples=False,
      batch_size=eval_batch_size,
      max_length=config.max_eval_target_length)

  predict_ds = preprocess_data(
      eval_data,
      shuffle=False,
      pack_examples=False,
      batch_size=eval_batch_size,
      max_length=config.max_predict_length,
      drop_remainder=False)

  return train_ds, eval_ds, predict_ds, sp_tokenizer


def decode_tokens(tokenizer, toks, eos_id = 2):
  valid_toks = toks[:np.argmax(toks == eos_id) + 1].astype(np.int32)
  return tokenizer.detokenize(valid_toks).numpy().decode("utf-8")

def encode_strings(tokenizer, strs, max_len):
  tokenized_batch = np.zeros((len(strs), max_len), np.int32)
  for i, s in enumerate(strs):
    toks = tokenizer.tokenize(s).numpy()
    # Remove EOS token in prompt.
    tokenized_batch[i, :toks.shape[0]-1] = toks[:-1]
  return tokenized_batch


# -----------------------------------------------------------------------------
# Data Iterators
# -----------------------------------------------------------------------------

# A generator shoud return (input, target, segments, positions)

def right_shift(arr):
  result = np.zeros_like(arr)
  result[:, 1:] = arr[:, :-1]
  return result


# Make fake repetitive data
def make_fake_gen(batch_size, seq_len, repeats, vocab_size):
  while True:
    x = np.random.randint(
        low=1, high=vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    tokens = np.concatenate([x]*repeats, axis=1)
    shifted = right_shift(tokens)
    yield shifted, tokens, None, None

fake_gen = make_fake_gen(batch_size=256, seq_len=32, repeats=4, vocab_size=128)

# Load lm1b data
train_ds, eval_ds, predict_ds, sp_tokenizer = get_datasets(
    n_devices=jax.local_device_count(), config=config, vocab_path=None)

def make_lm1b_gen():
  train_iter = iter(train_ds)
  while True:
    step_data = jax.tree_util.tree_map(np.asarray, next(train_iter))
    tokens = step_data['inputs']
    shifted = right_shift(tokens)
    yield shifted, tokens, step_data['inputs_segmentation'], step_data[
        'inputs_position']

lm1b_gen = make_lm1b_gen()


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
  rng1, rng2 = random.split(dropout_rng)

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


ckpt_path = '~/flaxformer/lm1b'

def train_loop(config, steps, data_gen, state = None):
  nextrng = random.PRNGKey(0)
  model = Transformer(config)
  # with mesh, nn_partitioning.axis_rules(logical_axis_rules):
  #   vars = p_init({'params': random.PRNGKey(0), 'dropout': random.PRNGKey(0)},
  #                 jnp.ones((16, 128)), jnp.ones((16, 128)))
  vars = model.init({
      'params': random.PRNGKey(0),
      'dropout': random.PRNGKey(0)
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
def decode_without_kvc(state, model, shifted, data):
  shifted_in = shifted + 0.0
  shifted_in[:, 17:] = 0
  for i in range(15):
    output = model.apply({'params': state.params},
                         shifted_in,
                         data,
                         rngs={'dropout': random.PRNGKey(0)})
    updates = jnp.argmax(jax.nn.softmax(output), axis=2)
    shifted_in[:, 17+i] = updates[:, 16+i]  # greedy sampling

  print(shifted_in.shape)
  return shifted_in[0]


# Decoding with KV cache
def decode_with_kvc(state, model, shifted, data):
  output, updated_vars = model.apply({'params': state.params},
                                     data,
                                     data,
                                     decode=True,
                                     max_decode_length=config.max_target_length,
                                     rngs={'dropout': random.PRNGKey(0)},
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
    output, updated_vars = model.apply({'params': state.params, 'cache': cache},
                            data_in[:, i:i+1], data[:, i:i+1],
                            rngs={},
                            decode=True, max_decode_length=config.max_target_length,
                            mutable=('cache',))
    if i == 0:
      print(data_in[:, i:i+1].shape)
      print(output.shape)
    updates = jnp.argmax(jax.nn.softmax(output), axis=2)
    # print(i, data_in[0, i:i+1], updates[0])
    cache = updated_vars['cache']
    if i >= 16:
      data_in[:, i+1:i+2] = updates[:]  #greedy sampling

  return data_in[0]
