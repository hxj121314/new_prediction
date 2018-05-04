import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.ops.losses import util
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


TOP = np.pi
BOTTLE = -np.pi
BOUNDVALUE = TOP - BOTTLE
IND_NUM = 3
AXISWEIGHT = [1.0, 2.0, 10.0]

##########################################################MY_LOSS
class Reduction(object):
  """Types of loss reduction."""

  # Un-reduced weighted losses with the same shape as input.
  NONE = "none"

  # Scalar sum of `NONE`.
  SUM = "weighted_sum"

  # Scalar `SUM` divided by sum of weights.
  MEAN = "weighted_mean"

  # Scalar `SUM` divided by number of non-zero weights.
  SUM_BY_NONZERO_WEIGHTS = "weighted_sum_by_nonzero_weights"

  @classmethod
  def all(cls):
    return (
        cls.NONE,
        cls.SUM,
        cls.MEAN,
        cls.SUM_BY_NONZERO_WEIGHTS)

  @classmethod
  def validate(cls, key):
    if key not in cls.all():
      raise ValueError("Invalid ReductionKey %s." % key)


def _safe_div(numerator, denominator, name="value"):
  """Computes a safe divide which returns 0 if the denominator is zero.
  Note that the function contains an additional conditional check that is
  necessary for avoiding situations where the loss is zero causing NaNs to
  creep into the gradient computation.
  Args:
    numerator: An arbitrary `Tensor`.
    denominator: `Tensor` whose shape matches `numerator` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.
  Returns:
    The element-wise value of the numerator divided by the denominator.
  """
  return array_ops.where(
      math_ops.greater(denominator, 0),
      math_ops.div(numerator, array_ops.where(
          math_ops.equal(denominator, 0),
          array_ops.ones_like(denominator), denominator)),
      array_ops.zeros_like(numerator),
      name=name)


def _safe_mean(losses, num_present):
  """Computes a safe mean of the losses.
  Args:
    losses: `Tensor` whose elements contain individual loss measurements.
    num_present: The number of measurable elements in `losses`.
  Returns:
    A scalar representing the mean of `losses`. If `num_present` is zero,
      then zero is returned.
  """
  total_loss = math_ops.reduce_sum(losses)
  return _safe_div(total_loss, num_present)


def _num_present(losses, weights, per_batch=False):
  """Computes the number of elements in the loss function induced by `weights`.
  A given weights tensor induces different numbers of usable elements in the
  `losses` tensor. The `weights` tensor is broadcast across `losses` for all
  possible dimensions. For example, if `losses` is a tensor of dimension
  `[4, 5, 6, 3]` and `weights` is a tensor of shape `[4, 5]`, then `weights` is,
  in effect, tiled to match the shape of `losses`. Following this effective
  tile, the total number of present elements is the number of non-zero weights.
  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: `Tensor` of shape `[]`, `[batch_size]` or
      `[batch_size, d1, ... dK]`, where K < N.
    per_batch: Whether to return the number of elements per batch or as a sum
      total.
  Returns:
    The number of present (non-zero) elements in the losses tensor. If
      `per_batch` is `True`, the value is returned as a tensor of size
      `[batch_size]`. Otherwise, a single scalar tensor is returned.
  """
  with ops.name_scope(None, "num_present", (losses, weights)) as scope:
    weights = math_ops.to_float(weights)
    present = array_ops.where(
        math_ops.equal(weights, 0.0),
        array_ops.zeros_like(weights),
        array_ops.ones_like(weights))
    present = weights_broadcast_ops.broadcast_weights(present, losses)
    if per_batch:
      return math_ops.reduce_sum(
          present, axis=math_ops.range(1, array_ops.rank(present)),
          keep_dims=True, name=scope)
    return math_ops.reduce_sum(present, name=scope)


def compute_weighted_loss(
    losses, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Computes the weighted loss.
  Args:
    losses: `Tensor` of shape `[batch_size, d1, ... dN]`.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `losses`, and must be broadcastable to `losses` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: the scope for the operations performed in computing the loss.
    loss_collection: the loss will be added to these collections.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss `Tensor` of the same type as `losses`. If `reduction` is
    `NONE`, this has the same shape as `losses`; otherwise, it is scalar.
  Raises:
    ValueError: If `weights` is `None` or the shape is not compatible with
      `losses`, or if the number of dimensions (rank) of either `losses` or
      `weights` is missing.
  """
  Reduction.validate(reduction)
  with ops.name_scope(scope, "weighted_loss", (losses, weights)):
    with ops.control_dependencies((
        weights_broadcast_ops.assert_broadcastable(weights, losses),)):
      losses = ops.convert_to_tensor(losses)
      input_dtype = losses.dtype
      losses = math_ops.to_float(losses)
      weights = math_ops.to_float(weights)
      weighted_losses = math_ops.multiply(losses, weights)
      if reduction == Reduction.NONE:
        loss = weighted_losses
      else:
        loss = math_ops.reduce_sum(weighted_losses)
        if reduction == Reduction.MEAN:
          loss = _safe_mean(
              loss,
              math_ops.reduce_sum(array_ops.ones_like(losses) * weights))
        elif reduction == Reduction.SUM_BY_NONZERO_WEIGHTS:
          loss = _safe_mean(loss, _num_present(losses, weights))

      # Convert the result back to the input type.
      loss = math_ops.cast(loss, input_dtype)
      util.add_loss(loss, loss_collection)
      return loss

def mean_squared_error(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Adds a Sum-of-Squares loss to the training procedure.
  `weights` acts as a coefficient for the loss. If a scalar is provided, then
  the loss is simply scaled by the given value. If `weights` is a tensor of size
  [batch_size], then the total loss for each sample of the batch is rescaled
  by the corresponding element in the `weights` vector. If the shape of
  `weights` matches the shape of `predictions`, then the loss of each
  measurable element of `predictions` is scaled by the corresponding value of
  `weights`.
  Args:
    labels: The ground truth output tensor, same dimensions as 'predictions'.
    predictions: The predicted outputs.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `losses` dimension).
    scope: The scope for the operations performed in computing the loss.
    loss_collection: collection to which the loss will be added.
    reduction: Type of reduction to apply to loss.
  Returns:
    Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
    shape as `labels`; otherwise, it is scalar.
  Raises:
    ValueError: If the shape of `predictions` doesn't match that of `labels` or
      if the shape of `weights` is invalid.  Also if `labels` or `predictions`
      is None.
  """
  if labels is None:
    raise ValueError("labels must not be None.")
  if predictions is None:
    raise ValueError("predictions must not be None.")
  with ops.name_scope(scope, "mean_squared_error",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    losses = tf.square(predictions-labels)
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)

def my_loss(
    labels, predictions, weights=1.0, scope=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS):
  if labels is None:
    raise ValueError("labels must not be None.")
  if predictions is None:
    raise ValueError("predictions must not be None.")
  with ops.name_scope(scope, "mean_squared_error",
                      (predictions, labels, weights)) as scope:
    predictions = math_ops.to_float(predictions)
    labels = math_ops.to_float(labels)
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())
    bound = tf.constant(np.asarray([BOUNDVALUE] * IND_NUM, 'float').tolist())
    v1 = tf.abs(predictions-labels)
    v2 = tf.abs(bound - v1)
    axisweight = tf.constant(AXISWEIGHT)
    losses = tf.where(tf.greater(v2,v1),tf.square(v1), tf.square(v2)) * axisweight
    return compute_weighted_loss(
        losses, weights, scope, loss_collection, reduction=reduction)
