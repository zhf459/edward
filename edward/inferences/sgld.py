from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.models import Trace
from edward.inferences import docstrings as doc
from edward.inferences.inference import (
    call_function_up_to_args, make_intercept)


@doc.set_doc(
    args=(doc.arg_model +
          doc.arg_align_latent_monte_carlo +
          doc.arg_align_data +
          doc.arg_step_size +
          doc.arg_auto_transform +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_samples,
    notes_conditional_inference=doc.notes_conditional_inference)
def sgld(model, align_latent, align_data, step_size=0.25,
         auto_transform=True, collections=None, *args, **kwargs):
  """Stochastic gradient Langevin dynamics [@welling2011bayesian].

  SGLD simulates Langevin dynamics using a discretized integrator. Its
  discretization error goes to zero as the learning rate decreases.

  Args:
  @{args}

  Returns:
  @{returns}

  #### Notes

  @{notes_conditional_inference}

  #### Examples

  ```python
  def model():
    mu = Normal(loc=0.0, scale=1.0, name="mu")
    x = Normal(loc=mu, scale=1.0, sample_shape=10, name="x")

  samples = ed.sgld(model,
                    align_latent=lambda name: "qmu" if name == "mu" else None,
                    align_data=lambda name: "x_data" if name == "x" else None,
                    x_data=x_data)
  ```
  """
  # Trace one execution of model to collect states. The list of states
  # (and order) may vary across executions.
  with Trace() as model_trace:
    call_function_up_to_args(model, *args, **kwargs)
  states = []
  for name, node in six.iteritems(model_trace):
    if align_latent(name) is not None:
      z = node.value
      states.append(z)

  def _target_log_prob_fn(*fargs):
    """Target's unnormalized log-joint density as a function of states."""
    posterior_trace = {align_latent(state.name): arg
                       for state, arg in zip(states, fargs)}
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    with Trace(intercept=intercept) as model_trace:
      # Note program may not run into same list of states. For newly
      # unseen states, program uses them as is; for states that are
      # passed-in but unseen, program doesn't use them.
      call_function_up_to_args(model, *args, **kwargs)

    p_log_prob = 0.0
    for name, node in six.iteritems(model_trace):
      rv = node.value
      p_log_prob += tf.reduce_sum(rv.log_prob(rv.value))
    return p_log_prob

  next_states = _sgld_kernel(
      target_log_prob_fn=_target_log_prob_fn,
      states=states,
      step_sizes=step_size)
  return {align_latent(state.name): next_state
          for state, next_state in zip(states, next_states)}


def _sgld_kernel(target_log_prob_fn,
                 states,
                 counter,
                 momentums,
                 learning_rate,
                 preconditioner_decay_rate=0.95,
                 num_pseudo_batches=1,
                 burnin=25,
                 diagonal_bias=1e-8,
                 independent_chain_ndims=0,
                 return_additional_state=False,
                 target_log_prob=None,
                 grads_target_log_prob=None,
                 name=None):
  """tf.contrib.bayesflow.SGLDOptimizer re-implemented as a pure function.

  Args:
    ...
    counter: Counter for iteration number, namely, to determine if
      past burnin phase.
    momentums: List of Tensors, representing exponentially weighted
      moving average of each squared gradient with respect to a state.
      It is recommended to initialize it with tf.ones.
    learning_rate: From tf.contrib.bayesflow.SGLDOptimizer.
    preconditioner_decay_rate: From tf.contrib.bayesflow.SGLDOptimizer.
    num_pseudo_batches: From tf.contrib.bayesflow.SGLDOptimizer.
    burnin: From tf.contrib.bayesflow.SGLDOptimizer.
    diagonal_bias: From tf.contrib.bayesflow.SGLDOptimizer.
    ...
  """
  with tf.name_scope(name, "_sgld_kernel", states):
    with tf.name_scope("init"):
      if target_log_prob is None:
        target_log_prob = target_log_prob_fn(*states)
      if grads_target_log_prob is None:
        grads_target_log_prob = tf.gradients(target_log_prob, states)

    next_states = [
        state - learning_rate *
        _apply_noisy_update(mom, grad, counter, burnin, learning_rate,
                            diagonal_bias, num_pseudo_batches)
        for state, mom, grad in zip(states, momentums, grads_target_log_prob)]
    if not return_additional_state:
      return next_states

    counter += 1
    momentums = [(1.0 - preconditioner_decay_rate) *
                 (math_ops.square(grad) - mom)
                 for mom, grad in zip(momentums, grads_target_log_prob)]
    return [
        next_states,
        counter,
        momentums,
    ]


def _apply_noisy_update(mom, grad, counter, burnin, learning_rate,
                        diagonal_bias, num_pseudo_batches):
  """Adapted from tf.contrib.bayesflow.SGLDOptimizer._apply_noisy_update."""
  from tensorflow.python.ops import array_ops
  from tensorflow.python.ops import math_ops
  from tensorflow.python.ops import random_ops
  # Compute and apply the gradient update following
  # preconditioned Langevin dynamics
  stddev = array_ops.where(
      array_ops.squeeze(counter > burnin),
      math_ops.cast(math_ops.rsqrt(learning_rate), grad.dtype),
      array_ops.zeros([], grad.dtype))

  preconditioner = math_ops.rsqrt(
      mom + math_ops.cast(diagonal_bias, grad.dtype))
  return (
      0.5 * preconditioner * grad * math_ops.cast(num_pseudo_batches,
                                                  grad.dtype) +
      random_ops.random_normal(array_ops.shape(grad), 1.0, dtype=grad.dtype) *
      stddev * math_ops.sqrt(preconditioner))
