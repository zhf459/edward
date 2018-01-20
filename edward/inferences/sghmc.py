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
    args_part_one=(doc.arg_model +
                   doc.arg_align_latent_monte_carlo +
                   doc.arg_align_data +
                   doc.arg_step_size)[:-1],
    args_part_two=(doc.arg_auto_transform +
                   doc.arg_collections +
                   doc.arg_args_kwargs)[:-1],
    returns=doc.return_samples,
    notes_conditional_inference=doc.notes_conditional_inference)
def sghmc(model, align_latent, align_data,
          step_size=0.25, friction=0.1,
          auto_transform=True, collections=None, *args, **kwargs):
  """Stochastic gradient Hamiltonian Monte Carlo [@chen2014stochastic].

  SGHMC simulates Hamiltonian dynamics with friction using a discretized
  integrator. Its discretization error goes to zero as the learning
  rate decreases. Namely, it implements the update equations from (15)
  of @chen2014stochastic.

  This function implements an adaptive mass matrix using RMSProp.
  Namely, it uses the update from pre-conditioned SGLD
  [@li2016preconditioned] extended to second-order Langevin dynamics
  (SGHMC): the preconditioner is equal to the inverse of the mass
  matrix [@chen2014stochastic].

  Args:
  @{args_part_one}
    friction: float, optional.
      Constant scale on the friction term in the Hamiltonian system.
      The implementation may be extended in the future to enable a
      friction per random variable (`friction` would be a callable).
  @{args_part_two}

  Returns:
  @{returns}

  #### Notes

  Probabilistic programs may have latent variables which vary across
  executions. The MCMC algorithm transitions across latent variables
  instantiated during one execution of the model.

  @{notes_conditional_inference}

  #### Examples

  ```python
  def model():
    mu = Normal(loc=0.0, scale=1.0, name="mu")
    x = Normal(loc=mu, scale=1.0, sample_shape=10, name="x")

  samples = ed.sghmc(model,
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

  deltas = _sghmc_kernel(
      target_log_prob_fn=_target_log_prob_fn,
      states=states,
      step_sizes=step_size,
      frictions=friction)
  return {align_latent(state.name): state + delta
          for state, delta in zip(states, deltas)}


def _sghmc_kernel(target_log_prob_fn,
                  states,
                  counter,
                  momentums,
                  velocities,
                  friction,
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
  """Pre-conditioned SGHMC.

  Args:
    ...
    counter:
    momentums:
    momentums_states: Auxiliary momentums for states (the other is
      momentum for the preconditioner RMSProp.)
    friction:
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

    next_states = []
    next_momentums_states = []
    for state, mom, grad in zip(states, momentums, grads_target_log_prob):
      state_update, mom_state_update = _apply_noisy_update(
          mom, grad, counter, burnin, learning_rate,
          friction, mom_state,
          diagonal_bias, num_pseudo_batches)
      next_state = state - learning_rate * state_update
      next_mom_state = mom - learning_rate * mom_state_update
      next_states.append(next_state)
      next_momentums_states.append(next_mom_state)

    if not return_additional_state:
      return next_states

    counter += 1
    momentums = [(1.0 - preconditioner_decay_rate) *
                 (math_ops.square(grad) - mom)
                 for mom, grad in zip(momentums, grads_target_log_prob)]
    return [
        next_states,
        counter,
        next_momentum_states,
        momentums,
    ]


def _apply_noisy_update(mom, grad, counter, burnin, learning_rate,
                        friction, mom_state,
                        diagonal_bias, num_pseudo_batches):
  """Adapted from tf.contrib.bayesflow.SGLDOptimizer._apply_noisy_update."""
  from tensorflow.python.ops import array_ops
  from tensorflow.python.ops import math_ops
  from tensorflow.python.ops import random_ops
  stddev = array_ops.where(
      array_ops.squeeze(counter > burnin),
      math_ops.cast(math_ops.rsqrt(learning_rate * friction), grad.dtype),
      array_ops.zeros([], grad.dtype))

  preconditioner = math_ops.rsqrt(
      mom + math_ops.cast(diagonal_bias, grad.dtype))
  state_update = preconditioner * mom_state
  # TODO is this true for preconditioner?
  mom_state_update = (
      (1.0 - 0.5 * friction) * mom_state +
      preconditioner * grad * math_ops.cast(num_pseudo_batches,
                                            grad.dtype) +
      random_ops.random_normal(array_ops.shape(grad), 1.0, dtype=grad.dtype) *
      stddev * math_ops.sqrt(preconditioner))
  return state_update, mom_state_update
