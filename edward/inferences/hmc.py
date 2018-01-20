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
def hmc(model, align_latent, align_data,
        step_size=0.25, num_leapfrog_steps=2,
        auto_transform=True, collections=None, *args, **kwargs):
  """Hamiltonian Monte Carlo, also known as hybrid Monte Carlo
  [@duane1987hybrid; @neal2011mcmc].

  HMC simulates Hamiltonian dynamics using a numerical integrator. The
  integrator has a discretization error and is corrected with a
  Metropolis accept-reject step.

  Args:
  @{args_part_one}
    num_leapfrog_steps: int, optional.
      Number of steps of numerical integrator.
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

  samples = ed.hmc(model,
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

  next_states = _hmc_kernel(
      num_leapfrog_steps=num_leapfrog_steps,
      target_log_prob_fn=_target_log_prob_fn,
      states=states,
      step_sizes=step_size)
  return {align_latent(state.name): next_state
          for state, next_state in zip(states, next_states)}


def _hmc_kernel(num_leapfrog_steps,
                target_log_prob_fn,
                states=(),
                step_sizes=(),
                independent_chain_ndims=0,
                return_additional_state=False,
                target_log_prob=None,
                grads_target_log_prob=None,
                seed=None,
                name=None):
  from edward.inferences.metropolis_hastings import _metropolis_hastings_kernel
  with tf.name_scope(name, "_hmc_kernel", states):
    with tf.name_scope("init"):
      if target_log_prob is None:
        target_log_prob = target_log_prob_fn(*states)
      if grads_target_log_prob is None:
        grads_target_log_prob = tf.gradients(target_log_prob, states)
      log_potential = -target_log_prob
      gradients = map(tf.negative, grads_target_log_prob)
      momentums = map(
          lambda s: tf.random_normal(shape=tf.shape(s),
                                     dtype=s.dtype.base_dtype),
          states)

    def _proposal_fn(*state_tensors):
      [
          proposed_deltas,
          proposed_log_potential,
          proposed_gradients,
          proposed_momentums,
      ] = _leapfrog_integrator(num_leapfrog_steps,
                               target_log_prob_fn,
                               states,
                               step_sizes,
                               log_potential,
                               gradients,
                               momentums)
      proposed_states = states + proposed_deltas
      energy_change = _compute_energy_change(log_potential,
                                             momentums,
                                             proposed_log_potential,
                                             proposed_momentums,
                                             independent_chain_ndims)
      log_hastings_ratio = energy_change
      return proposed_states, log_hastings_ratio

    out = _metropolis_hastings_kernel(
        states=states,
        target_log_prob_fn=target_log_prob_fn,
        proposal_fn=_proposal_fn,
        independent_chain_ndims=independent_chain_ndims,
        return_additional_state=return_additional_state,
        target_log_prob=target_log_prob,
        seed=seed)
    if not return_additional_state:
      next_states = out
      return next_states

    next_states, log_accept_prob, next_log_prob, proposed_states = out
    next_gradients = [tf.where(is_accepted, proposed_gradient, gradient)
                      for proposed_gradient, gradient
                      in zip(proposed_gradients, gradients)]
    return [
        next_states,
        log_accepted_prob,
        next_log_prob,
        next_gradients,
        proposed_states,
    ]


def _leapfrog_integrator(num_steps,
                        target_log_prob_fn,
                        state_parts,
                        step_sizes,
                        log_potential,
                        gradients,
                        momentums,
                        name=None):
  """Applies `n_steps` steps of the leapfrog integrator.

  This just wraps `leapfrog_step()` in a `tf.while_loop()`, reusing
  gradient computations where possible.

  Args:
    step_size: Scalar step size or array of step sizes for the
      leapfrog integrator. Broadcasts to the shape of
      `initial_position`. Larger step sizes lead to faster progress, but
      too-large step sizes lead to larger discretization error and
      worse energy conservation.
    n_steps: Number of steps to run the leapfrog integrator.
    initial_position: Tensor containing the value(s) of the position variable(s)
      to update.
    initial_momentum: Tensor containing the value(s) of the momentum variable(s)
      to update.
    potential_and_grad: Python callable that takes a position tensor like
      `initial_position` and returns the potential energy and its gradient at
      that position.
    initial_grad: Tensor with the value of the gradient of the potential energy
      at `initial_position`.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    updated_position: Updated value of the position.
    updated_momentum: Updated value of the momentum.
    new_potential: Potential energy of the new position. Has shape matching
      `potential_and_grad(initial_position)`.
    new_grad: Gradient from potential_and_grad() evaluated at the new position.
      Has shape matching `initial_position`.

  Example: Simple quadratic potential.

  ```python
  def potential_and_grad(position):
    return tf.reduce_sum(0.5 * tf.square(position)), position
  position = tf.placeholder(np.float32)
  momentum = tf.placeholder(np.float32)
  potential, grad = potential_and_grad(position)
  new_position, new_momentum, new_potential, new_grad = hmc.leapfrog_integrator(
    0.1, 3, position, momentum, potential_and_grad, grad)

  sess = tf.Session()
  position_val = np.random.randn(10)
  momentum_val = np.random.randn(10)
  potential_val, grad_val = sess.run([potential, grad],
                                     {position: position_val})
  positions = np.zeros([100, 10])
  for i in xrange(100):
    position_val, momentum_val, potential_val, grad_val = sess.run(
      [new_position, new_momentum, new_potential, new_grad],
      {position: position_val, momentum: momentum_val})
    positions[i] = position_val
  # Should trace out sinusoidal dynamics.
  plt.plot(positions[:, 0])
  ```
  """
  def _loop_body(step, deltas, ignore_log_potential, gradients, momentums):
    return [step + 1] + list(_leapfrog_step(
        target_log_prob_fn, state_parts, step_sizes,
        deltas, gradients, momentums))

  with tf.name_scope(name, "_leapfrog_integrator",
                     (list(state_parts) + list(step_sizes) + [log_potential] +
                      gradients + momentums)):
    return tf.while_loop(
        cond=lambda step, *args: step < num_steps,
        body=_loop_body,
        loop_vars=[0,                                # step
                   map(tf.zeros_like, state_parts),  # deltas
                   log_potential,
                   gradients,
                   momentums],
        back_prop=False)[1:]


def _leapfrog_step(target_log_prob_fn, state_parts, step_sizes,
                  deltas, gradients, momentums,
                  name=None):
  """Applies one step of the leapfrog integrator.

  Assumes a simple quadratic kinetic energy function: 0.5 * ||momentum||^2.

  Args:
    step_size: Scalar step size or array of step sizes for the
      leapfrog integrator. Broadcasts to the shape of
      `position`. Larger step sizes lead to faster progress, but
      too-large step sizes lead to larger discretization error and
      worse energy conservation.
    position: Tensor containing the value(s) of the position variable(s)
      to update.
    momentum: Tensor containing the value(s) of the momentum variable(s)
      to update.
    potential_and_grad: Python callable that takes a position tensor like
      `position` and returns the potential energy and its gradient at that
      position.
    grad: Tensor with the value of the gradient of the potential energy
      at `position`.
    name: Python `str` name prefixed to Ops created by this function.

  Returns:
    updated_deltas: ...
    new_potential: Potential energy of the new position. Has shape matching
      `potential_and_grad(position)`.
    updated_momentum: Updated value of the momentum.
    new_grad: Gradient from potential_and_grad() evaluated at the new position.
      Has shape matching `position`.

  Example: Simple quadratic potential.

  ```python
  def potential_and_grad(position):
    # Simple quadratic potential
    return tf.reduce_sum(0.5 * tf.square(position)), position
  position = tf.placeholder(np.float32)
  momentum = tf.placeholder(np.float32)
  potential, grad = potential_and_grad(position)
  new_position, new_momentum, new_potential, new_grad = hmc.leapfrog_step(
    0.1, position, momentum, potential_and_grad, grad)

  sess = tf.Session()
  position_val = np.random.randn(10)
  momentum_val = np.random.randn(10)
  potential_val, grad_val = sess.run([potential, grad],
                                     {position: position_val})
  positions = np.zeros([100, 10])
  for i in xrange(100):
    position_val, momentum_val, potential_val, grad_val = sess.run(
      [new_position, new_momentum, new_potential, new_grad],
      {position: position_val, momentum: momentum_val})
    positions[i] = position_val
  # Should trace out sinusoidal dynamics.
  plt.plot(positions[:, 0])
  ```
  """
  def _increment(xs, scalar, steps, ys):  # x + scalar * step * y
    with tf.name_scope("_increment"):
      return [x + scalar * step * y for x, step, y in zip(xs, steps, ys)]

  with ops.name_scope(
      name, 'leapfrog_step', (list(state_parts) + list(step_sizes) +
                              [deltas, gradients, momentums])):
    new_momentums = _increment(momentums, -0.5, step_sizes, gradients)
    new_deltas = _increment(deltas, 1., step_sizes, new_momentums)
    new_state_parts = [state_part + new_delta
                       for state_part, new_delta
                       in zip(state_parts, new_deltas)]
    new_log_potential = -target_log_prob_fn(*new_state_parts)
    new_gradients = tf.gradients(new_log_potential, new_state_parts)
    if any(new_gradient is None for new_gradient in new_gradients):
        raise ValueError(
            "Encountered `None` gradient. Does your `target_log_prob_fn` "
            "access all `tf.Variable`s via `tf.get_variable`?\n"
            "  state_parts: {}\n"
            "  new_state_parts: {}\n"
            "  new_gradients: {}".format(
                state_parts, new_state_parts, new_gradients))
    new_momentums = _increment(new_momentums, -0.5, step_sizes, new_gradients)
    return new_deltas, new_log_potential, new_gradients, new_momentums


def _compute_energy_change(log_potential,
                           momentums,
                           proposed_log_potential,
                           proposed_momentums,
                           independent_chain_ndims,
                           name=None):
  # Abbreviate lk0=log_kinetic_energy and lk1=proposed_log_kinetic_energy
  # since they're a mouthful and lets us inline more.
  lk0, lk1 = [], []
  for momentum, proposed_momentum in zip(momentums, proposed_momentums):
    axis = tf.range(independent_chain_ndims, tf.rank(momentum))

    # tf.squeeze expects a Python list for the `axis` argument.
    # Until this is resolved (b/72225430), we require the axis (a range) to
    # be known prior to graph execution.
    from tensorflow.python.framework import tensor_util
    axis = tensor_util.constant_value(axis)
    if axis is None:
      raise NotImplementedError("Cannot workaround the fact that `tf.squeeze`"
                                "requires Python list for `axis` argument.")
    else:
      axis = axis.tolist()

    log_sum_sq = lambda x: tf.reduce_logsumexp(2. * tf.log(tf.abs(x)), axis)
    lk0.append(log_sum_sq(momentum))
    lk1.append(log_sum_sq(proposed_momentum))
  lk0 = -np.log(2.) + tf.reduce_logsumexp(tf.stack(lk0, axis=-1), axis=-1)
  lk1 = -np.log(2.) + tf.reduce_logsumexp(tf.stack(lk1, axis=-1), axis=-1)
  x = tf.stack([proposed_log_potential, tf.exp(lk1),
                -log_potential, -tf.exp(lk0)], axis=-1)

  # The sum is NaN if any element is NaN or we see both +Inf and -Inf.
  # This we will replace such rows with infinite energy change thus implying
  # rejection.
  is_indeterminate_sum = (
      tf.reduce_any(tf.is_nan(x) | (~tf.is_finite(x) & (x < 0.)), axis=-1) &
      tf.reduce_any(tf.is_nan(x) | (~tf.is_finite(x) & (x > 0.)), axis=-1))

  is_indeterminate_sum = tf.tile(
      is_indeterminate_sum[..., tf.newaxis],
      multiples=tf.concat([tf.shape(is_indeterminate_sum), [4]], axis=0))

  return tf.reduce_sum(
      tf.where(
          is_indeterminate_sum,
          tf.fill(tf.shape(x), value=x.dtype.as_numpy_dtype(np.inf)),
          x),
      axis=-1)
