from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.models import Trace
from edward.inferences import docstrings as doc
from edward.inferences.inference import (
    call_function_up_to_args, make_intercept)
from collections import OrderedDict


@doc.set_doc(
    arg_model=doc.arg_model[:-1],
    args=(doc.arg_align_latent_monte_carlo +
          doc.arg_align_data +
          doc.arg_auto_transform +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_samples,
    notes_conditional_inference=doc.notes_conditional_inference)
def metropolis_hastings(model, proposal, align_latent, align_data,
                        auto_transform=True, collections=None, *args, **kwargs):
  """Metropolis-Hastings [@metropolis1953equation; @hastings1970monte].

  It draws sample from the proposal given the last sample. The
  accept or reject the sample is based on the ratio,

  $\\text{ratio} =
        \log p(x, z^{\\text{new}}) - \log p(x, z^{\\text{old}}) -
        \log g(z^{\\text{new}} \mid z^{\\text{old}}) +
        \log g(z^{\\text{old}} \mid z^{\\text{new}})$

  Args:
  {@arg_model}
    proposal: function whose inputs are a subset of `args` (e.g.,
      for amortized). Output is not used. Collection of random
      variables to perform inference on; each is binded to a proposal
      distribution $g(z' \mid z)$.
  {@args}

  #### Notes

  @{notes_conditional_inference}

  TODO The function assumes the proposal distribution has the
  same support as the prior. auto_transform does X.

  #### Examples

  ```python
  def model():
    mu = Normal(loc=0.0, scale=1.0, name="mu")
    x = Normal(loc=mu, scale=1.0, sample_shape=10, name="x")

  def proposal(mu):
    qmu = Normal(loc=mu, scale=0.5, name="qmu")

  samples = ed.metropolis_hastings(
      model, proposal,
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

  def _proposal_fn(*fargs):
    """Takes inputted states and returns (proposed states, log Hastings ratio).

    This implementation doesn't let `proposal take *args, **kwargs as
    input (i.e., it cannot be amortized). We also assume proposal
    returns same size and order as inputted states.
    """
    with Trace() as new_trace:
      # Build g(new | old) with newly drawn states given inputted ones.
      call_function_up_to_args(proposal, *fargs)
    new_states = [new_trace[align_latent(state.name)] for state in states]
    old_states_trace = {align_latent(state.name): arg
                        for state, arg in zip(states, fargs)}
    intercept = make_intercept(
        old_states_trace, align_data, align_latent, args, kwargs)
    with Trace(intercept=intercept) as old_trace:
      # Build g(old | new) where all rv values are set to old states.
      call_function_up_to_args(proposal, *new_states)
    # Compute log p(old | new) - log p(new | old).
    log_hastings_ratio = 0.0
    for state in states:
      old_state = old_trace[align_latent(state.name)].value
      new_state = new_trace[align_latent(state.name)].value
      log_hastings_ratio += tf.reduce_sum(old_state.log_prob(old_state.value))
      log_hastings_ratio -= tf.reduce_sum(new_state.log_prob(new_state.value))
    return new_states, log_hastings_ratio

  # TODO independent_chain_ndims
  # TODO target_log_prob and previous sample to plug in?
  next_states = _metropolis_hastings_kernel(
      target_log_prob_fn=_target_log_prob_fn,
      proposal_fn=_proposal_fn,
      states=states,
      target_log_prob=target_log_prob)
  return {align_latent(state.name): next_state
          for state, next_state in zip(states, next_states)}


def _metropolis_hastings_kernel(target_log_prob_fn,
                                proposal_fn,
                                states,
                                independent_chain_ndims=0,
                                return_additional_state=False,
                                target_log_prob=None,
                                seed=None,
                                name=None):
  """Runs one iteration of Metropolis-Hastings.

  Args:
    states: list of `Tensor`s each representing part of the
      chain's state.
    target_log_prob_fn: Python callable which takes an argument like
      `*state_tensors` (i.e., Python expanded) and returns the target
      distribution's (possibly unnormalized) log-density. Output has
      same shape as `target_log_prob`.
    proposal_fn: Python callable which takes an argument like
      `*state_tensors` (i.e., Python expanded) and returns a tuple of
      a list of proposed states of same size as input, and a log
      Hastings ratio `Tensor` of same shape as `target_log_prob`. If
      proposal is symmetric, set the second value to `None` to enable
      more efficient computation than explicitly supplying a tensor of
      zeros.
    independent_chain_ndims: Scalar `int` `Tensor` representing the number of
      leading dimensions (in each state) which index independent chains.
      Default value: `0` (i.e., only one chain).
    return_additional_state: Boolean determining whether to return
      additional state information.
    target_log_prob: `Tensor` of shape (independent_chain_dims,) if
      independent_chain_ndims == 1 else ().
    seed: A Python integer. Used to create a random seed for the distribution.
      See
      @{tf.set_random_seed}
      for behavior.
    name: A name of the operation (optional).
  """
  with tf.name_scope(name, "_metropolis_hastings_kernel", states):
    with tf.name_scope("init"):
      if target_log_prob is None:
        target_log_prob = target_log_prob_fn(*states)
    proposed_states, log_hastings_ratio = proposal_fn(*states)
    if log_hastings_ratio is None:
      # Assume proposal is symmetric so log Hastings ratio is zero,
      # log p(old | new) - log p(new | old) = 0.
      log_hastings_ratio = 0.

    target_log_prob_proposed_states = target_log_prob_fn(proposed_states)

    with tf.name_scope(
            "accept_reject",
            states + [target_log_prob, target_log_prob_proposed_states]):
      log_accept_prob = (target_log_prob_proposed_states - target_log_prob +
                         log_hastings_ratio)
      log_draws = tf.log(tf.random_uniform(tf.shape(log_accept_prob),
                                           seed=seed,
                                           dtype=log_accept_prob.dtype))
      is_proposal_accepted = log_draws < log_accept_prob
      next_states = [tf.where(is_proposal_accepted, proposed_state, state)
                     for proposed_state, state in zip(proposed_states, states)]

    if not return_additional_state:
      return next_states

    next_log_prob = tf.where(is_proposal_accepted,
                             target_log_prob_proposed_states,
                             target_log_prob)
    return [
        next_states,
        log_accept_prob,
        next_log_prob,
        proposed_states,
    ]
