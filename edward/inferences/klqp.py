from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.models import Trace
from edward.inferences import docstrings as doc
from edward.inferences.inference import (
    call_function_up_to_args, make_intercept)

try:
  from edward.models import Normal
  from tensorflow.contrib.distributions import kl_divergence
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))

tfd = tf.contrib.distributions


@doc.set_doc(
    args=(doc.arg_model +
          doc.arg_variational +
          doc.arg_align_latent +
          doc.arg_align_data +
          doc.arg_scale +
          doc.arg_n_samples +
          doc.arg_kl_scaling +
          doc.arg_auto_transform +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_loss_surrogate_loss,
    notes_model_parameters=doc.notes_model_parameters,
    notes_conditional_inference=doc.notes_conditional_inference_samples,
    notes_regularization_losses=doc.notes_regularization_losses)
def klqp(model, variational, align_latent, align_data,
         scale=lambda name: 1.0, n_samples=1, kl_scaling=lambda name: 1.0,
         auto_transform=True, collections=None, *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This function returns a loss and surrogate loss
  [@schulman2015stochastic; @ruiz2016generalized; @ritchie2016deep].
  The surrogate loss' autodiff automates selection of two black box
  gradient estimators given a variational factor:

  1. score function gradients [@paisley2012variational] with
     Rao-Blackwellization [@ranganath2014black];
  2. reparameterization gradients [@kingma2014auto].

  If the KL divergence between a variational factor and its aligned
  prior is tractable, then the loss function can be written as

  $-\mathbb{E}_{q(z; \lambda)}[\log p(x \mid z)] +
      \\text{KL}( q(z; \lambda) \| p(z) ),$

  where the KL term is computed analytically [@kingma2014auto]. We
  compute this automatically when $p(z)$ and $q(z; \lambda)$ are
  Normal.

  Current Rao-Blackwellization is limited to Rao-Blackwellizing across
  stochastic nodes in the computation graph. It does not
  Rao-Blackwellize within a node such as when a node represents
  multiple random variables via non-scalar batch shape.
  Rao-Blackwellization is performed at runtime for each sample.

  Args:
  @{args}

  Returns:
  @{returns}

  #### Notes

  Probabilistic programs may have random variables which vary across
  executions. The algorithm returns calculations following `n_samples`
  executions of the model and variational programs.

  @{notes_model_parameters}

  @{notes_conditional_inference}

  @{notes_regularization_losses}

  #### Examples

  ```python
  def model():
    mu = Normal(loc=0.0, scale=1.0, name="mu")
    x = Normal(loc=mu, scale=1.0, sample_shape=10, name="x")

  def variational():
    qmu = Normal(loc=tf.get_variable("loc", []),
                 scale=tf.nn.softplus(tf.get_variable("shape", [])),
                 name="qmu")

  loss, surrogate_loss = ed.klqp(
      model, variational,
      align_latent=lambda name: "qmu" if name == "mu" else None,
      align_data=lambda name: "x" if name == "x" else None,
      x=x_data)
  ```
  """
  # TODO control variates
  # + baseline, learnable baseline
  # + Ruiz+ 2016
  # + Tucker+ 2017; Cremer+ 2017
  # + Miller+ 2017
  # TODO analytic stuff
  # + Roeder+ 2017
  p_log_prob = [None] * n_samples
  q_log_prob = [None] * n_samples
  surrogate_loss = [None] * n_samples
  kl_penalty = 0.0
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    with Trace(intercept=intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    # Collect key-value pairs of (rv, rv's (scaled) log prob).
    p_dict = {}
    q_dict = {}
    inverse_align_latent = {}
    for name, node in six.iteritems(model_trace):
      rv = node.value
      posterior_node = posterior_trace.get(align_latent(name), None)
      if posterior_node is None:
        # Likelihood
        scale_factor = scale(name)
        p_dict[rv] = tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
      else:
        qz = posterior_node.value
        # For pairs with analytic KL, accumulate KL divergences for
        # first iteration in loop.
        if isinstance(rv, Normal) and isinstance(qz, Normal):
          if s == 0:
            kl_penalty += tf.reduce_sum(
                kl_scaling(name) * kl_divergence(qz, rv))
        else:
          scale_factor = scale(name)
          p_dict[rv] = tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
          q_dict[qz] = tf.reduce_sum(scale_factor * qz.log_prob(qz.value))
          inverse_align_latent[qz] = rv

    # Build surrogate loss.
    scaled_q_log_prob = 0.0
    for qz, log_prob in six.iteritems(q_dict):
      if qz.reparameterization_type == tfd.FULLY_REPARAMETERIZED:
        scale_factor = 1.0
      else:
        scale_factor = 0.0
        for rv in qz.get_blanket(q_rvs) + [qz]:
          scale_factor += q_dict[rv]
          scale_factor -= p_dict[inverse_align_latent[qz]]
      scaled_q_log_prob += scale_factor * log_prob

    p_log_prob_s = tf.reduce_sum(list(six.itervalues(p_dict)))
    p_log_prob[s] = p_log_prob_s
    q_log_prob[s] = tf.reduce_sum(list(six.itervalues(q_dict)))
    surrogate_loss[s] = scaled_q_log_prob - p_log_prob_s

  p_log_prob = tf.reduce_mean(p_log_prob)
  q_log_prob = tf.reduce_mean(q_log_prob)
  surrogate_loss = tf.reduce_mean(surrogate_loss) + kl_penalty

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  surrogate_loss += reg_penalty

  if collections is not None:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=collections)
    tf.summary.scalar("loss/q_log_prob", q_log_prob,
                      collections=collections)
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=collections)

  loss = q_log_prob - p_log_prob + kl_penalty + reg_penalty
  return loss, surrogate_loss


@doc.set_doc(
    args=(doc.arg_model +
          doc.arg_variational +
          doc.arg_align_latent +
          doc.arg_align_data +
          doc.arg_scale +
          doc.arg_n_samples +
          doc.arg_auto_transform +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_loss,
    notes_model_parameters=doc.notes_model_parameters,
    notes_conditional_inference=doc.notes_conditional_inference_samples,
    notes_regularization_losses=doc.notes_regularization_losses)
def klqp_reparameterization(model, variational, align_latent, align_data,
                            scale=lambda name: 1.0, n_samples=1,
                            auto_transform=True, collections=None,
                            *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This function builds a loss function equal to KL(q||p) up to a
  constant. Its automatic differentiation is a stochastic gradient of

  $-\\text{ELBO} =
      -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]$

  based on the reparameterization trick [@kingma2014auto].

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.

  Args:
  @{args}

  Returns:
  @{returns}

  #### Notes

  Probabilistic programs may have random variables which vary across
  executions. The algorithm returns calculations following `n_samples`
  executions of the model and variational programs.

  @{notes_model_parameters}

  @{notes_conditional_inference}

  @{notes_regularization_losses}

  #### Examples

  ```python
  def model():
    mu = Normal(loc=0.0, scale=1.0, name="mu")
    x = Normal(loc=mu, scale=1.0, sample_shape=10, name="x")

  def variational():
    qmu = Normal(loc=tf.get_variable("loc", []),
                 scale=tf.nn.softplus(tf.get_variable("shape", [])),
                 name="qmu")

  loss = ed.klqp_reparameterization(
      model, variational,
      align_latent=lambda name: "qmu" if name == "mu" else None,
      align_data=lambda name: "x" if name == "x" else None,
      x=x_data)
  ```
  """
  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    with Trace(intercept=intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    for name, node in six.iteritems(model_trace):
      rv = node.value
      scale_factor = scale(name)
      p_log_prob[s] += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
      posterior_node = posterior_trace.get(align_latent(name), None)
      if posterior_node is not None:
        qz = posterior_node.value
        q_log_prob[s] += tf.reduce_sum(scale_factor * qz.log_prob(qz.value))

  p_log_prob = tf.reduce_mean(p_log_prob)
  q_log_prob = tf.reduce_mean(q_log_prob)
  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  if collections is not None:
    tf.summary.scalar("loss/p_log_prob", p_log_prob,
                      collections=collections)
    tf.summary.scalar("loss/q_log_prob", q_log_prob,
                      collections=collections)
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=collections)
  loss = q_log_prob - p_log_prob + reg_penalty
  return loss


@doc.set_doc(
    args=(doc.arg_model +
          doc.arg_variational +
          doc.arg_align_latent +
          doc.arg_align_data +
          doc.arg_scale +
          doc.arg_n_samples +
          doc.arg_kl_scaling +
          doc.arg_auto_transform +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_loss,
    notes_model_parameters=doc.notes_model_parameters,
    notes_conditional_inference=doc.notes_conditional_inference_samples,
    notes_regularization_losses=doc.notes_regularization_losses)
def klqp_reparameterization_kl(model, variational, align_latent, align_data,
                               scale=lambda name: 1.0, n_samples=1,
                               kl_scaling=lambda name: 1.0,
                               auto_transform=True, collections=None,
                               *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This function builds a loss function equal to KL(q||p) up to a
  constant. Its automatic differentiation is a stochastic gradient of

  $-\\text{ELBO} =
      -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]$

  based on the reparameterization trick [@kingma2014auto].

  It assumes the KL is analytic.

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.

  Args:
  @{args}

  Returns:
  @{returns}

  #### Notes

  Probabilistic programs may have random variables which vary across
  executions. The algorithm returns calculations following `n_samples`
  executions of the model and variational programs.

  @{notes_model_parameters}

  @{notes_conditional_inference}

  @{notes_regularization_losses}

  #### Examples

  ```python
  def model():
    mu = Normal(loc=0.0, scale=1.0, name="mu")
    x = Normal(loc=mu, scale=1.0, sample_shape=10, name="x")

  def variational():
    qmu = Normal(loc=tf.get_variable("loc", []),
                 scale=tf.nn.softplus(tf.get_variable("shape", [])),
                 name="qmu")

  loss = ed.klqp_reparameterization_kl(
      model, variational,
      align_latent=lambda name: "qmu" if name == "mu" else None,
      align_data=lambda name: "x" if name == "x" else None,
      x=x_data)
  ```
  """
  p_log_lik = [0.0] * n_samples
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    with Trace(intercept=intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    for name, node in six.iteritems(model_trace):
      rv = node.value
      scale_factor = scale(name)
      p_log_lik[s] += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))

  p_log_lik = tf.reduce_mean(p_log_lik)

  kl_penalty = 0.0
  for name, node in six.iteritems(model_trace):
    rv = node.value
    posterior_node = posterior_trace.get(align_latent(name), None)
    if posterior_node is not None:
      qz = posterior_node.value
      kl_penalty += tf.reduce_sum(kl_scaling(name) * kl_divergence(qz, rv))

  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  if collections is not None:
    tf.summary.scalar("loss/p_log_lik", p_log_lik,
                      collections=collections)
    tf.summary.scalar("loss/kl_penalty", kl_penalty,
                      collections=collections)
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=collections)
  loss = -p_log_lik + kl_penalty + reg_penalty
  return loss


@doc.set_doc(
    args=(doc.arg_model +
          doc.arg_variational +
          doc.arg_align_latent +
          doc.arg_align_data +
          doc.arg_scale +
          doc.arg_n_samples +
          doc.arg_auto_transform +
          doc.arg_collections +
          doc.arg_args_kwargs)[:-1],
    returns=doc.return_loss_surrogate_loss,
    notes_model_parameters=doc.notes_model_parameters,
    notes_conditional_inference=doc.notes_conditional_inference_samples,
    notes_regularization_losses=doc.notes_regularization_losses)
def klqp_score(model, variational, align_latent, align_data,
               scale=lambda name: 1.0, n_samples=1, auto_transform=True,
               collections=None, *args, **kwargs):
  """Variational inference with the KL divergence

  $\\text{KL}( q(z; \lambda) \| p(z \mid x) ).$

  This function builds a loss function equal to KL(q||p) up to a
  constant. It also builds a surrogate loss whose automatic
  differentiation is a stochastic gradient of

  $-\\text{ELBO} =
      -\mathbb{E}_{q(z; \lambda)} [ \log p(x, z) - \log q(z; \lambda) ]$

  based on the score function estimator [@paisley2012variational].

  Computed by sampling from $q(z;\lambda)$ and evaluating the
  expectation using Monte Carlo sampling.

  Args:
  @{args}

  Returns:
  @{returns}

  #### Notes

  Probabilistic programs may have random variables which vary across
  executions. The algorithm returns calculations following `n_samples`
  executions of the model and variational programs.

  @{notes_model_parameters}

  @{notes_conditional_inference}

  @{notes_regularization_losses}

  #### Examples

  ```python
  def model():
    mu = Normal(loc=0.0, scale=1.0, name="mu")
    x = Normal(loc=mu, scale=1.0, sample_shape=10, name="x")

  def variational():
    qmu = Normal(loc=tf.get_variable("loc", []),
                 scale=tf.nn.softplus(tf.get_variable("shape", [])),
                 name="qmu")

  loss, surrogate_loss = ed.klqp_score(
      model, variational,
      align_latent=lambda name: "qmu" if name == "mu" else None,
      align_data=lambda name: "x" if name == "x" else None,
      x=x_data)
  ```
  """
  p_log_prob = [0.0] * n_samples
  q_log_prob = [0.0] * n_samples
  for s in range(n_samples):
    with Trace() as posterior_trace:
      call_function_up_to_args(variational, *args, **kwargs)
    intercept = make_intercept(
        posterior_trace, align_data, align_latent, args, kwargs)
    with Trace(intercept=intercept) as model_trace:
      call_function_up_to_args(model, *args, **kwargs)

    for name, node in six.iteritems(model_trace):
      rv = node.value
      scale_factor = scale(name)
      p_log_prob[s] += tf.reduce_sum(scale_factor * rv.log_prob(rv.value))
      posterior_node = posterior_trace.get(align_latent(name), None)
      if posterior_node is not None:
        qz = posterior_node.value
        q_log_prob[s] += tf.reduce_sum(
            scale_factor * qz.log_prob(tf.stop_gradient(qz.value)))

  p_log_prob = tf.stack(p_log_prob)
  q_log_prob = tf.stack(q_log_prob)
  reg_penalty = tf.reduce_sum(tf.losses.get_regularization_losses())
  if collections is not None:
    tf.summary.scalar("loss/p_log_prob", tf.reduce_mean(p_log_prob),
                      collections=collections)
    tf.summary.scalar("loss/q_log_prob", tf.reduce_mean(q_log_prob),
                      collections=collections)
    tf.summary.scalar("loss/reg_penalty", reg_penalty,
                      collections=collections)
  losses = q_log_prob - p_log_prob
  loss = tf.reduce_mean(losses) + reg_penalty
  surrogate_loss = (tf.reduce_mean(q_log_prob * tf.stop_gradient(losses)) +
                    reg_penalty)
  return loss, surrogate_loss


def _default_constructor(latent_vars):
  if isinstance(latent_vars, list):
    with tf.variable_scope(None, default_name="posterior"):
      latent_vars_dict = {}
      continuous = \
          ('01', 'nonnegative', 'simplex', 'real', 'multivariate_real')
      for z in latent_vars:
        if not hasattr(z, 'support') or z.support not in continuous:
          raise AttributeError(
              "Random variable {} is not continuous or a random "
              "variable with supported continuous support.".format(z))
        batch_event_shape = z.batch_shape.concatenate(z.event_shape)
        loc = tf.Variable(tf.random_normal(batch_event_shape))
        scale = tf.nn.softplus(
            tf.Variable(tf.random_normal(batch_event_shape)))
        latent_vars_dict[z] = Normal(loc=loc, scale=scale)
      latent_vars = latent_vars_dict
  return latent_vars
