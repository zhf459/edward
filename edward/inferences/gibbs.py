from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import six
import tensorflow as tf

from collections import OrderedDict
from edward.inferences.conjugacy import complete_conditional


# TODO do we really need  gibbs and not just cycling complete
# conditionals? (maybe for train()))?
def gibbs(proposal=None, scan_order='random'):
  """Gibbs sampling [@geman1984stochastic].

  Args:
    proposal:
      Collection of random variables to perform inference on; each is
      binded to its complete conditionals which Gibbs cycles draws on.
        If not specified, default is to use `ed.complete_conditional`.
    scan_order: list or str, optional.
      The scan order for each Gibbs update. If list, it is the
      deterministic order of latent variables. An element in the list
      can be a `RandomVariable` or itself a list of
      `RandomVariable`s (this defines a blocked Gibbs sampler). If
      'random', will use a random order at each update.

  TODO The function assumes the proposal distribution has the
  same support as the prior. auto_transform does X.

  #### Examples

  ```python
  x_data = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1])

  p = Beta(1.0, 1.0)
  x = Bernoulli(probs=p, sample_shape=10)

  qp = Empirical(tf.Variable(tf.zeros(500)))
  inference = ed.Gibbs({p: qp}, data={x: x_data})
  ```
  """
  if proposal_vars is None:
    proposal_vars = {z: complete_conditional(z)
                     for z in six.iterkeys(latent_vars)}
  else:
    proposal_vars = check_and_maybe_build_latent_vars(proposal_vars)

  def update(self, feed_dict=None):
    """Run one iteration of sampling.

    Args:
      feed_dict: dict, optional.
        Feed dictionary for a TensorFlow session run. It is used to feed
        placeholders that are not fed during initialization.

    Returns:
      dict.
      Dictionary of algorithm-specific information. In this case, the
      acceptance rate of samples since (and including) this iteration.
    """
    sess = get_session()
    if not self.feed_dict:
      # Initialize feed for all conditionals to be the draws at step 0.
      samples = OrderedDict(self.latent_vars)
      inits = sess.run([qz.params[0] for qz in six.itervalues(samples)])
      for z, init in zip(six.iterkeys(samples), inits):
        self.feed_dict[z] = init

      for key, value in six.iteritems(self.data):
        if isinstance(key, tf.Tensor) and "Placeholder" in key.op.type:
          self.feed_dict[key] = value
        elif isinstance(key, RandomVariable) and \
                isinstance(value, (tf.Tensor, tf.Variable)):
          self.feed_dict[key] = sess.run(value)

    if feed_dict is None:
      feed_dict = {}

    self.feed_dict.update(feed_dict)

    # Determine scan order.
    if self.scan_order == 'random':
      scan_order = list(six.iterkeys(self.latent_vars))
      random.shuffle(scan_order)
    else:  # list
      scan_order = self.scan_order

    # Fetch samples by iterating over complete conditional draws.
    for z in scan_order:
      if isinstance(z, RandomVariable):
        draw = sess.run(self.proposal_vars[z], self.feed_dict)
        self.feed_dict[z] = draw
      else:  # list
        draws = sess.run([self.proposal_vars[zz] for zz in z], self.feed_dict)
        for zz, draw in zip(z, draws):
          self.feed_dict[zz] = draw

    # Assign the samples to the Empirical random variables.
    _, accept_rate = sess.run(
        [self.train, self.n_accept_over_t], self.feed_dict)
    t = sess.run(self.increment_t)

    if self.debug:
      sess.run(self.op_check, self.feed_dict)

    if self.logging and self.n_print != 0:
      if t == 1 or t % self.n_print == 0:
        summary = sess.run(self.summarize, self.feed_dict)
        self.train_writer.add_summary(summary, t)

    return {'t': t, 'accept_rate': accept_rate}
