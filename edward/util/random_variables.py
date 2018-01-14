from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six
import tensorflow as tf

from edward.models.random_variable import RandomVariable
from edward.models.random_variables import TransformedDistribution
from edward.models import PointMass
from edward.util.graphs import random_variables

tfb = tf.contrib.distributions.bijectors


def get_ancestors(x, collection=None):
  """Get ancestor random variables of input.

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find ancestors of.
    collection: list of RandomVariable, optional.
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Ancestor random variables of x.

  #### Examples
  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(0.0, 1.0)
  d = Normal(b * c, 1.0)
  assert set(ed.get_ancestors(d)) == set([a, b, c])
  ```
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value: node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set()
  visited = set()
  nodes = {x}
  while nodes:
    node = nodes.pop()

    if node in visited:
      continue
    visited.add(node)

    if isinstance(node, RandomVariable):
      node = node.value

    candidate_node = node_dict.get(node, None)
    if candidate_node is not None and candidate_node != x:
      output.add(candidate_node)

    nodes.update(node.op.inputs)

  return list(output)


def get_blanket(x, collection=None):
  """Get Markov blanket of input, which consists of its parents, its
  children, and the other parents of its children.

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find Markov blanket of.
    collection: list of RandomVariable, optional.
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Markov blanket of x.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(0.0, 1.0)
  c = Normal(a * b, 1.0)
  d = Normal(0.0, 1.0)
  e = Normal(c * d, 1.0)
  assert set(ed.get_blanket(c)) == set([a, b, d, e])
  ```
  """
  output = set()
  output.update(get_parents(x, collection))
  children = get_children(x, collection)
  output.update(children)
  for child in children:
    output.update(get_parents(child, collection))

  output.discard(x)
  return list(output)


def get_children(x, collection=None):
  """Get child random variables of input.

  Args:
    x: RandomVariable or tf.Tensor>
      Query node to find children of.
    collection: list of RandomVariable, optional>
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Child random variables of x.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(a, 1.0)
  d = Normal(c, 1.0)
  assert set(ed.get_children(a)) == set([b, c])
  ```
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value: node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set()
  visited = set()
  nodes = {x}
  while nodes:
    node = nodes.pop()

    if node in visited:
      continue
    visited.add(node)

    if isinstance(node, RandomVariable):
      node = node.value

    candidate_node = node_dict.get(node, None)
    if candidate_node is not None and candidate_node != x:
      output.add(candidate_node)
    else:
      for op in node.consumers():
        nodes.update(op.outputs)

  return list(output)


def get_descendants(x, collection=None):
  """Get descendant random variables of input.

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find descendants of.
    collection: list of RandomVariable, optional.
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Descendant random variables of x.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(a, 1.0)
  d = Normal(c, 1.0)
  assert set(ed.get_descendants(a)) == set([b, c, d])
  ```
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value: node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set()
  visited = set()
  nodes = {x}
  while nodes:
    node = nodes.pop()

    if node in visited:
      continue
    visited.add(node)

    if isinstance(node, RandomVariable):
      node = node.value

    candidate_node = node_dict.get(node, None)
    if candidate_node is not None and candidate_node != x:
      output.add(candidate_node)

    for op in node.consumers():
      nodes.update(op.outputs)

  return list(output)


def get_parents(x, collection=None):
  """Get parent random variables of input.

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find parents of.
    collection: list of RandomVariable, optional.
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Parent random variables of x.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(0.0, 1.0)
  d = Normal(b * c, 1.0)
  assert set(ed.get_parents(d)) == set([b, c])
  ```
  """
  if collection is None:
    collection = random_variables()

  node_dict = {node.value: node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set()
  visited = set()
  nodes = {x}
  while nodes:
    node = nodes.pop()

    if node in visited:
      continue
    visited.add(node)

    if isinstance(node, RandomVariable):
      node = node.value

    candidate_node = node_dict.get(node, None)
    if candidate_node is not None and candidate_node != x:
      output.add(candidate_node)
    else:
      nodes.update(node.op.inputs)

  return list(output)


def get_siblings(x, collection=None):
  """Get sibling random variables of input.

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find siblings of.
    collection: list of RandomVariable, optional.
      The collection of random variables to check with respect to;
      defaults to all random variables in the graph.

  Returns:
    list of RandomVariable.
    Sibling random variables of x.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(a, 1.0)
  assert ed.get_siblings(b) == [c]
  ```
  """
  parents = get_parents(x, collection)
  siblings = set()
  for parent in parents:
    siblings.update(get_children(parent, collection))

  siblings.discard(x)
  return list(siblings)


def get_variables(x, collection=None):
  """Get parent TensorFlow variables of input.

  Args:
    x: RandomVariable or tf.Tensor.
      Query node to find parents of.
    collection: list of tf.Variable, optional.
      The collection of variables to check with respect to; defaults to
      all variables in the graph.

  Returns:
    list of tf.Variable.
    TensorFlow variables that x depends on.

  #### Examples

  ```python
  a = tf.Variable(0.0)
  b = tf.Variable(0.0)
  c = Normal(a * b, 1.0)
  assert set(ed.get_variables(c)) == set([a, b])
  ```
  """
  if collection is None:
    collection = tf.global_variables()

  node_dict = {node.name: node for node in collection}

  # Traverse the graph. Add each node to the set if it's in the collection.
  output = set()
  visited = set()
  nodes = {x}
  while nodes:
    node = nodes.pop()

    if node in visited:
      continue
    visited.add(node)

    if isinstance(node, RandomVariable):
      node = node.value

    candidate_node = node_dict.get(node.name, None)
    if candidate_node is not None and candidate_node != x:
      output.add(candidate_node)

    nodes.update(node.op.inputs)

  return list(output)


def is_independent(a, b, condition=None):
  """Assess whether a is independent of b given the random variables in
  condition.

  Implemented using the Bayes-Ball algorithm [@schachter1998bayes].

  Args:
    a: RandomVariable or list of RandomVariable.
       Query node(s).
    b: RandomVariable or list of RandomVariable.
       Query node(s).
    condition: RandomVariable or list of RandomVariable, optional.
       Random variable(s) to condition on.

  Returns:
    bool.
    True if a is independent of b given the random variables in condition.

  #### Examples

  ```python
  a = Normal(0.0, 1.0)
  b = Normal(a, 1.0)
  c = Normal(a, 1.0)
  assert ed.is_independent(b, c, condition=a)
  ```
  """
  if condition is None:
    condition = []
  if not isinstance(a, list):
    a = [a]
  if not isinstance(b, list):
    b = [b]
  if not isinstance(condition, list):
    condition = [condition]
  A = set(a)
  B = set(b)
  condition = set(condition)

  top_marked = set()
  # The Bayes-Ball algorithm will traverse the belief network
  # and add each node that is relevant to B given condition
  # to the set bottom_marked. A and B are conditionally
  # independent if no node in A is in bottom_marked.
  bottom_marked = set()

  schedule = [(node, "child") for node in B]
  while schedule:
    node, came_from = schedule.pop()

    if node not in condition and came_from == "child":
      if node not in top_marked:
        top_marked.add(node)
        for parent in get_parents(node):
          schedule.append((parent, "child"))

      if not isinstance(node, PointMass) and node not in bottom_marked:
        bottom_marked.add(node)
        if node in A:
          return False  # node in A is relevant to B
        for child in get_children(node):
          schedule.append((child, "parent"))

    elif came_from == "parent":
      if node in condition and node not in top_marked:
        top_marked.add(node)
        for parent in get_parents(node):
          schedule.append((parent, "child"))

      elif node not in condition and node not in bottom_marked:
        bottom_marked.add(node)
        if node in A:
          return False  # node in A is relevant to B
        for child in get_children(node):
          schedule.append((child, "parent"))

  return True


def transform(x, *args, **kwargs):
  """Transform a continuous random variable to the unconstrained space.

  `transform` selects among a number of default transformations which
  depend on the support of the provided random variable:

  + $[0, 1]$ (e.g., Beta): Inverse of sigmoid.
  + $[0, \infty)$ (e.g., Gamma): Inverse of softplus.
  + Simplex (e.g., Dirichlet): Inverse of softmax-centered.
  + $(-\infty, \infty)$ (e.g., Normal, MultivariateNormalTriL): None.

  Args:
    x: RandomVariable.
      Continuous random variable to transform.
    *args, **kwargs: optional.
      Arguments to overwrite when forming the `TransformedDistribution`.
      For example, manually specify the transformation by passing in
      the `bijector` argument.

  Returns:
    RandomVariable.
    A `TransformedDistribution` random variable, or the provided random
    variable if no transformation was applied.

  #### Examples

  ```python
  x = Gamma(1.0, 1.0)
  y = ed.transform(x)
  sess = tf.Session()
  sess.run(y)
  -2.2279539
  ```
  """
  if len(args) != 0 or kwargs.get('bijector', None) is not None:
    return TransformedDistribution(x, *args, **kwargs)

  try:
    support = x.support
  except AttributeError as e:
    msg = """'{}' object has no 'support'
             so cannot be transformed.""".format(type(x).__name__)
    raise AttributeError(msg)

  if support == '01':
    bij = tfb.Invert(tfb.Sigmoid())
    new_support = 'real'
  elif support == 'nonnegative':
    bij = tfb.Invert(tfb.Softplus())
    new_support = 'real'
  elif support == 'simplex':
    bij = tfb.Invert(tfb.SoftmaxCentered(event_ndims=1))
    new_support = 'multivariate_real'
  elif support in ('real', 'multivariate_real'):
    return x
  else:
    msg = "'transform' does not handle supports of type '{}'".format(support)
    raise ValueError(msg)

  new_x = TransformedDistribution(x, bij, *args, **kwargs)
  new_x.support = new_support
  return new_x


def compute_multinomial_mode(probs, total_count=1, seed=None):
  """Compute the mode of a Multinomial random variable.

  Args:
    probs: 1-D Numpy array of Multinomial class probabilities
    total_count: integer number of trials in single Multinomial draw
    seed: a Python integer. Used to create a random seed for the
      distribution

  #### Examples

  ```python
  # returns either [2, 2, 1], [2, 1, 2] or [1, 2, 2]
  probs = np.array(3 * [1/3])
  total_count = 5
  compute_multinomial_mode(probs, total_count)

  # returns [3, 2, 0]
  probs = np.array(3 * [1/3])
  total_count = 5
  compute_multinomial_mode(probs, total_count)
  ```
  """
  def softmax(vec):
    numerator = np.exp(vec)
    return numerator / numerator.sum(axis=0)

  random_state = np.random.RandomState(seed)
  mode = np.zeros_like(probs, dtype=np.int32)
  if total_count == 1:
    mode[np.argmax(probs)] += 1
    return list(mode)
  remaining_count = total_count
  uniform_prob = 1 / total_count

  while remaining_count > 0:
    if (probs < uniform_prob).all():
      probs = softmax(probs)
    mask = probs >= uniform_prob
    overflow_count = int(mask.sum() - remaining_count)
    if overflow_count > 0:
      hot_indices = np.where(mask)[0]
      cold_indices = random_state.choice(hot_indices, overflow_count,
                                         replace=False)
      mask[cold_indices] = False
    mode[mask] += 1
    probs[mask] -= uniform_prob
    remaining_count -= np.sum(mask)
  return mode
