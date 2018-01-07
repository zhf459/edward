from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class Trace(object):
  """Context manager with two objects:

  + The trace stack stores executions from each primitive fn.
  + (Optional) The intercept callable intercepts the continuation of a function.

  Optionally, the trace stack stores the function call, its inputs,
  its parent primitives, and its children primitives. This lets us trace
  the continuation structure. Storing inputs can be memory-intensive as it
  prevents garbage collection; hence it's optional.
  """
  def __init__(self, intercept=None, trace_continuation=False):
    self._intercept = intercept
    self._trace_continuation = trace_continuation
    # We use OrderedDict. It is essentially a stack where each element is a node
    # (value) and its name (key); the name is a pointer to the node.
    self._trace_stack = collections.OrderedDict({})

  def __enter__(self):
    # Note if Trace's are nested, global vars are set
    # to the innermost context's variables.
    if self._intercept is not None:
      global _INTERCEPT
      _INTERCEPT = self._intercept
    global _TRACE_CONTINUATION, _TRACE_STACK
    _TRACE_CONTINUATION = self._trace_continuation
    _TRACE_STACK = self._trace_stack
    return self

  def __exit__(self, t, v, tb):
    global _INTERCEPT, _TRACE_CONTINUATION, _TRACE_STACK
    try:
      del _INTERCEPT
    except:
      pass
    del _TRACE_CONTINUATION
    del _TRACE_STACK

  # operator-overloading for convenience
  def __repr__(self):
    return self._trace_stack.__repr__()

  def __str__(self):
    return self._trace_stack.__str__()

  def __delitem__(self, key):
    del self._trace_stack[key]

  def __getitem__(self, key):
    return self._trace_stack[key]

  def __setitem__(self, key, value):
    self._trace_stack[key] = value

  def get(self, key, value=None):
    return self._trace_stack.get(key, value)

  def iteritems(self):
    return self._trace_stack.items()

  def iterkeys(self):
    return self._trace_stack.keys()

  def itervalues(self):
    return self._trace_stack.values()

  def items(self):
    return self._trace_stack.items()

  def keys(self):
    return self._trace_stack.keys()

  def values(self):
    return self._trace_stack.values()


class Node(object):
  """Node in trace stack. Collection of nodes forms a directed acyclic graph."""
  __slots__ = ['value', 'f', 'args', 'kwargs', 'parents', 'children']

  def __init__(self, value, f=None, args=None, kwargs=None,
               parents=None, children=None):
    self.value = value
    self.f = f
    self.args = args
    self.kwargs = kwargs
    self.parents = parents
    self.children = children


def primitive(fn):
  """Wraps function so its continuation can be intercepted
  and its execution can be written to a stack.

  Apply this to decorate primitive functions.
  """
  def wrapped_fn(*args, **kwargs):
    global _INTERCEPT, _TRACE_CONTINUATION, _TRACE_STACK
    if '_INTERCEPT' in globals():
      out = _INTERCEPT(fn, *args, **kwargs)
    else:
      out = fn(*args, **kwargs)
    if '_TRACE_CONTINUATION' in globals() and '_TRACE_STACK' in globals():
      if _TRACE_CONTINUATION:
        parents = [v for v in list(args) + kwargs.values()
                   if hasattr(v, "name") and v.name in _TRACE_STACK]
        node = Node(out, fn, args, kwargs, parents, [])
        _TRACE_STACK[out.name] = node
        for parent in parents:
          parent.children.append(node)
      else:
        _TRACE_STACK[out.name] = Node(out)
    return out
  return wrapped_fn


# TODO(trandustin): wrapping via init, not primitive() so wrapped
# class still belongs in RandomVariable. Is this distinction
# necessary?
def primitive_cls(__init__):
  """Wraps class' __init__ so its continuation can be intercepted
  and its execution can be written to a stack.

  Apply this to decorate primitive classes.
  """
  def wrapped_fn(self, *args, **kwargs):
    global _INTERCEPT, _TRACE_CONTINUATION, _TRACE_STACK
    if '_INTERCEPT' in globals():
      _INTERCEPT(__init__, self, *args, **kwargs)
    else:
      __init__(self, *args, **kwargs)
    if '_TRACE_CONTINUATION' in globals() and '_TRACE_STACK' in globals():
      if _TRACE_CONTINUATION:
        parents = [v for v in list(args) + kwargs.values()
                   if hasattr(v, "name") and v.name in _TRACE_STACK]
        node = Node(self, __init__, args, kwargs, parents, [])
        _TRACE_STACK[self.name] = node
        for parent in parents:
          parent.children.append(node)
      else:
        _TRACE_STACK[self.name] = Node(self)
  return wrapped_fn
