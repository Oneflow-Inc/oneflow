import itertools
from collections.abc import Iterable

def GenCartesianProduct(sets):
  assert isinstance(sets, Iterable)
  for set in sets:
    assert isinstance(set, Iterable)
  return itertools.product(*sets) 
