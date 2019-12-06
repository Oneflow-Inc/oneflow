import itertools
from collections.abc import Iterable
import os
import numpy as np


def GenCartesianProduct(sets):
    assert isinstance(sets, Iterable)
    for set in sets:
        assert isinstance(set, Iterable)
    return itertools.product(*sets)


def GenArgList(args):
    return GenCartesianProduct(args)


def GetSavePath():
    return "/tmp/op_unit_test/"


# Save func for flow.watch
def Save(name):
    path = GetSavePath()
    if not os.path.isdir(path):
        os.makedirs(path)

    def _save(x):
        np.save(os.path.join(path, name), x)

    return _save
