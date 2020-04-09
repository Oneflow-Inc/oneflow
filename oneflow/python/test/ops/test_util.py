import os
import numpy as np
import itertools
from collections import OrderedDict
from collections.abc import Iterable

import oneflow as flow


def GenCartesianProduct(sets):
    assert isinstance(sets, Iterable)
    for set in sets:
        assert isinstance(set, Iterable)
    return itertools.product(*sets)


def GenArgList(arg_dict):
    assert isinstance(arg_dict, OrderedDict)
    sets = [arg_set for _, arg_set in arg_dict.items()]
    return GenCartesianProduct(sets)


def GetSavePath():
    return "./log/op_unit_test/"


# Save func for flow.watch
def Save(name):
    path = GetSavePath()
    if not os.path.isdir(path):
        os.makedirs(path)

    def _save(x):
        np.save(os.path.join(path, name), x.ndarray())

    return _save


type_name_to_flow_type = {
    "float16": flow.float16,
    "float32": flow.float32,
    "double": flow.double,
    "int8": flow.int8,
    "int32": flow.int32,
    "int64": flow.int64,
    "char": flow.char,
    "uint8": flow.uint8,
}

type_name_to_np_type = {
    "float16": np.float16,
    "float32": np.float32,
    "double": np.float64,
    "int8": np.int8,
    "int32": np.int32,
    "int64": np.int64,
    "char": np.byte,
    "uint8": np.uint8,
}
