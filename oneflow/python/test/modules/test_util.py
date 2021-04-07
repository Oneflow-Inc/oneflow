"""
Copyright 2020 The OneFlow Authors. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import itertools
import os
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import oneflow as flow
import oneflow.typing as oft

import test_global_storage


def GenCartesianProduct(sets):
    assert isinstance(sets, Iterable)
    for set in sets:
        assert isinstance(set, Iterable)
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            if "gpu" in set:
                set.remove("gpu")
    return itertools.product(*sets)


def GenArgList(arg_dict):
    assert isinstance(arg_dict, OrderedDict)
    assert all([isinstance(x, list) for x in arg_dict.values()])
    sets = [arg_set for _, arg_set in arg_dict.items()]
    return GenCartesianProduct(sets)


def GenArgDict(arg_dict):
    return [dict(zip(arg_dict.keys(), x)) for x in GenArgList(arg_dict)]


class Args:
    def __init__(self, flow_args, tf_args=None):
        super().__init__()
        if tf_args is None:
            tf_args = flow_args
        self.flow_args = flow_args
        self.tf_args = tf_args

    def __str__(self):
        return "flow_args={} tf_args={}".format(self.flow_args, self.tf_args)

    def __repr__(self):
        return self.__str__()


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


def FlattenArray(input_array):
    output_array = list()
    for x in np.nditer(input_array):
        output_array.append(x.tolist())
    return output_array


def Array2Numpy(input_array, target_shape):
    return np.array(input_array).reshape(target_shape, order="C")


def Index2Coordinate(idx, tensor_shape):
    coordinate = []
    tmp = idx
    for i in range(len(tensor_shape) - 1, -1, -1):
        axis_size = tensor_shape[i]
        coor = tmp % axis_size
        coordinate.insert(0, int(coor))
        tmp = (tmp - coor) / axis_size
    return coordinate


def Coordinate2Index(coordinate, tensor_shape):
    if len(coordinate) != len(tensor_shape):
        raise "wrong coordinate or shape"
    idx = 0
    for i, coor in enumerate(coordinate):
        size_at_axis = coor
        for j in range(i + 1, len(tensor_shape)):
            size_at_axis *= tensor_shape[j]

        idx += size_at_axis
    return idx
