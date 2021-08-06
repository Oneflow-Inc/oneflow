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

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft


def GenCartesianProduct(sets):
    assert isinstance(sets, Iterable)
    for set in sets:
        assert isinstance(set, Iterable)
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            if "gpu" in set:
                set.remove("gpu")
            if "cuda" in set:
                set.remove("cuda")
    return itertools.product(*sets)


def GenArgList(arg_dict):
    assert isinstance(arg_dict, OrderedDict)
    assert all([isinstance(x, list) for x in arg_dict.values()])
    sets = [arg_set for (_, arg_set) in arg_dict.items()]
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


def Coordinate2Index(coordinate, tensor_shape):
    if len(coordinate) != len(tensor_shape):
        raise "wrong coordinate or shape"
    idx = 0
    for (i, coor) in enumerate(coordinate):
        size_at_axis = coor
        for j in range(i + 1, len(tensor_shape)):
            size_at_axis *= tensor_shape[j]
        idx += size_at_axis
    return idx
