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

import os
import itertools
import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow

import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *


def _test_eager_boxing_normal_1d_exhaustive_testing(
    test_case, shape, in_device, out_device, in_device_list, out_device_list
):
    sbps = [
        flow.sbp.split(0),
        flow.sbp.split(1),
        flow.sbp.broadcast,
        flow.sbp.partial_sum,
    ]
    in_placement = flow.placement(type=in_device, ranks=in_device_list)
    out_placement = flow.placement(type=out_device, ranks=out_device_list)
    rand_tensor = random_tensor(len(shape), *shape, requires_grad=False).oneflow
    for elem in itertools.product(sbps, sbps):
        x = rand_tensor.to_global(placement=in_placement, sbp=elem[0])
        y = x.to_global(placement=out_placement, sbp=elem[1])
        test_case.assertTrue(np.allclose(y.numpy(), x.numpy(), 1e-3, 1e-3))


def _test_eager_boxing_symmetric_2d_exhaustive_testing(
    test_case, in_device, out_device
):
    sbps = [
        flow.sbp.split(0),
        flow.sbp.split(1),
        flow.sbp.broadcast,
        flow.sbp.partial_sum,
    ]
    nd_sbps = itertools.product(
        itertools.product(sbps, sbps), itertools.product(sbps, sbps)
    )
    shape = (8, 8, 16)
    in_placement = flow.placement(type=in_device, ranks=[[0, 1], [2, 3]])
    out_placement = flow.placement(type=out_device, ranks=[[0, 1], [2, 3]])
    rand_tensor = random_tensor(len(shape), *shape, requires_grad=False).oneflow
    for elem in nd_sbps:
        x = rand_tensor.to_global(placement=in_placement, sbp=elem[0])
        y = x.to_global(placement=out_placement, sbp=elem[1])
        test_case.assertTrue(np.allclose(y.numpy(), x.numpy(), 1e-3, 1e-3))


def _test_eager_boxing_1d_special_split_axis(
    test_case, in_device, out_device, in_device_list, out_device_list
):
    sbps = [
        flow.sbp.split(2),
        flow.sbp.split(3),
        flow.sbp.broadcast,
        flow.sbp.partial_sum,
    ]
    shape = (4, 4, 5, 7)
    in_placement = flow.placement(type=in_device, ranks=in_device_list)
    out_placement = flow.placement(type=out_device, ranks=out_device_list)
    rand_tensor = random_tensor(len(shape), *shape, requires_grad=False).oneflow
    for elem in itertools.product(sbps, sbps):
        x = rand_tensor.to_global(placement=in_placement, sbp=elem[0])
        y = x.to_global(placement=out_placement, sbp=elem[1])
        test_case.assertTrue(np.allclose(y.numpy(), x.numpy(), 1e-3, 1e-3))


def _test_eager_boxing_2d_special_split_axis(test_case, in_device, out_device):
    sbps = [
        flow.sbp.split(2),
        flow.sbp.split(4),
        flow.sbp.broadcast,
        flow.sbp.partial_sum,
    ]
    nd_sbps = itertools.product(
        itertools.product(sbps, sbps), itertools.product(sbps, sbps)
    )
    shape = (4, 8, 4, 8, 4)
    in_placement = flow.placement(type=in_device, ranks=[[0, 1], [2, 3]])
    out_placement = flow.placement(type=out_device, ranks=[[0, 1], [2, 3]])
    rand_tensor = random_tensor(len(shape), *shape, requires_grad=False).oneflow
    for elem in nd_sbps:
        x = rand_tensor.to_global(placement=in_placement, sbp=elem[0])
        y = x.to_global(placement=out_placement, sbp=elem[1])
        test_case.assertTrue(np.allclose(y.numpy(), x.numpy(), 1e-3, 1e-3))


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerBoxingSymmetricExhaustiveTesting(flow.unittest.TestCase):
    @globaltest
    def test_eager_boxing_normal_1d_exhaustive_testing(test_case):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(4, 4), (6, 8), (5, 7)]
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        arg_dict["in_device_list"] = [[0, 1], [1, 2, 3], [0, 1, 2, 3]]
        arg_dict["out_device_list"] = [[0, 1, 3], [0, 1, 2, 3]]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_normal_1d_exhaustive_testing(test_case, *arg)

    @globaltest
    def test_eager_boxing_symmetric_2d_exhaustive_testing(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_symmetric_2d_exhaustive_testing(test_case, *arg)


@flow.unittest.skip_unless_1n4d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestEagerBoxingSpecialSplitAxisExhaustiveTesting(flow.unittest.TestCase):
    @globaltest
    def test_eager_boxing_1d_special_split_axis(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        arg_dict["in_device_list"] = [[0, 1], [1, 2, 3], [0, 1, 2, 3]]
        arg_dict["out_device_list"] = [[0, 1, 3], [0, 1, 2, 3]]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_1d_special_split_axis(test_case, *arg)

    @globaltest
    def test_eager_boxing_2d_special_split_axis(test_case):
        arg_dict = OrderedDict()
        arg_dict["in_device"] = ["cpu", "cuda"]
        arg_dict["out_device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            _test_eager_boxing_2d_special_split_axis(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
