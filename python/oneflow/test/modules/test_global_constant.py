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

import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import GenArgDict


def _test_global_new_full(test_case, shape, full_value, placement, sbp):

    np_res = np.full(shape, full_value)
    x = flow.ones(shape)
    y = x.new_full(shape, full_value, placement=placement, sbp=sbp)

    test_case.assertEqual(y.shape, flow.Size(shape))
    test_case.assertEqual(y.sbp, sbp)
    test_case.assertEqual(y.placement, placement)

    y = y.to_global(
        placement=placement,
        sbp=[flow.sbp.broadcast for _ in range(len(placement.ranks.shape))],
    ).to_local()
    test_case.assertTrue(np.array_equal(y.numpy(), np_res))


def _test_global_graph_new_full(test_case, shape, full_value, placement, sbp):

    np_res = np.full(shape, full_value)

    class GlobalNewFullGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()

        def build(self,):
            x = flow.ones(shape)
            y = x.new_full(shape, full_value, placement=placement, sbp=sbp)
            return y

    model = GlobalNewFullGraph()
    y = model()

    test_case.assertEqual(y.shape, flow.Size(shape))
    test_case.assertEqual(y.sbp, sbp)
    test_case.assertEqual(y.placement, placement)

    y = y.to_global(
        placement=placement,
        sbp=[flow.sbp.broadcast for _ in range(len(placement.ranks.shape))],
    ).to_local()
    test_case.assertTrue(np.array_equal(y.numpy(), np_res))


def _test_global_constant(test_case, func, shape, placement, sbp):
    func2 = None
    if func == "ones":
        func = flow.ones
        np_res = np.ones(shape)
    elif func == "zeros":
        func = flow.zeros
        np_res = np.zeros(shape)
    elif func == "new_zeros":
        func = flow.zeros
        np_res = np.zeros(shape)
        func2 = flow.new_zeros
    else:
        raise NotImplementedError

    x = func(*shape, placement=placement, sbp=sbp)
    if func2:
        x = func2(x)

    test_case.assertEqual(x.shape, flow.Size(shape))
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)

    x = x.to_global(
        placement=placement,
        sbp=[flow.sbp.broadcast for _ in range(len(placement.ranks.shape))],
    ).to_local()
    test_case.assertTrue(np.array_equal(x.numpy(), np_res))


def _test_graph_constant(test_case, func, shape, placement, sbp):
    func2 = None
    if func == "ones":
        func = flow.ones
        np_res = np.ones(shape)
    elif func == "zeros":
        func = flow.zeros
        np_res = np.zeros(shape)
    elif func == "new_zeros":
        func = flow.zeros
        np_res = np.zeros(shape)
        func2 = flow.new_zeros
    else:
        raise NotImplementedError

    class GlobalConstantGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()

        def build(self):
            x = func(*shape, placement=placement, sbp=sbp)
            if func2:
                x = func2(x)
            return x

    model = GlobalConstantGraph()
    x = model()

    test_case.assertEqual(x.shape, flow.Size(shape))
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)

    x = x.to_global(
        placement=placement,
        sbp=[flow.sbp.broadcast for _ in range(len(placement.ranks.shape))],
    ).to_local()
    test_case.assertTrue(np.array_equal(x.numpy(), np_res))


class TestConstantGlobal(flow.unittest.TestCase):
    @globaltest
    def test_constant_global(test_case):
        shapes = [(8,), (8, 8,), (8, 8, 8)]
        functions = [
            "ones",
            "zeros",
            "new_zeros",
        ]
        for func in functions:
            for shape in shapes:
                for placement in all_placement():
                    for sbp in all_sbp(
                        placement, max_dim=len(shape), except_partial_sum=True
                    ):
                        _test_global_constant(test_case, func, shape, placement, sbp)

        full_values = [2, 3, 4]
        for full_value in full_values:
            for shape in shapes:
                for placement in all_placement():
                    for sbp in all_sbp(placement, max_dim=len(shape),):
                        _test_global_new_full(
                            test_case, shape, full_value, placement, sbp
                        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n2d()
    def test_constant_graph(test_case):
        arg_dict = OrderedDict()
        arg_dict["func"] = ["ones", "zeros", "new_zeros"]
        arg_dict["shape"] = [(8,), (8, 8,), (8, 8, 8)]
        arg_dict["placement"] = [
            # 1d
            flow.placement("cpu", ranks=[0, 1]),
            flow.placement("cuda", ranks=[0, 1]),
            # 2d
            flow.placement("cpu", ranks=[[0, 1],]),
            flow.placement("cuda", ranks=[[0, 1],]),
        ]

        for args in GenArgDict(arg_dict):
            func = args["func"]
            shape = args["shape"]
            placement = args["placement"]
            for sbp in all_sbp(placement, max_dim=len(shape), except_partial_sum=True):
                _test_graph_constant(test_case, func, shape, placement, sbp)
        full_values = [2, 3, 4]
        shapes = [(8,), (8, 8,), (8, 8, 8)]
        for full_value in full_values:
            for shape in shapes:
                for placement in all_placement():
                    for sbp in all_sbp(placement, max_dim=len(shape)):
                        _test_global_graph_new_full(
                            test_case, shape, full_value, placement, sbp
                        )


if __name__ == "__main__":
    unittest.main()
