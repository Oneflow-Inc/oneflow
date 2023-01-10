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

import oneflow as flow
import numpy as np
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import GenArgDict


def _test_global_randn(test_case, shape, placement, sbp):
    x1 = flow.randn(*shape, placement=placement, sbp=sbp)
    x2 = flow.randn(*shape, placement=placement, sbp=sbp)
    test_case.assertTrue(not np.allclose(x1.numpy(), x2.numpy(), atol=1e-4, rtol=1e-4))
    test_case.assertEqual(x1.shape, flow.Size(shape))
    test_case.assertEqual(x1.sbp, sbp)
    test_case.assertEqual(x1.placement, placement)


def _test_different_dtype(test_case, shape, placement, sbp):
    x1 = flow.randn(*shape, dtype=flow.float32, placement=placement, sbp=sbp)
    x2 = flow.randn(*shape, dtype=flow.float64, placement=placement, sbp=sbp)
    test_case.assertTrue(not np.allclose(x1.numpy(), x2.numpy(), atol=1e-4, rtol=1e-4))
    test_case.assertEqual(x1.shape, flow.Size(shape))


def _test_backward(test_case, shape, placement, sbp):
    x = flow.randn(*shape, placement=placement, sbp=sbp, requires_grad=True)
    y = x.sum()
    y.backward()
    test_case.assertTrue(
        np.allclose(np.ones(shape), x.grad.numpy(), atol=1e-4, rtol=1e-4)
    )


def _test_with_generator(test_case, shape, placement, sbp):
    gen = flow.Generator()
    gen.manual_seed(0)
    y1 = flow.randn(*shape, placement=placement, sbp=sbp, generator=gen)
    gen.manual_seed(0)
    y2 = flow.randn(*shape, placement=placement, sbp=sbp, generator=gen)
    test_case.assertTrue(np.allclose(y1.numpy(), y2.numpy(), atol=1e-4, rtol=1e-4))


def _test_randn_tuple_shape(test_case, shape, placement, sbp):
    y1 = flow.randn(*shape, placement=placement, sbp=sbp)
    y2 = flow.randn(*shape, placement=placement, sbp=sbp)

    test_case.assertTrue(not np.array_equal(y1.numpy(), y2.numpy()))
    test_case.assertTrue(shape == y1.shape)


def _test_graph_randn(test_case, shape, placement, sbp):
    class GlobalRandnGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()

        def build(self):
            x = flow.randn(*shape, placement=placement, sbp=sbp)
            return x

    model = GlobalRandnGraph()
    x = model()

    test_case.assertEqual(x.shape, flow.Size(shape))
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


class TestRandnGlobal(flow.unittest.TestCase):
    @globaltest
    def test_randn_global(test_case):
        shapes = [(8,), (8, 8,), (8, 8, 8)]
        for shape in shapes:
            for placement in all_placement():
                for sbp in all_sbp(
                    placement, max_dim=len(shape), except_partial_sum=True
                ):
                    _test_global_randn(test_case, shape, placement, sbp)
                    _test_different_dtype(test_case, shape, placement, sbp)
                    _test_backward(test_case, shape, placement, sbp)
                    _test_with_generator(test_case, shape, placement, sbp)
                    _test_randn_tuple_shape(test_case, shape, placement, sbp)

    @flow.unittest.skip_unless_1n2d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @globaltest
    def test_randn_graph(test_case):
        arg_dict = OrderedDict()
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
            shape = args["shape"]
            placement = args["placement"]
            for sbp in all_sbp(placement, max_dim=len(shape), except_partial_sum=True):
                _test_graph_randn(test_case, shape, placement, sbp)


if __name__ == "__main__":
    unittest.main()
