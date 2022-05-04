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
from oneflow.test_utils.automated_test_util import *

from oneflow.test_utils.test_util import GenArgDict
import math


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_arange_with_random_data(test_case, placement, sbp):
    start = random(0, 10).to(int).value()
    end = start + random(0, 10).to(int).value()
    step = random(1, max(2, end - start)).to(int).value()
    start = start * 8
    end = end * 8
    x = torch.arange(start=start, end=end, step=step)
    x.oneflow = flow.arange(
        start=start, end=end, step=step, placement=placement, sbp=sbp
    )
    return x


@autotest(n=1, auto_backward=True, check_graph=False)
def _test_arange_with_float_delta(test_case, placement, sbp):
    start = random(0, 10).to(int).value()
    end = start + random(0, 10).to(int).value()
    step = random(1, max(2, end - start)).to(float).value()
    start = start * 8
    end = end * 8
    x = torch.arange(start=start, end=end, step=step, requires_grad=True)
    x.oneflow = flow.arange(
        start=start,
        end=end,
        step=step,
        placement=placement,
        sbp=sbp,
        requires_grad=True,
    )
    return x


class TestArange(flow.unittest.TestCase):
    @globaltest
    def test_arange(test_case):
        for placement in all_placement():
            # arange does not support split and partial_sum currently.
            for sbp in all_sbp(
                placement, max_dim=1, except_split=True, except_partial_sum=True
            ):
                _test_arange_with_random_data(test_case, placement, sbp)
                _test_arange_with_float_delta(test_case, placement, sbp)


@autotest(n=1, check_graph=False)
def _test_consistent_arange(test_case, start, end, step, placement, sbp):
    if (math.ceil((end - start) / step)) % 2 == 1:
        end = end + step
    x = flow.arange(start, end, step, placement=placement, sbp=sbp)
    y1 = x.to_global(placement=placement, sbp=sbp)
    y2 = np.arange(start, end, step)
    test_case.assertTrue(np.allclose(y1.numpy(), y2, atol=1e-4, rtol=1e-4))
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


@autotest(n=1, check_graph=False)
def _test_graph_arange(test_case, start, end, step, placement, sbp):
    class ConsistentArangeGraph(flow.nn.Graph):
        def __init__(self,):
            super().__init__()

        def build(self):
            x = flow.arange(start, end, step, placement=placement, sbp=sbp)
            return x

    model = ConsistentArangeGraph()
    x = model()
    y = np.arange(start, end, step)
    test_case.assertTrue(np.allclose(x.numpy(), y, atol=1e-4, rtol=1e-4))
    test_case.assertEqual(x.sbp, sbp)
    test_case.assertEqual(x.placement, placement)


class TestArangeConsistent(flow.unittest.TestCase):
    @globaltest
    def test_arange_consistent(test_case):
        arg_dict = OrderedDict()
        arg_dict["start"] = [i for i in range(1, 5, 1)]
        arg_dict["end"] = [i for i in range(10, 50, 10)]
        arg_dict["step"] = [i for i in range(1, 5, 1)]
        for args in GenArgDict(arg_dict):
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=1, except_partial_sum=True):
                    _test_consistent_arange(
                        test_case, **args, placement=placement, sbp=sbp
                    )

    @flow.unittest.skip_unless_1n2d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @globaltest
    def test_arange_graph(test_case):
        arg_dict = OrderedDict()
        arg_dict["start"] = [i for i in range(1, 5, 1)]
        arg_dict["end"] = [i for i in range(10, 30, 10)]
        arg_dict["step"] = [i for i in range(1, 5, 1)]
        arg_dict["placement"] = [
            # 1d
            flow.placement("cpu", ranks=[0, 1]),
            flow.placement("cuda", ranks=[0, 1]),
            # 2d
            flow.placement("cpu", ranks=[[0, 1],]),
            flow.placement("cuda", ranks=[[0, 1],]),
        ]
        for args in GenArgDict(arg_dict):
            start = args["start"]
            end = args["end"]
            step = args["step"]
            if (math.ceil((end - start) / step)) % 2 == 1:
                end = end + step
            placement = args["placement"]
            for sbp in all_sbp(placement, max_dim=1, except_partial_sum=True):
                _test_graph_arange(test_case, start, end, step, placement, sbp)


if __name__ == "__main__":
    unittest.main()
