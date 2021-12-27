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

from test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


@flow.unittest.skip_unless_1n1d()
class TestLinspace(flow.unittest.TestCase):
    @autotest(n=30, auto_backward=False, rtol=1e-5, atol=1e-5, check_graph=True)
    def test_linspace_int_with_random_data(test_case):
        start = random().to(int)
        end = start + random().to(int)
        steps = random(0, end - start).to(int)
        x = torch.linspace(start=start, end=end, steps=steps)
        device = random_device()
        x.to(device)
        return x

    @autotest(n=30, auto_backward=False, rtol=1e-5, atol=1e-5, check_graph=True)
    def test_linspace_float_with_random_data(test_case):
        start = random()
        end = start + random()
        steps = random(0, end - start).to(int)
        x = torch.linspace(start=start, end=end, steps=steps)
        device = random_device()
        x.to(device)
        return x

    def test_consistent_naive(test_case):
        placement = flow.placement("cpu", {0: [0]})
        sbp = (flow.sbp.broadcast,)
        x = flow.linspace(start=0, end=10, steps=2, placement=placement, sbp=sbp)
        test_case.assertEqual(x.sbp, sbp)
        test_case.assertEqual(x.placement, placement)


if __name__ == "__main__":
    unittest.main()
