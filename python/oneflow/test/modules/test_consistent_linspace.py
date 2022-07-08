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
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_linspace_int_impl(test_case, placement, sbp):
    start = random().to(int)
    end = start + random().to(int)
    steps = random(0, end - start).to(int)
    x = torch.linspace(start=start, end=end, steps=steps).to_global(
        placement=placement, sbp=sbp
    )
    return x


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_linspace_float_impl(test_case, placement, sbp):
    start = random()
    end = start + random()
    steps = random(0, end - start).to(int)
    x = torch.linspace(start=start, end=end, steps=steps).to_global(
        placement=placement, sbp=sbp
    )
    return x


class TestLinspace(flow.unittest.TestCase):
    @globaltest
    def test_linspace(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement):
                _test_linspace_int_impl(test_case, placement, sbp)
                _test_linspace_float_impl(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
