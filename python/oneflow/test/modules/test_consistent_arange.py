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
import numpy as np
import unittest

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


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


if __name__ == "__main__":
    unittest.main()
