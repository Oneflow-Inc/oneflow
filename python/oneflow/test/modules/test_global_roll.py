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

import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=True)
def _test_roll_impl(test_case, placement, sbp):
    shifts = (
        random(-100, 100).to(int).value(),
        random(-100, 100).to(int).value(),
        random(-100, 100).to(int).value(),
        random(-100, 100).to(int).value(),
    )
    dims = (0, 1, 2, 3)
    x_dims = [random(2, 4) * 8 for _ in range(4)]
    x = random_tensor(4, *x_dims)
    y = x.to_global(placement=placement, sbp=sbp)
    z = torch.roll(y, shifts, dims)
    return z


class TestRollGlobal(flow.unittest.TestCase):
    @globaltest
    def test_roll(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                _test_roll_impl(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
