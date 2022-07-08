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


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_interpolate_impl(test_case, ndim, placement, sbp):
    dims = [random(1, 3).to(int) * 8 for i in range(ndim)]
    x = random_tensor(ndim, *dims, requires_grad=False)
    x = x.to_global(placement=placement, sbp=sbp)
    y = torch.nn.functional.interpolate(
        x, scale_factor=2.0, mode=oneof("linear", "bicubic", "area")
    )
    return y


class TestInterpolateConsistent(flow.unittest.TestCase):
    @globaltest
    def test_interpolate(test_case):
        ndim = random(1, 5).to(int).value()
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=min(2, ndim)):
                _test_interpolate_impl(test_case, ndim, placement, sbp)


if __name__ == "__main__":
    unittest.main()
