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


@autotest(n=1, check_graph=False)
def _test_reflection_pad2d_impl(test_case, padding, placement, sbp):
    m = torch.nn.ReflectionPad2d(padding=padding)
    dims = [random(2, 4) * 8 for _ in range(4)]
    x = random_tensor(4, *dims)
    y = x.to_global(placement=placement, sbp=sbp)
    z = m(y)
    return z


class TestReflectionPad2dGlobal(flow.unittest.TestCase):
    @globaltest
    def test_reflection_pad2d(test_case):
        padding = [
            (2, 2, 1, 1),
            1,
            (1, 0, 1, 0),
            (0, 1, 0, 1),
        ]
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=4):
                for pad in padding:
                    _test_reflection_pad2d_impl(test_case, pad, placement, sbp)


if __name__ == "__main__":
    unittest.main()
