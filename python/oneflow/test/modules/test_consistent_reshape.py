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
def test_reshape_impl(test_case, pair, placement, sbp):
    shape, to_shape = pair
    x = random_tensor(len(shape), *shape)
    y = x.to_global(placement=placement, sbp=sbp)
    z = y.reshape(to_shape)
    return z


class TestReshapeConsistent(flow.unittest.TestCase):
    @globaltest
    def test_reshape(test_case):
        shape_pairs = [
            ((8, 16), (8 * 16,)),
            ((8, 16), (8 * 4, 4)),
            ((8, 16, 24), (64, 6, 8)),
            ((8, 16), (64, 1, -1)),
            ((8, 16), (-1,)),
        ]
        for pair in shape_pairs:
            for placement in all_placement():
                for sbp in all_sbp(placement, max_dim=len(pair[0])):
                    test_reshape_impl(test_case, pair, placement, sbp)


if __name__ == "__main__":
    unittest.main()
