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


@autotest(n=1, check_graph=False, auto_backward=False)
def _test_tensor_new(test_case, placement, sbp):
    x = random_tensor(1, 64).to_global(placement=placement, sbp=sbp).oneflow
    y = x.new()
    test_case.assertTrue(x.dtype == y.dtype)
    for x_sbp, y_sbp in zip(x.sbp, y.sbp):
        test_case.assertTrue(x_sbp == y_sbp)
    test_case.assertTrue(x.placement == y.placement)

    y = x.new(1, 2, 3)
    test_case.assertTrue(list(y.shape) == [1, 2, 3])
    test_case.assertTrue(x.dtype == y.dtype)
    for x_sbp, y_sbp in zip(x.sbp, y.sbp):
        test_case.assertTrue(x_sbp == y_sbp)
    test_case.assertTrue(x.placement == y.placement)

    y = x.new([1, 2, 3])
    test_case.assertTrue(list(y.shape) == [3])
    test_case.assertTrue(x.dtype == y.dtype)
    for x_sbp, y_sbp in zip(x.sbp, y.sbp):
        test_case.assertTrue(x_sbp == y_sbp)
    test_case.assertTrue(x.placement == y.placement)


class TestTensorNew(flow.unittest.TestCase):
    @globaltest
    def test_tensor_new(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, valid_split_axis=0):
                _test_tensor_new(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
