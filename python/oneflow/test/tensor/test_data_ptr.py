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


class TestDataPtr(unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_equality(test_case):
        x = flow.ones(2, 3)
        y = flow.ones(2, 3)
        test_case.assertNotEqual(x.data_ptr(), y.data_ptr())

        test_case.assertEqual(x.data_ptr(), x.data.data_ptr())

        x_ptr = x.data_ptr()
        x[:] = 2
        test_case.assertEqual(x_ptr, x.data_ptr())

    @flow.unittest.skip_unless_1n2d()
    def test_global_tensor(test_case):
        x = flow.randn(
            2, 3, placement=flow.placement.all("cpu"), sbp=flow.sbp.broadcast
        )
        test_case.assertEqual(x.data_ptr(), x.to_local().data_ptr())


if __name__ == "__main__":
    unittest.main()
