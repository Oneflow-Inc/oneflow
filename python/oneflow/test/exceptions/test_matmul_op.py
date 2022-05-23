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


@flow.unittest.skip_unless_1n1d()
class TestMatmulOp(flow.unittest.TestCase):
    def test_matmul_error_msg(test_case):
        x = flow.randn((3, 4))
        y = flow.randn((4, 5, 6))

        res = flow.matmul(x, y)
        # with test_case.assertRaises(RuntimeError) as ctx:
        #     res = flow.matmul(x, y)
        #     test_case.assertTrue("" in str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
