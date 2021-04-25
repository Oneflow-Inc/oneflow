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
import oneflow as flow
import unittest
import numpy as np


# TODO(): OnesLike module
def ones_like(x):
    op = flow.builtin_op("ones_like").Input("like").Output("out").Build()
    return op(x)[0]


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_ones_like_case1(test_case):
        x = flow.Tensor(2, 3)
        y = ones_like(x)
        test_case.assertTrue(y.dtype is flow.float32)
        test_case.assertTrue(y.shape == x.shape)
        # TODO(): Use mirrored mode
        # test_case.assertTrue(y.device == x.device)

        y_numpy = np.ones_like(x.numpy())
        test_case.assertTrue(np.array_equal(y.numpy(), y_numpy))

    def test_ones_like_case2(test_case):
        x = flow.Tensor(2, 3, dtype=flow.int)
        y = ones_like(x)
        test_case.assertTrue(y.dtype is flow.int)
        test_case.assertTrue(y.shape == x.shape)
        # TODO(): Use mirrored mode
        # test_case.assertTrue(y.device == x.device)

        y_numpy = np.ones_like(x.numpy())
        test_case.assertTrue(np.array_equal(y.numpy(), y_numpy))

    def test_ones_like_case3(test_case):
        # TODO(): Test gpu device
        pass


if __name__ == "__main__":
    unittest.main()
