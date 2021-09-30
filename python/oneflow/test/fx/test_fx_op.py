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
import oneflow.unittest
import numpy as np
import unittest
from oneflow.fx import symbolic_trace
from oneflow.test_utils.automated_test_util import *


def sort_op(x):
    return flow.sort(x)


def ones_like_op(x):
    return flow.ones_like(x)


def avg_pool2d(x):
    return flow.nn.functional.avg_pool2d(x, 2)


@flow.unittest.skip_unless_1n1d()
class TestFX(flow.unittest.TestCase):
    # def test_sort_op(test_case):
    #     gm : flow.fx.GraphModule = symbolic_trace(sort_op)
    #     print(gm.graph)
    #     input = flow.randn(3, 4)
    #     assert(np.allclose(gm(input)[0].numpy(), sort_op(input)[0].numpy(), equal_nan=True))

    # def test_ones_like_op(test_case):
    #     gm : flow.fx.GraphModule = symbolic_trace(ones_like_op)
    #     input = flow.randn(3, 4)
    #     assert(np.allclose(gm(input).numpy(), ones_like_op(input).numpy(), equal_nan=True))

    def test_avg_pool2d_op(test_case):
        gm: flow.fx.GraphModule = symbolic_trace(avg_pool2d)
        input = flow.randn(1, 1, 4, 4)
        assert np.allclose(gm(input).numpy(), avg_pool2d(input).numpy(), equal_nan=True)


if __name__ == "__main__":
    unittest.main()
