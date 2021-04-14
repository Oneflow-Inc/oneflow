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
import collections.abc
from itertools import repeat
import unittest
from typing import Tuple, Union
import time

import numpy as np

import oneflow as flow
import oneflow.typing as tp


def np_relu(np_arr):
    return np.where(np_arr > 0, np_arr, 0)


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_matmul_speed_eager(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())
        func_config.default_placement_scope(flow.scope.placement("gpu", "0:0"))

        @flow.global_function(function_config=func_config)
        def job():
            op1 = (
                flow.builtin_op("constant")
                .Output("out")
                .Attr("is_floating_value", True)
                .Attr("floating_value", 3.1)
                .Attr("dtype", flow.float32)
                .Attr("shape", [1000, 1000])
                .Build()
            )
            op2 = (
                flow.builtin_op("matmul")
                .Input("a")
                .Input("b")
                .Attr("transpose_a", False)
                .Attr("transpose_b", False)
                .Output("out")
                .Build()
            )

            for _ in range(100):
                y = op1()[0]
                y = op2(y, y)[0]

        job()


if __name__ == "__main__":
    unittest.main()
