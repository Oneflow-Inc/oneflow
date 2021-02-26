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
from typing import Tuple

import numpy as np

import oneflow as flow
import oneflow.typing as tp


_counter = 0


def get_var_helper(shape):
    global _counter
    var = flow.get_variable(
        "x_" + str(_counter), shape=shape, initializer=flow.kaiming_initializer(shape)
    )
    _counter += 1
    return var


class TestContainer(flow.unittest.TestCase):
    def test_module_forward(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

            def forward(self, x):
                return x + self.w

        m1 = CustomModule(5)
        m2 = CustomModule(4)
        s = flow.nn.Sequential(m1, m2)

        test_case.assertEqual(s(1), 10)


if __name__ == "__main__":
    unittest.main()
