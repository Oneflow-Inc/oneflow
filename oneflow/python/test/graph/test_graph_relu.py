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
import os

import numpy as np

import oneflow
import oneflow.experimental as flow
import oneflow.python.framework.graph_build_util as graph_build_util


@unittest.skip(" nn.Graph cannnot run right now ")
class TestReluGraph(flow.unittest.TestCase):
    def test_relu_graph(test_case):
        data = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
        x = flow.tensor(data, dtype=flow.float32)

        MyRelu = flow.nn.ReLU()
        y_eager = MyRelu(x)
        print("eager out :", y_eager)

        class ReluGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()
                self.cc_relu = MyRelu

            def build(self, x):
                return self.cc_relu(x)

        relu_g = ReluGraph()
        y_lazy = relu_g(x)[0]
        print("lazy out :", y_lazy)
        test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


if __name__ == "__main__":
    unittest.main()
