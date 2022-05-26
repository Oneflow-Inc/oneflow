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
# RUN: python3 %s

from oneflow_iree.compiler import Runner
import oneflow as flow
import oneflow.unittest
import unittest
import numpy as np


class RELU(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = flow.nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class GraphModule(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.fw = RELU()

    def build(self, x):
        return self.fw(x)


def _test_check_iree_runner(test_case):
    func = Runner(GraphModule, return_numpy=True).cuda()
    # run on iree cuda backend
    input = flow.Tensor([-1.0, 1.0])
    output = func(input)
    test_case.assertTrue(np.allclose(output, [0., 1.]))
    # change input shape
    input = flow.Tensor([-1.0, 1.0, -1])
    output = func(input)
    test_case.assertTrue(np.allclose(output, [0., 1., 0.]))
    # change on iree cpu backend
    func = func.cpu()
    input = flow.Tensor([-1.0, 0.0, 1.0])
    output = func(input)
    test_case.assertTrue(np.allclose(output, [0., 0., 1.]))
    # change input shape
    input = flow.Tensor([-1, 1.0])
    output = func(input)
    test_case.assertTrue(np.allclose(output, [0., 1.]))



@flow.unittest.skip_unless_1n1d()
class TestCheckIreeRunner(oneflow.unittest.TestCase):
    def test_check_iree_runner(test_case):
        _test_check_iree_runner(test_case)


if __name__ == "__main__":
    unittest.main()
