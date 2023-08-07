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
# RUN: python3 -m oneflow.test_utils.throttle --with-cuda=%with_cuda python3 %s | FileCheck %s
# CHECK: jit

import unittest
import numpy as np

import os


import oneflow as flow
import oneflow.unittest


class MatMulModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = flow.nn.Parameter(flow.Tensor(5, 10))
        self.b = flow.nn.Parameter(flow.Tensor(10))

    def forward(self, x):
        return flow.matmul(x, self.w) + self.b


def do_matmul_graph(test_case, with_cuda=False):
    x = flow.randn(2, 5)
    module_to_run = MatMulModule()
    if with_cuda:
        x = x.cuda()
        module_to_run = module_to_run.to("cuda")
    y_eager = module_to_run(x)

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.fw = module_to_run

        def build(self, x):
            return self.fw(x)

    graph_to_run = GraphToRun()
    y_lazy = graph_to_run(x)
    test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestFuseCastScale(oneflow.unittest.MLIRTestCase):
    def setUp(self):
        os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
        os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"

    def test_relu_graph(test_case):
        import oneflow.sysconfig

        if oneflow.sysconfig.with_cuda():
            do_matmul_graph(test_case, True)

        do_matmul_graph(test_case)


if __name__ == "__main__":
    unittest.main()
