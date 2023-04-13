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

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"

import oneflow as flow
import oneflow.unittest


class CastModule(flow.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, scale):
        # TODO: also support scale as a scalar, for instance: scale = 7.7
        return x.to(dtype=flow.float32) * scale


def do_relu_graph(test_case, data, with_cuda):
    x = flow.tensor(data, dtype=flow.int64)
    scale = flow.tensor([7.7], dtype=flow.float32)
    if with_cuda:
        x = x.cuda()
        scale = scale.cuda()
    module_to_run = CastModule()
    y_eager = module_to_run(x, scale)

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.fw = module_to_run

        def build(self, x, scale):
            return self.fw(x, scale)

    graph_to_run = GraphToRun()
    y_lazy = graph_to_run(x, scale)
    test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestFuseCastScale(oneflow.unittest.TestCase):
    def test_relu_graph(test_case):
        import oneflow.sysconfig

        if oneflow.sysconfig.with_cuda():
            do_relu_graph(test_case, np.array([2.0, 1.0, 0.0, -1.0, -2.0]), True)
        do_relu_graph(
            test_case,
            np.array([[2.0, 1.0, 0.0, -1.0, -2.0], [2.0, 1.0, 0.0, -1.0, -2.0]]),
            False,
        )


if __name__ == "__main__":
    unittest.main()
