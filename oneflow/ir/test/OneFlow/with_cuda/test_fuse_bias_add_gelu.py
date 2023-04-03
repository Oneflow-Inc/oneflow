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
# CHECK-NOT: oneflow.bias_add

import unittest
import numpy as np

import os

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_STDOUT"] = "1"
import oneflow as flow
import oneflow.unittest
import oneflow.sysconfig


def do_bias_add_gelu_graph(test_case, with_cuda):
    x = flow.randn(2, 3, 4, 5)
    bias = flow.randn(5)
    gelu = flow.nn.GELU()
    if with_cuda:
        x = x.cuda()
        bias = bias.to("cuda")
        gelu.to("cuda")

    eager_res = gelu(flow._C.bias_add(x, bias, axis=3))

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.gelu = gelu

        def build(self, x, bias):
            return self.gelu(flow._C.bias_add(x, bias, axis=3))

    graph_to_run = GraphToRun()
    lazy_res = graph_to_run(x, bias)
    test_case.assertTrue(np.array_equal(eager_res.numpy(), lazy_res.numpy()))


@flow.unittest.skip_unless_1n1d()
@unittest.skipUnless(oneflow.sysconfig.with_cuda(), "needs -DBUILD_CUDA=ON")
class TestBiasAddGelu(oneflow.unittest.TestCase):
    def test_bias_add_gelu_graph(test_case):
        do_bias_add_gelu_graph(test_case, True)


if __name__ == "__main__":
    unittest.main()
