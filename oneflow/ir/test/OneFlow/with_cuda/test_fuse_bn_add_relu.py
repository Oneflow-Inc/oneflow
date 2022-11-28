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
# RUN: python3 %s | FileCheck %s
# CHECK-NOT: oneflow.bias_add

import unittest
import numpy as np

import os

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_NORMALIZATION_OPS"] = "1"
os.environ["ONEFLOW_MLIR_PRINT_STATS"] = "1"

import oneflow as flow
import oneflow.unittest
import oneflow.sysconfig


def do_bias_add_gelu_graph(test_case, with_cuda):
    def get_bn():
        return flow.nn.FusedBatchNorm3d(num_features=2, eps=1e-5, momentum=0.1).to(
            "cuda"
        )

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = get_bn()

        def build(self, x, addend):
            return self.m(x, addend=addend)

    graph_to_run = GraphToRun()
    x = flow.Tensor(np.random.randn(3, 2, 5, 8, 4)).to("cuda")
    addend = flow.Tensor(np.random.randn(3, 2, 5, 8, 4)).to("cuda")
    eager_res = get_bn()(x, addend=addend)
    lazy_res = graph_to_run(x, addend)
    test_case.assertTrue(np.array_equal(eager_res.numpy(), lazy_res.numpy()))


@flow.unittest.skip_unless_1n1d()
@unittest.skipUnless(oneflow.sysconfig.with_cuda(), "needs -DBUILD_CUDA=ON")
class TestBiasAddGelu(oneflow.unittest.TestCase):
    def test_bias_add_gelu_graph(test_case):
        do_bias_add_gelu_graph(test_case, True)


if __name__ == "__main__":
    unittest.main()
