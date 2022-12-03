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
os.environ["ONEFLOW_MLIR_GROUP_MATMUL"] = "1"
os.environ["ONEFLOW_MLIR_STDOUT"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_TIMING"] = "1"
os.environ["ONEFLOW_MLIR_PRINT_STATS"] = "1"

import oneflow as flow
import oneflow.unittest
import oneflow.sysconfig


def _matmul_bias(x, weight, bias):
    return flow._C.bias_add(
        flow._C.matmul(x, weight, transpose_b=True), bias, axis=len(x.shape) - 1
    )


def _matmul_bias1(x, w, bias):
    return flow._C.fused_matmul_bias(x, w, bias)


def do_fused_matmul_bias_graph(test_case, dev):
    x = np.random.uniform(low=-1, high=1, size=(8, 9))
    w = np.random.uniform(low=-1, high=1, size=(10, 9))
    bias = np.random.uniform(low=-1, high=1, size=(10))
    x = flow.from_numpy(x).to(dev).to(flow.float32)
    w = flow.from_numpy(w).to(dev).to(flow.float32)
    bias = flow.from_numpy(bias).to(dev).to(flow.float32)
    eager_res = _matmul_bias(x, w, bias) * 3

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x, w, bias):
            return (
                _matmul_bias1(x, w, bias)
                + _matmul_bias(x, w, bias)
                + _matmul_bias1(x, w, bias)
            )

    graph_to_run = GraphToRun()
    lazy_res = graph_to_run(x, w, bias)
    test_case.assertTrue(np.allclose(eager_res.numpy(), lazy_res.numpy()))


@flow.unittest.skip_unless_1n1d()
@unittest.skipUnless(oneflow.sysconfig.with_cuda(), "needs -DBUILD_CUDA=ON")
class TestBiasAddDropout(oneflow.unittest.TestCase):
    def test_fused_matmul_bias_graph(test_case):
        do_fused_matmul_bias_graph(test_case, "cuda")


if __name__ == "__main__":
    unittest.main()
