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
# CHECK: oneflow.transpose

import unittest
import numpy as np

import os

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_PREFER_NHWC"] = "1"

import oneflow as flow
import oneflow.unittest


def do_nhwc_bias_add(test_case, with_cuda):
    a = flow.randn(2, 3, 4, 5)
    b = flow.randn(3)
    if with_cuda:
        a = a.cuda()
        b = b.cuda()

    eager_bias_add_res = flow._C.bias_add(a, b, axis=1)

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, a, b):
            return flow._C.bias_add(a, b, axis=1)

    graph_to_run = GraphToRun()
    lazy_bias_add_res = graph_to_run(a, b)
    test_case.assertTrue(
        np.allclose(
            eager_bias_add_res.numpy(), lazy_bias_add_res.numpy(), rtol=1e-5, atol=1e-5
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestNhwcBiasAdd(oneflow.unittest.TestCase):
    def test_nhwc_bias_add_graph(test_case):
        import oneflow.sysconfig

        if oneflow.sysconfig.with_cuda():
            do_nhwc_bias_add(test_case, True)
        do_nhwc_bias_add(test_case, False)


if __name__ == "__main__":
    unittest.main()
