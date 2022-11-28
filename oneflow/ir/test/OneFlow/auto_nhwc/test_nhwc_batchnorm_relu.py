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


def do_nhwc_bacth_norm(test_case, with_cuda):
    x = flow.randn(2, 3, 4, 5)
    bn = flow.nn.BatchNorm2d(3)
    if with_cuda:
        x = x.cuda()
        bn.to("cuda")

    eager_batch_norm_res = flow.relu(bn(x))

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = bn

        def build(self, x):
            return flow.relu(self.m(x))

    graph_to_run = GraphToRun()
    lazy_batch_norm_res = graph_to_run(x)
    test_case.assertTrue(
        np.allclose(
            eager_batch_norm_res.numpy(),
            lazy_batch_norm_res.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestNhwcConv(oneflow.unittest.TestCase):
    def test_nhwc_conv_graph(test_case):
        import oneflow.sysconfig

        if oneflow.sysconfig.with_cuda():
            do_nhwc_bacth_norm(test_case, True)
        # do_nhwc_bacth_norm(test_case, False)


if __name__ == "__main__":
    unittest.main()
