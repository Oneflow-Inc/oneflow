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
# CHECK-NOT: oneflow.pad

import unittest
import numpy as np

import os

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_FORWARD_OPS"] = "1"
os.environ["ONEFLOW_MLIR_STDOUT"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_TIMING"] = "1"
os.environ["ONEFLOW_MLIR_PRINT_STATS"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_IR_PRINTING"] = "1"
import oneflow as flow
import oneflow.unittest
import oneflow.sysconfig


def do_pad_conv_graph(test_case, with_cuda, with_bias, with_nchw=True):
    if with_nchw:
        x = flow.randn(2, 3, 4, 5)
    else:
        x = flow.randn(2, 4, 5, 3)
    conv = flow.nn.Conv2d(3, 3, 2, 1, bias=with_bias)
    if with_cuda:
        x = x.cuda()
        conv.to("cuda")

    if with_nchw:
        pad_x = flow.nn.functional.pad(x, (1, 1, 1, 1))
    else:
        pad_x = flow.nn.functional.pad(x, (0, 0, 1, 1, 1, 1))
    eager_conv_x = conv(pad_x)

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.conv = conv

        def build(self, x):
            if with_nchw:
                pad_x = flow.nn.functional.pad(x, (1, 1, 1, 1))
            else:
                pad_x = flow.nn.functional.pad(x, (0, 0, 1, 1, 1, 1))
            return self.conv(pad_x)

    graph_to_run = GraphToRun()
    lazy_conv_x = graph_to_run(x)
    test_case.assertTrue(np.array_equal(eager_conv_x.numpy(), lazy_conv_x.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestFusePadConv(oneflow.unittest.TestCase):
    @unittest.skipUnless(oneflow.sysconfig.with_cuda(), "needs -DBUILD_CUDA=ON")
    def test_pad_conv_graph_cuda(test_case):
        do_pad_conv_graph(test_case, True, True)
        do_pad_conv_graph(test_case, True, False)
        do_pad_conv_graph(test_case, True, False, True)

    def test_pad_conv_graph_cpu(test_case):
        do_pad_conv_graph(test_case, False, True)
        do_pad_conv_graph(test_case, False, False)
        do_pad_conv_graph(test_case, False, False, True)


if __name__ == "__main__":
    unittest.main()
