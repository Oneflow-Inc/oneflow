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
# RUN: python3 -m oneflow.test_utils.throttle --with-cuda=%with_cuda python3 %s

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/..")

import unittest
import numpy as np

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_IR_PRINTING"] = "1"

import oneflow as flow
import oneflow.unittest
import oneflow.nn as nn


def _test_okl_conv2d(test_case):
    x = flow.randn(2, 3, 224, 224)
    x = x.cuda()
    conv = nn.Conv2d(3, 2, kernel_size=7, stride=2, padding=3, bias=False).to("cuda")

    eager_res = conv(x)

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()

            self.net = conv

        def build(self, x):
            return self.net(x)

    graph_to_run = GraphToRun()
    lazy_res = graph_to_run(x)
    test_case.assertTrue(
        np.allclose(eager_res.numpy(), lazy_res.numpy(), rtol=1e-4, atol=1e-4)
    )


@flow.unittest.skip_unless_1n1d()
class TestOKLResNet(flow.unittest.TestCase):
    @unittest.skipUnless(flow.sysconfig.with_cuda(), "only test cpu cases")
    def test_okl_conv2d(test_case):
        _test_okl_conv2d(test_case)


if __name__ == "__main__":
    unittest.main()
