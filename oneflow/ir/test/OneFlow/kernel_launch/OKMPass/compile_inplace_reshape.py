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
import unittest
import numpy as np
import oneflow as flow
import oneflow.unittest


class graphToTest(flow.nn.Graph):
    def __init__(self):
        super().__init__()

    def build(self, arg0):
        arg0 = flow.reshape(arg0, shape=[3, 4])
        output0 = flow.relu(arg0)
        output0 = flow.reshape(output0, shape=[3, 4])
        output2 = flow.relu(output0)
        output2 = flow.gelu(output0)
        return output2


@flow.unittest.skip_unless_1n1d()
class TestInplaceVMRunGraphPass(flow.unittest.MLIRTestCase):
    def setUp(self):
        os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
        # os.environ["ONEFLOW_MLIR_ENABLE_IR_PRINTING"] = "1"
        os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
        # os.environ["ONEFLOW_RUN_GRAPH_BY_VM"] = "1"

    def test_inplace_reshape(test_case):
        a = flow.randn(12, 1)
        g = graphToTest()
        lazy_b = g(a)
        lazy_b = g(a)
        lazy_b = g(a)
        lazy_b = g(a)
        eager_b = graphToTest.build(None, a)
        test_case.assertTrue(
            np.allclose(
                eager_b.numpy(),
                lazy_b.numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
