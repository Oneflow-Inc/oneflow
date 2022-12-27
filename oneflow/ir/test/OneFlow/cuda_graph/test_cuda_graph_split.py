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
# RUN: python3 %s
import unittest
import os

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
os.environ["ONEFLOW_KERNEL_ENABLE_CUDA_GRAPH"] = "1"

import oneflow as flow
import oneflow.unittest

def _test_okl_ops_with_cuda(test_case: flow.unittest.TestCase):
    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x):
            y = flow.relu(x)
            z = flow.tanh(y)
            return flow.sort(z)

    x = flow.Tensor([1, -1]).cuda()
    graph_to_run = GraphToRun()
    for i in range(10):
        lazy_relu = graph_to_run(x)



@flow.unittest.skip_unless_1n1d()
class TestOKLOps(flow.unittest.TestCase):
    @unittest.skipUnless(flow.sysconfig.with_cuda(), "only test cpu cases")
    def test_okl_ops_with_cuda(test_case):
        _test_okl_ops_with_cuda(test_case)


if __name__ == "__main__":
    unittest.main()

