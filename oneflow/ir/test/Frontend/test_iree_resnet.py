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

from oneflow_iree.compiler import Runner
from flowvision.models import resnet50
import oneflow as flow
import oneflow.unittest
import unittest
import os
import numpy as np
import time

os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = "1"


def _test_iree_resnet_cpu(test_case):
    model = resnet50(pretrained=True)
    model.eval()

    class GraphModuleForIree(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model

        def build(self, x):
            return self.model(x)

    class GraphModuleForOFMLIR(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model

        def build(self, x):
            return self.model(x)

    func = Runner(GraphModuleForIree, return_numpy=True)
    input = flow.ones([1, 3, 224, 224])
    f = GraphModuleForOFMLIR()
    for iter in range(2):
        iree_output = func(input)
        graph_output = f(input)
        graph_output = graph_output.cpu().detach().numpy()
        # the rtol accumulate layer by layer
        test_case.assertTrue(
            np.allclose(iree_output, graph_output, rtol=1.0e-1, atol=1e-3)
        )


def _test_iree_resnet_cuda(test_case):
    model = resnet50(pretrained=True).cuda()
    model.eval()

    class GraphModuleForIree(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model

        def build(self, x):
            return self.model(x)

    class GraphModuleForOFMLIR(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model

        def build(self, x):
            return self.model(x)

    func = Runner(GraphModuleForIree, return_numpy=True)
    input = flow.ones([1, 3, 224, 224]).cuda()
    f = GraphModuleForOFMLIR()
    for iter in range(2):
        iree_output = func(input)
        graph_output = f(input)
        graph_output = graph_output.cpu().detach().numpy()
        # the rtol accumulate layer by layer
        test_case.assertTrue(
            np.allclose(iree_output, graph_output, rtol=1.0e-1, atol=1e-3)
        )


@flow.unittest.skip_unless_1n1d()
class TestIreeResnet(oneflow.unittest.TestCase):
    def test_iree_resnet_cpu(test_case):
        _test_iree_resnet_cpu(test_case)

    @unittest.skipUnless(oneflow.sysconfig.with_cuda(), "only test cpu cases")
    def test_iree_resnet_cuda(test_case):
        _test_iree_resnet_cuda(test_case)


if __name__ == "__main__":
    unittest.main()
