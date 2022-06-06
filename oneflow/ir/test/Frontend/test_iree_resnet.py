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

from google.protobuf import text_format


os.environ["ONEFLOW_MLIR_ENABLE_INFERENCE_OPTIMIZATION"] = "1"


def _test_iree_resnet_cpu(test_case):
    model = resnet50(pretrained=True)
    model.eval()

    class GraphModule(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model

        def build(self, x):
            return self.model(x)

    for iter in range(3):
        print("======== in cpu iter" + str(iter + 1))
        func = Runner(GraphModule, return_numpy=True)
        data = np.ones([1, 3, 224, 224]).astype(np.float32)
        input = flow.tensor(data, requires_grad=False)
        iree_output = func(input)
        f = GraphModule()
        start_time = time.time()
        graph_output = f(input)
        gap = time.time() - start_time
        print("graph cost: " + str(gap))
        test_case.assertTrue(
            np.allclose(iree_output, graph_output.cpu().detach().numpy(), rtol=1.0e-1)
        )


def _test_iree_resnet_cuda(test_case):
    model = resnet50(pretrained=True).cuda()
    model.eval()

    class GraphModule(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = model

        def build(self, x):
            return self.model(x)

    for iter in range(3):
        print("======== in cuda iter" + str(iter + 1))
        func = Runner(GraphModule, return_numpy=True).cuda()
        data = np.ones([1, 3, 224, 224]).astype(np.float32)
        input = flow.tensor(data, requires_grad=False).cuda()
        iree_output = func(input)
        start_time = time.time()
        graph_output = GraphModule()(input)
        gap = time.time() - start_time
        print("graph cost: " + str(gap))
        test_case.assertTrue(
            np.allclose(iree_output, graph_output.cpu().detach().numpy(), rtol=1.0e-1)
        )


@flow.unittest.skip_unless_1n1d()
class TestIreeResnet(oneflow.unittest.TestCase):
    def test_iree_resnet_cpu(test_case):
        _test_iree_resnet_cpu(test_case)

    def test_iree_resnet_cuda(test_case):
        _test_iree_resnet_cuda(test_case)


if __name__ == "__main__":
    unittest.main()
