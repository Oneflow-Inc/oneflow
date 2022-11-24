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

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + "/..")

import unittest
import numpy as np


import oneflow as flow
import oneflow.unittest
import time

from networks.resnet50 import resnet50


def _test_okl_resnet(test_case):
    x = flow.randn(2, 3, 224, 224)
    resnet = resnet50()
    x = x.cuda()
    resnet.to("cuda")

    eager_res = resnet(x)

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.resnet = resnet

        def build(self, x):
            return self.resnet(x)

    graph_to_run = GraphToRun()
    lazy_res = graph_to_run(x)
    test_case.assertTrue(
        np.allclose(eager_res.numpy(), lazy_res.numpy(), rtol=1e-4, atol=1e-4)
    )

def _test_okl_resnet_in_okl(test_case, warm_times:int=3, test_times:int=10):
    os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
    os.environ["ONEFLOW_MLIR_FUSE_KERNEL_LAUNCH"] = "1"
    x = flow.randn(2, 3, 224, 224).cuda()
    resnet = resnet50().to("cuda")

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.resnet = resnet

        def build(self, x):
            return self.resnet(x)
    graph_to_run = GraphToRun()

    [graph_to_run(x) for _ in range(warm_times)]

    time_start = time.time()
    [graph_to_run(x) for _ in range(test_times)]
    print(f'''
OKL:
 - warm times: {warm_times}
 - test times: {test_times}
 - single cost: {(time.time() - time_start)/test_times:f}
    ''')

def _test_okl_resnet_in_eager(test_case, warm_times:int=3, test_times:int=10):
    x = flow.randn(2, 3, 224, 224).cuda()
    resnet = resnet50().to("cuda")
    [resnet(x) for _ in range(warm_times)]

    time_start = time.time()
    [resnet(x) for _ in range(test_times)]
    print(f'''
Eager:
 - warm times: {warm_times}
 - test times: {test_times}
 - single cost: {(time.time() - time_start)/test_times:f}
    ''')


@flow.unittest.skip_unless_1n1d()
class TestOKLResNet(flow.unittest.TestCase):
    # @unittest.skipUnless(flow.sysconfig.with_cuda(), "only test cpu cases")
    # def test_okl_resnet(test_case):
    #     _test_okl_resnet(test_case)

    @unittest.skipUnless(flow.sysconfig.with_cuda(), "only test cpu cases")
    def test_okl_resnet_tim(test_case):
        warm_times = 3;
        test_times = 10;
        _test_okl_resnet_in_eager(test_case, warm_times, test_times)
        _test_okl_resnet_in_okl(test_case, warm_times, test_times)


if __name__ == "__main__":
    unittest.main()
