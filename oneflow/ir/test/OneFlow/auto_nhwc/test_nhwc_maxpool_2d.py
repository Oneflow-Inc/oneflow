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


def do_nhwc_maxpool_2d(test_case, with_cuda, with_return_induces):
    x = flow.randn(1, 4, 4, 4)
    maxpool_2d = flow.nn.MaxPool2d(
        kernel_size=3, padding=1, stride=3, return_indices=with_return_induces
    )
    if with_cuda:
        x = x.cuda()
        maxpool_2d.to("cuda")

    eager_maxpool_2d_res = maxpool_2d(x)

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.m = maxpool_2d

        def build(self, x):
            return self.m(x)

    graph_to_run = GraphToRun()
    lazy_maxpool_2d_res = graph_to_run(x)
    if with_return_induces:
        test_case.assertTrue(
            np.allclose(
                eager_maxpool_2d_res[0].numpy(),
                lazy_maxpool_2d_res[0].numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
        )
    else:
        test_case.assertTrue(
            np.allclose(
                eager_maxpool_2d_res.numpy(),
                lazy_maxpool_2d_res.numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
        )


@flow.unittest.skip_unless_1n1d()
class TestNhwcMaxPool2d(oneflow.unittest.TestCase):
    def test_nhwc_maxpool_2d_graph(test_case):
        do_nhwc_maxpool_2d(test_case, True, True)
        do_nhwc_maxpool_2d(test_case, True, False)
        do_nhwc_maxpool_2d(test_case, False, True)
        do_nhwc_maxpool_2d(test_case, False, False)


if __name__ == "__main__":
    unittest.main()
