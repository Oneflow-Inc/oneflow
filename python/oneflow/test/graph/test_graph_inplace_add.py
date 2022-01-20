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
import os
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest


def _test_graph_lazy_inplace(test_case, x, y):
    class LazyInplaceAdd(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self, x, y):
            x += y
            return x

    z = LazyInplaceAdd()(x, y)
    test_case.assertTrue(np.allclose(z.numpy(), (x + y).numpy(), 1e-05, 1e-05))


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLocalInplace(oneflow.unittest.TestCase):
    def test_graph_inplace_gpu(test_case):
        x = flow.randn(10, 10, device=flow.device("cuda"))
        y = flow.ones(10, device=flow.device("cuda"))
        _test_graph_lazy_inplace(test_case, x, y)

    def test_graph_inplace_cpu(test_case):
        x = flow.randn(10, 10, device=flow.device("cpu"))
        y = flow.ones(10, device=flow.device("cpu"))
        _test_graph_lazy_inplace(test_case, x, y)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestConsistentInplace(oneflow.unittest.TestCase):
    def test_graph_inplace_gpu(test_case):
        x = flow.randn(
            10, 10, placement=flow.placement("cuda", {0: [0, 1]}), sbp=flow.sbp.split(1)
        )
        y = flow.ones(
            10, placement=flow.placement("cuda", {0: [0, 1]}), sbp=flow.sbp.broadcast
        )
        _test_graph_lazy_inplace(test_case, x, y)

    def test_graph_inplace_cpu(test_case):
        x = flow.randn(
            10, 10, placement=flow.placement("cpu", {0: [0, 1]}), sbp=flow.sbp.split(1)
        )
        y = flow.ones(
            10, placement=flow.placement("cpu", {0: [0, 1]}), sbp=flow.sbp.broadcast
        )
        _test_graph_lazy_inplace(test_case, x, y)


if __name__ == "__main__":
    unittest.main()
