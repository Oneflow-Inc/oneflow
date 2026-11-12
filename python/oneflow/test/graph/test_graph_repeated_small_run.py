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


class _TinyGraph(flow.nn.Graph):
    def build(self, x):
        return x * 2 + 1


def _test_repeated_small_run(test_case, device):
    # A graph this small can finish on the actor threads before the VM thread
    # that launched it returns, so every run exercises the launch/finish order.
    graph = _TinyGraph()
    x = flow.arange(4, dtype=flow.float32, device=device)
    expected = np.arange(4, dtype=np.float32) * 2 + 1
    for _ in range(2000):
        y = graph(x)
    test_case.assertTrue(np.array_equal(y.numpy(), expected))


@flow.unittest.skip_unless_1n1d()
class TestGraphRepeatedSmallRun(flow.unittest.TestCase):
    def test_repeated_small_run_cpu(test_case):
        _test_repeated_small_run(test_case, "cpu")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_repeated_small_run_cuda(test_case):
        _test_repeated_small_run(test_case, "cuda")


if __name__ == "__main__":
    unittest.main()
