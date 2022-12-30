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
import argparse
import numpy as np
import os
import time
import unittest

import oneflow as flow
import oneflow.unittest


def _test_long_add_n_graph(test_case, device):
    input_arr = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
            [0.8901075, -0.49976737, -1.07153746],
            [-0.44872912, -1.07275683, 0.06256855],
            [-0.22556897, 0.74798368, 0.90416439],
            [0.48339456, -2.32742195, -0.59321527],
        ],
        dtype=np.float32,
    )
    x0 = flow.tensor(input_arr, device=device)
    x1 = flow.tensor(input_arr, device=device)
    x2 = flow.tensor(input_arr, device=device)
    x3 = flow.tensor(input_arr, device=device)
    x4 = flow.tensor(input_arr, device=device)
    x5 = flow.tensor(input_arr, device=device)
    x6 = flow.tensor(input_arr, device=device)
    x7 = flow.tensor(input_arr, device=device)
    x8 = flow.tensor(input_arr, device=device)
    x9 = flow.tensor(input_arr, device=device)

    class AddNGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self):
            # Deprecated `temp = x0 + x0` to avoid unstable test
            # enable this after fix https://github.com/Oneflow-Inc/oneflow/issues/9431
            # temp = x0 + x0
            temp = x0
            temp = temp + x1  # test add_n1(add_n0(...), ...)
            temp = temp + temp  # test add_n1(add_n0(...), add_n0(...))
            temp = temp + x2
            temp = temp + x3
            temp = temp + x4
            temp = temp + x5
            temp = temp + x6
            temp = temp + x7
            other_add_n = x8 + x9
            temp = temp + other_add_n  # test add_n2(add_n0(), add_n1())
            return temp

    add_n_g = AddNGraph()
    of_lazy_out = add_n_g()
    test_case.assertTrue(np.allclose(input_arr * 12, of_lazy_out.numpy(), 1e-05, 1e-05))


def _test_add_n_consume_multi_add_n_graph(test_case, device):
    input_arr = np.array(
        [
            [-0.94630778, -0.83378579, -0.87060891],
            [2.0289922, -0.28708987, -2.18369248],
            [0.35217619, -0.67095644, -1.58943879],
            [0.08086036, -1.81075924, 1.20752494],
            [0.8901075, -0.49976737, -1.07153746],
            [-0.44872912, -1.07275683, 0.06256855],
            [-0.22556897, 0.74798368, 0.90416439],
            [0.48339456, -2.32742195, -0.59321527],
        ],
        dtype=np.float32,
    )
    x0 = flow.tensor(input_arr, device=device)

    class AddNGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()

        def build(self):
            temp = x0 + x0
            temp = temp + temp
            return temp

    add_n_g = AddNGraph()
    of_lazy_out = add_n_g()
    test_case.assertTrue(np.allclose(input_arr * 4, of_lazy_out.numpy(), 1e-05, 1e-05))


@unittest.skip("fail on ci")
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLongAddNGraph(oneflow.unittest.TestCase):
    def test_add_n(test_case):
        device = "cuda"
        _test_long_add_n_graph(test_case, device)

    def test_consume_multi_add_n(test_case):
        device = "cuda"
        _test_add_n_consume_multi_add_n_graph(test_case, device)


if __name__ == "__main__":
    unittest.main()
