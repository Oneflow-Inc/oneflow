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
import sys
import numpy as np
import oneflow.nn as nn
import oneflow as flow
import oneflow.unittest


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphInplaceOperations(flow.unittest.TestCase):
    def test_inplace_scalar_add(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self, input):
                    input += 1
                    x = flow.randn(4, 4, device=device)
                    x += 3
                    flow.add(x, 7, inplace=True)
                    return x

            graph = Graph()
            graph(flow.randn(4, 4, device=device))

        _test("cpu")
        _test("cuda")

    def test_inplace_scalar_sub(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self, input):
                    input -= 3
                    x = flow.randn(4, 4, device=device)
                    x -= 1
                    flow.sub(x, 2, inplace=True)
                    return x

            graph = Graph()
            graph(flow.randn(4, 4, device=device))

        _test("cpu")
        _test("cuda")

    def test_inplace_scalar_mul(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self, input):
                    input *= 10
                    x = flow.randn(4, 4, device=device)
                    x *= 5
                    flow.mul(x, 8, inplace=True)
                    return x

            graph = Graph()
            graph(flow.randn(4, 4, device=device))

        _test("cpu")
        _test("cuda")

    def test_inplace_add(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    y = flow.randn(4, 4, device=device)
                    x += y
                    flow.add(x, y, inplace=True)
                    flow.add([x, y], inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_sub(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    y = flow.randn(4, 4, device=device)
                    x -= y
                    flow.sub(x, y, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_pow(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    flow.pow(x, 2, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_floor_divide(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    flow.floor_divide(x, 2, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_fmod(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    flow.fmod(x, 2, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_relu(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    flow.relu(x, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_celu(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    flow.nn.functional.celu(x, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_hardsigmoid(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    flow.nn.functional.hardsigmoid(x, inplace=True)
                    return x

            graph = Graph()
            graph()

    def test_inplace_hardshrink(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    flow.nn.functional.hardshrink(x, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_leaky_relu(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    flow.nn.functional.leaky_relu(x, 1.0, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_tensor_scatter_nd_update(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.arange(8, device=device)
                    indices = flow.tensor([[1], [3], [5]], device=device)
                    updates = flow.tensor([-1, -2, -3], device=device)
                    flow._C.tensor_scatter_nd_update(x, indices, updates, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_slice_update(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.Tensor(
                        np.array([1, 1, 1, 1, 1]).astype(np.float32), device=device
                    )
                    update = flow.Tensor(
                        np.array([2, 3, 4]).astype(np.float32), device=device
                    )
                    flow.slice_update(
                        x, update, slice_tup_list=[[1, 4, 1]]
                    )  # slice_update_op is inplace by default
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_softshrink(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    flow.nn.functional.softshrink(x, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_dropout(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4, 4, device=device)
                    flow.nn.functional.dropout(x, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_broadcast(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4,4, device=device)
                    flow._C.broadcast(x, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    def test_inplace_local_all_reduce(test_case):
        def _test(device):
            class Graph(nn.Graph):
                def build(self):
                    x = flow.randn(4,4, device=device)
                    flow._C.local_all_reduce(x, inplace=True)
                    return x

            graph = Graph()
            graph()

        _test("cpu")
        _test("cuda")

    # def test_inplace_local_reduce(test_case):
    #     def _test(device):
    #         class Graph(nn.Graph):
    #             def build(self):
    #                 x = flow.randn(4,4, device=device)
    #                 flow._C.local_reduce(x, inplace=True, dst=0)
    #                 return x

    #         graph = Graph()
    #         graph()

    #     _test("cpu")
    #     _test("cuda")


if __name__ == "__main__":
    unittest.main()
