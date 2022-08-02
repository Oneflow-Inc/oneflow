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
from oneflow.ops.array_ops import parse_slice_tuple_list
import oneflow.unittest
import random


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestGraphInplaceOperations(flow.unittest.TestCase):
    def test_inplace_scalar_add(test_case):
        def _test(device):
            
            add_value = random.randint(1,100)
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.add(input, add_value, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.add(input, add_value)

            input = flow.randn(4,4,4, device = device)
            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            eq = flow.all(not_inplace_graph(input)==inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_scalar_sub(test_case):
        def _test(device):
            sub_value = random.randint(1,100)
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.sub(input, sub_value, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.sub(input, sub_value)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input)==inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_scalar_mul(test_case):
        def _test(device):
            mul_value = random.randint(1,100)
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.mul(input, mul_value, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.mul(input, mul_value)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input)==inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_add(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input1, input2):
                    flow.add(input1, input2, inplace=True)
                    return input1

            class NotInplaceGraph(nn.Graph):
                def build(self, input1, input2):
                    return flow.add(input1, input2)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input1 = flow.randn(4, 4, 4, device=device)
            input2 = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input1,input2) == inplace_graph(input1, input2))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_sub(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input1, input2):
                    flow.sub(input1, input2, inplace=True)
                    return input1

            class NotInplaceGraph(nn.Graph):
                def build(self, input1, input2):
                    return flow.sub(input1, input2)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input1 = flow.randn(4, 4, 4, device=device)
            input2 = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input1,input2) == inplace_graph(input1, input2))
            test_case.assertTrue(eq)
            

        _test("cpu")
        _test("cuda")

    def test_inplace_pow(test_case):
        def _test(device):
            exp = random.randint(1,5)
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.pow(input, exp, inplace=True)
                    return input
            
            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.pow(input, exp)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_floor_divide(test_case):
        def _test(device):
            div_value = random.randint(1,10)
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.floor_divide(input, div_value, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.floor_divide(input, div_value)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_fmod(test_case):
        def _test(device):
            mod_value = random.randint(1,10)
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.fmod(input, mod_value, inplace=True)
                    return input
            
            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.fmod(input, mod_value)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_relu(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.nn.functional.relu(input, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.nn.functional.relu(input)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_celu(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.nn.functional.celu(input, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.nn.functional.celu(input)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_hardsigmoid(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.nn.functional.hardsigmoid(input, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.nn.functional.hardsigmoid(input)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)

    def test_inplace_hardshrink(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.nn.functional.hardsigmoid(input, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.nn.functional.hardsigmoid(input)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_leaky_relu(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.nn.functional.leaky_relu(input, 1.0, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.nn.functional.leaky_relu(input, 1.0)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_tensor_scatter_nd_update(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input,indices, updates):
                    indices = flow.tensor([[1], [3], [5]], device=device)
                    updates = flow.tensor([-1, -2, -3], device=device)
                    flow._C.tensor_scatter_nd_update(input, indices, updates, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input, indices, updates):
                  
                    return flow._C.tensor_scatter_nd_update(input, indices, updates)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.arange(8, device=device)
            indices = flow.tensor([[1], [3], [5]], device=device)
            updates = flow.tensor([-1, -2, -3], device=device)
            eq = flow.all(not_inplace_graph(input, indices, updates) == inplace_graph(input, indices, updates))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_slice_update(test_case):
        def _test(device):
            slice_tup_list=[[1, 4, 1]]
            input = flow.Tensor(
                        np.array([1, 1, 1, 1, 1]).astype(np.float32), device=device
                    )
            update = flow.Tensor(
                        np.array([2, 3, 4]).astype(np.float32), device=device
                    )
            (start, stop, step) = parse_slice_tuple_list(slice_tup_list, input.shape)
            class InplaceGraph(nn.Graph):
                def build(self, input, update):
                    flow._C.slice_update(input, update, start, stop, step, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input, update):
                    return flow._C.slice_update(input, update, start, stop, step)
            
            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            eq = flow.all(not_inplace_graph(input, update) == inplace_graph(input, update))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_softshrink(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.nn.functional.softshrink(input, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.nn.functional.softshrink(input)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_dropout(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow.nn.functional.dropout(input, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow.nn.functional.dropout(input)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_broadcast(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow._C.broadcast(input, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow._C.broadcast(input)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)

        _test("cpu")
        _test("cuda")

    def test_inplace_local_all_reduce(test_case):
        def _test(device):
            class InplaceGraph(nn.Graph):
                def build(self, input):
                    flow._C.local_all_reduce(input, inplace=True)
                    return input

            class NotInplaceGraph(nn.Graph):
                def build(self, input):
                    return flow._C.local_all_reduce(input)

            inplace_graph = InplaceGraph()
            not_inplace_graph = NotInplaceGraph()
            input = flow.randn(4, 4, 4, device=device)
            eq = flow.all(not_inplace_graph(input) == inplace_graph(input))
            test_case.assertTrue(eq)


        _test("cpu")
        _test("cuda")

if __name__ == "__main__":
    unittest.main()
