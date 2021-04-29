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
import unittest
import oneflow as flow
import numpy as np
import os
import random
import oneflow.typing as oft
from collections import OrderedDict


@flow.unittest.skip_unless_1n1d()
class TestTensor(flow.unittest.TestCase):
    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_numpy_and_default_dtype(test_case):
        shape = (2, 3, 4, 5)
        tensor = flow.Tensor(*shape)
        tensor.set_data_initializer(flow.ones_initializer())
        test_case.assertTrue(tensor.dtype == flow.float32)
        test_case.assertTrue(
            np.array_equal(tensor.numpy(), np.ones(shape, dtype=np.float32))
        )

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_construct_from_numpy_or_list(test_case):
        shape = (2, 3, 4, 5)
        np_arr = np.random.rand(*shape)
        tensor = flow.Tensor(np_arr)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr))

        np_int_arr = np.random.randint(-100, high=100, size=shape, dtype=np.int32)
        tensor = flow.Tensor(np_int_arr)
        print("dtype: ", tensor.dtype, np_int_arr.dtype)
        test_case.assertTrue(tensor.dtype == flow.int32)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_int_arr))

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_tensor_init_methods(test_case):
        # test float dtype init
        shape = (2, 3, 4, 5)
        x = flow.Tensor(*shape)

        np_ones = np.ones(x.shape)
        np_zeros = np.zeros(x.shape)

        random_fill_val = random.uniform(-100.0, 100.0)
        x.fill_(random_fill_val)
        test_case.assertTrue(np.allclose(x.numpy(), random_fill_val * np_ones))

        flow.nn.init.ones_(x)
        test_case.assertTrue(np.array_equal(x.numpy(), np_ones))

        flow.nn.init.zeros_(x)
        test_case.assertTrue(np.array_equal(x.numpy(), np_zeros))

        flow.nn.init.constant_(x, random_fill_val)
        test_case.assertTrue(np.allclose(x.numpy(), random_fill_val * np_ones))

        z = flow.Tensor(5, 4, 3, 2)
        flow.nn.init.kaiming_normal_(z, a=0.1, mode="fan_out", nonlinearity="relu")
        flow.nn.init.kaiming_uniform_(z)
        flow.nn.init.xavier_normal_(z)
        flow.nn.init.xavier_uniform_(z)

        # test int dtype init
        x = flow.Tensor(*shape, dtype=flow.int32)
        np_ones = np.ones(x.shape, dtype=np.int32)
        np_zeros = np.zeros(x.shape, dtype=np.int32)

        random_fill_val = random.randint(-100, 100)
        x.fill_(random_fill_val)
        test_case.assertTrue(np.allclose(x.numpy(), random_fill_val * np_ones))

        flow.nn.init.ones_(x)
        test_case.assertTrue(np.array_equal(x.numpy(), np_ones))

        flow.nn.init.zeros_(x)
        test_case.assertTrue(np.array_equal(x.numpy(), np_zeros))

        flow.nn.init.constant_(x, random_fill_val)
        test_case.assertTrue(np.allclose(x.numpy(), random_fill_val * np_ones))

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_creating_consistent_tensor(test_case):
        shape = (2, 3)
        x = flow.Tensor(*shape, placement=flow.placement("gpu", ["0:0"], None))
        x.set_placement(flow.placement("cpu", ["0:0"], None))
        x.set_is_consistent(True)
        test_case.assertTrue(not x.is_cuda)
        x.determine()

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_tensor_device(test_case):
        shape = (2, 3, 4, 5)
        x = flow.Tensor(*shape)
        test_case.assertTrue(x.is_cuda)
        x = flow.Tensor(*shape, device=flow.device("cuda"))
        test_case.assertTrue(x.is_cuda)
        x = flow.Tensor(*shape, device=flow.device("cpu"))
        test_case.assertTrue(not x.is_cuda)

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_tensor_autograd_related_methods(test_case):
        shape = (2, 3, 4, 5)
        x = flow.Tensor(*shape)
        y = flow.Tensor(*shape, requires_grad=True)
        x.fill_(1.0)
        y.fill_(2.0)

        # TODO(Liang Depeng): change to `z = x + y`
        add_op = flow.builtin_op("add_n").Input("in", 2).Output("out").Build()
        z = add_op(x, y)[0]

        test_case.assertTrue(x.requires_grad == False and x.is_leaf == True)
        test_case.assertTrue(y.requires_grad == True and y.is_leaf == True)
        test_case.assertTrue(z.requires_grad == True and z.is_leaf == False)

        v = flow.Tensor(*shape, requires_grad=True)
        # NOTE(Liang Depeng): the following setting not working
        z.retain_grad()
        # TODO(Liang Depeng): change to `w = v + z`
        w = add_op(v, z)[0]

        grad = flow.Tensor(*shape)
        grad.fill_(1.0)
        grad.determine()
        w.backward(gradient=grad, retain_graph=True)

        test_case.assertTrue(v.grad != None)
        test_case.assertTrue(y.grad != None)
        # NOTE(Liang Depeng): call z.grad or x.grad will raise Exception
        # test_case.assertTrue(z.grad == None)
        # test_case.assertTrue(x.grad == None)
        w.backward(gradient=grad, retain_graph=True)

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_user_defined_data(test_case):
        list_data = [5, 5]
        tuple_data = (5, 5)
        numpy_data = np.array((5, 5))
        x = flow.Tensor(list_data)
        y = flow.Tensor(tuple_data)
        z = flow.Tensor(numpy_data)

        test_case.assertTrue(np.array_equal(x.numpy(), 5 * np.ones(x.shape)))
        test_case.assertTrue(np.array_equal(y.numpy(), 5 * np.ones(y.shape)))
        test_case.assertTrue(np.array_equal(z.numpy(), 5 * np.ones(z.shape)))

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_mirrored_tensor_and_op(test_case):
        x1 = flow.Tensor([[1.0, 2.0]])
        test_case.assertEqual(x1.dtype, flow.float64)
        test_case.assertEqual(x1.shape, flow.Size((1, 2)))
        x2 = flow.Tensor([[1.0], [2.0]])
        # TODO(Liang Depeng): change to MatMul module
        op = (
            flow.builtin_op("matmul")
            .Input("a")
            .Input("b")
            .Attr("transpose_a", False)
            .Attr("transpose_b", False)
            .Attr("alpha", float(1.0))
            .Output("out")
            .Build()
        )
        y = op(x1, x2)[0]
        test_case.assertTrue(
            np.array_equal(y.numpy(), np.array([[5.0]], dtype=np.float32))
        )

    @unittest.skipIf(
        # TODO(Liang Depeng): enable this test after tensor support indexing
        # not flow.unittest.env.eager_execution_enabled(),
        # "numpy doesn't work in lazy mode",
        True,
        "skip for now",
    )
    def test_indexing(test_case):
        class SliceExtracter:
            def __getitem__(self, key):
                return key

        se = SliceExtracter()

        def compare_getitem_with_numpy(tensor, slices):
            np_arr = tensor.numpy()
            test_case.assertTrue(np.array_equal(np_arr[slices], tensor[slices].numpy()))

        def compare_setitem_with_numpy(tensor, slices, value):
            np_arr = tensor.numpy()
            if isinstance(value, flow.Tensor):
                np_value = value.numpy()
            else:
                np_value = value
            np_arr[slices] = np_value
            tensor[slices] = value
            test_case.assertTrue(np.array_equal(np_arr, tensor.numpy()))

        x = flow.Tensor(5, 5)
        v = flow.Tensor([[0, 1, 2, 3, 4]])
        compare_getitem_with_numpy(x, se[-4:-1:2])
        compare_getitem_with_numpy(x, se[-1:])
        compare_setitem_with_numpy(x, se[-1:], v)
        compare_setitem_with_numpy(x, se[2::2], 2)

        flow.nn.init.kaiming_normal_(x, a=0.1, mode="fan_out", nonlinearity="relu")

        flow.nn.init.kaiming_uniform_(x)

        flow.nn.init.xavier_normal_(x)

        flow.nn.init.xavier_uniform_(x)

        test_case.assertEqual(flow.nn.init.calculate_gain("conv2d"), 1)
        test_case.assertEqual(flow.nn.init.calculate_gain("tanh"), 5.0 / 3)

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_div(test_case):
        x = flow.Tensor(np.random.randn(1, 1))
        y = flow.Tensor(np.random.randn(2, 3))
        of_out = x / y
        np_out = np.divide(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        of_out = x / 3
        np_out = np.divide(x.numpy(), 3)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        of_out = 3 / x
        np_out = np.divide(3, x.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(1))
        of_out = 3 / x
        np_out = np.divide(3, x.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_mul(test_case):
        x = flow.Tensor(np.random.randn(1, 1))
        y = flow.Tensor(np.random.randn(2, 3))
        of_out = x * y
        np_out = np.multiply(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        of_out = x * 3
        np_out = np.multiply(x.numpy(), 3)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        of_out = 3 * x
        np_out = np.multiply(3, x.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_add_tensor_method(test_case):
        x = flow.Tensor(np.random.randn(1, 1))
        y = flow.Tensor(np.random.randn(2, 3))
        of_out = x + y
        np_out = np.add(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        of_out = x + 3
        np_out = np.add(x.numpy(), 3)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        of_out = 3 + x
        np_out = np.add(3, x.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_sub_tensor_method(test_case):
        x = flow.Tensor(np.random.randn(1, 1))
        y = flow.Tensor(np.random.randn(2, 3))
        of_out = x - y
        np_out = np.subtract(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        of_out = x - 3
        np_out = np.subtract(x.numpy(), 3)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

        x = flow.Tensor(np.random.randn(2, 3))
        of_out = 3 - x
        np_out = np.subtract(3, x.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_sum(test_case):
        input = flow.Tensor(np.random.randn(4, 5, 6), dtype=flow.float32)
        of_out = input.sum(dim=(2, 1))
        np_out = np.sum(input.numpy(), axis=(2, 1))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_mean(test_case):
        input = flow.Tensor(np.random.randn(2, 3), dtype=flow.float32)
        of_out = input.mean(dim=0)
        np_out = np.mean(input.numpy(), axis=0)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_neg(test_case):
        input = flow.Tensor(np.random.randn(2, 3), dtype=flow.float32)
        of_out = -input
        np_out = -input.numpy()
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))

if __name__ == "__main__":
    unittest.main()
