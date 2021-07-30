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
from collections import OrderedDict

import numpy as np
from automated_test_util import *

import oneflow as flow
import oneflow.typing as oft
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTensor(flow.unittest.TestCase):
    def test_numpy_and_default_dtype(test_case):
        shape = (2, 3, 4, 5)
        tensor = flow.Tensor(*shape)
        flow.nn.init.ones_(tensor)
        test_case.assertTrue(tensor.dtype == flow.float32)
        test_case.assertTrue(
            np.array_equal(tensor.numpy(), np.ones(shape, dtype=np.float32))
        )

    def test_tensor_property(test_case):
        shape = (2, 3, 4, 5)
        tensor = flow.Tensor(*shape)
        test_case.assertEqual(tensor.storage_offset(), 0)
        test_case.assertEqual(tensor.stride(), (60, 20, 5, 1))
        test_case.assertEqual(tensor.is_cuda, False)
        test_case.assertTrue(tensor.is_contiguous())

    def test_copy_to_and_from_numpy(test_case):
        np_arr = np.array([4, 6], dtype=np.float32)
        tensor = flow.Tensor(np_arr, dtype=flow.float32)
        test_case.assertTrue(np.array_equal(tensor.numpy(), np_arr))
        test_case.assertEqual(np.float32, tensor.numpy().dtype)
        np_arr = np.array([4, 6], dtype=np.int32)
        tensor = flow.Tensor(np_arr, dtype=flow.int32)
        test_case.assertTrue(np.array_equal(tensor.numpy(), np_arr))
        test_case.assertEqual(np.int32, tensor.numpy().dtype)

    def test_construct_from_numpy_or_list(test_case):
        shape = (2, 3, 4, 5)
        np_arr = np.random.rand(*shape).astype(np.float32)
        tensor = flow.Tensor(np_arr)
        test_case.assertTrue(np.array_equal(tensor.numpy(), np_arr))
        np_int_arr = np.random.randint(-100, high=100, size=shape, dtype=np.int32)
        tensor = flow.Tensor(np_int_arr, dtype=flow.int32)
        test_case.assertEqual(tensor.dtype, flow.int32)
        test_case.assertTrue(np_arr.flags["C_CONTIGUOUS"])
        test_case.assertTrue(np.array_equal(tensor.numpy(), np_int_arr))
        np_arr = np.random.random((1, 256, 256, 3)).astype(np.float32)
        np_arr = np_arr.transpose(0, 3, 1, 2)
        tensor = flow.Tensor(np_arr)
        test_case.assertFalse(np_arr.flags["C_CONTIGUOUS"])
        test_case.assertTrue(np.array_equal(tensor.numpy(), np_arr))

    def test_construct_from_another_tensor(test_case):
        shape = (2, 3, 4, 5)
        np_arr = np.random.rand(*shape).astype(np.float32)
        tensor = flow.Tensor(np_arr)
        output = flow.Tensor(tensor)
        test_case.assertEqual(output.dtype, flow.float32)
        test_case.assertTrue(np.array_equal(output.numpy(), np_arr))

    def test_tensor_init_methods(test_case):
        shape = (2, 3, 4, 5)
        x = flow.Tensor(*shape)
        np_ones = np.ones(x.shape)
        np_zeros = np.zeros(x.shape)
        random_fill_val = np.random.uniform(-100.0, 100.0)
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
        x = flow.Tensor(*shape, dtype=flow.int32)
        np_ones = np.ones(x.shape, dtype=np.int32)
        np_zeros = np.zeros(x.shape, dtype=np.int32)
        random_fill_val = np.random.randint(-100, 100)
        x.fill_(random_fill_val)
        test_case.assertTrue(np.allclose(x.numpy(), random_fill_val * np_ones))
        flow.nn.init.ones_(x)
        test_case.assertTrue(np.array_equal(x.numpy(), np_ones))
        flow.nn.init.zeros_(x)
        test_case.assertTrue(np.array_equal(x.numpy(), np_zeros))
        flow.nn.init.constant_(x, random_fill_val)
        test_case.assertTrue(np.allclose(x.numpy(), random_fill_val * np_ones))
        test_case.assertEqual(flow.nn.init.calculate_gain("conv2d"), 1)
        test_case.assertEqual(flow.nn.init.calculate_gain("tanh"), 5.0 / 3)

    @unittest.skipIf(True, "consistent_tensor doesn't work right now")
    def test_creating_consistent_tensor(test_case):
        shape = (2, 3)
        x = flow.Tensor(*shape, placement=flow.placement("gpu", ["0:0"], None))
        x.set_placement(flow.placement("cpu", ["0:0"], None))
        x.set_is_consistent(True)
        test_case.assertTrue(not x.is_cuda)

    def test_tensor_with_single_int(test_case):
        x = flow.Tensor(5)
        test_case.assertEqual(x.shape, flow.Size([5]))
        x = flow.tensor(5)
        test_case.assertEqual(x.numpy().item(), 5)

    def test_tensor_device(test_case):
        shape = (2, 3, 4, 5)
        x = flow.Tensor(*shape)
        test_case.assertTrue(not x.is_cuda)
        x = flow.Tensor(*shape, device=flow.device("cuda"))
        test_case.assertTrue(x.is_cuda)
        x = flow.Tensor(*shape, device=flow.device("cpu"))
        test_case.assertTrue(not x.is_cuda)

    def test_tensor_autograd_related_methods(test_case):
        shape = (2, 3, 4, 5)
        x = flow.Tensor(*shape)
        y = flow.Tensor(*shape, requires_grad=True)
        x.fill_(1.0)
        y.fill_(2.0)
        z = x + y
        test_case.assertFalse(x.requires_grad)
        test_case.assertTrue(x.is_leaf)
        test_case.assertTrue(y.requires_grad)
        test_case.assertTrue(y.is_leaf)
        test_case.assertTrue(z.requires_grad)
        test_case.assertFalse(z.is_leaf)
        with flow.no_grad():
            m = x + y
        test_case.assertTrue(m.is_leaf)
        test_case.assertFalse(m.requires_grad)
        m.requires_grad = True
        v = flow.Tensor(*shape, requires_grad=True)
        z.retain_grad()
        w = v + z
        grad = flow.Tensor(*shape)
        grad.fill_(1.0)
        w.backward(gradient=grad, retain_graph=True)
        test_case.assertNotEqual(v.grad, None)
        test_case.assertNotEqual(y.grad, None)
        test_case.assertNotEqual(z.grad, None)
        test_case.assertIsNone(x.grad)
        w.backward(gradient=grad, retain_graph=True)

    def test_tensor_register_hook(test_case):
        shape = (2, 3)
        x = flow.Tensor(*shape, requires_grad=True)
        x.register_hook(lambda grad: grad * 2 + 1)
        y = x.sum() + (x * 2).sum()
        y.backward()
        test_case.assertTrue(np.array_equal(x.grad.numpy(), np.ones(shape) * 7))
        x = flow.Tensor(*shape, requires_grad=True)
        new_grad = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        x.register_hook(lambda _: new_grad)
        y = x.sum() + (x * 2).sum()
        y.backward()
        test_case.assertTrue(np.array_equal(x.grad.numpy(), new_grad.numpy()))
        grad_nonlocal = None

        def assign_nonlocal_variable_and_return_none(grad):
            nonlocal grad_nonlocal
            grad_nonlocal = grad

        x = flow.Tensor(*shape, requires_grad=True)
        new_grad = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        x.register_hook(assign_nonlocal_variable_and_return_none)
        y = x.sum() + (x * 2).sum()
        y.backward()
        test_case.assertTrue(np.array_equal(grad_nonlocal.numpy(), np.ones(shape) * 3))

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

    def test_mirrored_tensor_and_op(test_case):
        x1 = flow.Tensor([[1.0, 2.0]])
        test_case.assertEqual(x1.dtype, flow.float32)
        test_case.assertEqual(x1.shape, flow.Size((1, 2)))
        x2 = flow.Tensor([[1.0], [2.0]])
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

    def test_tensor_to_list(test_case):
        list_data = [[1.0, 3.0], [5.0, 6.0]]
        input = flow.Tensor(list_data)
        test_case.assertEqual(list_data, input.tolist())

    def test_tensor_nelement(test_case):
        shape = (2, 3, 4)
        input = flow.Tensor(*shape)
        test_case.assertEqual(input.nelement(), 24)

    def test_tensor_numel(test_case):
        shape = (2, 3, 4, 5)
        input = flow.Tensor(*shape)
        test_case.assertEqual(input.numel(), 120)

    def test_tensor_print(test_case):
        shape = (2, 3, 4, 5)
        input = flow.Tensor(*shape)
        input_str = str(input)
        test_case.assertTrue(input_str.startswith("tensor("))
        test_case.assertTrue("device=" not in input_str)
        gpu_input = flow.Tensor(*shape, device="cuda")
        gpu_input_str = str(gpu_input)
        test_case.assertTrue("device=" in gpu_input_str)
        test_case.assertTrue("cuda:0" in gpu_input_str)
        requires_grad_input = flow.Tensor(*shape, requires_grad=True)
        requires_grad_input_str = str(requires_grad_input)
        test_case.assertTrue("requires_grad=" in requires_grad_input_str)

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
        x = flow.Tensor(2, 3, 4)
        v = flow.Tensor(3)
        compare_setitem_with_numpy(x, se[:, :, 2], v)
        x = flow.Tensor(2, 3, 4)
        compare_setitem_with_numpy(x, se[1, :, 2], v)

    def test_div(test_case):
        x = flow.Tensor(np.random.randn(1, 1))
        y = flow.Tensor(np.random.randn(2, 3))
        of_out = x / y
        np_out = np.divide(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        x = flow.Tensor(np.random.randn(2, 3))
        of_out = x / 3
        np_out = np.divide(x.numpy(), 3)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        x = flow.Tensor(np.random.randn(2, 3))
        of_out = 3 / x
        np_out = np.divide(3, x.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        x = flow.Tensor(np.random.randn(1))
        of_out = 3 / x
        np_out = np.divide(3, x.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))

    def test_mul(test_case):
        x = flow.Tensor(np.random.randn(1, 1))
        y = flow.Tensor(np.random.randn(2, 3))
        of_out = x * y
        np_out = np.multiply(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        x = flow.Tensor(np.random.randn(2, 3))
        of_out = x * 3
        np_out = np.multiply(x.numpy(), 3)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        x = flow.Tensor(np.random.randn(2, 3))
        of_out = 3 * x
        np_out = np.multiply(3, x.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))

    def test_add_tensor_method(test_case):
        x = flow.Tensor(np.random.randn(1, 1))
        y = flow.Tensor(np.random.randn(2, 3))
        of_out = x + y
        np_out = np.add(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        x = flow.Tensor(np.random.randn(2, 3))
        of_out = x + 3
        np_out = np.add(x.numpy(), 3)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        x = flow.Tensor(np.random.randn(2, 3))
        of_out = 3 + x
        np_out = np.add(3, x.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))

    def test_sub_tensor_method(test_case):
        x = flow.Tensor(np.random.randn(1, 1))
        y = flow.Tensor(np.random.randn(2, 3))
        of_out = x - y
        np_out = np.subtract(x.numpy(), y.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        x = flow.Tensor(np.random.randn(2, 3))
        of_out = x - 3
        np_out = np.subtract(x.numpy(), 3)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        x = flow.Tensor(np.random.randn(2, 3))
        of_out = 3 - x
        np_out = np.subtract(3, x.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))

    def test_sum(test_case):
        input = flow.Tensor(np.random.randn(4, 5, 6), dtype=flow.float32)
        of_out = input.sum(dim=(2, 1))
        np_out = np.sum(input.numpy(), axis=(2, 1))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))

    def test_argwhere(test_case):
        shape = (2, 3, 4, 5)
        precision = 1e-5
        np_input = np.random.randn(*shape)
        input = flow.Tensor(np_input)
        of_out = input.argwhere()
        np_out = np.argwhere(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, precision, precision))
        test_case.assertTrue(np.array_equal(of_out.numpy().shape, np_out.shape))

    @autotest(n=5, auto_backward=False)
    def test_tensor_argmax_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 6).to(int)
        x = random_pytorch_tensor(ndim=ndim).to(device)
        y = x.argmax(dim=random(0, ndim).to(int), keepdim=random().to(bool))
        return y

    @autotest()
    def test_tensor_tanh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.tanh()
        return y

    @autotest()
    def test_flow_tensor_asin_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.5).to(device)
        y = x.asin()
        return y

    @autotest()
    def test_flow_tensor_arcsin_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.5).to(device)
        y = x.arcsin()
        return y

    @autotest()
    def test_flow_tensor_asinh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.asinh()
        return y

    @autotest()
    def test_flow_tensor_arcsinh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.arcsinh()
        return y

    @autotest()
    def test_flow_tensor_sinh_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.sinh()
        return y

    @autotest()
    def test_flow_tensor_atan2_with_random_data(test_case):
        device = random_device()
        x1 = random_pytorch_tensor(ndim=1, dim0=1).to(device)
        x2 = random_pytorch_tensor(ndim=1, dim0=1).to(device)
        y = x1.atan2(x2)
        return y

    @autotest()
    def test_arccosh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=2, high=3).to(device)
        y = x.arccosh()
        return y

    @autotest()
    def test_acosh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=2, high=3).to(device)
        y = x.acosh()
        return y

    def test_mean(test_case):
        input = flow.Tensor(np.random.randn(2, 3), dtype=flow.float32)
        of_out = input.mean(dim=0)
        np_out = np.mean(input.numpy(), axis=0)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))

    def test_neg(test_case):
        input = flow.Tensor(np.random.randn(2, 3), dtype=flow.float32)
        of_out = -input
        np_out = -input.numpy()
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))

    def test_negative(test_case):
        input = flow.Tensor(np.random.randn(2, 3), dtype=flow.float32)
        of_out = input.negative()
        np_out = -input.numpy()
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))

    def test_greater(test_case):
        input1 = flow.Tensor(np.array([1, 1, 4]).astype(np.float32), dtype=flow.float32)
        input2 = flow.Tensor(np.array([1, 2, 3]).astype(np.float32), dtype=flow.float32)
        of_out = input1.gt(input2)
        np_out = np.greater(input1.numpy(), input2.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))

    def test_less(test_case):
        input1 = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        input2 = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        of_out = input1.lt(input2)
        np_out = np.less(input1.numpy(), input2.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))

    def test_tensor_slice(test_case):
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        input = flow.Tensor(x)
        test_case.assertTrue(np.allclose(input[0].numpy(), x[0], 1e-05, 1e-05))
        test_case.assertTrue(np.allclose(input[1].numpy(), x[1], 1e-05, 1e-05))
        test_case.assertTrue(np.allclose(input[0, :].numpy(), x[0, :], 1e-05, 1e-05))
        test_case.assertTrue(
            np.allclose(input[0, :, 0:2].numpy(), x[0, :, 0:2], 1e-05, 1e-05)
        )

    def test_tensor_logical_slice_assign(test_case):
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        input = flow.Tensor(x)
        input[:, 0] = 3.1415926
        x[:, 0] = 3.1415926
        test_case.assertTrue(np.allclose(input.numpy(), x, 1e-05, 1e-05))
        input[:, 1:2] = 1
        x[:, 1:2] = 1
        test_case.assertTrue(np.allclose(input.numpy(), x, 1e-05, 1e-05))
        input[:] = 1.234
        x[:] = 1.234
        test_case.assertTrue(np.allclose(input.numpy(), x, 1e-05, 1e-05))
        input[0] = 0
        x[0] = 0
        test_case.assertTrue(np.allclose(input.numpy(), x, 1e-05, 1e-05))

    def test_zeros_(test_case):
        shape = (2, 3)
        x = flow.Tensor(np.random.randn(*shape), dtype=flow.float32)
        x.zeros_()
        test_case.assertTrue(np.array_equal(x.numpy(), np.zeros(shape)))

    def test_construct_small_tensor(test_case):
        shape = (2, 3, 4, 5)
        np_arr = np.random.rand(*shape).astype(np.float32)
        tensor = flow.tensor(np_arr)
        test_case.assertTrue(np.array_equal(tensor.numpy(), np_arr))
        test_case.assertEqual(tensor.dtype, flow.float32)
        np_int_arr = np.random.randint(-100, high=100, size=shape, dtype=np.int32)
        tensor = flow.tensor(np_int_arr, dtype=flow.int32)
        test_case.assertEqual(tensor.dtype, flow.int32)
        list_data = [[1, 2.0], [5, 3]]
        tensor = flow.tensor(list_data)
        test_case.assertEqual(tensor.dtype, flow.float32)
        test_case.assertTrue(
            np.allclose(tensor.numpy(), np.array(list_data), 0.0001, 0.0001)
        )
        tuple_data = ((1, 2, 5), (4, 3, 10))
        tensor = flow.tensor(tuple_data)
        test_case.assertEqual(tensor.dtype, flow.int64)
        test_case.assertTrue(np.array_equal(tensor.numpy(), np.array(tuple_data)))
        scalar = 5.5
        tensor = flow.tensor(scalar)
        test_case.assertEqual(tensor.dtype, flow.float32)
        test_case.assertTrue(
            np.allclose(tensor.numpy(), np.array(scalar), 0.0001, 0.0001)
        )

    def test_floor(test_case):
        input = flow.Tensor(np.random.randn(4, 5, 6), dtype=flow.float32)
        of_out = input.floor()
        np_out = np.floor(input.numpy())
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
        )

    def test_tensor_round(test_case):
        shape = (2, 3)
        np_input = np.random.randn(*shape)
        of_input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_out = flow.round(of_input)
        np_out = np.round(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(
            np.allclose(of_input.grad.numpy(), np.zeros(shape), 0.0001, 0.0001)
        )

    def _test_tensor_reshape(test_case):
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        ).astype(np.float32)
        input = flow.Tensor(x)
        of_shape = input.reshape(shape=[2, 2, 2, -1]).numpy().shape
        np_shape = (2, 2, 2, 2)
        test_case.assertTrue(np.array_equal(of_shape, np_shape))

    @autotest()
    def test_reshape_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4).to(device)
        y = x.reshape(shape=(-1,))
        return y

    @autotest()
    def test_tensor_squeeze_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.squeeze(random().to(int))
        return y

    @autotest()
    def test_flow_unsqueeze_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.unsqueeze(random(1, 3).to(int))
        return y

    @autotest()
    def test_permute_flow_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4).to(device)
        y = x.permute(
            random(0, 4).to(int),
            random(0, 4).to(int),
            random(0, 4).to(int),
            random(0, 4).to(int),
        )
        return y

    @autotest()
    def test_transpose_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=4).to(device)
        y = x.transpose(dim0=random(1, 3).to(int), dim1=random(1, 3).to(int))
        return y

    def test_tensor_where(test_case):
        x = flow.Tensor(
            np.array([[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]),
            dtype=flow.float32,
        )
        y = flow.Tensor(np.ones(shape=(3, 2)), dtype=flow.float32)
        condition = flow.Tensor(np.array([[0, 1], [1, 0], [1, 0]]), dtype=flow.int32)
        of_out = condition.where(x, y)
        np_out = np.array([[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]])
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_tensor_equal(test_case):
        arr1 = np.random.randint(1, 10, size=(2, 3, 4, 5))
        arr2 = np.random.randint(1, 10, size=(2, 3, 4, 5))
        input = flow.Tensor(arr1, dtype=flow.float32)
        other = flow.Tensor(arr2, dtype=flow.float32)
        of_out = input.eq(other)
        np_out = np.equal(arr1, arr2)
        test_case.assertTrue(np.array_equal(of_out.numpy(), np_out))

    def _test_tensor_atan(test_case, shape, device):
        np_input = np.random.randn(*shape)
        of_input = flow.Tensor(
            np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
        )
        of_out = of_input.atan()
        np_out = np.arctan(np_input)
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
        )
        of_out = of_out.sum()
        of_out.backward()
        np_out_grad = 1 / (1 + np_input ** 2)
        test_case.assertTrue(
            np.allclose(
                of_input.grad.numpy(), np_out_grad, 1e-05, 1e-05, equal_nan=True
            )
        )

    def _test_tensor_arctan(test_case, shape, device):
        np_input = np.random.randn(*shape)
        of_input = flow.Tensor(
            np_input, dtype=flow.float32, device=flow.device(device), requires_grad=True
        )
        of_out = of_input.arctan()
        np_out = np.arctan(np_input)
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
        )
        of_out = of_out.sum()
        of_out.backward()
        np_out_grad = 1 / (1 + np_input ** 2)
        test_case.assertTrue(
            np.allclose(
                of_input.grad.numpy(), np_out_grad, 1e-05, 1e-05, equal_nan=True
            )
        )

    def test_tensor_detach(test_case):
        shape = (2, 3, 4, 5)
        x = flow.Tensor(np.random.randn(*shape), dtype=flow.float32, requires_grad=True)
        test_case.assertTrue(np.allclose(x.detach().numpy(), x.numpy(), 0.0001, 0.0001))
        test_case.assertEqual(x.detach().requires_grad, False)
        y = x * 2
        z = y.detach()
        test_case.assertEqual(z.is_leaf, True)
        test_case.assertEqual(z.grad_fn, None)

    def test_tensor_clamp_(test_case):
        input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        of_out = input.clamp(0.1, 0.5)
        np_out = np.clip(input.numpy(), 0.1, 0.5)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))

    def test_tensor_clip_(test_case):
        input = flow.Tensor(np.random.randn(2, 6, 5, 3), dtype=flow.float32)
        of_out = input.clip(0.1, 0.5)
        np_out = np.clip(input.numpy(), 0.1, 0.5)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))

    def _test_cast_tensor_function(test_case):
        shape = (2, 3, 4, 5)
        np_arr = np.random.randn(*shape).astype(np.float32)
        input = flow.Tensor(np_arr, dtype=flow.float32)
        output = input.cast(flow.int8)
        np_out = np_arr.astype(np.int8)
        test_case.assertTrue(np.array_equal(output.numpy(), np_out))

    def _test_sin_tensor_function(test_case, shape, device):
        input = flow.Tensor(np.random.randn(2, 3, 4, 5))
        of_out = input.sin()
        np_out = np.sin(input.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))

    def test_cos_tensor_function(test_case):
        arr = np.random.randn(2, 3, 4, 5)
        input = flow.Tensor(arr, dtype=flow.float32)
        np_out = np.cos(arr)
        of_out = input.cos()
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))

    def test_std_tensor_function(test_case):
        np_arr = np.random.randn(9, 8, 7, 6)
        input = flow.Tensor(np_arr)
        of_out = input.std(dim=1, keepdim=False)
        np_out = np.std(np_arr, axis=1)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))

    def test_sqrt_tensor_function(test_case):
        input_arr = np.random.rand(1, 6, 3, 8)
        np_out = np.sqrt(input_arr)
        x = flow.Tensor(input_arr)
        of_out = x.sqrt()
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
        )

    def test_rsqrt_tensor_function(test_case):
        np_arr = np.random.rand(3, 2, 5, 7)
        np_out = 1 / np.sqrt(np_arr)
        x = flow.Tensor(np_arr)
        of_out = flow.rsqrt(input=x)
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
        )

    def test_square_tensor_function(test_case):
        np_arr = np.random.randn(2, 7, 7, 3)
        np_out = np.square(np_arr)
        x = flow.Tensor(np_arr)
        of_out = x.square()
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
        )

    @autotest()
    def test_addmm_tensor_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor(ndim=2, dim0=2, dim1=3).to(device)
        mat1 = random_pytorch_tensor(ndim=2, dim0=2, dim1=4).to(device)
        mat2 = random_pytorch_tensor(ndim=2, dim0=4, dim1=3).to(device)
        y = input.addmm(
            mat1,
            mat2,
            beta=random().to(float) | nothing(),
            alpha=random().to(float) | nothing(),
        )
        return y

    @autotest()
    def test_addmm_broadcast_tensor_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor(ndim=2, dim0=1, dim1=1).to(device)
        mat1 = random_pytorch_tensor(ndim=2, dim0=2, dim1=4).to(device)
        mat2 = random_pytorch_tensor(ndim=2, dim0=4, dim1=3).to(device)
        y = input.addmm(
            mat1,
            mat2,
            beta=random().to(float) | nothing(),
            alpha=random().to(float) | nothing(),
        )
        return y

    @autotest()
    def test_clamp_tensor_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        y = input.clamp(min=random().to(float), max=random().to(float))
        return y

    @autotest()
    def test_clamp_minnone_tensor_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        y = input.clamp(min=random().to(float) | nothing(), max=random().to(float))
        return y

    @autotest()
    def test_clamp_maxnone_tensor_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        y = input.clamp(min=random().to(float), max=random().to(float) | nothing())
        return y

    @autotest()
    def test_clip_tensor_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        y = input.clip(min=random().to(float), max=random().to(float))
        return y

    @autotest()
    def test_clip_minnone_tensor_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        y = input.clip(min=random().to(float) | nothing(), max=random().to(float))
        return y

    @autotest()
    def test_clip_maxnone_tensor_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        y = input.clip(min=random().to(float), max=random().to(float) | nothing())
        return y

    @autotest()
    def test_ceil_tensor_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        y = input.ceil()
        return y

    @autotest()
    def test_expm1_tensor_with_random_data(test_case):
        device = random_device()
        input = random_pytorch_tensor().to(device)
        y = input.expm1()
        return y

    @autotest()
    def test_floor_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.floor()
        return y

    def test_norm_tensor_function(test_case):
        input = flow.Tensor(
            np.array([[-4.0, -3.0, -2.0], [-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]),
            dtype=flow.float32,
        )
        of_out_1 = input.norm("fro")
        np_out_1 = np.linalg.norm(input.numpy(), "fro")
        of_out_2 = input.norm(2, dim=1)
        np_out_2 = np.linalg.norm(input.numpy(), ord=2, axis=1)
        of_out_3 = input.norm(float("inf"), dim=0, keepdim=True)
        np_out_3 = np.linalg.norm(
            input.numpy(), ord=float("inf"), axis=0, keepdims=True
        )
        test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out_1, 1e-05, 1e-05))
        test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out_2, 1e-05, 1e-05))
        test_case.assertTrue(np.allclose(of_out_3.numpy(), np_out_3, 1e-05, 1e-05))

    @autotest()
    def test_pow_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = random().to(float)
        z = x.pow(y)
        return z

    @autotest()
    def test_atanh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.49).to(device)
        y = x.atanh()
        return y

    @autotest()
    def test_acos_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.49).to(device)
        y = x.acos()
        return y

    @autotest()
    def test_acosh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=2.0, high=3.0).to(device)
        y = x.acosh()
        return y

    @autotest()
    def test_atan_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.atan()
        return y

    @autotest()
    def test_arctan_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.arctan()
        return y

    @autotest()
    def test_tan_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor().to(device)
        y = x.tan()
        return y

    @autotest()
    def test_tan2_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(ndim=2, dim1=3).to(device)
        y = random_pytorch_tensor(ndim=2, dim1=3).to(device)
        z = x.atan2(y)
        return z

    @autotest()
    def test_arctanh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_pytorch_tensor(low=-0.5, high=0.5).to(device)
        y = x.arctanh()
        return y

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_tensor_fmod(test_case):
        x = flow.Tensor(np.random.uniform(-100, 100, (5, 5)), requires_grad=True)
        y = np.random.uniform(-10, 10)
        of_out = x.fmod(y)
        np_out = np.sign(x.numpy()) * np.abs(np.fmod(x.numpy(), y))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(
            np.allclose(x.grad.numpy(), np.ones((5, 5)), 0.0001, 0.0001)
        )

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_magic_fmod(test_case):
        x = flow.Tensor(np.random.uniform(-100, 100, (5, 5)), requires_grad=True)
        y = np.random.uniform(-10, 10)
        of_out = x % y
        np_out = np.sign(x.numpy()) * np.abs(np.fmod(x.numpy(), y))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(
            np.allclose(x.grad.numpy(), np.ones((5, 5)), 0.0001, 0.0001)
        )

    def test_tensor_mish(test_case):
        def np_mish(x):
            f = 1 + np.exp(x)
            y = x * ((f * f - 1) / (f * f + 1))
            y_grad = (f * f - 1) / (f * f + 1) + x * (4 * f * (f - 1)) / (
                (f * f + 1) * (f * f + 1)
            )
            return [y, y_grad]

        np_input = np.random.randn(2, 4, 5, 6)
        of_input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_out = of_input.mish()
        (np_out, np_grad) = np_mish(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-05, 1e-05))

    def test_tensor_triu(test_case):
        def np_triu(x, diagonal):
            y = np.triu(x, diagonal)
            y_grad = np.triu(np.ones_like(x), diagonal)
            return [y, y_grad]

        diagonal_list = [2, -1]
        for diagonal in diagonal_list:
            np_input = np.random.randn(2, 4, 6)
            of_input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
            of_out = of_input.triu(diagonal)
            (np_out, np_grad) = np_triu(np_input, diagonal)
            test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
            of_out = of_out.sum()
            of_out.backward()
            test_case.assertTrue(
                np.allclose(of_input.grad.numpy(), np_grad, 1e-05, 1e-05)
            )

    def test_tensor_grad_assignment(test_case):
        np_input = np.random.randn(2, 4, 5, 6)
        of_input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_output = 2 * of_input
        of_output = of_output.sum()
        of_output.backward()
        new_grad = flow.Tensor(
            np.full(np_input.shape, np.random.randn(1)), dtype=flow.float32
        )
        of_input.grad = new_grad
        test_case.assertTrue(
            np.allclose(of_input.grad.detach().numpy(), new_grad.numpy(), 1e-05, 1e-05)
        )
        of_input.grad = None
        test_case.assertTrue(of_input.grad is None)

    def test_tensor_grad_assignment_sum(test_case):
        np_input = np.random.randn(1, 5, 7, 3)
        of_input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_output = of_input.sum()
        of_output.backward()
        rand_init = np.random.randn(1)
        rand_scale = np.random.randn(1)
        new_grad = flow.Tensor(np.full(np_input.shape, rand_init), dtype=flow.float32)
        of_input.grad = new_grad
        of_output = flow.Tensor(rand_scale, dtype=flow.float32) * of_input
        of_output = of_output.sum()
        of_output.backward()
        test_case.assertTrue(
            np.allclose(
                of_input.grad.detach().numpy(),
                np.full(np_input.shape, rand_init + rand_scale),
                1e-05,
                1e-05,
            )
        )
        of_input.grad = of_input.grad * 2
        test_case.assertTrue(
            np.allclose(
                of_input.grad.detach().numpy(),
                2 * np.full(np_input.shape, rand_init + rand_scale),
                1e-05,
                1e-05,
            )
        )

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_mish(test_case):
        def np_mish(x):
            f = 1 + np.exp(x)
            y = x * ((f * f - 1) / (f * f + 1))
            y_grad = (f * f - 1) / (f * f + 1) + x * (4 * f * (f - 1)) / (
                (f * f + 1) * (f * f + 1)
            )
            return [y, y_grad]

        np_input = np.random.randn(2, 4, 5, 6,)
        of_input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_out = of_input.mish()

        np_out, np_grad = np_mish(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-5, 1e-5))

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_silu(test_case):
        def np_silu(x):
            _sig = 1 / (1 + np.exp(-x))
            y = x * _sig
            y_grad = _sig * (1 + x * (1 - _sig))
            return [y, y_grad]

        np_input = np.random.randn(2, 4, 5, 6,)
        of_input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_out = of_input.silu()

        np_out, np_grad = np_silu(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-5, 1e-5))

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_selu(test_case):
        _scale = 1.0507009873554804934193349852946
        _alpha = 1.6732632423543772848170429916717

        def np_selu(x):
            y = np.where(x < 0, _scale * _alpha * (np.exp(x) - 1), _scale * x)
            y_grad = np.where(x < 0, _scale * _alpha * np.exp(x), _scale)
            return [y, y_grad]

        np_input = np.random.randn(2, 4, 5, 6,)
        of_input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_out = of_input.selu()

        np_out, np_grad = np_selu(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-5, 1e-5))

    @unittest.skip("still have error in ci")
    def test_tensor_softsign(test_case):
        def np_softsign(x):
            y = x / (1 + np.abs(x))
            y_grad = 1 / np.square(1 + np.abs(x))
            return [y, y_grad]

        np_input = np.random.randn(2, 4, 5, 6,)
        of_input = flow.Tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_out = of_input.softsign()

        np_out, np_grad = np_softsign(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-5, 1e-5))


if __name__ == "__main__":
    unittest.main()
