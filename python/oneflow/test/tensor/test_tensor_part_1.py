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

import copy
import os
import numpy as np
import unittest
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestTensor(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_numpy_and_default_dtype(test_case):
        shape = (2, 3, 4, 5)
        tensor = flow.Tensor(*shape)
        flow.nn.init.ones_(tensor)
        test_case.assertTrue(tensor.dtype == flow.float32)
        test_case.assertTrue(
            np.allclose(tensor.numpy(), np.ones(shape, dtype=np.float32))
        )

        shape = flow.Size((2, 3, 4, 5))
        tensor = flow.Tensor(shape)
        flow.nn.init.ones_(tensor)
        test_case.assertTrue(tensor.dtype == flow.float32)
        test_case.assertTrue(
            np.allclose(tensor.numpy(), np.ones(shape, dtype=np.float32))
        )

        shape = flow.Size((2, 3))
        tensor = flow.Tensor(shape)
        flow.nn.init.eye_(tensor)
        test_case.assertTrue(tensor.dtype == flow.float32)
        test_case.assertTrue(np.allclose(tensor.numpy(), np.eye(2, 3)))

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_deepcopy(test_case):
        shape = (2, 3)
        tensor1 = flow.ones(*shape).cuda()
        tensor2 = copy.deepcopy(tensor1)
        tensor1[0, 0] = 0
        test_case.assertEqual(tensor1.device, tensor2.device)
        test_case.assertEqual(tensor1[0, 0], 0)
        test_case.assertEqual(tensor2[0, 0], 1)

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_property(test_case):
        shape = (2, 3, 4, 5)
        tensor = flow.Tensor(*shape)
        test_case.assertEqual(tensor.storage_offset(), 0)
        test_case.assertEqual(tensor.stride(), (60, 20, 5, 1))
        test_case.assertEqual(tensor.is_cuda, False)
        test_case.assertTrue(tensor.is_contiguous())

    @flow.unittest.skip_unless_1n1d()
    def test_copy_to_and_from_numpy(test_case):
        np_arr = np.array([4, 6], dtype=np.float32)
        tensor = flow.tensor(np_arr, dtype=flow.float32)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr))
        test_case.assertEqual(np.float32, tensor.numpy().dtype)
        np_arr = np.array([4, 6], dtype=np.int32)
        tensor = flow.tensor(np_arr, dtype=flow.int32)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr))
        test_case.assertEqual(np.int32, tensor.numpy().dtype)
        np_arr = np.array([4, 6], dtype=np.float16)
        tensor = flow.tensor(np_arr, dtype=flow.float16)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr))
        test_case.assertEqual(np.float16, tensor.numpy().dtype)

    @flow.unittest.skip_unless_1n1d()
    def test_inplace_copy_from_contiguous_numpy(test_case):
        np_arr = np.arange(6).reshape(3, 2)
        tensor = flow.zeros(3, 2).to(flow.int64)
        tensor.copy_(np_arr)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr))

    @flow.unittest.skip_unless_1n1d()
    def test_inplace_copy_from_non_contiguous_numpy(test_case):
        np_arr = np.arange(6).reshape(2, 3).transpose(1, 0)
        tensor = flow.zeros(3, 2).to(flow.int64)
        tensor.copy_(np_arr)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr))

    @flow.unittest.skip_unless_1n1d()
    def test_construct_from_numpy_or_list(test_case):
        shape = (2, 3, 4, 5)
        np_arr = np.random.rand(*shape).astype(np.float32)
        tensor = flow.tensor(np_arr)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr))
        np_int_arr = np.random.randint(-100, high=100, size=shape, dtype=np.int32)
        tensor = flow.tensor(np_int_arr, dtype=flow.int32)
        test_case.assertEqual(tensor.dtype, flow.int32)
        test_case.assertTrue(np_arr.flags["C_CONTIGUOUS"])
        test_case.assertTrue(np.allclose(tensor.numpy(), np_int_arr))
        np_arr = np.random.random((1, 256, 256, 3)).astype(np.float32)
        np_arr = np_arr.transpose(0, 3, 1, 2)
        tensor = flow.tensor(np_arr)
        test_case.assertFalse(np_arr.flags["C_CONTIGUOUS"])
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr))

    @flow.unittest.skip_unless_1n1d()
    def test_construct_from_another_tensor(test_case):
        shape = (2, 3, 4, 5)
        np_arr = np.random.rand(*shape).astype(np.float32)
        tensor = flow.tensor(np_arr)
        output = flow.tensor(tensor)
        test_case.assertEqual(output.dtype, flow.float32)
        test_case.assertTrue(np.allclose(output.numpy(), np_arr))

    @flow.unittest.skip_unless_1n1d()
    def test_construct_np_array_from_tensor(test_case):
        tensor = flow.randn(5)
        np_arr = np.array(tensor)
        test_case.assertEqual(np_arr.shape, (5,))
        test_case.assertEqual(np_arr.dtype, np.float32)
        test_case.assertTrue(np.allclose(np_arr, tensor.numpy()))
        test_case.assertEqual(str(np_arr), str(tensor.numpy()))

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_sign_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.sign()
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_flow_tensor_gather_with_random_data(test_case):
        device = random_device()
        input = random_tensor(ndim=4, dim1=3, dim2=4, dim3=5).to(device)
        dim = random(0, 4).to(int).value()
        index = random_tensor(
            ndim=4,
            dim1=random(1, 3).to(int),
            dim2=random(1, 4).to(int),
            dim3=random(1, 5).to(int),
            low=0,
            high=1 if dim == 0 else dim,
            dtype=int,
        ).to(device)
        return input.gather(dim, index)

    def _test_tensor_init_methods(test_case, tensor_creator, get_numpy):
        for dtype in [flow.float32, flow.float16]:
            shape = (2, 3, 4, 5)
            x = tensor_creator(*shape).to(dtype)
            np_ones = np.ones(x.shape)
            np_zeros = np.zeros(x.shape)
            random_fill_val = 2.0
            x.fill_(random_fill_val)
            test_case.assertTrue(np.allclose(get_numpy(x), random_fill_val * np_ones))
            flow.nn.init.ones_(x)
            test_case.assertTrue(np.allclose(get_numpy(x), np_ones))
            flow.nn.init.zeros_(x)
            test_case.assertTrue(np.allclose(get_numpy(x), np_zeros))
            flow.nn.init.constant_(x, random_fill_val)
            test_case.assertTrue(np.allclose(get_numpy(x), random_fill_val * np_ones))
            z = tensor_creator(5, 4, 3, 2)
            flow.nn.init.kaiming_normal_(z, a=0.1, mode="fan_out", nonlinearity="relu")
            flow.nn.init.kaiming_uniform_(z)
            z.requires_grad_()
            flow.nn.init.xavier_normal_(z, flow.nn.init.calculate_gain("relu"))
            flow.nn.init.xavier_uniform_(z, flow.nn.init.calculate_gain("relu"))
            flow.nn.init.xavier_normal_(
                z, flow.nn.init.calculate_gain("leaky_relu", 0.2)
            )
            flow.nn.init.xavier_uniform_(
                z, flow.nn.init.calculate_gain("leaky_relu", 0.2)
            )
            flow.nn.init.trunc_normal_(z, mean=0.0, std=1.0, a=-2.0, b=2.0)
            flow.nn.init.normal_(z, mean=0.0, std=1.0)
            flow.nn.init.orthogonal_(z)

        x = tensor_creator(*shape).to(dtype=flow.int32)
        np_ones = np.ones(x.shape, dtype=np.int32)
        np_zeros = np.zeros(x.shape, dtype=np.int32)
        random_fill_val = -2
        x.fill_(random_fill_val)
        test_case.assertTrue(np.allclose(get_numpy(x), random_fill_val * np_ones))
        flow.nn.init.ones_(x)
        test_case.assertTrue(np.allclose(get_numpy(x), np_ones))
        flow.nn.init.zeros_(x)
        test_case.assertTrue(np.allclose(get_numpy(x), np_zeros))
        flow.nn.init.constant_(x, random_fill_val)
        test_case.assertTrue(np.allclose(get_numpy(x), random_fill_val * np_ones))
        x.zero_()
        test_case.assertTrue(np.array_equal(get_numpy(x), np_zeros))
        test_case.assertEqual(flow.nn.init.calculate_gain("conv2d"), 1)
        test_case.assertEqual(flow.nn.init.calculate_gain("tanh"), 5.0 / 3)

    def _test_non_contiguous_tensor_init_methods(test_case, tensor_creator, get_numpy):
        shape = (8, 8)
        x = flow.zeros(shape)
        sliced_x = x[::2, 1::2]
        not_sliced_x = x[1::2, ::2]
        random_fill_val = 923.53
        np_zeros = np.zeros((4, 4))
        # ones
        flow.nn.init.ones_(sliced_x)
        test_case.assertTrue(np.allclose(get_numpy(sliced_x), np.ones((4, 4))))
        test_case.assertTrue(np.allclose(get_numpy(not_sliced_x), np_zeros))
        # constant
        flow.nn.init.constant_(sliced_x, random_fill_val)
        test_case.assertTrue(
            np.allclose(get_numpy(sliced_x), np.ones((4, 4)) * random_fill_val)
        )
        test_case.assertTrue(np.allclose(get_numpy(not_sliced_x), np_zeros))
        # eye
        flow.nn.init.eye_(sliced_x)
        test_case.assertTrue(np.allclose(get_numpy(sliced_x), np.eye(4)))
        test_case.assertTrue(np.allclose(get_numpy(not_sliced_x), np_zeros))
        # kaiming_normal_
        flow.nn.init.kaiming_normal_(
            sliced_x, a=0.1, mode="fan_out", nonlinearity="relu"
        )
        test_case.assertTrue(np.allclose(get_numpy(not_sliced_x), np_zeros))
        # kaiming_uniform_
        flow.nn.init.kaiming_uniform_(sliced_x)
        test_case.assertTrue(np.allclose(get_numpy(not_sliced_x), np_zeros))
        # xavier_normal_ with relu gain
        flow.nn.init.xavier_normal_(sliced_x, flow.nn.init.calculate_gain("relu"))
        test_case.assertTrue(np.allclose(get_numpy(not_sliced_x), np_zeros))
        # xavier_uniform_ with relu gain
        flow.nn.init.xavier_uniform_(sliced_x, flow.nn.init.calculate_gain("relu"))
        test_case.assertTrue(np.allclose(get_numpy(not_sliced_x), np_zeros))
        # trunc_normal_
        flow.nn.init.trunc_normal_(sliced_x, mean=0.0, std=1.0, a=-2.0, b=2.0)
        test_case.assertTrue(np.allclose(get_numpy(not_sliced_x), np_zeros))
        # normal_
        flow.nn.init.normal_(sliced_x, mean=0.0, std=1.0)
        test_case.assertTrue(np.allclose(get_numpy(not_sliced_x), np_zeros))
        # orthogonal_
        flow.nn.init.orthogonal_(sliced_x)
        test_case.assertTrue(np.allclose(get_numpy(not_sliced_x), np_zeros))

    @flow.unittest.skip_unless_1n1d()
    def test_local_tensor_init_methods(test_case):
        for device in ["cpu", "cuda"]:
            test_case._test_tensor_init_methods(
                lambda *args, **kwargs: flow.Tensor(*args, **kwargs, device=device),
                lambda x: x.numpy(),
            )
            test_case._test_non_contiguous_tensor_init_methods(
                lambda *args, **kwargs: flow.Tensor(*args, **kwargs, device=device),
                lambda x: x.numpy(),
            )

    @flow.unittest.skip_unless_1n2d()
    def test_global_tensor_init_methods(test_case):
        for device in ["cpu", "cuda"]:
            test_case._test_tensor_init_methods(
                lambda *args, **kwargs: flow.Tensor(
                    *args,
                    **kwargs,
                    sbp=flow.sbp.broadcast,
                    placement=flow.placement(device, range(2))
                ),
                lambda x: x.to_global(sbp=flow.sbp.broadcast).to_local().numpy(),
            )

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_with_single_int(test_case):
        x = flow.Tensor(5)
        test_case.assertEqual(x.shape, flow.Size([5]))
        x = flow.tensor(5)
        test_case.assertEqual(x.numpy().item(), 5)

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_device(test_case):
        shape = (2, 3, 4, 5)
        x = flow.Tensor(*shape)
        test_case.assertTrue(not x.is_cuda)
        x = flow.Tensor(*shape, device=flow.device("cuda"))
        test_case.assertTrue(x.is_cuda)
        x = flow.Tensor(*shape, device=flow.device("cpu"))
        test_case.assertTrue(not x.is_cuda)

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=1, check_graph=True)
    def test_tensor_set_data_autograd_meta(test_case):
        x = torch.ones(2, 3).requires_grad_(True)
        y = x + x
        z = torch.zeros(2, 3)
        z.data = y
        return z.grad_fn, z.is_leaf

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_set_data(test_case):
        a = flow.ones(2, 3, requires_grad=False)
        b = flow.ones(4, 5, requires_grad=True).to("cuda")
        old_id = id(a)
        a.data = b
        test_case.assertEqual(old_id, id(a))
        test_case.assertTrue(a.shape == (4, 5))
        test_case.assertTrue(a.device == flow.device("cuda"))
        test_case.assertFalse(a.requires_grad)
        test_case.assertTrue(a.is_leaf)

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_unsupported_property(test_case):

        shape = (2, 3, 4, 5)
        x = flow.Tensor(*shape)
        test_case.assertTrue(x.is_local)

        with test_case.assertRaises(RuntimeError):
            x.global_id()

        with test_case.assertRaises(RuntimeError):
            x.sbp

        with test_case.assertRaises(RuntimeError):
            x.placement

        if x.dtype != flow.tensor_buffer:
            with test_case.assertRaises(RuntimeError):
                x._tensor_buffer_shapes_and_dtypes

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_to_bool(test_case):
        x = flow.tensor([0.0])
        test_case.assertFalse(bool(x))
        x = flow.tensor([0.0]).to("cuda")
        test_case.assertFalse(bool(x))
        x = flow.tensor([1.5])
        test_case.assertTrue(bool(x))
        x = flow.tensor([3])
        test_case.assertTrue(bool(x))
        with test_case.assertRaises(RuntimeError):
            bool(flow.tensor([1, 3, 5]))
            bool(flow.tensor([]))

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_autograd_fill_cpu(test_case):
        shape = (2, 3, 4, 5)
        x = flow.Tensor(*shape)
        y = flow.Tensor(*shape)
        x.fill_(1.0)
        y.fill_(flow.tensor(1.0))
        y.requires_grad = True
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
        v = flow.Tensor(*shape)
        v.requires_grad = True
        z.retain_grad()
        w = v + z
        grad = flow.Tensor(*shape)
        grad.fill_(1.0)
        w.backward(gradient=grad, retain_graph=True)
        test_case.assertTrue(
            np.allclose(v.grad.numpy(), np.ones(shape), atol=1e-4, rtol=1e-4)
        )
        test_case.assertTrue(
            np.allclose(y.grad.numpy(), np.ones(shape), atol=1e-4, rtol=1e-4)
        )
        test_case.assertTrue(
            np.allclose(z.grad.numpy(), np.ones(shape), atol=1e-4, rtol=1e-4)
        )
        test_case.assertIsNone(x.grad)
        test_case.assertIsNotNone(y.grad)
        w.backward(gradient=grad, retain_graph=True)
        # autocast test for fill_
        x = flow.tensor([2.4, 3.5], device="cuda", dtype=flow.float16)
        with flow.amp.autocast("cuda", flow.float16):
            y = x.clone()
            y.fill_(2.36)
            test_case.assertTrue(y.dtype == flow.float16)

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_autograd_fill_cuda(test_case):
        shape = (2, 3, 4, 5)
        x = flow.Tensor(*shape).to("cuda:0")
        y = flow.Tensor(*shape).to("cuda:0")
        x.fill_(1.0)
        y.fill_(flow.tensor(1.0).to("cuda:0"))
        y.requires_grad = True
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
        v = flow.Tensor(*shape).to("cuda:0")
        v.requires_grad = True
        z.retain_grad()
        w = v + z
        grad = flow.Tensor(*shape)
        grad.fill_(1.0)
        w.backward(gradient=grad, retain_graph=True)
        test_case.assertTrue(
            np.allclose(v.grad.numpy(), np.ones(shape), atol=1e-4, rtol=1e-4)
        )
        test_case.assertTrue(
            np.allclose(y.grad.numpy(), np.ones(shape), atol=1e-4, rtol=1e-4)
        )
        test_case.assertTrue(
            np.allclose(z.grad.numpy(), np.ones(shape), atol=1e-4, rtol=1e-4)
        )
        test_case.assertIsNone(x.grad)
        test_case.assertIsNotNone(y.grad)
        w.backward(gradient=grad, retain_graph=True)

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_register_post_grad_accumulation_hook(test_case):
        shape = (2, 3)
        x = flow.Tensor(*shape)
        x.requires_grad = True
        x._register_post_grad_accumulation_hook(lambda grad: grad * 2 + 1)
        y = x.sum() + (x * 2).sum()
        y.backward()
        test_case.assertTrue(
            np.allclose(x.grad.numpy(), np.ones(shape) * 7, atol=1e-4, rtol=1e-4)
        )

        x = flow.Tensor(*shape)
        x.requires_grad = True

        def inplace_add_and_return_none(x):
            x.add_(1)
            return None

        x._register_post_grad_accumulation_hook(inplace_add_and_return_none)
        y = x.sum() + (x * 2).sum()
        y.backward()
        test_case.assertTrue(
            np.allclose(x.grad.numpy(), np.ones(shape) * 4, atol=1e-4, rtol=1e-4)
        )

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_register_hook(test_case):
        shape = (2, 3)
        x = flow.Tensor(*shape)
        x.requires_grad = True
        x.register_hook(lambda grad: grad * 2 + 1)
        y = x.sum() + (x * 2).sum()
        y.backward()
        test_case.assertTrue(
            np.allclose(x.grad.numpy(), np.ones(shape) * 7, atol=1e-4, rtol=1e-4)
        )
        x = flow.Tensor(*shape)
        x.requires_grad = True
        new_grad = flow.Tensor([[1, 2, 3], [4, 5, 6]])
        x.register_hook(lambda _: new_grad)
        y = x.sum() + (x * 2).sum()
        y.backward()
        test_case.assertTrue(np.allclose(x.grad.numpy(), new_grad.numpy()))
        grad_nonlocal = None

        def assign_nonlocal_variable_and_return_none(grad):
            nonlocal grad_nonlocal
            grad_nonlocal = grad

        x = flow.Tensor(*shape)
        x.requires_grad = True
        new_grad = flow.tensor([[1, 2, 3], [4, 5, 6]], dtype=flow.float32)
        x.register_hook(assign_nonlocal_variable_and_return_none)
        y = x.sum() + (x * 2).sum()
        y.backward()
        test_case.assertTrue(np.allclose(grad_nonlocal.numpy(), np.ones(shape) * 3))

    @flow.unittest.skip_unless_1n1d()
    def test_non_leaf_tensor_register_hook(test_case):
        shape = (2, 3)
        x = flow.Tensor(*shape).requires_grad_()
        y = x + 1
        y.register_hook(lambda grad: grad * 2)
        z1 = y * 2
        z2 = y * 3
        loss = (z1 + z2).sum()
        loss.backward(retain_graph=True)
        loss.backward()
        test_case.assertTrue(np.allclose(x.grad.numpy(), np.ones(shape) * 20))

    @flow.unittest.skip_unless_1n1d()
    def test_user_defined_data(test_case):
        list_data = [5, 5]
        tuple_data = (5, 5)
        numpy_data = np.array((5, 5))
        x = flow.Tensor(list_data)
        y = flow.Tensor(tuple_data)
        z = flow.Tensor(numpy_data)
        test_case.assertTrue(np.allclose(x.numpy(), 5 * np.ones(x.shape)))
        test_case.assertTrue(np.allclose(y.numpy(), 5 * np.ones(y.shape)))
        test_case.assertTrue(np.allclose(z.numpy(), 5 * np.ones(z.shape)))

    @flow.unittest.skip_unless_1n1d()
    def test_local_tensor_and_op(test_case):
        x1 = flow.Tensor([[1.0, 2.0]])
        test_case.assertEqual(x1.dtype, flow.float32)
        test_case.assertEqual(x1.shape, flow.Size((1, 2)))
        x2 = flow.Tensor([[1.0], [2.0]])
        y = flow.matmul(x1, x2)
        test_case.assertTrue(
            np.allclose(y.numpy(), np.array([[5.0]], dtype=np.float32))
        )

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5, rtol=1e-2, atol=1e-3)
    def test_matmul_with_random_data(test_case):
        device = random_device()
        dim0 = random(low=2, high=10).to(int)
        dim1 = random(low=3, high=20).to(int)
        dim2 = random(low=2, high=11).to(int)
        a = random_tensor(ndim=2, dim0=dim0, dim1=dim1).to(device)
        b = random_tensor(ndim=2, dim0=dim1, dim1=dim2).to(device)
        return a @ b

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_mv_with_random_data(test_case):
        device = random_device()
        dim0 = random(low=2, high=10).to(int)
        dim1 = random(low=3, high=20).to(int)
        a = random_tensor(ndim=2, dim0=dim0, dim1=dim1).to(device)
        b = random_tensor(ndim=1, dim0=dim1).to(device)
        return a.mv(b)

    @flow.unittest.skip_unless_1n1d()
    @autotest(check_graph=True, rtol=1e-2, atol=1e-3)
    def test_mm_with_random_data(test_case):
        device = random_device()
        dim0 = random(low=2, high=10).to(int)
        dim1 = random(low=3, high=20).to(int)
        dim2 = random(low=2, high=11).to(int)
        a = random_tensor(ndim=2, dim0=dim0, dim1=dim1).to(device)
        b = random_tensor(ndim=2, dim0=dim1, dim1=dim2).to(device)
        return a.mm(b)

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_to_list(test_case):
        list_data = [[1.0, 3.0], [5.0, 6.0]]
        input = flow.Tensor(list_data)
        test_case.assertEqual(list_data, input.tolist())

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_nelement(test_case):
        shape = (2, 3, 4)
        input = flow.Tensor(*shape)
        test_case.assertEqual(input.nelement(), 24)

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_numel(test_case):
        shape = (2, 3, 4, 5)
        input = flow.Tensor(*shape)
        test_case.assertEqual(input.numel(), 120)

    @flow.unittest.skip_unless_1n1d()
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
        requires_grad_input = flow.Tensor(*shape)
        requires_grad_input.requires_grad = True
        requires_grad_input_str = str(requires_grad_input)
        test_case.assertTrue("requires_grad=" in requires_grad_input_str)

    @flow.unittest.skip_unless_1n1d()
    def test_indexing(test_case):
        class SliceExtracter:
            def __getitem__(self, key):
                return key

        se = SliceExtracter()

        def compare_getitem_with_numpy(tensor, slices):
            np_arr = tensor.numpy()
            test_case.assertTrue(np.allclose(np_arr[slices], tensor[slices].numpy()))

        def compare_setitem_with_numpy(tensor, slices, value):
            np_arr = tensor.numpy()
            if isinstance(value, flow.Tensor):
                np_value = value.numpy()
            else:
                np_value = value
            np_arr[slices] = np_value
            tensor[slices] = value
            test_case.assertTrue(np.allclose(np_arr, tensor.numpy(), rtol=1e-4))

        x = flow.randn(5, 5)
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

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5, auto_backward=False)
    def test_setitem_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=0, high=0, ndim=1, dim0=16, requires_grad=False).to(
            device
        )
        y = random_tensor(low=-2, high=2, ndim=1, dim0=16).to(device)
        idx = random_tensor(
            low=0, high=15, ndim=1, dim0=20, dtype=int, requires_grad=False
        ).to(device)

        getitem_of = y.oneflow[idx.oneflow]
        getitem_torch = y.pytorch[idx.pytorch]
        test_case.assertTrue(
            np.allclose(getitem_of.numpy(), getitem_torch.detach().cpu().numpy())
        )

        x.oneflow[idx.oneflow] = getitem_of
        x.pytorch[idx.pytorch] = getitem_torch
        return x

    @flow.unittest.skip_unless_1n1d()
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

    @flow.unittest.skip_unless_1n1d()
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

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_mul_inplace_tensor(test_case):
        device = random_device()
        rand_tensor = random_tensor(
            low=-2, high=2, ndim=4, dim0=16, dim1=9, dim2=4, dim3=7
        ).to(device)
        y = rand_tensor + 1
        x = random_tensor(low=-2, high=2, ndim=4, dim0=16, dim1=9, dim2=4, dim3=7).to(
            device
        )
        y.mul_(x)
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_broadcast_mul_inplace_tensor(test_case):
        device = random_device()
        rand_tensor = random_tensor(ndim=3, dim0=4, dim1=8, dim2=13).to(device)
        y = rand_tensor + 1
        x = random_tensor(ndim=2, dim0=8, dim1=13).to(device)
        y.mul_(x)
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_div_inplace_tensor(test_case):
        device = random_device()
        rand_tensor = random_tensor(
            low=-2, high=2, ndim=4, dim0=26, dim1=7, dim2=4, dim3=17
        ).to(device)
        y = rand_tensor + 1
        x = random_tensor(low=-2, high=2, ndim=4, dim0=26, dim1=7, dim2=4, dim3=17).to(
            device
        )
        y.div_(x)
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_broadcast_div_inplace_tensor(test_case):
        device = random_device()
        rand_tensor = random_tensor(ndim=3, dim0=4, dim1=8, dim2=13).to(device)
        y = rand_tensor + 1
        x = random_tensor(ndim=2, dim0=8, dim1=13).to(device)
        y.div_(x)
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_add_inplace_tensor(test_case):
        device = random_device()
        rand_tensor = random_tensor(
            low=-2, high=2, ndim=4, dim0=6, dim1=9, dim2=14, dim3=17
        ).to(device)
        y = rand_tensor + 1
        x = random_tensor(low=-2, high=2, ndim=4, dim0=6, dim1=9, dim2=14, dim3=17).to(
            device
        )
        y.add_(x)
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_broadcast_add_inplace_tensor(test_case):
        device = random_device()
        rand_tensor = random_tensor(ndim=3, dim0=5, dim1=9, dim2=23).to(device)
        y = rand_tensor + 1
        x = random_tensor(ndim=2, dim0=9, dim1=23).to(device)
        y.add_(x)
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_sub_inplace_tensor(test_case):
        device = random_device()
        rand_tensor = random_tensor(
            low=-2, high=2, ndim=4, dim0=6, dim1=9, dim2=14, dim3=17
        ).to(device)
        y = rand_tensor + 1
        x = random_tensor(low=-2, high=2, ndim=4, dim0=6, dim1=9, dim2=14, dim3=17).to(
            device
        )
        y.sub_(x)
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_broadcast_sub_inplace_tensor(test_case):
        device = random_device()
        rand_tensor = random_tensor(ndim=3, dim0=5, dim1=9, dim2=23).to(device)
        y = rand_tensor + 1
        x = random_tensor(ndim=2, dim0=9, dim1=23).to(device)
        y.sub_(x)
        return y

    @flow.unittest.skip_unless_1n1d()
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

    @flow.unittest.skip_unless_1n1d()
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

    @flow.unittest.skip_unless_1n1d()
    def test_sum(test_case):
        input = flow.tensor(np.random.randn(4, 5, 6), dtype=flow.float32)
        of_out = input.sum(dim=(2, 1))
        np_out = np.sum(input.numpy(), axis=(2, 1))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))

    @flow.unittest.skip_unless_1n1d()
    def test_argwhere(test_case):
        shape = (2, 3, 4, 5)
        precision = 1e-5
        np_input = np.random.randn(*shape)
        input = flow.Tensor(np_input)
        of_out = input.argwhere()
        np_out = np.argwhere(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, precision, precision))
        test_case.assertTrue(np.allclose(of_out.numpy().shape, np_out.shape))

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_tensor_argmax_with_random_data(test_case):
        device = random_device()
        ndim = random(1, 6).to(int)
        x = random_tensor(ndim=ndim).to(device)
        y = x.argmax(dim=random(0, ndim).to(int), keepdim=random().to(bool))
        return y

    @autotest(auto_backward=False, check_graph=False)
    def test_max_bool_input_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(
            device, dtype=torch.bool
        )
        return x.max(dim)

    @autotest(auto_backward=False, check_graph=False)
    def test_min_bool_input_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float, requires_grad=False).to(
            device, dtype=torch.bool
        )
        return x.min(dim)

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_tanh_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.tanh()
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_flow_tensor_asin_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-0.5, high=0.5).to(device)
        y = x.asin()
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_flow_tensor_arcsin_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-0.5, high=0.5).to(device)
        y = x.arcsin()
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_flow_tensor_asinh_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.asinh()
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_flow_tensor_arcsinh_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.arcsinh()
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_flow_tensor_sinh_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.sinh()
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_flow_tensor_atan2_with_random_data(test_case):
        device = random_device()
        x1 = random_tensor(ndim=1, dim0=1).to(device)
        x2 = random_tensor(ndim=1, dim0=1).to(device)
        y = x1.atan2(x2)
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_dot(test_case):
        device = random_device()
        k = random(10, 100)
        x = random_tensor(ndim=1, dim0=k).to(device)
        y = random_tensor(ndim=1, dim0=k).to(device)
        z = x.dot(y)
        return z

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_arccos_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=2, high=3).to(device)
        y = x.arccos()
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_arccosh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=2, high=3).to(device)
        y = x.arccosh()
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_acosh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=2, high=3).to(device)
        y = x.acosh()
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(auto_backward=False, check_graph=True)
    def test_sort_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = x.sort(dim=random(low=-4, high=4).to(int), descending=random_bool())
        return y[0], y[1]

    @flow.unittest.skip_unless_1n1d()
    @autotest(auto_backward=False, check_graph=True)
    def test_sort_tensor_return_type(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        result = x.sort(dim=random(low=-4, high=4).to(int), descending=random_bool())
        return result.values, result.indices

    @flow.unittest.skip_unless_1n1d()
    @autotest(auto_backward=False, check_graph=True)
    def test_argsort_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = x.argsort(dim=random(low=-4, high=4).to(int), descending=random_bool())
        return y

    @autotest(n=5)
    def test_mean_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(ndim=4, dtype=float).to(device)
        return x.mean(dim)

    @autotest(n=5)
    def test_log_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return x.log()

    @autotest(n=5)
    def test_log1p_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return x.log1p()

    @autotest(n=5)
    def test_log2_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return x.log2()

    @autotest(n=5)
    def test_log10_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return x.log10()

    @autotest(n=5)
    def test_neg_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return -x

    @autotest(n=5)
    def test_negative_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return x.negative()

    @autotest(n=5)
    def test_neg_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return x.neg()

    @autotest(auto_backward=False, check_graph=True)
    def test_greater_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=3, dim1=2, dim2=3).to(device)
        y = random_tensor(ndim=3, dim1=2, dim2=3).to(device)
        return x.gt(y)

    @autotest(auto_backward=False, check_graph=True)
    def test_less_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=3, dim1=2, dim2=3).to(device)
        y = random_tensor(ndim=3, dim1=2, dim2=3).to(device)
        return x.lt(y)

    @autotest(auto_backward=False, check_graph=True)
    def test_tensor_topk_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dim1=8, dim2=9, dim3=10).to(device)
        y = x.topk(
            random(low=1, high=8).to(int),
            dim=random(low=1, high=4).to(int) | nothing(),
            largest=random_bool() | nothing(),
            sorted=constant(True) | nothing(),
        )
        return y[0], y[1]

    @autotest(auto_backward=False, check_graph=True)
    def test_tensor_topk_return_type(test_case):
        device = random_device()
        x = random_tensor(ndim=4, dim1=8, dim2=9, dim3=10).to(device)
        result = x.topk(
            random(low=1, high=8).to(int),
            dim=random(low=1, high=4).to(int),
            largest=random_bool(),
            sorted=constant(True),
        )
        return result.values, result.indices

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_fmod_element_with_random_data(test_case):
        device = random_device()
        dim1 = random().to(int)
        dim2 = random().to(int)
        input = random_tensor(ndim=3, dim1=dim1, dim2=dim2).to(device)
        other = random_tensor(ndim=3, dim1=dim1, dim2=dim2).to(device)
        return input.fmod(other)

    @autotest(auto_backward=False, check_graph=True)
    def test_flow_fmod_broadcast_with_random_data(test_case):
        device = random_device()
        dim1 = random().to(int)
        dim2 = random().to(int)
        input = random_tensor(ndim=3, dim1=constant(1), dim2=dim2).to(device)
        other = random_tensor(ndim=3, dim1=dim1, dim2=constant(1)).to(device)
        return input.fmod(other)

    @autotest(auto_backward=True, check_graph=True)
    def test_flow_fmod_scalar_with_random_data(test_case):
        device = random_device()
        dim1 = random().to(int)
        dim2 = random().to(int)
        input = random_tensor(ndim=3, dim1=dim1, dim2=dim2).to(device)
        other = 3
        return input.fmod(other)

    @autotest(auto_backward=False, check_graph=True)
    def test_fmod_with_0_size_data(test_case):
        device = random_device()
        x = random_tensor(4, 2, 1, 0, 3).to(device)
        y = x.fmod(2)
        return y

    @autotest(n=5)
    def test_tensor_flip_list_with_random_data(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4, dim1=random().to(int), dim2=random().to(int), dim3=random().to(int)
        ).to(device)
        y = x.flip(constant([0, 1, 2]))
        return y

    @autotest(n=5)
    def test_tensor_flip_tuple_with_random_data(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4, dim1=random().to(int), dim2=random().to(int), dim3=random().to(int)
        ).to(device)
        y = x.flip(constant((0, 1, 2)))
        return y

    @autotest(n=5)
    def test_tensor_chunk_list_with_random_data(test_case):
        device = random_device()
        dim = random(1, 4).to(int)
        x = random_tensor(
            ndim=4,
            dim1=random(low=4, high=8).to(int),
            dim2=random(low=4, high=8).to(int),
            dim3=random(low=4, high=8).to(int),
        ).to(device)
        y = x.chunk(chunks=random(low=1, high=5).to(int), dim=dim)
        z = torch.cat(y, dim=dim)
        return z

    @autotest(n=5)
    def test_tensor_reciprocal_list_with_random_data(test_case):
        device = random_device()
        x = random_tensor(
            ndim=4, dim1=random().to(int), dim2=random().to(int), dim3=random().to(int)
        ).to(device)
        y = x.reciprocal()
        return y

    @flow.unittest.skip_unless_1n1d()
    def test_tensor_slice(test_case):
        x = np.random.randn(2, 3, 4, 5).astype(np.float32)
        input = flow.tensor(x)
        test_case.assertTrue(np.allclose(input[0].numpy(), x[0], 1e-05, 1e-05))
        test_case.assertTrue(np.allclose(input[1].numpy(), x[1], 1e-05, 1e-05))
        test_case.assertTrue(np.allclose(input[0, :].numpy(), x[0, :], 1e-05, 1e-05))
        test_case.assertTrue(
            np.allclose(input[0, :, 0:2].numpy(), x[0, :, 0:2], 1e-05, 1e-05)
        )

    @flow.unittest.skip_unless_1n1d()
    def test_zeros_(test_case):
        shape = (2, 3)
        x = flow.tensor(np.random.randn(*shape), dtype=flow.float32)
        x.zero_()
        test_case.assertTrue(np.allclose(x.numpy(), np.zeros(shape)))

    @flow.unittest.skip_unless_1n1d()
    def test_construct_small_tensor(test_case):
        shape = (2, 3, 4, 5)
        np_arr = np.random.rand(*shape).astype(np.float32)
        tensor = flow.tensor(np_arr)
        test_case.assertTrue(np.allclose(tensor.numpy(), np_arr))
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
        test_case.assertTrue(np.allclose(tensor.numpy(), np.array(tuple_data)))
        scalar = 5.5
        tensor = flow.tensor(scalar)
        test_case.assertEqual(tensor.dtype, flow.float32)
        test_case.assertTrue(
            np.allclose(tensor.numpy(), np.array(scalar), 0.0001, 0.0001)
        )

    @autotest(n=5)
    def test_tensor_floor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.floor()
        return y

    @autotest(n=5)
    def test_tensor_round_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.round()
        return y

    def _test_tensor_reshape(test_case):
        x = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
        ).astype(np.float32)
        input = flow.tensor(x)
        of_shape = input.reshape(2, 2, 2, -1).numpy().shape
        np_shape = (2, 2, 2, 2)
        test_case.assertTrue(np.allclose(of_shape, np_shape))

    @autotest(n=5)
    def test_flatten_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.flatten(
            start_dim=random(1, 6).to(int) | nothing(),
            end_dim=random(1, 6).to(int) | nothing(),
        )
        return y

    @autotest(n=5)
    def test_reshape_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = x.reshape(-1)
        return y

    @autotest(n=1)
    def test_reshape_tensor_with_random_data_and_keyword(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = x.reshape(shape=[-1,])
        return y

    @autotest(n=5)
    def test_reshape_as_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = x.reshape(-1)
        z = y.reshape_as(other=x)
        return z

    @autotest(n=5)
    def test_tensor_squeeze_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.squeeze(random().to(int))
        return y

    @autotest(n=5)
    def test_flow_unsqueeze_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.unsqueeze(random(1, 3).to(int))
        return y

    @autotest(n=3, auto_backward=False, check_graph=True)
    def test_flow_invert_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device, dtype=torch.bool)
        y = ~x
        return y

    def test_tensor_float(test_case):
        x = flow.tensor(1)
        y = float(x)
        test_case.assertTrue(np.array_equal(y, 1.0))

    def test_tensor_int(test_case):
        x = flow.tensor(2.3)
        y = int(x)
        test_case.assertTrue(np.array_equal(y, 2))

    def test_none_equal(test_case):
        xt = flow.randn(10)
        yt = flow.randn(10)
        z = None in [xt, yt]
        test_case.assertTrue(np.array_equal(z, False))
        zt = None
        z = None in [xt, yt, zt]
        test_case.assertTrue(np.array_equal(z, True))

    def test_half(test_case):
        x = flow.tensor([1], dtype=flow.int64)
        test_case.assertTrue(x.dtype == flow.int64)
        y = x.half()
        test_case.assertTrue(y.dtype == flow.float16)

    def test_byte(test_case):
        x = flow.tensor([1.2], dtype=flow.float32)
        test_case.assertTrue(x.dtype == flow.float32)
        y = x.byte()
        test_case.assertTrue(y.dtype == flow.uint8)

    def test_tensor_constructor(test_case):
        x = flow.tensor([1, 2, 3])
        test_case.assertTrue(np.array_equal(x.numpy(), [1, 2, 3]))
        test_case.assertEqual(x.dtype, flow.int64)
        x = flow.tensor([1.0, 2.0, 3.0])
        test_case.assertTrue(np.array_equal(x.numpy(), [1.0, 2.0, 3.0]))
        test_case.assertEqual(x.dtype, flow.float32)
        x = flow.tensor([1.0, 2.0, 3.0], dtype=flow.float64)
        test_case.assertTrue(np.array_equal(x.numpy(), [1.0, 2.0, 3.0]))
        test_case.assertEqual(x.dtype, flow.float64)
        np_arr = np.array([1, 2, 3])
        x = flow.tensor(np_arr)
        test_case.assertTrue(np.array_equal(x.numpy(), [1, 2, 3]))
        test_case.assertEqual(x.dtype, flow.int64)
        np_arr = np.array([1, 2, 3], dtype=np.float64)
        x = flow.tensor(np_arr)
        test_case.assertTrue(np.array_equal(x.numpy(), [1.0, 2.0, 3.0]))
        test_case.assertEqual(x.dtype, flow.float64)
        x = flow.tensor(np_arr, dtype=flow.float32)
        test_case.assertTrue(np.array_equal(x.numpy(), [1.0, 2.0, 3.0]))
        test_case.assertEqual(x.dtype, flow.float32)
        x = flow.tensor(np_arr, dtype=flow.int8)
        test_case.assertTrue(np.array_equal(x.numpy(), [1.0, 2.0, 3.0]))
        test_case.assertEqual(x.dtype, flow.int8)
        x = flow.tensor([flow.tensor([1, 2])] * 3, dtype=flow.float32)
        test_case.assertTrue(np.array_equal(x.numpy(), [[1, 2], [1, 2], [1, 2]]))
        test_case.assertEqual(x.dtype, flow.float32)

    def test_tensor_contains_magic_method(test_case):
        x = flow.tensor([[1, 2, 3], [4, 5, 6]])
        y = 1 in x
        test_case.assertEqual(y, True)

    @profile(torch.Tensor.fill_)
    def profile_fill_(test_case):
        torch.Tensor.fill_(torch.ones(1, 8, 16, 16), 2)
        torch.Tensor.fill_(torch.ones(1000, 1000), 2)
        torch.Tensor.fill_(torch.ones(1, 8, 16, 16), torch.tensor(2))
        torch.Tensor.fill_(torch.ones(1000, 1000), torch.tensor(2))


if __name__ == "__main__":
    unittest.main()
