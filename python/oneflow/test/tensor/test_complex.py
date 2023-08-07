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
import numpy as np
import torch as torch_original

import os
import unittest
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import (
    Array2Numpy,
    FlattenArray,
    GenArgList,
    Index2Coordinate,
)
from collections import OrderedDict

"""
TODO(lml): Support and test more apis.
Finished: 
flow.from_numpy()
flow.tensor()
flow.ones()
flow.zeros()
flow.full()
flow.add()
flow.sub()
flow.mul
flow.sum()
flow.equal()
flow.not_equal()
flow.cast()
Tensor.new_ones()
Tensor.new_zeros()
Tensor.new_full()
Tensor.real()
Tensor.imag()
Tensor.conj()
Tensor.conj_physical()

To complete:
flow.randn()
flow.div()
flow.pow()
Tensor.adjoint()
Tensor.conj_physical_()
Tensor.resolve_conj()
Tensor.chalf()
Tensor.cfloat(),
Tensor.cdouble()
More apis..
"""


def compare_result(a, b, rtol=1e-5, atol=1e-8):
    assert np.allclose(
        a, b, rtol=rtol, atol=atol
    ), f"\na\n{a}\n{'-' * 80}\nb:\n{b}\n{'*' * 80}\ndiff:\n{a - b}"


def _np_zero_pad2d_grad(src, dest, padding):
    (c_idx, h_idx, w_idx) = (1, 2, 3)
    pad_left = padding[0]
    pad_right = padding[1]
    pad_top = padding[2]
    pad_bottom = padding[3]
    (dx_height, dx_width) = (dest.shape[h_idx], dest.shape[w_idx])
    (dy_height, dy_width) = (src.shape[h_idx], src.shape[w_idx])
    numpy_src = np.ones(src.shape, np.int32)
    numpy_dest = np.zeros(dest.shape, np.int32)
    array_src = FlattenArray(numpy_src)
    array_dest = FlattenArray(numpy_dest)
    src_num = src.shape[c_idx] * src.shape[h_idx] * src.shape[w_idx]
    dest_num = dest.shape[c_idx] * dest.shape[h_idx] * dest.shape[w_idx]
    elements_num = src.shape[0] * src_num
    for iter_n in range(elements_num):
        coords = Index2Coordinate(iter_n, src.shape)
        (n, c, i, j) = (coords[0], coords[c_idx], coords[h_idx], coords[w_idx])
        ip_x = ip_y = 0
        if (
            j >= pad_left
            and j < dx_width + pad_left
            and (i >= pad_top)
            and (i < dx_height + pad_top)
        ):
            ip_x = j - pad_left
            ip_y = i - pad_top
            src_index = n * src_num + c * dy_width * dy_height + i * dy_width + j
            dest_index = (
                n * dest_num + c * dx_width * dx_height + ip_y * dx_width + ip_x
            )
            array_dest[dest_index] += array_src[src_index]
    numpy_dest = Array2Numpy(array_dest, dest.shape)
    return numpy_dest


def _test_ZeroPad2d(test_case, shape, padding, value, device, rtol, atol):
    np_input = np.random.random(shape)
    of_input = flow.tensor(
        np_input, dtype=test_case.dtype, device=flow.device(device), requires_grad=True
    )
    if isinstance(padding, int):
        np_boundary = ((0, 0), (0, 0), (padding, padding), (padding, padding))
    elif isinstance(padding, (tuple, int)) and len(padding) == 4:
        np_boundary = (
            (0, 0),
            (0, 0),
            (padding[2], padding[3]),
            (padding[0], padding[1]),
        )
    else:
        raise ValueError("padding must be in  or tuple!")
    layer = flow.nn.ZeroPad2d(padding=padding)
    of_out = layer(of_input)
    np_out = np.pad(np_input, np_boundary, mode="constant", constant_values=value)
    test_case.assertTrue(np.allclose(of_out.cpu().detach().numpy(), np_out, rtol, atol))
    of_out = of_out.sum()
    of_out.backward()
    np_out_grad = _np_zero_pad2d_grad(np_out, np_input, layer.padding)
    test_case.assertTrue(
        np.allclose(of_input.grad.cpu().detach().numpy(), np_out_grad, rtol, atol)
    )


class TestTensorComplex64(unittest.TestCase):
    def setUp(self):
        self.dtype = flow.cfloat
        self.complex_dtype = flow.complex64
        self.np_dtype = np.complex64
        self.type_str = "ComplexFloatTensor"
        self.real_dtype = flow.float
        self.np_real_dtype = np.float32
        self.rtol = 1e-5
        self.atol = 1e-5
        self.a = [1.0 + 1j, 2.0]
        self.np_a = np.array(self.a, dtype=self.np_dtype)
        self.b = [[1.0 + 1j, 2.0], [1.0, 2.0 - 1j], [-1.0, 1j]]
        self.np_b = np.array(self.b, dtype=self.np_dtype)

        self.lower_n_dims = 2
        self.upper_n_dims = 5
        self.shape = []
        for _ in range(10):
            num_dims = np.random.randint(self.lower_n_dims, self.upper_n_dims)
            shape_ = [np.random.randint(1, 11) * 4 for _ in range(num_dims)]
            self.shape.append(shape_)

    def test_from_numpy(self):
        a = flow.from_numpy(self.np_a)
        self.assertEqual(a.dtype, self.dtype)
        self.assertEqual(a.type(), "oneflow." + self.type_str)
        np_a = a.numpy()
        self.assertEqual(np_a.dtype, self.np_dtype)
        assert np.allclose(np_a, self.np_a)

        b = flow.from_numpy(self.np_b)
        self.assertEqual(b.dtype, self.dtype)
        self.assertEqual(b.type(), "oneflow." + self.type_str)
        np_b = b.numpy()
        self.assertEqual(np_b.dtype, self.np_dtype)
        assert np.allclose(np_b, self.np_b)

    def test_tensor(self):
        a = flow.tensor(self.a, dtype=self.dtype)
        self.assertEqual(a.dtype, self.dtype)
        self.assertEqual(a.type(), "oneflow." + self.type_str)
        np_a = a.numpy()
        self.assertEqual(np_a.dtype, self.np_dtype)
        assert np.allclose(np_a, self.np_a)

        a = flow.tensor(self.np_a, dtype=self.dtype)
        self.assertEqual(a.dtype, self.dtype)
        self.assertEqual(a.type(), "oneflow." + self.type_str)
        np_a = a.numpy()
        self.assertEqual(np_a.dtype, self.np_dtype)
        assert np.allclose(np_a, self.np_a)

    @unittest.skip("skip for now, becase it failed 6 times in past week")
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_tensor_cuda(self):
        a = flow.tensor(self.a, dtype=self.dtype, device="cuda")
        self.assertEqual(a.dtype, self.dtype)
        self.assertEqual(a.type(), "oneflow.cuda." + self.type_str)
        np_a = a.numpy()
        self.assertEqual(np_a.dtype, self.np_dtype)
        assert np.allclose(np_a, self.np_a)

        a = flow.tensor(self.np_a, dtype=self.dtype, device="cuda")
        self.assertEqual(a.dtype, self.dtype)
        self.assertEqual(a.type(), "oneflow.cuda." + self.type_str)
        np_a = a.numpy()
        self.assertEqual(np_a.dtype, self.np_dtype)
        assert np.allclose(np_a, self.np_a)

    @unittest.skip("skip for now, becase it failed 2 times in past week")
    def test_slice(self):
        a = flow.from_numpy(self.np_a)
        np_slice_a = a[1].numpy()
        self.assertEqual(np_slice_a.dtype, self.np_dtype)
        assert np.allclose(np_slice_a, self.np_a[1])

        b = flow.from_numpy(self.np_b)
        np_slice_b = b[1].numpy()
        self.assertEqual(np_slice_b.dtype, self.np_dtype)
        assert np.allclose(np_slice_b, self.np_b[1])

        c = flow.full((3, 2), 3.14 + 2j, dtype=self.dtype)
        np_slice_c = c[0:2, :].numpy()
        self.assertEqual(np_slice_c.dtype, self.np_dtype)
        assert np.allclose(
            np_slice_c, np.ones((2, 2), dtype=self.np_dtype) * (3.14 + 2j)
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_slice_cuda(self):
        a = flow.from_numpy(self.np_a).cuda()
        np_slice_a = a[1].cpu().numpy()
        self.assertEqual(np_slice_a.dtype, self.np_dtype)
        assert np.allclose(np_slice_a, self.np_a[1])

        b = flow.from_numpy(self.np_b).cuda()
        np_slice_b = b[1].cpu().numpy()
        self.assertEqual(np_slice_b.dtype, self.np_dtype)
        assert np.allclose(np_slice_b, self.np_b[1])

        c = flow.full((3, 2), 3.14 + 2j, dtype=self.dtype).cuda()
        np_slice_c = c[0:2, :].cpu().numpy()
        self.assertEqual(np_slice_c.dtype, self.np_dtype)
        assert np.allclose(
            np_slice_c, np.ones((2, 2), dtype=self.np_dtype) * (3.14 + 2j)
        )

    def test_new_tensor(self):
        a = flow.tensor(self.a, dtype=self.dtype)
        b = a.new_tensor(self.b)
        self.assertEqual(b.dtype, self.dtype)
        self.assertEqual(b.type(), "oneflow." + self.type_str)
        np_b = b.numpy()
        self.assertEqual(np_b.dtype, self.np_dtype)
        assert np.allclose(np_b, self.np_b)

    def test_new_empty(self):
        a = flow.tensor(self.a, dtype=self.dtype)
        c = a.new_empty((3, 2))
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)

    def test_ones(self):
        c = flow.ones((3, 2), dtype=self.dtype)
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_dtype))

    def test_new_ones(self):
        b = flow.tensor(self.b, dtype=self.dtype)
        c = b.new_ones((3, 2))
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_dtype))

    def test_zeros(self):
        c = flow.zeros((3, 2), dtype=self.dtype)
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)
        assert np.allclose(np_c, np.zeros((3, 2), dtype=self.np_dtype))

    def test_new_zeros(self):
        b = flow.tensor(self.b, dtype=self.dtype)
        c = b.new_zeros((3, 2))
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)
        assert np.allclose(np_c, np.zeros((3, 2), dtype=self.np_dtype))

    def test_full(self):
        c = flow.full((3, 2), 3.14 + 2j, dtype=self.dtype)
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_dtype) * (3.14 + 2j))

    def test_new_full(self):
        a = flow.tensor(self.a, dtype=self.dtype)
        c = a.new_full((3, 2), 3.14 + 2j)
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_dtype) * (3.14 + 2j))

    def test_real(self):
        c = flow.full((3, 2), 3.14 + 2j, dtype=self.dtype).real()
        self.assertEqual(c.dtype, self.real_dtype)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_real_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_real_dtype) * 3.14)

    def test_imag(self):
        c = flow.full((3, 2), 3.14 + 2j, dtype=self.dtype).imag()
        self.assertEqual(c.dtype, self.real_dtype)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_real_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_real_dtype) * 2)

    def test_conj(self):
        c = flow.full((3, 2), 3.14 + 2j, dtype=self.dtype).conj()
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_dtype) * (3.14 - 2j))

    def test_conj_physical(self):
        c = flow.full((3, 2), 3.14 + 2j, dtype=self.dtype).conj_physical()
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_dtype) * (3.14 - 2j))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_real_cuda(self):
        c = flow.full((3, 2), 3.14 + 2j, dtype=self.dtype, device="cuda").real()
        self.assertEqual(c.dtype, self.real_dtype)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_real_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_real_dtype) * 3.14)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_imag_cuda(self):
        c = flow.full((3, 2), 3.14 + 2j, dtype=self.dtype, device="cuda").imag()
        self.assertEqual(c.dtype, self.real_dtype)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_real_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_real_dtype) * 2)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_conj_cuda(self):
        c = flow.full((3, 2), 3.14 + 2j, dtype=self.dtype, device="cuda").conj()
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow.cuda." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_dtype) * (3.14 - 2j))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_conj_physical_cuda(self):
        c = flow.full(
            (3, 2), 3.14 + 2j, dtype=self.dtype, device="cuda"
        ).conj_physical()
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow.cuda." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)
        assert np.allclose(np_c, np.ones((3, 2), dtype=self.np_dtype) * (3.14 - 2j))

    def test_add_cpu(self):
        device = "cpu"
        for i, input_shape in enumerate(self.shape):
            np_x = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_x = np_x.astype(self.np_dtype)

            np_y = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_y = np_y.astype(self.np_dtype)

            flow_x = flow.from_numpy(np_x).to(device).requires_grad_(True)
            flow_y = flow.from_numpy(np_y).to(device).requires_grad_(True)
            self.assertEqual(flow_x.dtype, self.dtype)
            self.assertEqual(flow_y.dtype, self.dtype)

            # forward
            flow_ret = flow.add(flow_x, flow_y)
            np_ret = np_x + np_y
            compare_result(flow_ret, np_ret, self.rtol, self.atol)

            # backward
            flow_ret.sum().backward()
            compare_result(
                flow_x.grad.numpy(), np.ones(input_shape), self.rtol, self.atol
            )
            compare_result(
                flow_y.grad.numpy(), np.ones(input_shape), self.rtol, self.atol
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_add_cuda(self):
        device = "cuda"
        for i, input_shape in enumerate(self.shape):
            np_x = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_x = np_x.astype(self.np_dtype)

            np_y = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_y = np_y.astype(self.np_dtype)

            flow_x = flow.from_numpy(np_x).to(device).requires_grad_(True)
            flow_y = flow.from_numpy(np_y).to(device).requires_grad_(True)
            self.assertEqual(flow_x.dtype, self.dtype)
            self.assertEqual(flow_y.dtype, self.dtype)

            # forward
            flow_ret = flow.add(flow_x, flow_y)
            np_ret = np_x + np_y
            compare_result(flow_ret.cpu().detach(), np_ret, self.rtol, self.atol)

            # backward
            flow_ret.sum().backward()
            compare_result(
                flow_x.grad.cpu().detach().numpy(),
                np.ones(input_shape),
                self.rtol,
                self.atol,
            )
            compare_result(
                flow_y.grad.cpu().detach().numpy(),
                np.ones(input_shape),
                self.rtol,
                self.atol,
            )

    def test_sub_cpu(self):
        device = "cpu"
        for i, input_shape in enumerate(self.shape):
            np_x = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_x = np_x.astype(self.np_dtype)

            np_y = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_y = np_y.astype(self.np_dtype)

            flow_x = flow.from_numpy(np_x).to(device).requires_grad_(True)
            flow_y = flow.from_numpy(np_y).to(device).requires_grad_(True)
            self.assertEqual(flow_x.dtype, self.dtype)
            self.assertEqual(flow_y.dtype, self.dtype)

            # forward
            flow_ret = flow.sub(flow_x, flow_y)
            np_ret = np_x - np_y
            compare_result(flow_ret, np_ret, self.rtol, self.atol)

            # backward
            flow_ret.sum().backward()
            compare_result(
                flow_x.grad.numpy(), np.ones(input_shape), self.rtol, self.atol
            )
            compare_result(
                flow_y.grad.numpy(), -np.ones(input_shape), self.rtol, self.atol
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_sub_cuda(self):
        device = "cuda"
        for i, input_shape in enumerate(self.shape):
            np_x = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_x = np_x.astype(self.np_dtype)

            np_y = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_y = np_y.astype(self.np_dtype)

            flow_x = flow.from_numpy(np_x).to(device).requires_grad_(True)
            flow_y = flow.from_numpy(np_y).to(device).requires_grad_(True)
            self.assertEqual(flow_x.dtype, self.dtype)
            self.assertEqual(flow_y.dtype, self.dtype)

            # forward
            flow_ret = flow.sub(flow_x, flow_y)
            np_ret = np_x - np_y
            compare_result(flow_ret.cpu().detach(), np_ret, self.rtol, self.atol)

            # backward
            flow_ret.sum().backward()
            compare_result(
                flow_x.grad.cpu().detach().numpy(),
                np.ones(input_shape),
                self.rtol,
                self.atol,
            )
            compare_result(
                flow_y.grad.cpu().detach().numpy(),
                -np.ones(input_shape),
                self.rtol,
                self.atol,
            )

    def test_mul_cpu(self):
        device = "cpu"
        for i, input_shape in enumerate(self.shape):
            np_x = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_x = np_x.astype(self.np_dtype)

            np_y = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_y = np_y.astype(self.np_dtype)

            flow_x = flow.from_numpy(np_x).to(device).requires_grad_(True)
            flow_y = flow.from_numpy(np_y).to(device).requires_grad_(True)
            self.assertEqual(flow_x.dtype, self.dtype)
            self.assertEqual(flow_y.dtype, self.dtype)

            # forward
            flow_ret = flow.mul(flow_x, flow_y)
            np_ret = np_x * np_y
            compare_result(flow_ret, np_ret, self.rtol, self.atol)

            # backward
            flow_ret.sum().backward()
            compare_result(
                flow_x.grad.numpy(), flow_y.numpy().conjugate(), self.rtol, self.atol
            )
            compare_result(
                flow_y.grad.numpy(), flow_x.numpy().conjugate(), self.rtol, self.atol
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_mul_cuda(self):
        device = "cuda"
        for i, input_shape in enumerate(self.shape):
            np_x = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_x = np_x.astype(self.np_dtype)

            np_y = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_y = np_y.astype(self.np_dtype)

            flow_x = flow.from_numpy(np_x).to(device).requires_grad_(True)
            flow_y = flow.from_numpy(np_y).to(device).requires_grad_(True)
            self.assertEqual(flow_x.dtype, self.dtype)
            self.assertEqual(flow_y.dtype, self.dtype)

            # forward
            flow_ret = flow.mul(flow_x, flow_y)
            np_ret = np_x * np_y
            compare_result(flow_ret.cpu().detach(), np_ret, self.rtol, self.atol)

            # backward
            flow_ret.sum().backward()
            compare_result(
                flow_x.grad.cpu().detach().numpy(),
                flow_y.numpy().conjugate(),
                self.rtol,
                self.atol,
            )
            compare_result(
                flow_y.grad.cpu().detach().numpy(),
                flow_x.numpy().conjugate(),
                self.rtol,
                self.atol,
            )

    def test_sum_cpu(self):
        device = "cpu"
        for i, input_shape in enumerate(self.shape):
            n_dims = np.random.randint(1, len(input_shape))
            dims = np.random.choice(
                len(input_shape) - 1, n_dims, replace=False
            ).tolist()
            keepdim = True if np.random.randint(2) == 1 else False

            np_x = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_x = np_x.astype(self.np_dtype)

            flow_x = flow.from_numpy(np_x).to(device).requires_grad_(True)
            self.assertEqual(flow_x.dtype, self.dtype)

            # forward
            flow_ret = flow.sum(flow_x, dim=dims, keepdim=keepdim)
            np_ret = np.sum(np_x, axis=tuple(dims), keepdims=keepdim)
            compare_result(flow_ret, np_ret, self.rtol, self.atol * 1000)

            # backward
            flow_ret.sum().backward()
            compare_result(
                flow_x.grad.numpy(), np.ones(input_shape), self.rtol, self.atol
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_sum_cuda(self):
        device = "cuda"
        for i, input_shape in enumerate(self.shape):
            n_dims = np.random.randint(1, len(input_shape))
            dims = np.random.choice(
                len(input_shape) - 1, n_dims, replace=False
            ).tolist()
            keepdim = True if np.random.randint(2) == 1 else False

            np_x = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_x = np_x.astype(self.np_dtype)

            flow_x = flow.from_numpy(np_x).to(device).requires_grad_(True)
            self.assertEqual(flow_x.dtype, self.dtype)

            # forward
            flow_ret = flow.sum(flow_x, dim=dims, keepdim=keepdim)
            np_ret = np.sum(np_x, axis=tuple(dims), keepdims=keepdim)
            compare_result(flow_ret.cpu().detach(), np_ret, self.rtol, self.atol * 1000)

            # backward
            flow_ret.sum().backward()
            compare_result(
                flow_x.grad.cpu().detach().numpy(),
                np.ones(input_shape),
                self.rtol,
                self.atol,
            )

    def test_equal_cpu(self):
        device = "cpu"
        for i, input_shape in enumerate(self.shape):

            np_x = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_x = np_x.astype(self.np_dtype)

            np_y = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_y = np_y.astype(self.np_dtype)

            np_z = np.copy(np_x)

            flow_x = flow.from_numpy(np_x).to(device).requires_grad_(False)
            flow_y = flow.from_numpy(np_y).to(device).requires_grad_(False)
            flow_z = flow.from_numpy(np_z).to(device).requires_grad_(False)
            self.assertEqual(flow_x.dtype, self.dtype)
            self.assertEqual(flow_y.dtype, self.dtype)
            self.assertEqual(flow_z.dtype, self.dtype)

            # forward
            flow_ret = flow.equal(flow_x, flow_y)
            np_ret = np.equal(np_x, np_y)
            compare_result(flow_ret, np_ret, self.rtol, self.atol)

            flow_ret = flow.equal(flow_x, flow_z)
            compare_result(
                flow_ret, np.ones(flow_x.shape).astype(bool), self.rtol, self.atol
            )

            flow_ret = flow.not_equal(flow_x, flow_z)
            compare_result(
                flow_ret, np.zeros(flow_x.shape).astype(bool), self.rtol, self.atol
            )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_equal_cuda(self):
        device = "cuda"
        for i, input_shape in enumerate(self.shape):

            np_x = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_x = np_x.astype(self.np_dtype)

            np_y = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
            np_y = np_y.astype(self.np_dtype)

            np_z = np.copy(np_x)

            flow_x = flow.from_numpy(np_x).to(device).requires_grad_(False)
            flow_y = flow.from_numpy(np_y).to(device).requires_grad_(False)
            flow_z = flow.from_numpy(np_z).to(device).requires_grad_(False)
            self.assertEqual(flow_x.dtype, self.dtype)
            self.assertEqual(flow_y.dtype, self.dtype)
            self.assertEqual(flow_z.dtype, self.dtype)

            # forward
            flow_ret = flow.equal(flow_x, flow_y)
            np_ret = np.equal(np_x, np_y)
            compare_result(flow_ret, np_ret, self.rtol, self.atol)

            flow_ret = flow.equal(flow_x, flow_z)
            compare_result(
                flow_ret, np.ones(flow_x.shape).astype(bool), self.rtol, self.atol
            )

            flow_ret = flow.not_equal(flow_x, flow_z)
            compare_result(
                flow_ret.cpu().detach(),
                np.zeros(flow_x.shape).astype(bool),
                self.rtol,
                self.atol,
            )

    def test_constant_pad(self):
        arg_dict = OrderedDict()
        arg_dict["shape"] = [(1, 2, 3, 4), (8, 3, 4, 4)]
        arg_dict["padding"] = [2, (1, 1, 2, 2)]
        arg_dict["value"] = [0.0]
        arg_dict["device"] = (
            ["cpu", "cuda"] if os.getenv("ONEFLOW_TEST_CPU_ONLY") is None else ["cpu"]
        )
        arg_dict["rtol"] = [self.rtol]
        arg_dict["atol"] = [self.atol]
        for arg in GenArgList(arg_dict):
            _test_ZeroPad2d(self, *arg)

    def test_cast(self):
        dtype_pairs = [
            (np.uint8, "ByteTensor"),
            (np.int8, "CharTensor"),
            (np.int32, "IntTensor"),
            (np.int64, "LongTensor"),
            (np.float32, "FloatTensor"),
            (np.float64, "DoubleTensor"),
        ]
        shape = (3, 5, 2)
        for np_dtype, type_str in dtype_pairs:
            np_arr = np.random.randn(*shape).astype(np_dtype)
            flow_tensor = flow.from_numpy(np_arr)
            self.assertEqual(flow_tensor.type(), "oneflow." + type_str)
            np_out = np_arr.astype(self.np_dtype)
            flow_out = flow.cast(flow_tensor, dtype=self.complex_dtype)
            self.assertTrue(np.array_equal(flow_out.numpy(), np_out))

        # cp64 -> cp128
        np_arr = np.random.randn(*shape) + 1.0j * np.random.randn(*shape)
        np_arr = np_arr.astype(np.complex64)
        flow_tensor = flow.from_numpy(np_arr)
        self.assertEqual(flow_tensor.dtype, flow.complex64)

        np_out = np_arr.astype(np.complex128)
        flow_out = flow.cast(flow_tensor, dtype=flow.complex128)
        self.assertTrue(np.array_equal(flow_out.numpy(), np_out))

        # cp128 -> cp64
        np_out = np_out.astype(np.complex64)
        flow_out = flow.cast(flow_out, dtype=flow.complex64)
        self.assertTrue(np.array_equal(flow_out.numpy(), np_out))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_cast_cuda(self):
        dtype_pairs = [
            (np.uint8, "ByteTensor"),
            (np.int8, "CharTensor"),
            (np.int32, "IntTensor"),
            (np.int64, "LongTensor"),
            (np.float32, "FloatTensor"),
            (np.float64, "DoubleTensor"),
        ]
        shape = (7, 4, 11)
        for np_dtype, type_str in dtype_pairs:
            np_arr = np.random.randn(*shape).astype(np_dtype)
            flow_tensor = flow.from_numpy(np_arr).cuda()
            self.assertEqual(flow_tensor.type(), "oneflow.cuda." + type_str)
            np_out = np_arr.astype(self.np_dtype)
            flow_out = flow.cast(flow_tensor, dtype=self.complex_dtype)
            self.assertTrue(np.array_equal(flow_out.cpu().detach().numpy(), np_out))

        # cp64 -> cp128
        np_arr = np.random.randn(*shape) + 1.0j * np.random.randn(*shape)
        np_arr = np_arr.astype(np.complex64)
        flow_tensor = flow.from_numpy(np_arr).cuda()
        self.assertEqual(flow_tensor.dtype, flow.complex64)

        np_out = np_arr.astype(np.complex128)
        flow_out = flow.cast(flow_tensor, dtype=flow.complex128)
        self.assertTrue(np.array_equal(flow_out.cpu().detach().numpy(), np_out))

        # cp128 -> cp64
        np_out = np_out.astype(np.complex64)
        flow_out = flow.cast(flow_out, dtype=flow.complex64)
        self.assertTrue(np.array_equal(flow_out.cpu().detach().numpy(), np_out))


class TestTensorComplex128(TestTensorComplex64):
    def setUp(self):
        self.dtype = flow.cdouble
        self.complex_dtype = flow.complex128
        self.np_dtype = np.complex128
        self.type_str = "ComplexDoubleTensor"
        self.real_dtype = flow.double
        self.np_real_dtype = np.float64
        self.rtol = 1e-7
        self.atol = 1e-7
        self.a = [1.0 + 1j, 2.0]
        self.np_a = np.array(self.a, dtype=self.np_dtype)
        self.b = [[1.0 + 1j, 2.0], [1.0, 2.0 - 1j], [-1.0, 1j]]
        self.np_b = np.array(self.b, dtype=self.np_dtype)

        self.lower_n_dims = 2
        self.upper_n_dims = 5
        self.shape = []
        for _ in range(10):
            num_dims = np.random.randint(self.lower_n_dims, self.upper_n_dims)
            shape_ = [np.random.randint(1, 11) * 4 for _ in range(num_dims)]
            self.shape.append(shape_)


class TestAutograd(unittest.TestCase):
    def test_backward(self):
        a = flow.tensor([1.0 + 2j, 2.0 - 3j, 1j], dtype=flow.cfloat)
        a.requires_grad = True
        b = flow.conj(a)
        loss = flow.sum(a.real() + b.imag())
        loss.backward()
        assert np.allclose(a.grad.numpy(), np.ones((3,), dtype=np.complex64) * (1 - 1j))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_backward_cuda(self):
        a = flow.tensor([1.0 + 2j, 2.0 - 3j, 1j], dtype=flow.cfloat, device="cuda")
        a.requires_grad = True
        b = flow.conj(a)
        loss = flow.sum(a.real() + b.imag())
        loss.backward()
        assert np.allclose(a.grad.numpy(), np.ones((3,), dtype=np.complex64) * (1 - 1j))

    def test_grad(self):
        a = flow.tensor([1.0 + 2j, 2.0 - 3j, 1j], dtype=flow.cfloat)
        a.requires_grad = True
        b = flow.conj(a)
        c = a.real() + b.imag()
        np_dc = np.ones((3,), dtype=np.float32)
        dc = flow.tensor(np_dc)
        (da,) = flow.autograd.grad(c, a, dc)
        assert np.allclose(da.numpy(), np.ones((3,), dtype=np.complex64) * (1 - 1j))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_grad_cuda(self):
        a = flow.tensor([1.0 + 2j, 2.0 - 3j, 1j], dtype=flow.cfloat, device="cuda")
        a.requires_grad = True
        b = flow.conj(a)
        c = a.real() + b.imag()
        np_dc = np.ones((3,), dtype=np.float32)
        dc = flow.tensor(np_dc)
        (da,) = flow.autograd.grad(c, a, dc)
        assert np.allclose(da.numpy(), np.ones((3,), dtype=np.complex64) * (1 - 1j))


if __name__ == "__main__":
    unittest.main()
