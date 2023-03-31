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
import os
import unittest
import oneflow as flow


"""
TODO(lml): Support and test more apis.
Finished: 
flow.from_numpy()
flow.tensor()
flow.ones()
flow.zeros()
flow.full()
Tensor.new_ones()
Tensor.new_zeros()
Tensor.new_full()

TO add test:
Tensor.real()
Tensor.imag()
Tensor.conj()
Tensor.conj_physical()

To complete:
flow.randn()
Tensor.adjoint()
Tensor.conj_physical_()
Tensor.resolve_conj()
Tensor.chalf()
Tensor.cfloat(),
Tensor.cdouble()
More apis..
"""


class TestTensorComplex64(unittest.TestCase):
    def setUp(self):
        self.dtype = flow.cfloat
        self.np_dtype = np.complex64
        self.type_str = "ComplexFloatTensor"
        self.real_dtype = flow.float
        self.np_real_dtype = np.float32
        self.a = [1.0 + 1j, 2.0]
        self.np_a = np.array(self.a, dtype=self.np_dtype)
        self.b = [[1.0 + 1j, 2.0], [1.0, 2.0 - 1j], [-1.0, 1j]]
        self.np_b = np.array(self.b, dtype=self.np_dtype)

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

    def test_slice(self):
        a = flow.from_numpy(self.np_a)
        np_slice_a = a[1].numpy()
        self.assertEqual(np_slice_a.dtype, self.np_dtype)
        assert np.allclose(np_slice_a, self.np_a[1])

        b = flow.from_numpy(self.np_b)
        np_slice_b = b[1].numpy()
        self.assertEqual(np_slice_b.dtype, self.np_dtype)
        assert np.allclose(np_slice_b, self.np_b[1])

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
        
        shape = (5,6,8)
        np_c = np.random.randn(*shape) + 1.0j * np.random.randn(*shape)
        np_c = np_c.astype(self.np_dtype)
        c = flow.from_numpy(np_c)
        self.assertEqual(c.type(), "oneflow." + self.type_str)
        np_c = np.conj(np_c)
        c = flow.conj_physical(c)
        assert np.allclose(np_c, c.numpy())
        

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


class TestTensorComplex128(TestTensorComplex64):
    def setUp(self):
        self.dtype = flow.cdouble
        self.np_dtype = np.complex128
        self.type_str = "ComplexDoubleTensor"
        self.real_dtype = flow.double
        self.np_real_dtype = np.float64
        self.a = [1.0 + 1j, 2.0]
        self.np_a = np.array(self.a, dtype=self.np_dtype)
        self.b = [[1.0 + 1j, 2.0], [1.0, 2.0 - 1j], [-1.0, 1j]]
        self.np_b = np.array(self.b, dtype=self.np_dtype)


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
