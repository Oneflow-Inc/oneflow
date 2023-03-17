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
flow.new_ones()
flow.new_zeros()
flow.new_full()

To complete:
flow.randn()
Tensor.real()
Tensor.imag()
Tensor.conj()
Tensor.adjoint()
Tensor.conj_physical()
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
        self.a = [1.0 + 1j, 2.0]
        self.np_a = np.array(self.a, dtype=self.np_dtype)
        self.b = [[1.0 + 1j, 2.0], [1.0, 2.0 - 1j], [-1.0, 1j]]
        self.np_b = np.array(self.b, dtype=self.np_dtype)
        self.c = [
            [3.14 + 2j, 3.14 + 2j],
            [3.14 + 2j, 3.14 + 2j],
            [3.14 + 2j, 3.14 + 2j],
        ]
        self.np_c = np.array(self.c, dtype=self.np_dtype)

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
        assert np.allclose(np_c, self.np_c)

    def test_new_full(self):
        a = flow.tensor(self.a, dtype=self.dtype)
        c = a.new_full((3, 2), 3.14 + 2j)
        self.assertEqual(c.dtype, self.dtype)
        self.assertEqual(c.type(), "oneflow." + self.type_str)
        np_c = c.numpy()
        self.assertEqual(np_c.dtype, self.np_dtype)
        assert np.allclose(np_c, self.np_c)


class TestTensorComplex128(TestTensorComplex64):
    def setUp(self):
        self.dtype = flow.cdouble
        self.np_dtype = np.complex128
        self.type_str = "ComplexDoubleTensor"
        self.a = [1.0 + 1j, 2.0]
        self.np_a = np.array(self.a, dtype=self.np_dtype)
        self.b = [[1.0 + 1j, 2.0], [1.0, 2.0 - 1j], [-1.0, 1j]]
        self.np_b = np.array(self.b, dtype=self.np_dtype)
        self.c = [
            [3.14 + 2j, 3.14 + 2j],
            [3.14 + 2j, 3.14 + 2j],
            [3.14 + 2j, 3.14 + 2j],
        ]
        self.np_c = np.array(self.c, dtype=self.np_dtype)


if __name__ == "__main__":
    unittest.main()
