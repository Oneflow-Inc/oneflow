import numpy as np
import unittest
import oneflow as flow

class TestTensorComplex64(unittest.TestCase):

    def setUp(self):
        self.dtype = flow.cfloat
        self.np_dtype = np.complex64
        self.a = [1.0 + 1j, 2.0, 1j]
        self.np_a = np.array(self.a, dtype=self.np_dtype)
        self.b = [[1.0 + 1j, 2.0], [1.0, 2.0 - 1j], [-1.0, 1j]]
        self.np_b = np.array(self.b, dtype=self.np_dtype)

    def test_from_numpy(self):
        a = flow.from_numpy(self.np_a)
        self.assertEqual(a.dtype, self.dtype)
        np_a = a.numpy()
        self.assertEqual(np_a.dtype, self.np_dtype)
        assert np.allclose(np_a, self.np_a)

        b = flow.from_numpy(self.np_b)
        self.assertEqual(b.dtype, self.dtype)
        np_b = b.numpy()
        self.assertEqual(np_b.dtype, self.np_dtype)
        assert np.allclose(np_b, self.np_b)

    def test_tensor_cpu(self):
        a = flow.tensor(self.a, dtype=self.dtype, device='cpu')
        self.assertEqual(a.dtype, self.dtype)
        np_a = a.numpy()
        self.assertEqual(np_a.dtype, self.np_dtype)
        assert np.allclose(np_a, self.np_a)

        a = flow.tensor(self.np_a, dtype=self.dtype, device='cpu')
        self.assertEqual(a.dtype, self.dtype)
        np_a = a.numpy()
        self.assertEqual(np_a.dtype, self.np_dtype)
        assert np.allclose(np_a, self.np_a)

    def test_tensor_cuda(self):
        a = flow.tensor(self.a, dtype=self.dtype, device='cuda')
        self.assertEqual(a.dtype, self.dtype)
        np_a = a.numpy()
        self.assertEqual(np_a.dtype, self.np_dtype)
        assert np.allclose(np_a, self.np_a)

        a = flow.tensor(self.np_a, dtype=self.dtype, device='cuda')
        self.assertEqual(a.dtype, self.dtype)
        np_a = a.numpy()
        self.assertEqual(np_a.dtype, self.np_dtype)
        assert np.allclose(np_a, self.np_a)


    def test_slice(self):
        a = flow.from_numpy(self.np_a)[1]
        self.assertEqual(a.dtype, self.dtype)
        np_a = a.numpy()
        self.assertEqual(np_a.dtype, self.np_dtype)
        assert np.allclose(np_a, self.np_a[1])

        b = flow.from_numpy(self.np_b)[1]
        self.assertEqual(b.dtype, self.dtype)
        np_b = b.numpy()
        self.assertEqual(np_b.dtype, self.np_dtype)
        assert np.allclose(np_b, self.np_b[1])


# class TestTensorComplex128(TestTensorComplex64):
# 
#     def setUp(self):
#         self.dtype = flow.cdouble
#         self.np_dtype = np.complex128
#         self.a = [1.0 + 1j, 2.0, 1j]
#         self.np_a = np.array(self.a, dtype=self.np_dtype)
#         self.b = [[1.0 + 1j, 2.0], [1.0, 2.0 - 1j], [-1.0, 1j]]
#         self.np_b = np.array(self.b, dtype=self.np_dtype)


if __name__ == "__main__":
    unittest.main()
