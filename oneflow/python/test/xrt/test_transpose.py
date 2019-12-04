import unittest
import numpy as np

import oneflow as flow

def make_job(input_shape, permute, dtype=flow.float32):
    @flow.function
    def transpose_job(x = flow.input_blob_def(input_shape, dtype=dtype)):
        flow.config.use_xla_jit(False)
        flow.config.use_tensorrt(False)
        return flow.transpose(x, perm=permute)
    return transpose_job

def make_xla_job(input_shape, permute, dtype=flow.float32):
    @flow.function
    def xla_transpose_job(x = flow.input_blob_def(input_shape, dtype=dtype)):
        flow.config.use_xla_jit(True)
        flow.config.use_tensorrt(False)
        return flow.transpose(x, perm=permute)
    return xla_transpose_job

class TestTranspose(unittest.TestCase):
    def _test_body(self, x, permute, dtype=flow.float32):
        f1 = make_job(x.shape, permute, dtype=dtype)
        f2 = make_xla_job(x.shape, permute, dtype=dtype)
        a = f1(x).get()
        b = f2(x).get()
        print("without xla: ", a)
        print("with xla", b)
        self.assertTrue(a.shape == b.shape)
        self.assertTrue(np.allclose(a, b, rtol=1e-03, atol=1e-05))

        flow.clear_default_session()

    def _test_ones_body(self, shape, permute, dtype=flow.float32):
        np_dtype = flow.convert_of_dtype_to_numpy_dtype(dtype)
        x = np.ones(shape, dtype=np_dtype)
        self._test_body(x, permute, dtype=dtype)

    def _test_random_body(self, shape, permute, dtype=flow.float32):
        np_dtype = flow.convert_of_dtype_to_numpy_dtype(dtype)
        x = np.random.random(shape).astype(np_dtype)
        self._test_body(x, permute, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((1, 2), (1, 0))
        self._test_ones_body((2, 2, 2), (0, 2, 1))
        self._test_ones_body((2, 2, 2), (1, 0, 2))
        self._test_ones_body((2, 2, 2), (1, 2, 0))

    def test_random_input(self):
        self._test_random_body((1, 2), (1, 0))
        self._test_random_body((2, 2, 2), (0, 2, 1))
        self._test_random_body((2, 2, 2), (1, 0, 2))
        self._test_random_body((2, 2, 2), (1, 2, 0))

if __name__ == '__main__':
  unittest.main()

