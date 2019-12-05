import unittest
import numpy as np

import oneflow as flow

def make_job(x_shape, like_shape, dtype=flow.float32):
    @flow.function
    def reshape_like_job(x = flow.input_blob_def(x_shape, dtype=dtype),
                         like = flow.input_blob_def(like_shape, dtype=dtype)):
        flow.config.use_xla_jit(False)
        flow.config.use_tensorrt(False)
        return flow.reshape_like(x, like)
    return reshape_like_job

def make_xla_job(x_shape, like_shape, dtype=flow.float32):
    @flow.function
    def xla_reshape_like_job(x = flow.input_blob_def(x_shape, dtype=dtype),
                             like = flow.input_blob_def(like_shape, dtype=dtype)):
        flow.config.use_xla_jit(True)
        flow.config.use_tensorrt(False)
        return flow.reshape_like(x, like)
    return xla_reshape_like_job

class TestReshapeLike(unittest.TestCase):
    def _test_body(self, x, like, dtype=np.float32):
        f1 = make_job(x.shape, like.shape, dtype=flow.float32)
        f2 = make_xla_job(x.shape, like.shape, dtype=flow.float32)
        a = f1(x, like).get()
        b = f2(x, like).get()
        print("without xla: ", a)
        print("with xla", b)
        self.assertTrue(a.shape == b.shape)
        self.assertTrue(np.allclose(a, b , rtol=1e-03, atol=1e-05))

        flow.clear_default_session()

    def _test_ones_body(self, x_shape, like_shape, dtype=np.float32):
        x = np.ones(x_shape, dtype=dtype)
        like = np.ones(like_shape, dtype=dtype)
        self._test_body(x, like, dtype=dtype)

    def _test_random_body(self, x_shape, like_shape, dtype=np.float32):
        x = np.random.random(x_shape).astype(dtype)
        like = np.random.random(like_shape).astype(dtype)
        self._test_body(x, like, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((1, 10), (10,))
        self._test_ones_body((2, 10, 2), (4, 10))
        self._test_ones_body((2, 5, 2, 2), (2, 5, 4))

    def test_random_input(self):
        self._test_random_body((1, 10), (10,))
        self._test_random_body((2, 10, 2), (4, 10))
        self._test_random_body((2, 5, 2, 2), (2, 5, 4))

if __name__ == '__main__':
  unittest.main()
