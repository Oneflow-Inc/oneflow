import unittest

import numpy as np
import oneflow as flow

config = flow.function_config()


def make_job(x_shape, like_shape, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.function(config)
    def reshape_like_job(
        x=flow.FixedTensorDef(x_shape, dtype=dtype),
        like=flow.FixedTensorDef(like_shape, dtype=dtype),
    ):
        return flow.reshape_like(x, like)

    return reshape_like_job


def make_xla_job(x_shape, like_shape, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.function(config)
    def xla_reshape_like_job(
        x=flow.FixedTensorDef(x_shape, dtype=dtype),
        like=flow.FixedTensorDef(like_shape, dtype=dtype),
    ):
        return flow.reshape_like(x, like)

    return xla_reshape_like_job


def make_trt_job(x_shape, like_shape, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(True)

    @flow.function(config)
    def trt_reshape_like_job(
        x=flow.FixedTensorDef(x_shape, dtype=dtype),
        like=flow.FixedTensorDef(like_shape, dtype=dtype),
    ):
        return flow.reshape_like(x, like)

    return trt_reshape_like_job


class TestReshapeLike(unittest.TestCase):
    def _test_body(self, x, like, dtype=np.float32):
        f1 = make_job(x.shape, like.shape, dtype=flow.float32)
        f2 = make_xla_job(x.shape, like.shape, dtype=flow.float32)
        a = f1(x, like).get()
        b = f2(x, like).get()
        print("without xla: ", a)
        print("with xla: ", b)
        self.assertTrue(a.shape == b.shape)
        self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

        f3 = make_trt_job(x.shape, like.shape, dtype=flow.float32)
        c = f3(x, like).get()
        print("with tensorrt: ", c)
        self.assertTrue(a.shape == c.shape)
        self.assertTrue(np.allclose(a.ndarray(), c.ndarray(), rtol=1e-03, atol=1e-05))
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


if __name__ == "__main__":
    unittest.main()
