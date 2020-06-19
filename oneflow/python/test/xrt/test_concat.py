import unittest

import numpy as np
import oneflow as flow

config = flow.function_config()


def make_job(a_shape, b_shape, axis, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def concat_job(
        x=flow.FixedTensorDef(a_shape, dtype=dtype),
        y=flow.FixedTensorDef(b_shape, dtype=dtype),
    ):
        return flow.concat([x, y], axis=axis)

    return concat_job


def make_trt_job(a_shape, b_shape, axis, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(True)

    @flow.global_function(config)
    def trt_concat_job(
        x=flow.FixedTensorDef(a_shape, dtype=dtype),
        y=flow.FixedTensorDef(b_shape, dtype=dtype),
    ):
        return flow.concat([x, y], axis=axis)

    return trt_concat_job


class Testconcat(unittest.TestCase):
    def _test_body(self, x, y, axis, dtype=np.float32):
        f1 = make_job(x.shape, y.shape, axis, dtype=flow.float32)
        f2 = make_trt_job(x.shape, y.shape, axis, dtype=flow.float32)
        a = f1(x, y).get()
        b = f2(x, y).get()
        print("without xla: ", a)
        print("with tensorrt: ", b)
        self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, a_shape, b_shape, axis, dtype=np.float32):
        x = np.ones(a_shape, dtype=dtype)
        y = np.ones(b_shape, dtype=dtype)
        self._test_body(x, y, axis, dtype=dtype)

    def _test_random_body(self, a_shape, b_shape, axis, dtype=np.float32):
        x = np.random.random(a_shape).astype(dtype)
        y = np.random.random(b_shape).astype(dtype)
        self._test_body(x, y, axis, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((5, 2), (5, 3), axis=1)
        self._test_ones_body((5, 2), (5, 3), axis=-1)
        self._test_ones_body((5, 1, 2), (5, 1, 2), axis=1)
        self._test_ones_body((5, 1, 2), (5, 1, 2), axis=2)

    def test_random_input(self):
        self._test_random_body((5, 2), (5, 3), axis=1)
        self._test_random_body((5, 2), (5, 3), axis=-1)
        self._test_random_body((5, 1, 2), (5, 1, 2), axis=1)
        self._test_random_body((5, 3, 2), (5, 3, 2), axis=2)


if __name__ == "__main__":
    unittest.main()
