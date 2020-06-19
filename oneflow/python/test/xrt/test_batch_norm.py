import unittest

import numpy as np
import oneflow as flow

config = flow.function_config()


def make_job(input_shape, axis, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def batch_norm_job(x=flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.layers.batch_normalization(x, axis=axis)

    return batch_norm_job


def make_xla_job(input_shape, axis, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.global_function(config)
    def xla_batch_norm_job(x=flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.layers.batch_normalization(x, axis=axis)

    return xla_batch_norm_job


def make_trt_job(input_shape, axis, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(True)

    @flow.global_function(config)
    def trt_batch_norm_job(x=flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.layers.batch_normalization(x, axis=axis)

    return trt_batch_norm_job


class TestRelu(unittest.TestCase):
    def _test_body(self, x, axis, dtype=np.float32):
        f1 = make_job(x.shape, axis, dtype=flow.float32)
        f2 = make_xla_job(x.shape, axis, dtype=flow.float32)
        a = f1(x).get()
        b = f2(x).get()
        print("without xla: ", a)
        print("with xla: ", b)
        self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()
        f3 = make_trt_job(x.shape, axis, dtype=flow.float32)
        c = f3(x).get()
        print("with tensorrt: ", c)
        self.assertTrue(np.allclose(a.ndarray(), c.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, shape, axis, dtype=np.float32):
        x = np.ones(shape, dtype=dtype)
        self._test_body(x, axis, dtype=dtype)

    def _test_random_body(self, shape, axis, dtype=np.float32):
        x = np.random.random(shape).astype(dtype)
        self._test_body(x, axis, dtype=dtype)

    """
      TensorRT batch norm only support 4-d tensor (NCHW).
    """

    def test_ones_input(self):
        self._test_ones_body((2, 1, 2, 2), 1)
        self._test_ones_body((2, 5, 2, 2), 1)

    def test_random_input(self):
        self._test_random_body((2, 1, 2, 2), 1)
        self._test_random_body((2, 5, 2, 2), 1)


if __name__ == "__main__":
    unittest.main()
