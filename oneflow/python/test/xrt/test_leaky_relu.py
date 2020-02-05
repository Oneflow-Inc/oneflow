import unittest
import numpy as np

import oneflow as flow

config = flow.function_config()

def make_job(input_shape, alpha, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.function(config)
    def leaky_relu_job(x = flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.math.leaky_relu(x, alpha=alpha)
    return leaky_relu_job

def make_trt_job(input_shape, alpha, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(True)

    @flow.function(config)
    def trt_leaky_relu_job(x = flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.math.leaky_relu(x, alpha=alpha)
    return trt_leaky_relu_job

class TestLeakyRelu(unittest.TestCase):
    def _test_body(self, x, alpha, dtype=np.float32):
        f1 = make_job(x.shape, alpha, dtype=flow.float32)
        f2 = make_trt_job(x.shape, alpha, dtype=flow.float32)
        a = f1(x).get()
        b = f2(x).get()
        print("oneflow: ", a)
        print("oneflow with tensorrt: ", b)
        self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, shape, alpha=0.1, dtype=np.float32):
        x = np.ones(shape, dtype=dtype)
        self._test_body(x, alpha, dtype=dtype)

    def _test_random_body(self, shape, alpha=0.1, dtype=np.float32):
        # np.random.random generates float range from 0 to 1.
        x = 100 * (np.random.random(shape).astype(dtype) - 0.5)
        self._test_body(x, alpha, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((1), alpha=0.1)
        self._test_ones_body((1, 10), alpha=0.1)
        self._test_ones_body((2, 10, 2), alpha=0.1)
        self._test_ones_body((2, 5, 2, 2), alpha=0.1)

        self._test_ones_body((1), alpha=0.33)
        self._test_ones_body((1, 10), alpha=0.33)
        self._test_ones_body((2, 10, 2), alpha=0.33)
        self._test_ones_body((2, 5, 2, 2), alpha=0.33)

    def test_random_input(self):
        self._test_random_body((1), alpha=0.1)
        self._test_random_body((1, 10), alpha=0.1)
        self._test_random_body((2, 10, 2), alpha=0.1)
        self._test_random_body((2, 5, 2, 2), alpha=0.1)

        self._test_random_body((1), alpha=0.33)
        self._test_random_body((1, 10), alpha=0.33)
        self._test_random_body((2, 10, 2), alpha=0.33)
        self._test_random_body((2, 5, 2, 2), alpha=0.33)


if __name__ == '__main__':
    unittest.main()
