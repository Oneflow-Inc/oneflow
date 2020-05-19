import unittest
import numpy as np

import oneflow as flow

config = flow.function_config()

def make_job(shape, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.function(config)
    def tanh_grad_job(y = flow.FixedTensorDef(shape, dtype=dtype),
                      dy = flow.FixedTensorDef(shape, dtype=dtype)):
        return flow.keras.activations.tanh_grad(y, dy)
    return tanh_grad_job

def make_xla_job(shape, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.function(config)
    def xla_tanh_grad_job(y = flow.FixedTensorDef(shape, dtype=dtype),
                          dy = flow.FixedTensorDef(shape, dtype=dtype)):
        return flow.keras.activations.tanh_grad(y, dy)
    return xla_tanh_grad_job

class TestTanhGrad(unittest.TestCase):
    def _test_body(self, y, dy, dtype=np.float32):
        f1 = make_job(y.shape, dtype=flow.float32)
        f2 = make_xla_job(y.shape, dtype=flow.float32)
        a = f1(y, dy).get()
        b = f2(y, dy).get()
        print("without xla: ", a)
        print("with xla", b)
        self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, shape, dtype=np.float32):
        y = np.ones(shape, dtype=dtype)
        dy = np.ones(shape, dtype=dtype)
        self._test_body(y, dy, dtype=dtype)

    def _test_random_body(self, shape, dtype=np.float32):
        y = np.random.random(shape).astype(dtype)
        dy = np.random.random(shape).astype(dtype)
        self._test_body(y, dy, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((1))
        self._test_ones_body((1, 10))
        self._test_ones_body((2, 10, 2))
        self._test_ones_body((2, 5, 2, 2))

    def test_random_input(self):
        self._test_random_body((1))
        self._test_random_body((1, 10))
        self._test_random_body((2, 10, 2))
        self._test_random_body((2, 5, 2, 2))

if __name__ == '__main__':
    unittest.main()
