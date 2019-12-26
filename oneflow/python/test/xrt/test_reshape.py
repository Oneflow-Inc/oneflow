import unittest
import numpy as np

import oneflow as flow

config = flow.function_config()

def make_job(x_shape, shape, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.function(config)
    def reshape_job(x = flow.FixedTensorDef(x_shape, dtype=dtype)):
        return flow.reshape(x, shape)
    return reshape_job

def make_xla_job(x_shape, shape, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.function(config)
    def xla_reshape_job(x = flow.FixedTensorDef(x_shape, dtype=dtype)):
        return flow.reshape(x, shape)
    return xla_reshape_job

def make_trt_job(x_shape, shape, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(True)

    @flow.function(config)
    def trt_reshape_job(x = flow.FixedTensorDef(x_shape, dtype=dtype)):
        return flow.reshape(x, shape)
    return trt_reshape_job

class TestReshape(unittest.TestCase):
    def _test_body(self, x, shape, dtype=np.float32):
        f1 = make_job(x.shape, shape, dtype=flow.float32)
        f2 = make_xla_job(x.shape, shape, dtype=flow.float32)
        a = f1(x).get()
        b = f2(x).get()
        print("without xla: ", a)
        print("with xla: ", b)
        self.assertTrue(a.shape == b.shape)
        self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

        f3 = make_trt_job(x.shape, shape, dtype=flow.float32)
        c = f3(x).get()
        print("with tensorrt: ", c)
        self.assertTrue(a.shape == c.shape)
        self.assertTrue(np.allclose(a.ndarray(), c.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, x_shape, shape, dtype=np.float32):
        x = np.ones(x_shape, dtype=dtype)
        self._test_body(x, shape, dtype=dtype)

    def _test_random_body(self, x_shape, shape, dtype=np.float32):
        x = np.random.random(x_shape).astype(dtype)
        self._test_body(x, shape, dtype=dtype)

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
