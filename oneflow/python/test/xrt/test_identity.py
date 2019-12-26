import unittest
import numpy as np

import oneflow as flow

config = flow.function_config()

def make_job(input_shape, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.function(config)
    def identity_job(x = flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.identity(x)
    return identity_job

def make_xla_job(input_shape, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.function(config)
    def xla_identity_job(x = flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.identity(x)
    return xla_identity_job

def make_trt_job(input_shape, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(True)

    @flow.function(config)
    def trt_identity_job(x = flow.FixedTensorDef(input_shape, dtype=dtype)):
        return flow.identity(x)
    return trt_identity_job

class TestIdentity(unittest.TestCase):
    def _test_body(self, x, dtype=np.float32):
        f1 = make_job(x.shape, dtype=flow.float32)
        f2 = make_xla_job(x.shape, dtype=flow.float32)
        f3 = make_trt_job(x.shape, dtype=flow.float32)         
        a = f1(x).get()
        b = f2(x).get()
        c = f3(x).get()
        print("without xla: ", a)
        print("with xla: ", b)
        print("with tensorrt: ", c)
        self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        self.assertTrue(np.allclose(a.ndarray(), c.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, shape, dtype=np.float32):
        x = np.ones(shape, dtype=dtype)
        self._test_body(x, dtype=dtype)

    def _test_random_body(self, shape, dtype=np.float32):
        x = np.random.random(shape).astype(dtype)
        self._test_body(x, dtype=dtype)

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
