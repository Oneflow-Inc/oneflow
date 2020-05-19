import unittest
import numpy as np

import oneflow as flow

config = flow.function_config()

def make_job(x_shape, b_shape, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(False)

    @flow.function(config)
    def bias_add_job(x = flow.FixedTensorDef(x_shape, dtype=dtype),
                     bias = flow.FixedTensorDef(b_shape, dtype=dtype)):
        return flow.nn.bias_add(x, bias)
    return bias_add_job

def make_xla_job(x_shape, b_shape, dtype=flow.float32):
    config.use_xla_jit(True)
    config.use_tensorrt(False)

    @flow.function(config)
    def xla_bias_add_job(x = flow.FixedTensorDef(x_shape, dtype=dtype),
                         bias = flow.FixedTensorDef(b_shape, dtype=dtype)):
        return flow.nn.bias_add(x, bias)
    return xla_bias_add_job

def make_trt_job(x_shape, b_shape, dtype=flow.float32):
    config.use_xla_jit(False)
    config.use_tensorrt(True)

    @flow.function(config)
    def trt_bias_add_job(x = flow.FixedTensorDef(x_shape, dtype=dtype),
                         bias = flow.FixedTensorDef(b_shape, dtype=dtype)):
        return flow.nn.bias_add(x, bias)
    return trt_bias_add_job

class TestBiasAdd(unittest.TestCase):
    def _test_body(self, x, bias, dtype=np.float32):
        f1 = make_job(x.shape, bias.shape, dtype=flow.float32)
        f2 = make_xla_job(x.shape, bias.shape, dtype=flow.float32)
        a = f1(x, bias).get()
        b = f2(x, bias).get()
        print("without xla: ", a)
        print("with xla: ", b)
        self.assertTrue(np.allclose(a.ndarray(), b.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

        f3 = make_trt_job(x.shape, bias.shape, dtype=flow.float32)
        c = f3(x, bias).get()
        print("with tensorrt: ", c)
        self.assertTrue(np.allclose(a.ndarray(), c.ndarray(), rtol=1e-03, atol=1e-05))
        flow.clear_default_session()

    def _test_ones_body(self, x_shape, bias_shape, dtype=np.float32):
        x = np.ones(x_shape, dtype=dtype)
        b = np.ones(bias_shape, dtype=dtype)
        self._test_body(x, b, dtype=dtype)

    def _test_random_body(self, x_shape, bias_shape, dtype=np.float32):
        x = np.random.random(x_shape).astype(dtype)
        b = np.random.random(bias_shape).astype(dtype)
        self._test_body(x, b, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((1, 10), (10))
        self._test_ones_body((2, 10, 2), (10))
        self._test_ones_body((2, 5, 2, 2), (5))

    def test_random_input(self):
        self._test_random_body((1, 10), (10))
        self._test_random_body((2, 10, 2), (10))
        self._test_random_body((2, 5, 2, 2), (5))

if __name__ == '__main__':
    unittest.main()
